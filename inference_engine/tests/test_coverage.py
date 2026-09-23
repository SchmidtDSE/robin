"""What the engine finished, and whether it answers the work it claims to answer."""

import numpy as np
import pytest
from pydantic import ValidationError

from robin_contracts.cards import ModelRef
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.records import ClassScore
from robin_contracts.results import ArtifactRecord, InferenceSuccess
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled, WindowGeometry
from robin_contracts.work import (
    AudioInput,
    FileDigest,
    InferenceWork,
    ModelSelection,
    RecordingRef,
    work_digest,
)
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.coverage import CoverageBuilder, check_completion_evidence

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
RECORD_DIGEST = f"sha256:v1:{HEX}"

MODEL_REF = ModelRef(name="owl", version="1", digest=RECORD_DIGEST)
GEOMETRY = WindowGeometry(window=3.0, hop=3.0, pad="drop")


def build_window(**overrides) -> AcceptedWindow:
    # A window carries a score by default so that only the case naming the empty
    # window depends on what an empty one is counted as.
    fields = {
        "recording_index": 0,
        "start": 0.0,
        "end": 3.0,
        "scores": (ClassScore("gull", 0.9),),
        "embedding": None,
    }
    return AcceptedWindow(**(fields | overrides))


def build_scores_request(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_work(indices=(0,), outputs=None) -> InferenceWork:
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=tuple(
            RecordingRef(
                index=index, namespace="soundhub", value=str(index), audio_uri=f"s3://b/{index}.wav"
            )
            for index in indices
        ),
        model=ModelSelection(
            ref=MODEL_REF,
            files=(
                FileDigest(
                    role="weights", uri="s3://b/owl.tflite", digest=FILE_DIGEST, size_bytes=8
                ),
            ),
        ),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=outputs if outputs is not None else (build_scores_request(),),
    )


def build_recipe() -> Recipe:
    return Recipe(
        model=MODEL_REF,
        backend="tflite",
        audio=AudioSpec(
            sample_rate=48000,
            window=3.0,
            hop=3.0,
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="drop",
        ),
        embedding_transform={"kind": "identity"},
        dtype="float32",
    )


def build_artifact(**overrides) -> ArtifactRecord:
    fields = {
        "kind": "scores",
        "contract_id": "robin.scores.arrow/1",
        "uri": "s3://bucket/scores.arrow",
        "checksum": FILE_DIGEST,
        "size_bytes": 64,
        "rows": 0,
    }
    return ArtifactRecord(**(fields | overrides))


def build_map() -> ArtifactRecord:
    return build_artifact(
        kind="recording_map",
        contract_id="robin.recording-map.json/1",
        uri="s3://bucket/map.json",
        rows=1,
    )


def cover(work: InferenceWork, **per_recording):
    """One coverage row per recording, each with one completed window."""
    builder = CoverageBuilder([recording.index for recording in work.recordings])
    for recording in work.recordings:
        builder.begin_recording(recording.index)
        builder.record(build_window(recording_index=recording.index, **per_recording))
        builder.end_recording()
    return builder.build()


def cover_nothing(work: InferenceWork):
    """One coverage row per recording, each declaring that it completed no window."""
    builder = CoverageBuilder([recording.index for recording in work.recordings])
    for recording in work.recordings:
        builder.begin_recording(recording.index)
        builder.end_recording(zero_window_reason="shorter_than_window")
    return builder.build()


def build_success(work: InferenceWork, **overrides) -> InferenceSuccess:
    coverage = overrides.pop("coverage") if "coverage" in overrides else cover(work)
    fields = {
        "schema_version": "robin.inference-result/1",
        "work_digest": work_digest(work),
        "recipe": build_recipe(),
        "model": work.model,
        "window_geometry": GEOMETRY,
        "artifacts": (build_artifact(rows=sum(row.score_rows for row in coverage)),),
        "recording_map": build_map(),
        "coverage": coverage,
        "resolved_scores_request": build_scores_request(),
    }
    return InferenceSuccess(**(fields | overrides))


# --- CoverageBuilder --------------------------------------------------------


def test_success_factory_preserves_empty_coverage():
    with pytest.raises(ValidationError):
        build_success(build_work(), coverage=())


def test_begin_recording_refuses_to_discard_unfinished_coverage():
    builder = CoverageBuilder([0, 1])
    builder.begin_recording(0)
    builder.record(build_window())

    with pytest.raises(RuntimeError):
        builder.begin_recording(1)

    builder.end_recording()
    builder.begin_recording(1)
    builder.end_recording(zero_window_reason="shorter_than_window")
    first, second = builder.build()

    assert first.recording_index == 0
    assert first.windows_completed == 1
    assert first.score_rows == 1
    assert second.recording_index == 1
    assert second.windows_completed == 0


def test_a_recording_with_no_windows_needs_a_reason():
    builder = CoverageBuilder([0])
    builder.begin_recording(0)

    with pytest.raises(errors.EngineError) as raised:
        builder.end_recording()

    assert raised.value.code == errors.UNEXPLAINED_ZERO_WINDOWS
    assert raised.value.stage == errors.INFER
    assert raised.value.recording_index == 0

    explained = CoverageBuilder([0])
    explained.begin_recording(0)
    explained.end_recording(zero_window_reason="shorter_than_window")
    (row,) = explained.build()

    assert row.windows_completed == 0
    assert row.zero_window_reason == "shorter_than_window"
    assert row.first_window_start_s is None
    assert row.last_window_end_s is None


def test_a_completed_window_is_counted_even_with_no_scores():
    builder = CoverageBuilder([0])
    builder.begin_recording(0)
    builder.record(build_window(scores=()))
    builder.end_recording()
    (row,) = builder.build()

    assert row.windows_completed == 1
    assert row.score_rows == 0
    assert row.embedding_rows == 0
    assert row.zero_window_reason is None


def test_score_rows_count_every_accepted_score():
    builder = CoverageBuilder([0])
    builder.begin_recording(0)
    builder.record(build_window(scores=(ClassScore("gull", 0.9), ClassScore("tern", 0.1))))
    builder.record(build_window(start=3.0, end=6.0, scores=(ClassScore("gull", 0.4),)))
    builder.end_recording()
    (row,) = builder.build()

    assert row.windows_completed == 2
    assert row.score_rows == 3


def test_embedding_rows_count_only_the_windows_carrying_one():
    builder = CoverageBuilder([0])
    builder.begin_recording(0)
    builder.record(build_window(embedding=np.zeros(4, dtype=np.float32)))
    builder.record(build_window(start=3.0, end=6.0))
    builder.end_recording()
    (row,) = builder.build()

    assert row.windows_completed == 2
    assert row.embedding_rows == 1


def test_bounds_span_the_first_start_and_the_greatest_end():
    builder = CoverageBuilder([0])
    builder.begin_recording(0)
    builder.record(build_window(start=0.0, end=5.0))
    builder.record(build_window(start=3.0, end=8.0))
    builder.record(build_window(start=6.0, end=7.0))
    builder.end_recording()
    (row,) = builder.build()

    assert row.first_window_start_s == 0.0
    assert row.last_window_end_s == 8.0


def test_rows_are_built_in_index_order_whatever_the_processing_order():
    builder = CoverageBuilder([0, 2, 5])
    for index in (5, 0, 2):
        builder.begin_recording(index)
        builder.record(build_window(recording_index=index))
        builder.end_recording()

    assert [row.recording_index for row in builder.build()] == [0, 2, 5]


def test_a_recording_is_begun_once_and_must_be_one_of_the_works():
    builder = CoverageBuilder([0])

    with pytest.raises(RuntimeError):
        builder.begin_recording(1)

    builder.begin_recording(0)
    builder.record(build_window())
    builder.end_recording()

    with pytest.raises(RuntimeError):
        builder.begin_recording(0)


def test_build_refuses_a_recording_that_was_never_finished():
    never_begun = CoverageBuilder([0, 1])
    never_begun.begin_recording(0)
    never_begun.record(build_window())
    never_begun.end_recording()

    with pytest.raises(RuntimeError):
        never_begun.build()

    left_open = CoverageBuilder([0])
    left_open.begin_recording(0)
    left_open.record(build_window())

    with pytest.raises(RuntimeError):
        left_open.build()


def test_a_reason_on_a_completed_recording_is_a_defect():
    builder = CoverageBuilder([0])
    builder.begin_recording(0)
    builder.record(build_window())

    with pytest.raises(RuntimeError, match="supplied zero-window reason.*shorter_than_window"):
        builder.end_recording(zero_window_reason="shorter_than_window")


def test_a_window_with_no_recording_open_is_a_defect():
    builder = CoverageBuilder([0])

    with pytest.raises(RuntimeError):
        builder.record(build_window())

    builder.begin_recording(0)
    builder.record(build_window())
    builder.end_recording()

    with pytest.raises(RuntimeError):
        builder.record(build_window(start=3.0, end=6.0))


def test_a_window_naming_another_recording_is_a_defect():
    builder = CoverageBuilder([0, 1])
    builder.begin_recording(0)

    with pytest.raises(RuntimeError):
        builder.record(build_window(recording_index=1))


# --- check_completion_evidence ----------------------------------------------


def test_coverage_must_name_exactly_the_works_recordings():
    work = build_work(indices=(0, 1))
    two_recordings = build_work(indices=(0, 1, 2))

    extra = build_success(work, coverage=cover(two_recordings))
    with pytest.raises(RuntimeError):
        check_completion_evidence(work, extra)

    missing = build_success(work, coverage=cover(build_work(indices=(0,))))
    with pytest.raises(RuntimeError):
        check_completion_evidence(work, missing)


def test_every_requested_kind_has_an_artifact_and_no_other_does():
    work = build_work()

    with pytest.raises(RuntimeError):
        check_completion_evidence(
            work,
            build_success(
                work,
                coverage=cover_nothing(work),
                artifacts=(),
                resolved_scores_request=None,
            ),
        )

    detections = build_artifact(kind="detections", contract_id="robin.detections.parquet/1")
    unrequested = build_success(
        work,
        coverage=cover_nothing(work),
        artifacts=(build_artifact(), detections),
        resolved_detection_policy=ThresholdPolicy(min_score=0.5),
    )
    with pytest.raises(RuntimeError):
        check_completion_evidence(work, unrequested)

    embeddings_only = build_work(
        outputs=(EmbeddingsRequest(contract_id="robin.embeddings.arrow/1"),)
    )
    check_completion_evidence(
        embeddings_only,
        build_success(
            embeddings_only,
            coverage=cover_nothing(embeddings_only),
            artifacts=(
                build_artifact(kind="embeddings", contract_id="robin.embeddings.arrow/1"),
            ),
            resolved_scores_request=None,
        ),
    )


def test_the_result_must_carry_this_works_digest():
    work = build_work()
    other = build_work(indices=(0, 1))

    success = build_success(work, work_digest=work_digest(other))

    with pytest.raises(RuntimeError):
        check_completion_evidence(work, success)


def test_a_consistent_result_passes():
    work = build_work(
        indices=(0, 1),
        outputs=(
            build_scores_request(),
            DetectionsRequest(
                contract_id="robin.detections.parquet/1",
                policy=ThresholdPolicy(min_score=0.5),
            ),
        ),
    )
    coverage = cover(work)
    success = build_success(
        work,
        coverage=coverage,
        artifacts=(
            build_artifact(rows=sum(row.score_rows for row in coverage)),
            build_artifact(kind="detections", contract_id="robin.detections.parquet/1"),
        ),
        resolved_detection_policy=ThresholdPolicy(min_score=0.5),
    )

    assert check_completion_evidence(work, success) is None
