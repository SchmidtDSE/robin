"""What the engine finished, and whether it answers the work it claims to answer."""

import re

import numpy as np
import pytest
from pydantic import ValidationError

from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled, model_ref
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.records import ClassScore
from robin_contracts.results import ArtifactRecord, InferenceSuccess
from robin_contracts.specs import AudioSpec, Recipe, WindowGeometry
from robin_contracts.work import (
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
    work_digest,
)
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.coverage import CoverageBuilder, check_completion_evidence

REGISTRY_DIGEST = "sha256:" + "a" * 64

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
RECORD_DIGEST = f"sha256:v1:{HEX}"

CARD = ModelCard(
    model_name="owl",
    model_version="1",
    runtime="tensorflow",
    window_duration=3.0,
    sample_rate=48000,
    min_detection_threshold=0.0,
    score_domain="sigmoid",
    taxa_registry_digest=REGISTRY_DIGEST,
    audio=AudioGeometry(
        downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
    ),
    backend="tensorflow",
    dtype="float32",
)
GEOMETRY = WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="drop")


def build_window(**overrides) -> AcceptedWindow:
    # A window carries a score by default so that only the case naming the empty
    # window depends on what an empty one is counted as.
    fields = {
        "recording": RecordingRef(namespace="soundhub", value="0", audio_uri="s3://b/0.wav"),
        "start": 0.0,
        "end": 3.0,
        "scores": (ClassScore("gull", 0.9),),
        "embedding": None,
    }
    return AcceptedWindow(**(fields | overrides))


def build_scores_request(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.parquet/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_recordings(values=("0",)) -> tuple[RecordingRef, ...]:
    return tuple(
        RecordingRef(namespace="soundhub", value=value, audio_uri=f"s3://b/{value}.wav")
        for value in values
    )


def build_work(values=("0",), outputs=None) -> InferenceWork:
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=build_recordings(values),
        model=PinnedModel(
            card=CARD,
            files={"weights": PinnedFile(uri="s3://b/owl.tflite", digest=FILE_DIGEST, size_bytes=8)},
        ),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=outputs if outputs is not None else (build_scores_request(),),
    )


def build_recipe() -> Recipe:
    return Recipe(
        model=model_ref(CARD),
        backend="tflite",
        audio=AudioSpec(
            sample_rate=48000,
            window_duration=3.0,
            window_overlap=0.0,
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="drop",
        ),
        dtype="float32",
        settings={},
    )


def build_artifact(**overrides) -> ArtifactRecord:
    fields = {
        "kind": "scores",
        "contract_id": "robin.scores.parquet/1",
        "namespace": "soundhub",
        "value": "0",
        "uri": "s3://bucket/scores.parquet",
        "checksum": FILE_DIGEST,
        "size_bytes": 64,
        "rows": 1,
    }
    return ArtifactRecord(**(fields | overrides))


def cover(recordings, detection_rows=0, **per_recording):
    """One coverage row per recording, each with one completed window."""
    builder = CoverageBuilder(recordings)
    for position, recording in enumerate(recordings):
        builder.begin_recording(position)
        builder.record(build_window(recording=recording, **per_recording))
        builder.end_recording(detection_rows=detection_rows)
    return builder.build()


def cover_nothing(recordings):
    """One coverage row per recording, each declaring that it completed no window."""
    builder = CoverageBuilder(recordings)
    for position in range(len(recordings)):
        builder.begin_recording(position)
        builder.end_recording(zero_window_reason="shorter_than_window")
    return builder.build()


def scores_records(coverage) -> tuple[ArtifactRecord, ...]:
    """One scores record for each recording that counted score rows."""
    return tuple(
        build_artifact(value=row.value, rows=row.score_rows)
        for row in coverage
        if row.score_rows
    )


def build_success(work: InferenceWork, **overrides) -> InferenceSuccess:
    coverage = overrides.pop("coverage") if "coverage" in overrides else cover(work.recordings)
    fields = {
        "schema_version": "robin.inference-result/1",
        "work_digest": work_digest(work),
        "recipe": build_recipe(),
        "model": work.model,
        "window_geometry": GEOMETRY,
        "artifacts": scores_records(coverage),
        "coverage": coverage,
        "resolved_scores_request": build_scores_request(),
    }
    return InferenceSuccess(**(fields | overrides))


# --- CoverageBuilder --------------------------------------------------------

ONE = build_recordings(("0",))
TWO = build_recordings(("0", "1"))


def test_success_factory_preserves_empty_coverage():
    with pytest.raises(ValidationError):
        build_success(build_work(), coverage=())


def test_begin_recording_refuses_to_discard_unfinished_coverage():
    builder = CoverageBuilder(TWO)
    builder.begin_recording(0)
    builder.record(build_window())

    with pytest.raises(RuntimeError):
        builder.begin_recording(1)

    builder.end_recording()
    builder.begin_recording(1)
    builder.end_recording(zero_window_reason="shorter_than_window")
    first, second = builder.build()

    assert (first.namespace, first.value) == (TWO[0].namespace, TWO[0].value)
    assert first.windows_completed == 1
    assert first.score_rows == 1
    assert (second.namespace, second.value) == (TWO[1].namespace, TWO[1].value)
    assert second.windows_completed == 0


def test_a_recording_with_no_windows_needs_a_reason():
    builder = CoverageBuilder(ONE)
    builder.begin_recording(0)

    with pytest.raises(errors.EngineError) as raised:
        builder.end_recording()

    assert raised.value.code == errors.UNEXPLAINED_ZERO_WINDOWS
    assert raised.value.stage == errors.INFER
    assert raised.value.recording is ONE[0]

    explained = CoverageBuilder(ONE)
    explained.begin_recording(0)
    explained.end_recording(zero_window_reason="shorter_than_window")
    (row,) = explained.build()

    assert row.windows_completed == 0
    assert row.zero_window_reason == "shorter_than_window"
    assert row.first_window_start_s is None
    assert row.last_window_end_s is None


def test_a_completed_window_is_counted_even_with_no_scores():
    builder = CoverageBuilder(ONE)
    builder.begin_recording(0)
    builder.record(build_window(scores=()))
    builder.end_recording()
    (row,) = builder.build()

    assert row.windows_completed == 1
    assert row.score_rows == 0
    assert row.embedding_rows == 0
    assert row.zero_window_reason is None


def test_score_rows_count_every_accepted_score():
    builder = CoverageBuilder(ONE)
    builder.begin_recording(0)
    builder.record(build_window(scores=(ClassScore("gull", 0.9), ClassScore("tern", 0.1))))
    builder.record(build_window(start=3.0, end=6.0, scores=(ClassScore("gull", 0.4),)))
    builder.end_recording()
    (row,) = builder.build()

    assert row.windows_completed == 2
    assert row.score_rows == 3


def test_embedding_rows_count_only_the_windows_carrying_one():
    builder = CoverageBuilder(ONE)
    builder.begin_recording(0)
    builder.record(build_window(embedding=np.zeros(4, dtype=np.float32)))
    builder.record(build_window(start=3.0, end=6.0))
    builder.end_recording()
    (row,) = builder.build()

    assert row.windows_completed == 2
    assert row.embedding_rows == 1


def test_detection_rows_are_the_count_passed_when_the_recording_ends():
    builder = CoverageBuilder(TWO)
    builder.begin_recording(0)
    builder.record(build_window(scores=(ClassScore("gull", 0.9), ClassScore("tern", 0.1))))
    builder.end_recording(detection_rows=2)
    builder.begin_recording(1)
    builder.record(build_window(recording=TWO[1]))
    builder.end_recording()
    counted, defaulted = builder.build()

    assert counted.detection_rows == 2
    assert defaulted.detection_rows == 0


def test_bounds_span_the_first_start_and_the_greatest_end():
    builder = CoverageBuilder(ONE)
    builder.begin_recording(0)
    builder.record(build_window(start=0.0, end=5.0))
    builder.record(build_window(start=3.0, end=8.0))
    builder.record(build_window(start=6.0, end=7.0))
    builder.end_recording()
    (row,) = builder.build()

    assert row.first_window_start_s == 0.0
    assert row.last_window_end_s == 8.0


def test_rows_are_built_in_work_order_whatever_the_processing_order():
    recordings = build_recordings(("b", "a", "c"))
    builder = CoverageBuilder(recordings)
    for position in (2, 0, 1):
        builder.begin_recording(position)
        builder.record(build_window())
        builder.end_recording()

    assert [(row.namespace, row.value) for row in builder.build()] == [
        (one.namespace, one.value) for one in recordings
    ]


@pytest.mark.parametrize("position", [1, -1])
def test_a_recording_must_be_one_of_the_works(position):
    with pytest.raises(RuntimeError, match="not one of this work's"):
        CoverageBuilder(ONE).begin_recording(position)


def test_a_recording_is_begun_once():
    builder = CoverageBuilder(ONE)

    builder.begin_recording(0)
    builder.record(build_window())
    builder.end_recording()

    with pytest.raises(RuntimeError):
        builder.begin_recording(0)


def test_build_refuses_a_recording_that_was_never_finished():
    never_begun = CoverageBuilder(TWO)
    never_begun.begin_recording(0)
    never_begun.record(build_window())
    never_begun.end_recording()

    with pytest.raises(RuntimeError):
        never_begun.build()

    left_open = CoverageBuilder(ONE)
    left_open.begin_recording(0)
    left_open.record(build_window())

    with pytest.raises(RuntimeError):
        left_open.build()


def test_a_reason_on_a_completed_recording_is_a_defect():
    builder = CoverageBuilder(ONE)
    builder.begin_recording(0)
    builder.record(build_window())

    with pytest.raises(RuntimeError, match="supplied zero-window reason.*shorter_than_window"):
        builder.end_recording(zero_window_reason="shorter_than_window")


def test_a_window_with_no_recording_open_is_a_defect():
    builder = CoverageBuilder(ONE)

    with pytest.raises(RuntimeError):
        builder.record(build_window())

    builder.begin_recording(0)
    builder.record(build_window())
    builder.end_recording()

    with pytest.raises(RuntimeError):
        builder.record(build_window(start=3.0, end=6.0))


# --- check_completion_evidence ----------------------------------------------


def test_coverage_must_name_exactly_the_works_recordings():
    work = build_work(values=("0", "1"))

    extra = build_success(work, coverage=cover(build_recordings(("0", "1", "2"))))
    with pytest.raises(RuntimeError, match=re.escape("('soundhub', '2')")):
        check_completion_evidence(work, extra)

    missing = build_success(work, coverage=cover(build_recordings(("0",))))
    with pytest.raises(RuntimeError, match=re.escape("('soundhub', '1')")):
        check_completion_evidence(work, missing)


def test_coverage_must_follow_the_works_order():
    work = build_work(values=("0", "1"))
    reordered = build_success(work, coverage=cover(tuple(reversed(work.recordings))))

    with pytest.raises(RuntimeError, match="order"):
        check_completion_evidence(work, reordered)


def test_a_requested_kind_may_have_no_artifacts():
    work = build_work()

    success = build_success(work, coverage=cover_nothing(work.recordings), artifacts=())

    assert check_completion_evidence(work, success) is None


def test_no_artifact_has_a_kind_the_work_did_not_request():
    work = build_work()
    coverage = cover(work.recordings, detection_rows=1)

    detections = build_artifact(kind="detections", contract_id="robin.detections.parquet/1")
    unrequested = build_success(
        work,
        coverage=coverage,
        artifacts=(*scores_records(coverage), detections),
        resolved_detection_policy=ThresholdPolicy(min_score=0.5),
    )
    with pytest.raises(RuntimeError, match="detections"):
        check_completion_evidence(work, unrequested)

    embedded = cover(work.recordings, embedding=np.zeros(4, dtype=np.float32))
    embeddings = build_artifact(kind="embeddings", contract_id="robin.embeddings.parquet/1")
    with pytest.raises(RuntimeError, match="embeddings"):
        check_completion_evidence(
            work,
            build_success(
                work, coverage=embedded, artifacts=(*scores_records(embedded), embeddings)
            ),
        )

    embeddings_only = build_work(
        outputs=(EmbeddingsRequest(contract_id="robin.embeddings.parquet/1"),)
    )
    check_completion_evidence(
        embeddings_only,
        build_success(
            embeddings_only,
            coverage=cover(embeddings_only.recordings, scores=(), embedding=np.zeros(4)),
            artifacts=(embeddings,),
            resolved_scores_request=None,
        ),
    )


def test_the_result_must_carry_this_works_digest():
    work = build_work()
    other = build_work(values=("0", "1"))

    success = build_success(work, work_digest=work_digest(other))

    with pytest.raises(RuntimeError):
        check_completion_evidence(work, success)


def test_a_consistent_result_passes():
    work = build_work(
        values=("0", "1"),
        outputs=(
            build_scores_request(),
            DetectionsRequest(
                contract_id="robin.detections.parquet/1",
                policy=ThresholdPolicy(min_score=0.5),
            ),
        ),
    )
    coverage = cover(work.recordings, detection_rows=1)
    success = build_success(
        work,
        coverage=coverage,
        artifacts=(
            *scores_records(coverage),
            *(
                build_artifact(
                    kind="detections", contract_id="robin.detections.parquet/1", value=row.value
                )
                for row in coverage
            ),
        ),
        resolved_detection_policy=ThresholdPolicy(min_score=0.5),
    )

    assert check_completion_evidence(work, success) is None
