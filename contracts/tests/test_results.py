"""What the engine hands back: its artifacts, its completion evidence, its failure."""

import re

import pytest
from pydantic import TypeAdapter, ValidationError

from robin_contracts.canonical import canonical_json_bytes, sha256_v1
from robin_contracts.cards import ModelCard, model_ref
from robin_contracts.output_contracts import ScoresRequest, ThresholdPolicy
from robin_contracts.results import (
    ArtifactRecord,
    FailureReport,
    InferenceFailure,
    InferenceResult,
    InferenceSuccess,
    RecordingCoverage,
)
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled, WindowGeometry
from robin_contracts.work import PinnedFile, PinnedModel

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
RECORD_DIGEST = f"sha256:v1:{HEX}"

CARD = ModelCard(
    model_name="owl",
    model_version="1",
    runtime="tensorflow",
    segment_duration=3.0,
    sample_rate=48000,
    min_detection_threshold=0.0,
)
MODEL_REF = model_ref(CARD)
GEOMETRY = WindowGeometry(window=3.0, hop=3.0, pad="drop")

DECLARED_STAGES = (
    "validate_request",
    "resolve_model",
    "load_registry",
    "acquire_audio",
    "read_input_artifact",
    "construct_model",
    "infer",
    "accept_window",
    "write_artifact",
    "aggregate",
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


def build_pinned_model() -> PinnedModel:
    return PinnedModel(
        card=CARD,
        files={"weights": PinnedFile(uri="s3://b/owl.tflite", digest=FILE_DIGEST, size_bytes=8)},
    )


def build_artifact(**overrides) -> ArtifactRecord:
    fields = {
        "kind": "scores",
        "contract_id": "robin.scores.arrow/1",
        "uri": "s3://bucket/scores.arrow",
        "checksum": FILE_DIGEST,
        "size_bytes": 1024,
        "rows": 0,
    }
    return ArtifactRecord(**(fields | overrides))


def build_coverage(**overrides) -> RecordingCoverage:
    fields = {
        "namespace": "soundhub",
        "value": "42",
        "audio_digest": None,
        "windows_completed": 2,
        "score_rows": 0,
        "embedding_rows": 0,
        "first_window_start_s": 0.0,
        "last_window_end_s": 6.0,
    }
    return RecordingCoverage(**(fields | overrides))


def build_scores_request(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_success(**overrides) -> InferenceSuccess:
    fields = {
        "schema_version": "robin.inference-result/1",
        "work_digest": RECORD_DIGEST,
        "recipe": build_recipe(),
        "model": build_pinned_model(),
        "window_geometry": GEOMETRY,
        "artifacts": (),
        "coverage": (build_coverage(),),
    }
    return InferenceSuccess(**(fields | overrides))


def build_report(**overrides) -> FailureReport:
    fields = {
        "code": "unknown_label",
        "stage": "accept_window",
        "detail": "label 'gull' is not declared by this registry",
    }
    return FailureReport(**(fields | overrides))


def build_failure(**overrides) -> InferenceFailure:
    fields = {
        "schema_version": "robin.inference-result/1",
        "work_digest": RECORD_DIGEST,
        "failure": build_report(),
    }
    return InferenceFailure(**(fields | overrides))


# --- ArtifactRecord ---------------------------------------------------------


def test_an_artifact_record_round_trips():
    record = build_artifact(rows=7)

    assert ArtifactRecord(**record.model_dump(mode="json")) == record


@pytest.mark.parametrize("contract_id", ["robin.inference-work/1", "robin.scores.arrow/2"])
def test_an_artifact_record_refuses_an_unknown_contract_id(contract_id):
    with pytest.raises(ValidationError):
        build_artifact(contract_id=contract_id)


@pytest.mark.parametrize("checksum", [RECORD_DIGEST, HEX])
def test_an_artifact_checksum_refuses_the_canonical_family(checksum):
    with pytest.raises(ValidationError):
        build_artifact(checksum=checksum)


def test_zero_rows_is_a_real_value():
    assert build_artifact(rows=0).rows == 0


@pytest.mark.parametrize("overrides", [{"rows": -1}, {"size_bytes": -1}])
def test_negative_rows_or_size_are_refused(overrides):
    with pytest.raises(ValidationError):
        build_artifact(**overrides)


# --- RecordingCoverage ------------------------------------------------------


def test_zero_windows_needs_a_declared_reason():
    with pytest.raises(ValidationError):
        build_coverage(windows_completed=0, first_window_start_s=None, last_window_end_s=None)

    row = build_coverage(
        windows_completed=0,
        first_window_start_s=None,
        last_window_end_s=None,
        zero_window_reason="shorter_than_window",
    )

    assert row.first_window_start_s is None
    assert row.last_window_end_s is None


def test_a_completed_recording_refuses_a_zero_window_reason():
    with pytest.raises(ValidationError):
        build_coverage(zero_window_reason="shorter_than_window")


@pytest.mark.parametrize(
    ("windows_completed", "first", "last", "reason", "accepted"),
    [
        (0, None, None, "no_input_windows", True),
        (0, 0.0, 3.0, "no_input_windows", False),
        (2, None, None, None, False),
        (2, 0.0, 6.0, None, True),
    ],
)
def test_window_bounds_are_null_exactly_when_no_window_completed(
    windows_completed, first, last, reason, accepted
):
    fields = {
        "windows_completed": windows_completed,
        "first_window_start_s": first,
        "last_window_end_s": last,
        "zero_window_reason": reason,
    }

    if accepted:
        assert build_coverage(**fields).windows_completed == windows_completed
        return
    with pytest.raises(ValidationError):
        build_coverage(**fields)


@pytest.mark.parametrize("overrides", [{"score_rows": 1}, {"embedding_rows": 1}])
def test_a_recording_with_no_windows_carries_no_rows(overrides):
    with pytest.raises(ValidationError):
        build_coverage(
            windows_completed=0,
            first_window_start_s=None,
            last_window_end_s=None,
            zero_window_reason="shorter_than_window",
            **overrides,
        )


def test_embedding_rows_never_exceed_completed_windows():
    assert build_coverage(windows_completed=2, embedding_rows=2).embedding_rows == 2

    with pytest.raises(ValidationError):
        build_coverage(windows_completed=2, embedding_rows=3)


@pytest.mark.parametrize(
    ("first", "last"),
    [
        (float("nan"), 6.0),
        (0.0, float("inf")),
        (-0.5, 6.0),
        (3.0, 3.0),
        (3.0, 1.0),
    ],
)
def test_window_bounds_must_be_finite_and_ordered(first, last):
    with pytest.raises(ValidationError):
        build_coverage(first_window_start_s=first, last_window_end_s=last)


@pytest.mark.parametrize(
    "overrides",
    [
        {"windows_completed": -1},
        {"score_rows": -1},
        {"embedding_rows": -1},
    ],
)
def test_negative_counts_are_refused(overrides):
    with pytest.raises(ValidationError):
        build_coverage(**overrides)


# --- InferenceSuccess -------------------------------------------------------


def test_a_minimal_success_round_trips():
    success = build_success()

    rebuilt = TypeAdapter(InferenceResult).validate_python(success.model_dump(mode="json"))

    assert rebuilt == success
    assert isinstance(rebuilt.coverage, tuple)
    assert isinstance(rebuilt.artifacts, tuple)


@pytest.mark.parametrize(
    "coverage",
    [
        pytest.param((), id="empty"),
        pytest.param(
            (build_coverage(value="42"), build_coverage(value="42")),
            id="repeated",
        ),
    ],
)
def test_coverage_rows_are_present_and_name_each_recording_once(coverage):
    with pytest.raises(ValidationError):
        build_success(coverage=coverage)


def test_coverage_order_is_left_to_the_work():
    # The result does not hold the work, so it cannot know the work's order.
    coverage = (build_coverage(value="43"), build_coverage(value="42"))

    assert [row.value for row in build_success(coverage=coverage).coverage] == [
        "43",
        "42",
    ]


def test_the_same_value_in_two_namespaces_is_two_coverage_rows():
    coverage = (
        build_coverage(namespace="soundhub", value="42"),
        build_coverage(namespace="arbimon", value="42"),
    )

    assert len(build_success(coverage=coverage).coverage) == 2


@pytest.mark.parametrize("digest", [None, FILE_DIGEST])
def test_coverage_keeps_the_audio_digest_exactly_null_included(digest):
    row = build_coverage(audio_digest=digest)

    assert RecordingCoverage.model_validate(row.model_dump(mode="json")).audio_digest == digest
    assert row.model_dump(mode="json")["audio_digest"] == digest


def test_coverage_audio_digest_refuses_the_canonical_family():
    with pytest.raises(ValidationError):
        build_coverage(audio_digest=RECORD_DIGEST)


def test_a_coverage_refusal_names_the_recording():
    with pytest.raises(ValidationError, match=re.escape("('soundhub', '42')")):
        build_coverage(
            windows_completed=0,
            first_window_start_s=None,
            last_window_end_s=None,
            zero_window_reason="shorter_than_window",
            score_rows=1,
        )


def test_score_rows_sum_to_the_scores_artifact_rows():
    scores = build_artifact(rows=5)
    coverage = (
        build_coverage(value="42", score_rows=2),
        build_coverage(value="43", score_rows=3),
    )

    success = build_success(
        artifacts=(scores,),
        coverage=coverage,
        resolved_scores_request=build_scores_request(),
    )

    assert success.artifacts[0].rows == 5

    with pytest.raises(ValidationError):
        build_success(
            artifacts=(build_artifact(rows=6),),
            coverage=coverage,
            resolved_scores_request=build_scores_request(),
        )


def test_embedding_rows_sum_to_the_embeddings_artifact_rows():
    embeddings = build_artifact(
        kind="embeddings", contract_id="robin.embeddings.arrow/1", rows=3
    )
    coverage = (
        build_coverage(value="42", embedding_rows=2),
        build_coverage(value="43", embedding_rows=1),
    )

    assert build_success(artifacts=(embeddings,), coverage=coverage).artifacts[0].rows == 3

    with pytest.raises(ValidationError):
        build_success(
            artifacts=(
                build_artifact(kind="embeddings", contract_id="robin.embeddings.arrow/1", rows=4),
            ),
            coverage=coverage,
        )


def test_a_kind_with_no_artifact_carries_no_rows():
    assert build_success(coverage=(build_coverage(embedding_rows=0),)).artifacts == ()

    with pytest.raises(ValidationError):
        build_success(coverage=(build_coverage(embedding_rows=2),))


def test_a_requested_kind_with_zero_rows_still_gets_an_artifact():
    success = build_success(
        artifacts=(build_artifact(rows=0),),
        coverage=(build_coverage(score_rows=0),),
        resolved_scores_request=build_scores_request(),
    )

    assert success.artifacts[0].rows == 0


def test_an_artifact_kind_appears_at_most_once():
    with pytest.raises(ValidationError):
        build_success(
            artifacts=(build_artifact(rows=0), build_artifact(rows=0)),
            resolved_scores_request=build_scores_request(),
        )


def test_a_resolved_request_accompanies_its_artifact():
    detections = build_artifact(kind="detections", contract_id="robin.detections.parquet/1")
    policy = ThresholdPolicy(min_score=0.5)

    with pytest.raises(ValidationError):
        build_success(artifacts=(build_artifact(),))

    with pytest.raises(ValidationError):
        build_success(resolved_scores_request=build_scores_request())

    with pytest.raises(ValidationError):
        build_success(artifacts=(detections,))

    with pytest.raises(ValidationError):
        build_success(resolved_detection_policy=policy)

    success = build_success(
        artifacts=(build_artifact(), detections),
        resolved_scores_request=build_scores_request(),
        resolved_detection_policy=policy,
    )

    assert success.resolved_detection_policy == policy


# --- InferenceFailure and FailureReport -------------------------------------


def test_a_failure_names_no_artifacts():
    assert set(InferenceFailure.model_fields) == {
        "schema_version",
        "outcome",
        "work_digest",
        "failure",
    }


def test_a_failure_report_refuses_an_undeclared_stage():
    with pytest.raises(ValidationError):
        build_report(stage="publish")

    for stage in DECLARED_STAGES:
        assert build_report(stage=stage).stage == stage


def test_a_failure_report_refuses_an_empty_code():
    with pytest.raises(ValidationError):
        build_report(code="")


def test_a_failure_report_locates_itself_only_when_it_can():
    report = build_report()

    assert (report.namespace, report.value) == (None, None)
    assert report.window_start_s is None
    located = build_report(namespace="soundhub", value="42")
    assert (located.namespace, located.value) == ("soundhub", "42")

    with pytest.raises(ValidationError):
        build_report(window_start_s=float("nan"))


@pytest.mark.parametrize(
    "half", [{"namespace": "soundhub"}, {"value": "42"}], ids=["namespace", "value"]
)
def test_a_failure_report_names_a_whole_recording_or_none(half):
    with pytest.raises(ValidationError, match="together"):
        build_report(**half)


def test_the_result_union_discriminates_on_outcome():
    adapter = TypeAdapter(InferenceResult)

    success = adapter.validate_python(build_success().model_dump(mode="json"))
    failure = adapter.validate_python(build_failure().model_dump(mode="json"))

    assert isinstance(success, InferenceSuccess)
    assert isinstance(failure, InferenceFailure)

    with pytest.raises(ValidationError):
        adapter.validate_python(build_failure().model_dump(mode="json") | {"outcome": "partial"})


# --- Canonical encoding -----------------------------------------------------


def test_a_result_replays_byte_identically():
    first = build_success()
    second = build_success()

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_v1(first) == sha256_v1(first.model_dump(mode="json"))


def test_the_work_digest_field_refuses_the_file_digest_family():
    with pytest.raises(ValidationError):
        build_success(work_digest=FILE_DIGEST)

    with pytest.raises(ValidationError):
        build_failure(work_digest=FILE_DIGEST)


# --- The registry binding ----------------------------------------------------


@pytest.mark.parametrize(
    "binding",
    [
        pytest.param({"registry_uri": "file:///registry.csv"}, id="uri_without_fingerprint"),
        pytest.param({"registry_fingerprint": FILE_DIGEST}, id="fingerprint_without_uri"),
    ],
)
def test_a_success_refuses_half_a_registry_binding(binding):
    with pytest.raises(ValidationError) as exc:
        build_success(**binding)

    assert "registry_uri" in str(exc.value)
    assert "registry_fingerprint" in str(exc.value)


def test_a_success_accepts_a_whole_registry_binding():
    success = build_success(
        registry_uri="file:///registry.csv", registry_fingerprint=FILE_DIGEST
    )

    assert success.registry_uri == "file:///registry.csv"
    assert success.registry_fingerprint == FILE_DIGEST


def test_a_success_accepts_no_registry_binding():
    success = build_success()

    assert success.registry_uri is None
    assert success.registry_fingerprint is None
