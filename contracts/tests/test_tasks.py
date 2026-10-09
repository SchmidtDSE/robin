"""The files the control plane and the worker exchange for one task."""

import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pytest
from pydantic import ValidationError

from robin_contracts.cards import AudioGeometry, HeadCard, ModelCard, RunnerResampled, model_ref
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.results import (
    ArtifactRecord,
    FailureReport,
    InferenceCompleted,
    InferenceFailure,
    RecordingCoverage,
    RecordingFailed,
)
from robin_contracts.specs import AudioSpec, Recipe, WindowGeometry
from robin_contracts.tasks import (
    InferenceTask,
    InferenceTaskResult,
    ModelOutputVersion,
    TaskRefusal,
    WorkerError,
    output_version_id,
    task_result_problem,
)
from robin_contracts.work import (
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
    work_digest,
)

REGISTRY_DIGEST = "sha256:" + "a" * 64
HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
TASK_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
OUTPUT_ROOT = "s3://bucket/runs/r1/tasks/t1/outputs"

CARD = ModelCard(
    model_name="owl",
    model_version="v4",
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


def build_head(**overrides) -> HeadCard:
    fields = {
        "model_name": "amy-head",
        "model_version": "1",
        "runtime": "onnx",
        "backbone": model_ref(CARD),
        "embedding_dim": 1280,
        "min_detection_threshold": 0.0,
        "score_domain": "sigmoid",
        "taxa_registry_digest": REGISTRY_DIGEST,
    }
    return HeadCard(**(fields | overrides))


def build_recording(**overrides) -> RecordingRef:
    fields = {"namespace": "soundhub", "value": "42", "audio_uri": "s3://bucket/42.wav"}
    return RecordingRef(**(fields | overrides))


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (build_recording(), build_recording(value="43")),
        "model": PinnedModel(
            card=CARD,
            files={"weights": PinnedFile(uri="s3://bucket/owl.tflite", digest=FILE_DIGEST, size_bytes=8)},
        ),
        "input": AudioInput(),
        "settings": {},
        "resources": {},
        "outputs": (ScoresRequest(contract_id="robin.scores.parquet/1", retention="full"),),
    }
    return InferenceWork(**(fields | overrides))


def build_task(**overrides) -> InferenceTask:
    fields = {
        "schema_version": "robin.inference-task/1",
        "task_id": TASK_ID,
        "output_root": OUTPUT_ROOT,
        "output_version": ModelOutputVersion(model="model:owl/v4", version=1),
        "work": build_work(),
    }
    return InferenceTask(**(fields | overrides))


# --- output_version_id and ModelOutputVersion -------------------------------

def test_a_base_model_is_identified_by_its_entry_point_key():
    assert output_version_id(CARD) == "model:owl/v4"


def test_every_head_on_one_runtime_shares_one_identifier():
    assert output_version_id(build_head()) == "head_runtime:onnx"
    assert output_version_id(build_head(model_name="other-head")) == "head_runtime:onnx"


@pytest.mark.parametrize("model", ["model:owl/v4", "head_runtime:onnx"])
def test_an_output_version_accepts_each_identifier_kind(model):
    assert ModelOutputVersion(model=model, version=1).model == model


@pytest.mark.parametrize("model", ["owl/v4", "model:", "head_runtime:", "", "runtime:onnx"])
def test_an_output_version_refuses_a_malformed_identifier(model):
    with pytest.raises(ValidationError):
        ModelOutputVersion(model=model, version=1)


@pytest.mark.parametrize("version", [0, -1, True, "1", 1.0])
def test_an_output_version_is_a_strict_integer_of_at_least_one(version):
    with pytest.raises(ValidationError):
        ModelOutputVersion(model="model:owl/v4", version=version)


# --- InferenceTask -----------------------------------------------------------

def test_a_task_round_trips_through_its_json():
    task = build_task()

    assert InferenceTask.model_validate_json(task.model_dump_json()) == task


def test_a_task_refuses_another_schema_version():
    with pytest.raises(ValidationError):
        build_task(schema_version="robin.inference-task/2")


def test_a_task_refuses_unknown_fields():
    data = build_task().model_dump(mode="json") | {"run_id": "r1"}

    with pytest.raises(ValidationError):
        InferenceTask.model_validate(data)


def test_a_task_refuses_an_empty_output_root():
    with pytest.raises(ValidationError):
        build_task(output_root="")


def test_a_task_whose_version_names_another_model_still_parses():
    # The worker refuses it with its own code; the parser must not hide that.
    task = build_task(output_version=ModelOutputVersion(model="model:perch/v8", version=3))

    assert output_version_id(task.work.model.card) != task.output_version.model


# --- InferenceTaskResult -----------------------------------------------------

STARTED = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
WORK_FILE_DIGEST = "sha256:" + "b" * 64
WORK_DIGEST = "sha256:v1:" + "c" * 64


def build_refusal(**overrides) -> TaskRefusal:
    fields = {"code": "output_version_mismatch", "detail": "the worker runs version 2"}
    return TaskRefusal(**(fields | overrides))


def build_work_failure() -> InferenceFailure:
    return InferenceFailure(
        schema_version="robin.inference-result/1",
        work_digest=WORK_DIGEST,
        failure=FailureReport(code="model_file_unavailable", stage="acquire_model", detail="gone"),
    )


def build_result(**overrides) -> InferenceTaskResult:
    fields = {
        "schema_version": "robin.inference-task-result/1",
        "task_id": TASK_ID,
        "work_file_digest": WORK_FILE_DIGEST,
        "robin_release": "robin@sha256:" + "d" * 64,
        "started_at": STARTED,
        "finished_at": STARTED + timedelta(minutes=5),
        "result": build_work_failure(),
    }
    return InferenceTaskResult(**(fields | overrides))


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(build_work_failure(), id="failure"),
        pytest.param(build_refusal(), id="refused"),
        pytest.param(WorkerError(detail="OSError: disk full"), id="error"),
    ],
)
def test_a_result_round_trips_as_the_kind_of_outcome_it_holds(outcome):
    result = build_result(result=outcome)

    rebuilt = InferenceTaskResult.model_validate_json(result.model_dump_json())

    assert type(rebuilt.result) is type(outcome)
    assert rebuilt == result


def test_a_result_refuses_an_unknown_outcome():
    data = build_result().model_dump(mode="json")
    data["result"] = {"outcome": "success", "detail": "x"}

    with pytest.raises(ValidationError):
        InferenceTaskResult.model_validate(data)


def test_a_refusal_refuses_an_unknown_code():
    with pytest.raises(ValidationError):
        build_refusal(code="disk_full")


def test_a_result_refuses_a_time_without_a_time_zone():
    with pytest.raises(ValidationError):
        build_result(started_at=STARTED.replace(tzinfo=None))


def test_a_result_refuses_finishing_before_it_started():
    with pytest.raises(ValidationError):
        build_result(finished_at=STARTED - timedelta(seconds=1))


def test_a_result_may_finish_when_it_started():
    assert build_result(finished_at=STARTED).finished_at == STARTED


def test_a_result_refuses_an_empty_release():
    with pytest.raises(ValidationError):
        build_result(robin_release="")


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(build_refusal(code="work_file_unreadable"), id="unreadable"),
        pytest.param(WorkerError(detail="MemoryError"), id="error"),
    ],
)
def test_a_result_may_omit_the_digest_when_the_bytes_were_not_read(outcome):
    assert build_result(work_file_digest=None, result=outcome).work_file_digest is None


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(build_work_failure(), id="failure"),
        pytest.param(build_refusal(code="work_file_invalid"), id="invalid"),
        pytest.param(build_refusal(code="output_version_mismatch"), id="mismatch"),
    ],
)
def test_a_result_needs_the_digest_when_the_bytes_were_read(outcome):
    with pytest.raises(ValidationError):
        build_result(work_file_digest=None, result=outcome)


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(build_refusal(code="work_file_unreadable"), id="unreadable"),
        pytest.param(build_refusal(code="work_file_invalid"), id="invalid"),
        pytest.param(build_refusal(code="unknown_schema_version"), id="unknown_schema"),
        pytest.param(WorkerError(detail="MemoryError"), id="error"),
    ],
)
def test_a_result_may_omit_the_task_id_when_the_envelope_was_not_parsed(outcome):
    assert build_result(task_id=None, result=outcome).task_id is None


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(build_work_failure(), id="failure"),
        pytest.param(build_refusal(code="model_identifier_mismatch"), id="identifier"),
        pytest.param(build_refusal(code="output_version_mismatch"), id="version"),
    ],
)
def test_a_result_needs_the_task_id_when_the_envelope_was_parsed(outcome):
    with pytest.raises(ValidationError):
        build_result(task_id=None, result=outcome)


# --- task_result_problem -----------------------------------------------------

def build_artifact(value: str, *, root: str = OUTPUT_ROOT, **overrides) -> ArtifactRecord:
    fields = {
        "kind": "scores",
        "contract_id": "robin.scores.parquet/1",
        "namespace": "soundhub",
        "value": value,
        "uri": f"{root}/scores/recording_namespace=soundhub/recording_value={value}/scores.parquet",
        "checksum": FILE_DIGEST,
        "size_bytes": 100,
        "rows": 3,
    }
    return ArtifactRecord(**(fields | overrides))


def build_covered(value: str) -> RecordingCoverage:
    return RecordingCoverage(
        namespace="soundhub",
        value=value,
        windows_completed=3,
        score_rows=3,
        embedding_rows=0,
        detection_rows=0,
        first_window_start_s=0.0,
        last_window_end_s=9.0,
    )


def build_failed(value: str) -> RecordingFailed:
    return RecordingFailed(
        namespace="soundhub",
        value=value,
        failure=FailureReport(code="input_unavailable", stage="acquire_input", detail="gone"),
    )


def build_completed(task: InferenceTask, **overrides) -> InferenceCompleted:
    card = task.work.model.card
    fields = {
        "schema_version": "robin.inference-result/1",
        "work_digest": work_digest(task.work),
        "recipe": Recipe(
            model=model_ref(card),
            backend="tensorflow",
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
        ),
        "model": task.work.model,
        "window_geometry": WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="drop"),
        "resolved_scores_request": task.work.outputs[0],
        "artifacts": (build_artifact("42", root=task.output_root),),
        "coverage": (build_covered("42"),),
        "failed": (build_failed("43"),),
    }
    return InferenceCompleted(**(fields | overrides))


def split_location(location: str) -> tuple[str, str] | None:
    """A stand-in for storage: `s3://` keys as written, and `enc://` paths decoded."""
    scheme, separator, rest = location.partition("://")
    store, _, path = rest.partition("/")
    if not separator or scheme not in ("s3", "enc") or not store:
        return None
    return f"{scheme}://{store}", unquote(path) if scheme == "enc" else path


def problem(result_outcome, task: InferenceTask | None = None, **result_fields) -> str | None:
    task = task or build_task()
    result = build_result(result=result_outcome, **result_fields)
    return task_result_problem(
        result, task=task, work_file_digest=WORK_FILE_DIGEST, split_location=split_location
    )


def artifact_problem(uri: str, root: str = OUTPUT_ROOT) -> str | None:
    task = build_task(output_root=root)
    outcome = build_completed(task, artifacts=(build_artifact("42", uri=uri),))
    return problem(outcome, task)


def test_a_completed_result_round_trips_as_completed():
    result = build_result(result=build_completed(build_task()))

    rebuilt = InferenceTaskResult.model_validate_json(result.model_dump_json())

    assert type(rebuilt.result) is InferenceCompleted
    assert rebuilt == result


def test_a_completed_result_for_its_task_passes():
    task = build_task()

    assert problem(build_completed(task), task) is None


def test_a_work_failure_for_its_task_passes():
    task = build_task()
    failure = build_work_failure().model_copy(update={"work_digest": work_digest(task.work)})

    assert problem(failure, task) is None


def test_a_refusal_of_an_unreadable_work_file_passes_without_binding():
    # The control plane then records the task as failed with the refusal code.
    outcome = build_refusal(code="work_file_unreadable")

    assert problem(outcome, task_id=None, work_file_digest=None) is None


def test_a_result_from_another_task_with_the_same_work_is_refused():
    task = build_task()
    other = uuid.UUID("00000000-0000-4000-8000-000000000002")

    reason = problem(build_completed(task), task, task_id=other)

    assert str(other) in reason and str(TASK_ID) in reason


def test_a_result_that_read_another_work_file_is_refused():
    task = build_task()
    other = "sha256:" + "e" * 64

    reason = problem(build_completed(task), task, work_file_digest=other)

    assert other in reason and WORK_FILE_DIGEST in reason


def test_a_refusal_that_read_another_work_file_is_refused():
    assert problem(build_refusal(), work_file_digest="sha256:" + "e" * 64) is not None


def test_a_result_for_another_work_is_refused():
    task = build_task()
    outcome = build_completed(task, work_digest=WORK_DIGEST)

    reason = problem(outcome, task)

    assert WORK_DIGEST in reason and work_digest(task.work) in reason


def test_a_work_failure_for_another_work_is_refused():
    assert WORK_DIGEST in problem(build_work_failure())


def test_a_result_that_omits_an_assigned_recording_is_refused():
    task = build_task()
    outcome = build_completed(task, failed=())

    assert "'43'" in problem(outcome, task)


def test_a_result_that_names_an_unassigned_recording_is_refused():
    task = build_task()
    outcome = build_completed(task, failed=(build_failed("43"), build_failed("44")))

    assert "'44'" in problem(outcome, task)


@pytest.mark.parametrize(
    "uri",
    [
        pytest.param(f"{OUTPUT_ROOT}-old/scores/x.parquet", id="look_alike_folder"),
        pytest.param(f"{OUTPUT_ROOT}/../../t2/outputs/x.parquet", id="parent_segment"),
        pytest.param(f"{OUTPUT_ROOT}/./x.parquet", id="current_segment"),
        pytest.param(f"{OUTPUT_ROOT}//x.parquet", id="empty_segment"),
        pytest.param(OUTPUT_ROOT, id="the_root_itself"),
        pytest.param(f"{OUTPUT_ROOT}/", id="the_root_with_a_slash"),
        pytest.param("s3://other/runs/r1/tasks/t1/outputs/x.parquet", id="other_bucket"),
        pytest.param("enc://bucket/runs/r1/tasks/t1/outputs/x.parquet", id="other_store"),
        pytest.param("gs://bucket/runs/r1/tasks/t1/outputs/x.parquet", id="unsupported"),
    ],
)
def test_an_artifact_outside_the_output_root_is_refused(uri):
    assert uri in artifact_problem(uri)


@pytest.mark.parametrize(
    "root",
    [
        pytest.param(OUTPUT_ROOT, id="plain"),
        pytest.param(f"{OUTPUT_ROOT}/", id="trailing_slash"),
        pytest.param("s3://bucket", id="whole_bucket"),
        pytest.param("s3://bucket/", id="whole_bucket_with_a_slash"),
    ],
)
def test_an_artifact_under_the_output_root_passes(root):
    assert artifact_problem(f"{OUTPUT_ROOT}/scores/x.parquet", root=root) is None


@pytest.mark.parametrize(
    "root",
    [
        pytest.param("gs://bucket/runs", id="unsupported"),
        pytest.param("s3://bucket/runs/../runs", id="parent_segment"),
        pytest.param("s3://bucket/runs//r1", id="empty_segment"),
    ],
)
def test_no_artifact_is_under_an_output_root_storage_cannot_name(root):
    assert artifact_problem(f"{root}/x.parquet", root=root) is not None


ENCODED_ROOT = "enc://disk/data/outputs"


@pytest.mark.parametrize(
    "uri",
    [
        pytest.param(f"{ENCODED_ROOT}/%2e%2e/other/x.parquet", id="encoded_dots"),
        pytest.param(f"{ENCODED_ROOT}/a%2F..%2F..%2Fother/x.parquet", id="encoded_slash"),
    ],
)
def test_the_check_reads_the_path_storage_reads(uri):
    # The stand-in decodes `enc://` paths, so these leave the root once decoded.
    assert uri in artifact_problem(uri, root=ENCODED_ROOT)


def test_a_root_is_matched_in_the_form_storage_reads():
    assert artifact_problem(f"{ENCODED_ROOT}/x.parquet", root="enc://disk/data/%6futputs") is None


def test_a_key_storage_does_not_decode_is_compared_as_written():
    assert artifact_problem(f"{OUTPUT_ROOT}/%2e%2e/x.parquet") is None
