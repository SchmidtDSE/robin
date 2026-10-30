"""One work file in, one result file out, in the test process with the test model."""

import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone

import ondio
import pytest
from robin_adapters.storage.location import split_location
from robin_contracts.canonical import checksum_file
from robin_contracts.local_files import get_local_path
from robin_contracts.results import InferenceCompleted
from robin_contracts.tasks import (
    InferenceTaskResult,
    ModelOutputVersion,
    task_result_problem,
)

from robin_worker import run, storage
from worker_doubles import TEST_VERSIONS, TaskBuilder


@pytest.fixture
def builder(tmp_path, test_model_site):
    return TaskBuilder(tmp_path / "task")


def run_and_read(builder, *, versions=TEST_VERSIONS, log=None):
    lines = [] if log is None else log
    code = run.run_task(
        str(builder.work_path), versions=versions, release="test-release", log=lines.append
    )
    result = InferenceTaskResult.model_validate_json(builder.result_path.read_bytes())
    return code, result


def problem_of(result, task, builder):
    return task_result_problem(
        result,
        task=task,
        work_file_digest=checksum_file(builder.work_path),
        split_location=split_location,
    )


# ---------------------------------------------------------------------------
# Works the engine runs.
# ---------------------------------------------------------------------------


def test_a_completed_work_writes_a_result_the_control_plane_accepts(builder):
    work_path, task = builder.write()

    code, result = run_and_read(builder)

    assert code == 0
    outcome = result.result
    assert isinstance(outcome, InferenceCompleted)
    assert {one.value for one in outcome.coverage} == {"r1", "r2"}
    assert [one.kind for one in outcome.artifacts] == ["scores", "scores"]
    assert result.task_id == task.task_id
    assert result.work_file_digest == checksum_file(work_path)
    assert result.robin_release == "test-release"
    assert result.started_at.tzinfo is not None and result.finished_at.tzinfo is not None
    assert result.finished_at >= result.started_at
    assert problem_of(result, task, builder) is None


def test_a_missing_input_fails_only_its_recording(builder):
    work = builder.work((builder.recording("r1"), builder.recording("r2")))
    (builder.directory / "audio" / "r2.wav").unlink()
    builder.write(work=work)

    code, result = run_and_read(builder)

    assert code == 0
    assert result.result.outcome == "completed"
    assert [one.value for one in result.result.coverage] == ["r1"]
    (failed,) = result.result.failed
    assert (failed.value, failed.failure.stage) == ("r2", "acquire_input")


def test_an_input_on_an_unsupported_scheme_fails_only_its_recording(builder):
    recordings = (builder.recording("r1"), builder.recording("r2", audio_uri="example://b/a.wav"))
    builder.write(work=builder.work(recordings))

    code, result = run_and_read(builder)

    assert code == 0
    assert [one.value for one in result.result.coverage] == ["r1"]
    (failed,) = result.result.failed
    assert failed.failure.stage == "acquire_input"
    assert "example://b/a.wav" in failed.failure.detail


def test_a_model_file_at_the_wrong_digest_fails_the_whole_work(builder):
    builder.files["weights"] = builder.files["weights"].model_copy(
        update={"digest": "sha256:" + "0" * 64}
    )
    builder.write()

    code, result = run_and_read(builder)

    assert code == 0
    assert result.result.outcome == "failure"
    assert result.result.failure.stage == "acquire_model"


@pytest.mark.parametrize(
    ("location", "code"),
    [
        # The bytes robin-models ships are not the test model's registry.
        pytest.param("robin-models:owl/v4/taxa_registry.csv", "model_file_digest_mismatch",
                     id="other_bytes"),
        pytest.param("robin-models:owl/v9/taxa_registry.csv", "model_file_unavailable",
                     id="not_shipped"),
    ],
)
def test_a_bundled_registry_this_worker_cannot_match_fails_the_whole_work(
    builder, location, code
):
    builder.files["taxa_registry"] = builder.files["taxa_registry"].model_copy(
        update={"uri": location}
    )
    builder.write()

    exit_code, result = run_and_read(builder)

    assert exit_code == 0
    assert result.result.outcome == "failure"
    assert (result.result.failure.stage, result.result.failure.code) == ("acquire_model", code)


# ---------------------------------------------------------------------------
# Refusals before inference.
# ---------------------------------------------------------------------------


def envelope(builder, **changes):
    """The work file's JSON as a dict, with `changes` applied; None removes a field."""
    value = json.loads(builder.task().model_dump_json())
    for name, new in changes.items():
        if new is None:
            del value[name]
        else:
            value[name] = new
    return value


def write_json(builder, value):
    builder.write_bytes(json.dumps(value).encode())


REFUSALS_OF_UNPARSED_FILES = [
    ("not utf-8", lambda b: b.write_bytes(b"\xff"), "work_file_invalid"),
    ("not an object", lambda b: b.write_bytes(b"[]"), "work_file_invalid"),
    (
        "no schema_version",
        lambda b: write_json(b, envelope(b, schema_version=None)),
        "work_file_invalid",
    ),
    (
        "a later schema",
        lambda b: write_json(b, envelope(b, schema_version="robin.inference-task/2")),
        "unknown_schema_version",
    ),
    ("no task_id", lambda b: write_json(b, envelope(b, task_id=None)), "work_file_invalid"),
    # Python's JSON parser raises errors other than JSONDecodeError for these two.
    ("a huge integer", lambda b: b.write_bytes(b"1" * 5000), "work_file_invalid"),
    ("deep nesting", lambda b: b.write_bytes(b"[" * 200_000), "work_file_invalid"),
]


def assert_refused(builder, result, code, *, task):
    assert result.result.outcome == "refused"
    assert result.result.code == code
    assert not builder.outputs.exists()
    if task is not None:
        assert problem_of(result, task, builder) is None


def test_a_missing_work_file_is_refused_with_no_digest(builder):
    code, result = run_and_read(builder)

    assert code == 1
    assert_refused(builder, result, "work_file_unreadable", task=None)
    assert result.task_id is None and result.work_file_digest is None


@pytest.mark.parametrize(
    ("write", "refusal"),
    [case[1:] for case in REFUSALS_OF_UNPARSED_FILES],
    ids=[case[0] for case in REFUSALS_OF_UNPARSED_FILES],
)
def test_a_work_file_that_does_not_parse_is_refused_with_its_digest(builder, write, refusal):
    write(builder)

    code, result = run_and_read(builder)

    assert code == 1
    assert_refused(builder, result, refusal, task=None)
    assert result.task_id is None
    assert result.work_file_digest == checksum_file(builder.work_path)
    task = builder.task()
    assert problem_of(result, task, builder) is None


def unusable_roots(builder):
    return ["relative/outputs", f"{builder.task_dir}/a/../outputs", "example://b/x"]


@pytest.mark.parametrize("which", range(3), ids=["relative", "dot-dot", "example"])
def test_an_output_root_that_cannot_hold_artifacts_is_refused(builder, which):
    root = unusable_roots(builder)[which]
    work_path, task = builder.write(output_root=root)

    code, result = run_and_read(builder)

    assert code == 1
    assert_refused(builder, result, "work_file_invalid", task=task)
    assert root in result.result.detail
    assert result.task_id == task.task_id
    assert result.work_file_digest == checksum_file(work_path)


@pytest.mark.parametrize(
    ("output_version", "versions", "refusal"),
    [
        (("model:other/1", 1), TEST_VERSIONS, "model_identifier_mismatch"),
        (("model:test-model/1", 2), TEST_VERSIONS, "output_version_mismatch"),
        (("model:test-model/1", 1), {}, "output_version_mismatch"),
    ],
    ids=["other model", "other version", "not in the table"],
)
def test_a_task_for_other_code_is_refused(builder, output_version, versions, refusal):
    model, version = output_version
    output_version = ModelOutputVersion(model=model, version=version)
    work_path, task = builder.write(output_version=output_version)

    code, result = run_and_read(builder, versions=versions)

    assert code == 1
    assert_refused(builder, result, refusal, task=task)
    assert result.task_id == task.task_id
    assert result.work_file_digest == checksum_file(work_path)


# ---------------------------------------------------------------------------
# Worker errors and the result file.
# ---------------------------------------------------------------------------


def raising(exc):
    def call(*args, **kwargs):
        raise exc

    return call


def test_an_unexpected_exception_is_reported_as_a_worker_error(builder, monkeypatch):
    work_path, task = builder.write()
    monkeypatch.setattr(run, "run_work", raising(RuntimeError("boom")))

    code, result = run_and_read(builder)

    assert code == 1
    assert result.result.outcome == "error"
    assert result.result.detail.startswith("RuntimeError: boom")
    assert result.task_id == task.task_id
    assert result.work_file_digest == checksum_file(work_path)


def test_a_long_worker_error_is_cut_to_the_detail_limit(builder, monkeypatch):
    builder.write()
    monkeypatch.setattr(run, "run_work", raising(RuntimeError("x" * 5000)))

    _, result = run_and_read(builder)

    assert len(result.result.detail) == 2000


def test_an_existing_result_is_not_changed(builder):
    builder.write()
    builder.result_path.write_bytes(b"earlier")
    lines = []

    code = run.run_task(
        str(builder.work_path), versions=TEST_VERSIONS, release="r", log=lines.append
    )

    assert code == 3
    assert builder.result_path.read_bytes() == b"earlier"
    assert any("already exists" in line for line in lines)


def test_each_execution_writes_its_outputs_in_its_own_folder(builder):
    builder.write()
    code, result = run_and_read(builder)
    committed = builder.result_path.read_bytes()

    second = run.run_task(
        str(builder.work_path), versions=TEST_VERSIONS, release="r", log=[].append
    )

    assert (code, second) == (0, 3)
    assert builder.result_path.read_bytes() == committed
    folders = sorted(builder.outputs.iterdir())
    assert len(folders) == 2
    assert all(one.is_dir() and uuid.UUID(one.name) for one in folders)
    artifacts = result.result.artifacts
    paths = [get_local_path(one.uri) for one in artifacts]
    assert len({path.relative_to(builder.outputs).parts[0] for path in paths}) == 1
    for artifact, path in zip(artifacts, paths, strict=True):
        assert checksum_file(path) == artifact.checksum


@pytest.mark.parametrize("form", ["path_with_slash", "file_uri_with_slash"])
def test_an_output_root_with_a_trailing_slash_gives_artifacts_that_pass_the_check(
    builder, form
):
    root = {
        "path_with_slash": f"{builder.outputs}/",
        "file_uri_with_slash": f"{builder.outputs.as_uri()}/",
    }[form]
    _, task = builder.write(output_root=root)

    code, result = run_and_read(builder)

    assert code == 0
    assert result.result.outcome == "completed"
    assert problem_of(result, task, builder) is None


def test_a_failed_result_write_says_the_result_is_unknown(builder, monkeypatch):
    builder.write()
    monkeypatch.setattr(storage, "create", raising(ondio.OndioError("disk full")))
    lines = []

    code = run.run_task(
        str(builder.work_path), versions=TEST_VERSIONS, release="r", log=lines.append
    )

    assert code == 4
    assert any("may or may not exist" in line and "disk full" in line for line in lines)


# ---------------------------------------------------------------------------
# The scratch folder.
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_cannot_be_removed(monkeypatch):
    """`shutil.rmtree` fails for the worker's scratch folder and works for anything else."""
    real = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if str(path).rsplit("/", 1)[-1].startswith("robin-worker-"):
            raise OSError("busy")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(run.shutil, "rmtree", rmtree)


def test_a_scratch_folder_that_stays_does_not_change_a_completed_result(
    builder, scratch_cannot_be_removed
):
    builder.write()
    lines = []

    code, result = run_and_read(builder, log=lines)

    assert code == 0
    assert result.result.outcome == "completed"
    assert any("robin-worker-" in line and "OSError: busy" in line for line in lines)


def test_a_scratch_folder_that_stays_does_not_change_a_stop(
    builder, scratch_cannot_be_removed, monkeypatch
):
    builder.write()
    monkeypatch.setattr(run, "run_work", raising(KeyboardInterrupt()))
    lines = []

    with pytest.raises(KeyboardInterrupt):
        run.run_task(str(builder.work_path), versions=TEST_VERSIONS, release="r", log=lines.append)

    assert not builder.result_path.exists()
    assert any("OSError: busy" in line for line in lines)


# ---------------------------------------------------------------------------
# Logs.
# ---------------------------------------------------------------------------


def test_engine_logs_reach_the_log_and_nothing_reaches_stdout(builder, capsys):
    builder.write()
    lines = []

    run_and_read(builder, log=lines)

    assert any("windows accepted" in line for line in lines)
    assert capsys.readouterr().out == ""


def test_a_clock_that_steps_back_still_writes_the_result(builder, monkeypatch):
    builder.write()
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    times = iter([start, start - timedelta(seconds=5)])
    monkeypatch.setattr(run, "_now", lambda: next(times))

    code, result = run_and_read(builder)

    assert code == 0
    assert result.result.outcome == "completed"
    assert result.finished_at == result.started_at == start


def test_a_root_through_a_link_gives_artifacts_that_pass_the_check(builder):
    real = builder.directory / "real-outputs"
    real.mkdir()
    linked = builder.directory / "linked-outputs"
    linked.symlink_to(real, target_is_directory=True)
    _, task = builder.write(output_root=str(linked))

    code, result = run_and_read(builder)

    assert code == 0
    assert all("linked-outputs" in one.uri for one in result.result.artifacts)
    assert problem_of(result, task, builder) is None


def test_a_work_uri_in_a_missing_folder_writes_nothing(tmp_path, test_model_site):
    missing = tmp_path / "typo" / "tasks" / "t"
    lines = []

    code = run.run_task(
        str(missing / "work.json"), versions=TEST_VERSIONS, release="r", log=lines.append
    )

    assert code == 4
    assert not (tmp_path / "typo").exists()
    assert any("may or may not exist" in line for line in lines)
