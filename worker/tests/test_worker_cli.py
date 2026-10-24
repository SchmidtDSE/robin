"""The `robin-worker` command, run as a separate process."""

import json
import signal
import subprocess
import sys
import time

import pytest
from robin_adapters.location import split_location
from robin_contracts.canonical import checksum_file
from robin_contracts.local_files import get_local_path
from robin_contracts.tasks import InferenceTaskResult, ModelOutputVersion, task_result_problem
from robin_models.output_versions import OUTPUT_VERSIONS

from robin_worker import cli

from worker_doubles import (
    STARTED_VARIABLE,
    TaskBuilder,
    popen_worker,
    run_worker,
)


@pytest.fixture
def builder(tmp_path):
    return TaskBuilder(tmp_path / "task")


def read_result(builder) -> InferenceTaskResult:
    return InferenceTaskResult.model_validate_json(builder.result_path.read_bytes())


def problem_of(result, task, builder):
    return task_result_problem(
        result,
        task=task,
        work_file_digest=checksum_file(builder.work_path),
        split_location=split_location,
    )


# ---------------------------------------------------------------------------
# `run`.
# ---------------------------------------------------------------------------


def test_the_worker_runs_a_task_end_to_end(tmp_path, builder, test_model_site):
    work, task = builder.write()

    done = run_worker(tmp_path, ["run", str(work)], site=test_model_site)

    assert done.returncode == 0, done.stderr
    assert done.stdout == ""
    result = read_result(builder)
    assert result.result.outcome == "completed"
    assert result.robin_release == "test-release"
    assert problem_of(result, task, builder) is None
    paths = [get_local_path(artifact.uri) for artifact in result.result.artifacts]
    assert len(paths) == 2 and all(path.is_file() for path in paths)


def test_the_release_is_unreleased_when_no_release_is_set(tmp_path, builder, test_model_site):
    work, _ = builder.write()

    done = run_worker(
        tmp_path, ["run", str(work)], site=test_model_site, env={"ROBIN_RELEASE": None}
    )

    assert done.returncode == 0, done.stderr
    assert read_result(builder).robin_release == "unreleased"


@pytest.mark.parametrize("as_uri", [False, True], ids=["path", "file-uri"])
def test_a_local_work_path_can_contain_question_marks_and_hashes(
    tmp_path, test_model_site, as_uri
):
    builder = TaskBuilder(tmp_path / "a?b#c")
    work, task = builder.write()
    location = work.as_uri() if as_uri else str(work)

    done = run_worker(tmp_path, ["run", location], site=test_model_site)

    assert done.returncode == 0, done.stderr
    assert done.stdout == ""
    result = read_result(builder)
    assert result.result.outcome == "completed"
    assert problem_of(result, task, builder) is None


def write_unreadable(builder):
    return None


def write_not_json(builder):
    builder.write_bytes(b"not json")
    return None


def write_root_with_dot_dot(builder):
    return builder.write(output_root=f"{builder.task_dir}/a/../outputs")[1]


def write_later_schema(builder):
    value = json.loads(builder.task().model_dump_json())
    value["schema_version"] = "robin.inference-task/2"
    builder.write_bytes(json.dumps(value).encode())
    return None


def write_other_model(builder):
    return builder.write(output_version=ModelOutputVersion(model="model:other/1", version=1))[1]


def write_test_model(builder):
    return builder.write()[1]


REFUSALS = [
    (write_unreadable, "work_file_unreadable", False, False),
    (write_not_json, "work_file_invalid", False, True),
    (write_root_with_dot_dot, "work_file_invalid", True, True),
    (write_later_schema, "unknown_schema_version", False, True),
    (write_other_model, "model_identifier_mismatch", True, True),
    (write_test_model, "output_version_mismatch", True, True),
]


@pytest.mark.parametrize(
    ("write", "code", "has_task_id", "has_digest"),
    REFUSALS,
    ids=[case[0].__name__ for case in REFUSALS],
)
def test_each_refusal_through_the_console_script(
    tmp_path, builder, test_model_site, write, code, has_task_id, has_digest
):
    task = write(builder)

    done = run_worker(
        tmp_path, ["run", str(builder.work_path)], site=test_model_site, test_versions=False
    )

    assert done.returncode == 1, done.stderr
    result = read_result(builder)
    assert result.result.outcome == "refused"
    assert result.result.code == code
    assert result.task_id == (builder.task_id if has_task_id else None)
    assert result.work_file_digest == (checksum_file(builder.work_path) if has_digest else None)
    if task is not None:
        assert problem_of(result, task, builder) is None
    assert not builder.outputs.exists()


# ---------------------------------------------------------------------------
# `output-versions`.
# ---------------------------------------------------------------------------


def test_output_versions_prints_the_real_table(tmp_path):
    done = run_worker(tmp_path, ["output-versions"], test_versions=False)

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [one.model_dump(mode="json") for one in OUTPUT_VERSIONS]


# Records each attempt to import an ML runtime, and refuses it. The test environment
# has no ML runtime, so checking `sys.modules` alone could never fail.
NO_ML_RUNTIME_CHECK = """
import sys

RUNTIMES = {"tensorflow", "onnxruntime", "sox_tensorflow"}
attempted = []


class RecordRuntimeImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in RUNTIMES:
            attempted.append(name)
            raise ImportError(f"{name} must not be imported")
        return None


sys.meta_path.insert(0, RecordRuntimeImports())
from robin_worker.cli import main

assert main(["output-versions"]) == 0
assert not attempted, attempted
"""


def run_check(source):
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)


def test_output_versions_loads_no_ml_runtime():
    done = run_check(NO_ML_RUNTIME_CHECK)

    assert done.returncode == 0, done.stderr


def test_the_runtime_check_fails_when_a_runtime_is_imported():
    # The check must be able to fail, even when the import itself is caught.
    source = NO_ML_RUNTIME_CHECK.replace(
        "from robin_worker.cli import main",
        "try:\n    import tensorflow\nexcept ImportError:\n    pass\n"
        "from robin_worker.cli import main",
    )

    done = run_check(source)

    assert done.returncode != 0
    assert "tensorflow" in done.stderr


# ---------------------------------------------------------------------------
# Wrong arguments.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args", [[], ["unknown"], ["run", "/x/a/work.json", "/x/b/work.json"]], ids=str
)
def test_wrong_arguments_exit_with_usage(tmp_path, args):
    done = run_worker(tmp_path, args, test_versions=False)

    assert done.returncode == 2


def test_a_relative_work_path_is_refused_as_not_absolute(tmp_path):
    done = run_worker(tmp_path, ["run", "runs/r/tasks/t/work.json"], test_versions=False)

    assert done.returncode == 2
    assert "absolute local path" in done.stderr
    assert "usage: robin-worker run" in done.stderr


@pytest.mark.parametrize("name", ["task.json", "work.json"], ids=["not work.json", "gs"])
def test_a_work_uri_with_no_result_location_writes_nothing(tmp_path, name):
    folder = tmp_path / "tasks" / "t"
    folder.mkdir(parents=True)
    uri = str(folder / name) if name == "task.json" else "gs://b/tasks/t/work.json"

    done = run_worker(tmp_path, ["run", uri], test_versions=False)

    assert done.returncode == 2
    assert uri in done.stderr and "no result was written" in done.stderr
    assert list(folder.iterdir()) == []


# ---------------------------------------------------------------------------
# Stop signals.
# ---------------------------------------------------------------------------


def wait_for(path, worker, *, seconds):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if worker.poll() is not None:
            raise AssertionError(f"the worker ended first: {worker.communicate()[1]}")
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} did not appear in {seconds} seconds")
        time.sleep(0.05)


@pytest.mark.parametrize(
    ("number", "exit_code"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)], ids=["TERM", "INT"]
)
def test_a_stop_signal_writes_no_result_and_cleans_up(
    tmp_path, builder, test_model_site, number, exit_code
):
    work, _ = builder.write()
    marker = tmp_path / "model-started"
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    env = {STARTED_VARIABLE: str(marker), "TMPDIR": str(temporary)}
    worker = popen_worker(tmp_path, ["run", str(work)], site=test_model_site, env=env)
    try:
        wait_for(marker, worker, seconds=60)
        assert list(temporary.glob("robin-work-*")), "the engine's work folder should exist"
        worker.send_signal(number)
        _, stderr = worker.communicate(timeout=60)
    finally:
        worker.kill()

    assert worker.returncode == exit_code, stderr
    assert "stopped by signal" in stderr
    assert "unless the stop came while the result was being written" in stderr
    assert not builder.result_path.exists()
    assert list(temporary.iterdir()) == []


def test_a_stop_ignores_later_stop_signals_so_cleanup_can_finish():
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with pytest.raises(cli._Stopped):
            cli._stop(signal.SIGTERM, None)

        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
