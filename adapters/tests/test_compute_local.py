import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from robin_contracts.ports.compute import ComputeRequest, ComputeTask, SubmissionRejected

from robin_adapters.compute.local.adapter import LocalCompute, get_state_folder

# Stands in for `robin-worker`. It writes a marker that holds its arguments and a file
# that holds the variables of its environment whose names start with `ROBIN_TEST_` or
# `FAKE_WORKER_`, and no others, so no credential is written to disk. It prints a line
# to stdout and to stderr, and then acts by its mode. A child that it starts ends its
# arguments with `child <uri>`, so only the process group links the child to the task.
# For a work URI that is not a `file:` URI, it writes in the folder that
# `FAKE_WORKER_FOLDER` names.
WORKER_SCRIPT = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    from pathlib import Path
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    if sys.argv[-1].startswith("file:"):
        folder = Path(url2pathname(urlsplit(sys.argv[-1]).path)).parent
    else:
        folder = Path(os.environ["FAKE_WORKER_FOLDER"])

    def write_marker(name, text):
        temporary = folder / f".{name}.tmp"
        temporary.write_text(text)
        os.replace(temporary, folder / name)

    def wait_for_release():
        deadline = time.monotonic() + 60
        while not (folder / "release").exists() and time.monotonic() < deadline:
            time.sleep(0.05)

    if sys.argv[1] == "child":
        write_marker(f"child-{os.getpid()}", "")
        wait_for_release()
        sys.exit()

    reported = {
        name: value for name, value in os.environ.items()
        if name.startswith(("ROBIN_TEST_", "FAKE_WORKER_"))
    }
    write_marker(f"environment-{os.getpid()}", json.dumps(reported))
    write_marker(f"marker-{os.getpid()}", json.dumps(sys.argv[1:]))
    print("to stdout", flush=True)
    print("to stderr", file=sys.stderr, flush=True)
    mode = os.environ["FAKE_WORKER_MODE"]
    if mode in ("child", "child-wait"):
        subprocess.Popen([sys.executable, __file__, "child", sys.argv[-1]])
    if mode in ("wait", "child-wait"):
        wait_for_release()
    """
)


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(WORKER_SCRIPT)
    return path


@pytest.fixture
def state_folder(tmp_path):
    return tmp_path / "state"


@pytest.fixture
def make_task(tmp_path, state_folder):
    """Makes a task with its own folder. At teardown, it ends every process in the folder.

    With `work_on_s3=True`, the task's work URI is an `s3://` URI, and its folder is in
    `state_folder`, where the adapter makes it.
    """
    folders = []

    def make(compute_ref=None, *, work_on_s3=False, storage_env_vars=None):
        task_id = uuid4()
        if work_on_s3:
            folder = state_folder / "tasks" / str(task_id)
            work_uri = f"s3://bucket/robin/runs/R/tasks/{task_id}/work.json"
        else:
            folder = tmp_path / f"task-{len(folders)}"
            folder.mkdir()
            work_uri = (folder / "work.json").as_uri()
        folders.append(folder)
        return ComputeTask(
            task_id=task_id,
            run_id=uuid4(),
            setup="local",
            work_uri=work_uri,
            settings={"kind": "local"},
            request=ComputeRequest(),
            compute_ref=compute_ref,
            storage_env_vars=storage_env_vars or {},
        )

    yield make
    for folder in folders:
        if not folder.exists():
            continue
        (folder / "release").touch()
        for marker in [*folder.glob("marker-*"), *folder.glob("child-*")]:
            try:
                os.killpg(int(marker.name.split("-")[1]), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _compute(script, mode="wait", *, prefix=()):
    return LocalCompute(
        command=(sys.executable, *prefix, str(script)), env={"FAKE_WORKER_MODE": mode}
    )


def _s3_compute(script, state_folder, task, mode="wait"):
    return LocalCompute(
        command=(sys.executable, str(script)),
        env={
            "FAKE_WORKER_MODE": mode,
            "FAKE_WORKER_FOLDER": str(_state_task_folder(state_folder, task)),
        },
        state_folder=state_folder,
    )


def _folder(task):
    return Path(task.work_uri.removeprefix("file://")).parent


def _state_task_folder(state_folder, task):
    return state_folder / "tasks" / str(task.task_id)


def _with_ref(task, compute_ref):
    return task.model_copy(update={"compute_ref": compute_ref})


def _wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not hold in time")
        time.sleep(0.05)


def _markers(task, kind="marker"):
    return _markers_in(_folder(task), kind)


def _markers_in(folder, kind="marker"):
    return sorted(folder.glob(f"{kind}-*"))


def _read_worker_environment(folder):
    [environment] = _markers_in(folder, "environment")
    return json.loads(environment.read_text())


def _is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_submit_starts_the_command_with_the_work_path(script, make_task):
    task = make_task()

    compute_ref = _compute(script).submit(task)

    _wait_for(lambda: _markers(task))
    [marker] = _markers(task)
    assert json.loads(marker.read_text()) == ["run", task.work_uri]
    assert compute_ref == marker.name.split("-")[1]
    log = _folder(task) / "worker.log"
    _wait_for(lambda: "to stderr" in log.read_text())
    assert "to stdout" in log.read_text()


def test_a_repeated_submit_while_the_task_runs_starts_nothing(script, make_task):
    task = make_task()
    compute = _compute(script)
    first = compute.submit(task)
    _wait_for(lambda: _markers(task))

    assert compute.submit(task) == first
    time.sleep(0.5)
    assert len(_markers(task)) == 1


def test_concurrent_submits_start_one_process(script, make_task):
    task = make_task()
    barrier = threading.Barrier(2)
    refs = []

    def submit():
        compute = _compute(script)
        barrier.wait()
        refs.append(compute.submit(task))

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    _wait_for(lambda: _markers(task))
    time.sleep(0.5)
    assert len(_markers(task)) == 1
    assert refs[0] == refs[1]


def _submit_from_another_process(script, task, mode="wait"):
    code = textwrap.dedent(
        f"""
        import sys
        from robin_contracts.ports.compute import ComputeTask
        from robin_adapters.compute.local.adapter import LocalCompute, get_state_folder

        task = ComputeTask.model_validate_json({task.model_dump_json()!r})
        compute = LocalCompute(
            command=(sys.executable, {str(script)!r}), env={{"FAKE_WORKER_MODE": {mode!r}}}
        )
        print(compute.submit(task))
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_the_process_outlives_the_caller(script, make_task):
    task = make_task()

    compute_ref = _submit_from_another_process(script, task)

    status = _compute(script).describe(_with_ref(task, compute_ref))
    assert (status.state, status.compute_ref) == ("running", compute_ref)


@pytest.mark.parametrize("started_by", ["this adapter", "another process"])
def test_describe_gives_running_then_not_found_after_the_process_ends(
    script, make_task, started_by
):
    task = make_task()
    compute = _compute(script)
    if started_by == "this adapter":
        compute_ref = compute.submit(task)
    else:
        compute_ref = _submit_from_another_process(script, task)
    task = _with_ref(task, compute_ref)
    assert compute.describe(task).state == "running"

    (_folder(task) / "release").touch()

    _wait_for(lambda: compute.describe(task).state == "not_found")


def test_describe_leaves_other_child_processes_alone(script, make_task):
    other = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    time.sleep(1)

    assert _compute(script).describe(make_task()).state == "not_found"

    assert other.wait(timeout=5) == 7


def test_describe_ignores_a_process_for_another_task(script, make_task):
    task, other_task = make_task(), make_task()
    _compute(script).submit(other_task)
    _wait_for(lambda: _markers(other_task))
    near_misses = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", *arguments],
            start_new_session=True,
        )
        for arguments in (["run", task.work_uri + ".bak"], ["run", task.work_uri, "extra"])
    ]
    try:
        time.sleep(0.5)
        assert _compute(script).describe(task).state == "not_found"
    finally:
        for process in near_misses:
            process.kill()
            process.wait()


def test_a_changed_command_still_finds_the_running_worker(script, make_task):
    task = make_task()
    first = _compute(script).submit(task)
    _wait_for(lambda: _markers(task))
    changed = _compute(script, prefix=("-X", "dev"))

    assert changed.submit(task) == first
    assert changed.describe(task).state == "running"
    time.sleep(0.5)
    assert len(_markers(task)) == 1


def test_a_worker_child_that_outlives_the_worker_keeps_the_task_running(script, make_task):
    task = make_task()
    compute = _compute(script, "child")
    task = _with_ref(task, compute.submit(task))
    _wait_for(lambda: _markers(task, "child"))
    worker = int(task.compute_ref)

    def has_worker_ended():
        compute.describe(task)  # collects the worker after it exits
        return not _is_alive(worker)

    _wait_for(has_worker_ended)
    assert compute.describe(task).state == "running"

    (_folder(task) / "release").touch()
    _wait_for(lambda: compute.describe(task).state == "not_found")


def test_a_submit_while_a_worker_child_runs_starts_nothing(script, make_task):
    task = make_task()
    compute = _compute(script, "child")
    task = _with_ref(task, compute.submit(task))
    _wait_for(lambda: _markers(task, "child"))
    worker = int(task.compute_ref)

    def has_worker_ended():
        compute.describe(task)  # collects the worker after it exits
        return not _is_alive(worker)

    _wait_for(has_worker_ended)

    assert compute.submit(task) == task.compute_ref
    assert len(_markers(task)) == 1


@pytest.mark.parametrize(
    "ps_script", ["exit 1", "echo 'not a process table'"], ids=["fails", "unparsable"]
)
def test_a_failed_process_table_read_is_unknown(script, make_task, tmp_path, monkeypatch, ps_script):
    bin_folder = tmp_path / "bin"
    bin_folder.mkdir()
    ps = bin_folder / "ps"
    ps.write_text(f"#!/bin/sh\n{ps_script}\n")
    ps.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_folder}{os.pathsep}{os.environ['PATH']}")
    task = make_task()
    compute = _compute(script)

    with pytest.raises(Exception):
        compute.describe(task)
    with pytest.raises(Exception) as raised:
        compute.submit(task)

    assert not isinstance(raised.value, SubmissionRejected)
    time.sleep(0.5)
    assert _markers(task) == []


@pytest.mark.parametrize("problem", ["missing", "not executable"])
def test_a_command_that_cannot_start_is_submit_rejected(make_task, tmp_path, problem):
    command = tmp_path / "robin-worker"
    if problem == "not executable":
        command.write_text("#!/bin/sh\n")
        command.chmod(0o644)

    with pytest.raises(SubmissionRejected):
        LocalCompute(command=(str(command),)).submit(make_task())


def test_stop_ends_the_process_group(script, make_task):
    task = make_task()
    compute = _compute(script, "child-wait")
    task = _with_ref(task, compute.submit(task))
    _wait_for(lambda: _markers(task, "child"))
    [child_marker] = _markers(task, "child")
    child = int(child_marker.name.split("-")[1])

    compute.stop(task)

    _wait_for(lambda: compute.describe(task).state == "not_found")
    _wait_for(lambda: not _is_alive(child))
    compute.stop(task)


def test_release_changes_nothing(script, make_task):
    task = make_task(compute_ref="12345")
    folder = _folder(task)
    (folder / "work.json").write_bytes(b"work")
    (folder / "result.json").write_bytes(b"result")
    (folder / "outputs").mkdir()
    (folder / "outputs" / "scores.parquet").write_bytes(b"scores")

    _compute(script).release(task)

    assert sorted(path.name for path in folder.rglob("*")) == [
        "outputs", "result.json", "scores.parquet", "work.json"
    ]
    assert (folder / "work.json").read_bytes() == b"work"
    assert (folder / "result.json").read_bytes() == b"result"
    assert (folder / "outputs" / "scores.parquet").read_bytes() == b"scores"


def test_submit_gives_the_worker_the_storage_environment(script, make_task):
    task = make_task(storage_env_vars={"ROBIN_TEST_STORAGE_VARIABLE": "set"})

    _compute(script).submit(task)

    _wait_for(lambda: _markers(task))
    environment = _read_worker_environment(_folder(task))
    assert environment["ROBIN_TEST_STORAGE_VARIABLE"] == "set"
    assert environment["FAKE_WORKER_MODE"] == "wait"


def test_an_empty_storage_environment_changes_no_variable(script, make_task, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_MODE", "wait")
    monkeypatch.setenv("ROBIN_TEST_KEPT_VARIABLE", "kept")
    task = make_task()

    LocalCompute(command=(sys.executable, str(script))).submit(task)

    _wait_for(lambda: _markers(task))
    environment = _read_worker_environment(_folder(task))
    assert environment == {"FAKE_WORKER_MODE": "wait", "ROBIN_TEST_KEPT_VARIABLE": "kept"}


def test_submit_of_an_s3_work_uri_starts_the_worker_with_that_uri(script, make_task, state_folder):
    task = make_task(work_on_s3=True)
    folder = _state_task_folder(state_folder, task)

    compute_ref = _s3_compute(script, state_folder, task).submit(task)

    _wait_for(lambda: _markers_in(folder))
    [marker] = _markers_in(folder)
    assert json.loads(marker.read_text()) == ["run", task.work_uri]
    assert compute_ref == marker.name.split("-")[1]
    log = folder / "worker.log"
    _wait_for(lambda: "to stderr" in log.read_text())
    assert "to stdout" in log.read_text()


def test_a_repeated_submit_of_an_s3_work_uri_returns_the_running_worker(
    script, make_task, state_folder
):
    task = make_task(work_on_s3=True)
    folder = _state_task_folder(state_folder, task)
    compute = _s3_compute(script, state_folder, task)
    first = compute.submit(task)
    _wait_for(lambda: _markers_in(folder))

    assert compute.submit(task) == first
    time.sleep(0.5)
    assert len(_markers_in(folder)) == 1


def test_two_submits_at_once_of_an_s3_work_uri_start_one_worker(script, make_task, state_folder):
    task = make_task(work_on_s3=True)
    folder = _state_task_folder(state_folder, task)
    barrier = threading.Barrier(2)
    refs = []

    def submit():
        compute = _s3_compute(script, state_folder, task)
        barrier.wait()
        refs.append(compute.submit(task))

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    _wait_for(lambda: _markers_in(folder))
    time.sleep(0.5)
    assert len(_markers_in(folder)) == 1
    assert refs[0] == refs[1]


def test_describe_and_stop_find_an_s3_worker_without_a_compute_ref(
    script, make_task, state_folder
):
    task = make_task(work_on_s3=True)
    folder = _state_task_folder(state_folder, task)
    starting = _s3_compute(script, state_folder, task)
    compute_ref = starting.submit(task)
    _wait_for(lambda: _markers_in(folder))
    recovering = _s3_compute(script, state_folder, task)

    status = recovering.describe(task)
    recovering.stop(task)

    assert (status.state, status.compute_ref) == ("running", compute_ref)
    # Only the adapter that started the worker collects it after it ends.
    _wait_for(lambda: starting.describe(task).state == "not_found")
    assert recovering.describe(task).state == "not_found"


def test_release_of_an_s3_task_is_safe(script, make_task, state_folder):
    task = make_task(work_on_s3=True, compute_ref="12345")
    compute = _s3_compute(script, state_folder, task)

    compute.release(task)
    compute.release(task)


@pytest.mark.parametrize("is_set", [True, False], ids=["set", "not set"])
def test_the_state_folder_comes_from_xdg_state_home(tmp_path, monkeypatch, is_set):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if is_set:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        expected = tmp_path / "state" / "robin" / "local-compute"
    else:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        expected = tmp_path / "home" / ".local" / "state" / "robin" / "local-compute"

    assert get_state_folder() == expected


@pytest.mark.parametrize("method", ["submit", "describe", "stop", "release"])
@pytest.mark.parametrize(
    "work_uri",
    [
        "runs/R/tasks/T/work.json",
        "s3://bucket/my work.json",
        "/data/my work.json",
        "file://otherhost/work.json",
    ],
    ids=["relative path", "whitespace", "whitespace in a path", "file URI on another host"],
)
def test_a_work_uri_without_a_scheme_or_with_whitespace_is_refused(
    script, make_task, state_folder, method, work_uri
):
    task = make_task().model_copy(update={"work_uri": work_uri})
    compute = LocalCompute(command=(sys.executable, str(script)), state_folder=state_folder)

    with pytest.raises(ValueError):
        getattr(compute, method)(task)
    assert not state_folder.exists()
