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

from robin_adapters.compute.local import LocalCompute

# Stands in for `robin-worker`. It writes a marker that holds its arguments, prints a line
# to stdout and to stderr, and then acts by its mode. A child that it starts ends its
# arguments with `child <uri>`, so only the process group links the child to the task.
WORKER_SCRIPT = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    from pathlib import Path
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    folder = Path(url2pathname(urlsplit(sys.argv[-1]).path)).parent

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
def make_task(tmp_path):
    """Makes a task with its own folder. At teardown, every process in it is ended."""
    folders = []

    def make(compute_ref=None):
        folder = tmp_path / f"task-{len(folders)}"
        folder.mkdir()
        folders.append(folder)
        return ComputeTask(
            task_id=uuid4(),
            work_uri=(folder / "work.json").as_uri(),
            settings={"kind": "local"},
            request=ComputeRequest(),
            compute_ref=compute_ref,
        )

    yield make
    for folder in folders:
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


def _folder(task):
    return Path(task.work_uri.removeprefix("file://")).parent


def _with_ref(task, compute_ref):
    return task.model_copy(update={"compute_ref": compute_ref})


def _wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not hold in time")
        time.sleep(0.05)


def _markers(task, kind="marker"):
    return sorted(_folder(task).glob(f"{kind}-*"))


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
        from robin_adapters.compute.local import LocalCompute

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


@pytest.mark.parametrize("method", ["submit", "describe", "stop", "release"])
def test_a_work_uri_that_is_not_local_is_refused(script, make_task, method):
    task = make_task().model_copy(update={"work_uri": "s3://bucket/work.json"})

    with pytest.raises(ValueError):
        getattr(_compute(script), method)(task)
