"""Compute on this computer: `LocalCompute` runs the worker as a detached process."""

import fcntl
import os
import signal
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import ComputeStatus, ComputeTask, SubmissionRejected

_PROCESS_TABLE_COMMAND = ("ps", "-ww", "-A", "-o", "pid=,pgid=,args=")
_PROCESS_TABLE_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class _Process:
    pid: int
    group: int
    arguments: tuple[str, ...]


class LocalCompute:
    """Runs `[*command, "run", <work URI>]` in a new session, one process for each task.

    The process outlives the caller. A later call finds it again in the process table,
    by its last two arguments. `compute_ref` is the worker's process ID, which is also
    its process group, so the task runs while any process stays in that group. A
    process that starts a session of its own leaves the group, and this adapter does
    not see it. The worker's stdout and stderr go to `worker.log` in the task folder.

    `env` adds variables to this process's environment for the worker. The adapter
    ignores the compute request: a local worker is not held to its requests, limits,
    or deadline.
    """

    def __init__(
        self, command: Sequence[str] = ("robin-worker",), env: Mapping[str, str] | None = None
    ) -> None:
        self._command = tuple(command)
        self._env = None if env is None else {**os.environ, **env}
        self._started: list[subprocess.Popen[bytes]] = []
        self._started_lock = threading.Lock()

    def submit(self, task: ComputeTask) -> str:
        work_path = _check_work_path(task)
        work_uri = work_path.as_uri()
        with _lock_task_folder(work_path.parent):
            self._collect_started()
            compute_ref = _find_execution(_read_process_table(), task, work_uri)
            if compute_ref is not None:
                return compute_ref
            return str(self._start(work_path.parent, work_uri))

    def describe(self, task: ComputeTask) -> ComputeStatus:
        work_uri = _check_work_path(task).as_uri()
        self._collect_started()
        compute_ref = _find_execution(_read_process_table(), task, work_uri)
        if compute_ref is not None:
            return ComputeStatus(state="running", compute_ref=compute_ref)
        return ComputeStatus(state="not_found")

    def stop(self, task: ComputeTask) -> None:
        work_uri = _check_work_path(task).as_uri()
        group = task.compute_ref
        if group is None:
            worker = _find_worker(_read_process_table(), work_uri)
            if worker is None:
                return
            group = str(worker.pid)
        try:
            os.killpg(int(group), signal.SIGTERM)
        except ProcessLookupError:
            pass

    def release(self, task: ComputeTask) -> None:
        # Local compute keeps no record of a task, so nothing is freed. The check makes a
        # task sent to the wrong adapter fail, so that it is not marked as released.
        _check_work_path(task)

    def _start(self, folder: Path, work_uri: str) -> int:
        with open(folder / "worker.log", "ab") as log:
            try:
                process = subprocess.Popen(
                    [*self._command, "run", work_uri],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._env,
                    start_new_session=True,
                )
            except (FileNotFoundError, PermissionError) as error:
                raise SubmissionRejected(
                    f"the worker command cannot start: {error.strerror}"
                ) from None
        with self._started_lock:
            self._started.append(process)
        return process.pid

    def _collect_started(self) -> None:
        # Only this adapter's own processes are polled, so a finished worker is not
        # seen as running, and no other child process of the caller loses its status.
        with self._started_lock:
            self._started = [process for process in self._started if process.poll() is None]


def _check_work_path(task: ComputeTask) -> Path:
    """The local path of the task's work file. Raises `ValueError` if the URI is not local."""
    return get_local_path(task.work_uri)


@contextmanager
def _lock_task_folder(folder: Path) -> Iterator[None]:
    # The lock covers the search and the start, so two callers cannot both start the task.
    descriptor = os.open(folder / ".compute.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _read_process_table() -> list[_Process]:
    """Every process on this computer. Raises if `ps` fails or its output cannot be parsed."""
    done = subprocess.run(
        _PROCESS_TABLE_COMMAND,
        capture_output=True,
        text=True,
        check=True,
        timeout=_PROCESS_TABLE_TIMEOUT_SECONDS,
    )
    processes = []
    for line in done.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
            raise ValueError("the process table has a line that cannot be parsed")
        processes.append(_Process(int(fields[0]), int(fields[1]), tuple(fields[2:])))
    if not processes:
        raise ValueError("the process table is empty")
    return processes


def _find_execution(processes: list[_Process], task: ComputeTask, work_uri: str) -> str | None:
    """The `compute_ref` of the task's execution, or None if none runs.

    The execution runs while its worker runs, or while a process stays in the group of
    the task's `compute_ref`.
    """
    worker = _find_worker(processes, work_uri)
    if worker is not None:
        return str(worker.pid)
    if task.compute_ref is not None:
        group = int(task.compute_ref)
        if any(process.group == group for process in processes):
            return task.compute_ref
    return None


def _find_worker(processes: list[_Process], work_uri: str) -> _Process | None:
    # A file URI has no whitespace, so it is one argument in the table's text.
    for process in processes:
        if process.arguments[-2:] == ("run", work_uri):
            return process
    return None
