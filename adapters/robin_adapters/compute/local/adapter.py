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

from robin_contracts.local_files import get_local_path, has_uri_scheme
from robin_contracts.ports.compute import ComputeStatus, ComputeTask, SubmissionRejected

_PROCESS_TABLE_COMMAND = ("ps", "-ww", "-A", "-o", "pid=,pgid=,args=")
_PROCESS_TABLE_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class _Process:
    pid: int
    group: int
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class _TaskFolder:
    """The folder that holds a task's lock and log, and whether the adapter makes it."""

    path: Path
    is_made_by_adapter: bool


class LocalCompute:
    """Runs each task's worker as a process on this computer.

    For each task, it starts `[*command, "run", <work URI>]` in a new session. The
    process keeps running after the caller ends. A later call finds it again in the
    process table, by its last two arguments.

    The worker's process ID is the task's `compute_ref`, and it is also the ID of the
    worker's process group. The processes that the worker starts join that group, so
    the task is running while any process in the group is running. A process that
    starts its own session leaves the group, and the adapter cannot see it.

    Each task has a folder that holds its lock and `worker.log`. The log gets the
    worker's stdout and stderr.

    - For a local work file, the folder is the one that holds `work.json`.
    - For a work file at a URI with another scheme, such as `s3://`, the folder is
      `tasks/<task ID>` in `state_folder`. By default, `state_folder` is
      `get_state_folder()`.

    The adapter never reads the work file, so it does not check which stores Robin
    supports.

    The worker's environment is this process's environment, with the variables in
    `env` and the task's `storage_env_vars` added. The adapter ignores the compute
    request: a local worker is not held to its requests, limits, or deadline.
    """

    def __init__(
        self,
        command: Sequence[str] = ("robin-worker",),
        env: Mapping[str, str] | None = None,
        state_folder: Path | None = None,
    ) -> None:
        self._command = tuple(command)
        self._env = None if env is None else {**os.environ, **env}
        self._state_folder = get_state_folder() if state_folder is None else state_folder
        self._started: list[subprocess.Popen[bytes]] = []
        self._started_lock = threading.Lock()

    def submit(self, task: ComputeTask) -> str:
        folder = self._get_task_folder(task)
        if folder.is_made_by_adapter:
            self._make_task_folder(folder.path)
        with _lock_task_folder(folder.path):
            self._collect_started()
            compute_ref = _find_execution(_read_process_table(), task)
            if compute_ref is not None:
                return compute_ref
            return str(self._start(task, folder.path))

    def describe(self, task: ComputeTask) -> ComputeStatus:
        self._get_task_folder(task)
        self._collect_started()
        compute_ref = _find_execution(_read_process_table(), task)
        if compute_ref is not None:
            return ComputeStatus(state="running", compute_ref=compute_ref)
        return ComputeStatus(state="not_found")

    def stop(self, task: ComputeTask) -> None:
        self._get_task_folder(task)
        group = task.compute_ref
        if group is None:
            worker = _find_worker(_read_process_table(), task.work_uri)
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
        self._get_task_folder(task)

    def _get_task_folder(self, task: ComputeTask) -> _TaskFolder:
        """The folder that holds the task's lock and log, and whether `submit` makes it.

        - For a local work file, given as an absolute path or a `file:` URI, it is the
          folder that holds `work.json`. The run manager made it when it wrote the file.
        - For a work file at a URI with another scheme, such as `s3://`, it is
          `tasks/<task ID>` in the state folder. `submit` makes it.

        Raises `ValueError` if the work URI holds whitespace, is a relative path, or is a
        `file:` URI that does not name a local path.
        """
        work_uri = task.work_uri
        # The adapter finds the worker by its last argument, so the URI must not split
        # into two arguments.
        if any(character.isspace() for character in work_uri):
            raise ValueError("the work URI holds whitespace")
        if not has_uri_scheme(work_uri) or work_uri.lower().startswith("file:"):
            # get_local_path refuses a relative path and a file: URI on another host.
            return _TaskFolder(get_local_path(work_uri).parent, is_made_by_adapter=False)
        folder = self._state_folder / "tasks" / str(task.task_id)
        return _TaskFolder(folder, is_made_by_adapter=True)

    def _make_task_folder(self, folder: Path) -> None:
        # Only the user who runs Robin can open these folders, because a worker log can
        # show storage paths and error messages.
        for path in (self._state_folder, self._state_folder / "tasks", folder):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _get_process_env_vars(self, task: ComputeTask) -> dict[str, str] | None:
        """Returns the environment for the worker's process, or None to use this
        process's environment unchanged."""
        if self._env is None and not task.storage_env_vars:
            return None
        return {**(self._env or os.environ), **task.storage_env_vars}

    def _start(self, task: ComputeTask, folder: Path) -> int:
        with open(folder / "worker.log", "ab") as log:
            try:
                process = subprocess.Popen(
                    [*self._command, "run", task.work_uri],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._get_process_env_vars(task),
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


def get_state_folder() -> Path:
    """Returns the folder for the locks and worker logs of tasks whose work file is not
    on this computer.

    The folder is `$XDG_STATE_HOME/robin/local-compute`. If `XDG_STATE_HOME` is not set,
    or is not an absolute path, it is `~/.local/state/robin/local-compute`.
    """
    state_home = os.environ.get("XDG_STATE_HOME", "")
    base = Path(state_home) if os.path.isabs(state_home) else Path.home() / ".local" / "state"
    return base / "robin" / "local-compute"


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


def _find_execution(processes: list[_Process], task: ComputeTask) -> str | None:
    """Finds the task's execution and returns its `compute_ref`, or None if no part of
    it is running.

    An execution is the worker and every process that it starts. They share one
    process group, and the group's ID is the worker's process ID, which is the
    `compute_ref`.

    - If the worker is running, found by its last two arguments, it returns the
      worker's process ID.
    - If the worker ended, but a process in the group of the task's `compute_ref` is
      still running, it returns that `compute_ref`.
    """
    worker = _find_worker(processes, task.work_uri)
    if worker is not None:
        return str(worker.pid)
    if task.compute_ref is not None:
        group = int(task.compute_ref)
        if any(process.group == group for process in processes):
            return task.compute_ref
    return None


def _find_worker(processes: list[_Process], work_uri: str) -> _Process | None:
    # `_get_task_folder` refuses a work URI with whitespace, so the URI is one field in
    # the process table.
    for process in processes:
        if process.arguments[-2:] == ("run", work_uri):
            return process
    return None
