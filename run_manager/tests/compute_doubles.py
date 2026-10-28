"""A compute adapter for tests, which a test tells what each call does."""

import itertools
import threading
from collections import defaultdict, deque
from uuid import UUID

from robin_contracts.ports.compute import ComputeStatus, ComputeTask

# An action: keep the execution, then raise TimeoutError, as a lost reply would.
ACCEPT_THEN_TIMEOUT = "accept_then_timeout"


class ScriptedCompute:
    """Keeps one execution for each task ID, and records each call in `calls`.

    `script(method, *actions)` sets what the next calls of `method` do, one action for
    each call:

    - an exception: raise it, and keep no execution;
    - `ACCEPT_THEN_TIMEOUT`: keep the execution, then raise `TimeoutError`;
    - a `threading.Event`: wait for it, then act as with no action;
    - a `ComputeStatus`: `describe` gives it.

    With no action, `submit` keeps an execution, or returns the `compute_ref` of the
    one it kept; `describe` gives `running` for a task with an execution and
    `not_found` for one without.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ComputeTask]] = []
        self.executions: dict[UUID, str] = {}
        self._actions: dict[str, deque] = defaultdict(deque)
        self._numbers = itertools.count(1)
        self._lock = threading.Lock()

    def script(self, method: str, *actions) -> None:
        with self._lock:
            self._actions[method].extend(actions)

    def get_calls(self, method: str) -> list[ComputeTask]:
        return [task for name, task in self.calls if name == method]

    def submit(self, task: ComputeTask) -> str:
        action = self._take("submit", task)
        with self._lock:
            if task.task_id not in self.executions:
                self.executions[task.task_id] = f"ref-{next(self._numbers)}"
            compute_ref = self.executions[task.task_id]
        if action == ACCEPT_THEN_TIMEOUT:
            raise TimeoutError("the reply was lost")
        return compute_ref

    def describe(self, task: ComputeTask) -> ComputeStatus:
        action = self._take("describe", task)
        if isinstance(action, ComputeStatus):
            return action
        compute_ref = self.executions.get(task.task_id)
        if compute_ref is None:
            return ComputeStatus(state="not_found")
        return ComputeStatus(state="running", compute_ref=compute_ref)

    def stop(self, task: ComputeTask) -> None:
        self._take("stop", task)

    def release(self, task: ComputeTask) -> None:
        self._take("release", task)

    def _take(self, method: str, task: ComputeTask):
        with self._lock:
            self.calls.append((method, task))
            queue = self._actions[method]
            action = queue.popleft() if queue else None
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, threading.Event):
            assert action.wait(30)
            return None
        return action
