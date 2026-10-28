"""The compute port: how the run manager starts a task's execution and follows it.

The run manager gives each call a `ComputeTask`. An adapter promises this:

- `submit` starts an execution of the task and returns its `compute_ref`. It never
  starts a second execution while an earlier one may still run: for a task that has an
  execution, it returns that execution's `compute_ref`. It raises `SubmissionRejected` only
  when it is sure that nothing runs and nothing will run. Any other exception means
  that the submit is uncertain: an execution may have started.
- `describe` gives the state of the task's execution. `waiting` means that the compute
  service accepted the task and did not start it. `ended` means a confirmed end.
  `not_found` means that no execution of the task exists and none is active. An
  exception means that the state is unknown.
- `stop` asks the execution to end. `release` frees what the compute service keeps for
  the task after its outcome is recorded. Both are safe to repeat. `release` never
  deletes `work.json`, `result.json`, or the task's outputs.

Robin reports an exception that is not `SubmissionRejected` by its type only, because its
text can hold a credential.
"""

from collections.abc import Mapping
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue

from robin_contracts.work import NonEmptyText


class ComputeRequest(BaseModel, frozen=True, extra="forbid"):
    """What the task asks of the compute service: resource requests, limits, and a deadline.

    Names and values use the compute service's own forms, such as `cpu: "2"` or
    `memory: 8Gi`.
    """

    requests: Mapping[NonEmptyText, NonEmptyText] = {}
    limits: Mapping[NonEmptyText, NonEmptyText] = {}
    active_deadline_seconds: Annotated[int, Field(strict=True, ge=1)] | None = None


class ComputeTask(BaseModel, frozen=True):
    """The task that a compute call acts on.

    `settings` are the run's frozen compute settings. `request` is the request stored on
    the task when its slot was reserved. `compute_ref` is None until a submit is
    confirmed.
    """

    task_id: UUID
    work_uri: NonEmptyText
    settings: Mapping[str, JsonValue]
    request: ComputeRequest
    compute_ref: str | None = None


class ComputeStatus(BaseModel, frozen=True):
    """The state of a task's execution, as `describe` gives it."""

    state: Literal["waiting", "running", "ended", "not_found"]
    compute_ref: str | None = None


class SubmissionRejected(Exception):
    """The compute service refused the task. Nothing runs and nothing will run.

    Its text is stored on the task and shown, so it holds no credential and no profile
    value.
    """


class ComputeAdapter(Protocol):
    def submit(self, task: ComputeTask) -> str: ...
    def describe(self, task: ComputeTask) -> ComputeStatus: ...
    def stop(self, task: ComputeTask) -> None: ...
    def release(self, task: ComputeTask) -> None: ...
