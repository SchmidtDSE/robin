"""The files the control plane and the worker exchange for one task.

The control plane writes the work file, an `InferenceTask`. The worker runs it and
writes the result file, an `InferenceTaskResult`, beside it.
"""

from collections.abc import Callable
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, BaseModel, Field, model_validator

from robin_contracts.cards import HeadCard, ModelCard, model_ref
from robin_contracts.output_contracts import TaskContractId, TaskResultContractId
from robin_contracts.results import InferenceCompleted, InferenceFailure
from robin_contracts.work import BytesDigest, InferenceWork, NonEmptyText, work_digest

_OUTPUT_VERSION_PREFIXES = ("model:", "head_runtime:")


def output_version_id(card: ModelCard | HeadCard) -> str:
    """The identifier of the code that runs `card`: its entry point's group and key.

    A base model is `model:<name>/<version>`. A head is `head_runtime:<runtime>`,
    because every head on one runtime runs the same code.
    """
    if isinstance(card, HeadCard):
        return f"head_runtime:{card.runtime}"
    return f"model:{model_ref(card).id}"


def _an_output_version_id(value: str) -> str:
    for prefix in _OUTPUT_VERSION_PREFIXES:
        if value.startswith(prefix) and len(value) > len(prefix):
            return value
    raise ValueError(
        f"expected 'model:<name>/<version>' or 'head_runtime:<runtime>', got {value!r}"
    )


class ModelOutputVersion(BaseModel, frozen=True, extra="forbid"):
    """The output version of the code that runs one model.

    The version increases when the outputs of that code change.
    """

    model: Annotated[str, AfterValidator(_an_output_version_id)]
    version: Annotated[int, Field(strict=True, ge=1)]


class InferenceTask(BaseModel, frozen=True, extra="forbid"):
    """The work file: one work, and what the worker needs to run it as one task.

    The worker writes the work's files under `output_root`. The worker, not this
    model, refuses a task whose `output_version` is not its own code's.
    """

    schema_version: TaskContractId
    task_id: UUID
    output_root: NonEmptyText
    output_version: ModelOutputVersion
    work: InferenceWork


# Splits a location into the store it is in and the path in that store that storage
# reads, without a leading `/`. It returns None for a location storage does not support.
SplitLocation = Callable[[str], tuple[str, str] | None]

TaskRefusalCode = Literal[
    "work_file_unreadable",
    "work_file_invalid",
    "unknown_schema_version",
    "model_identifier_mismatch",
    "output_version_mismatch",
]

# A worker that could not read the work file has no checksum of its bytes.
_UNREAD_REFUSALS = frozenset({"work_file_unreadable"})
# A worker that could not parse the work file's envelope has no task ID.
_UNPARSED_REFUSALS = frozenset(
    {"work_file_unreadable", "work_file_invalid", "unknown_schema_version"}
)


class TaskRefusal(BaseModel, frozen=True, extra="forbid"):
    """The worker refused the task before inference started. The work did not fail."""

    outcome: Literal["refused"] = "refused"
    code: TaskRefusalCode
    detail: str


class WorkerError(BaseModel, frozen=True, extra="forbid"):
    """An unexpected exception the worker caught at its top level: its type and message."""

    outcome: Literal["error"] = "error"
    detail: str


class InferenceTaskResult(BaseModel, frozen=True, extra="forbid"):
    """The result file: what happened to one task, bound to the work file it read.

    `work_file_digest` is the checksum of the work file's bytes. The times are when
    the worker started reading the work file and when it finished writing outputs.
    """

    schema_version: TaskResultContractId
    task_id: UUID | None
    work_file_digest: BytesDigest | None
    robin_release: NonEmptyText
    started_at: AwareDatetime
    finished_at: AwareDatetime
    result: Annotated[
        InferenceCompleted | InferenceFailure | TaskRefusal | WorkerError,
        Field(discriminator="outcome"),
    ]

    @model_validator(mode="after")
    def _finishes_after_it_starts(self) -> "InferenceTaskResult":
        if self.finished_at < self.started_at:
            raise ValueError(
                f"finished_at {self.finished_at} is before started_at {self.started_at}"
            )
        return self

    @model_validator(mode="after")
    def _binds_to_its_task_whenever_it_can(self) -> "InferenceTaskResult":
        if self.work_file_digest is None and not self._before(_UNREAD_REFUSALS):
            raise ValueError(
                "a worker that read the work file reports its work_file_digest"
            )
        if self.task_id is None and not self._before(_UNPARSED_REFUSALS):
            raise ValueError("a worker that parsed the work file reports its task_id")
        return self

    def _before(self, refusals: frozenset[str]) -> bool:
        # A worker error can happen at any point, so it may lack either value.
        outcome = self.result
        if isinstance(outcome, WorkerError):
            return True
        return isinstance(outcome, TaskRefusal) and outcome.code in refusals


def task_result_problem(
    result: InferenceTaskResult,
    *,
    task: InferenceTask,
    work_file_digest: str,
    split_location: SplitLocation,
) -> str | None:
    """Why `result` cannot be recorded as the outcome of `task`, or None if it can.

    `work_file_digest` is the checksum of the work file's bytes, kept from when the
    work file was written. `split_location` comes from the storage that wrote the
    artifacts. A refusal or a worker error passes when it names this task's work file
    or names none; the task is then recorded as failed.
    """
    if result.task_id is not None and result.task_id != task.task_id:
        return f"the result names task {result.task_id}, not {task.task_id}"
    if result.work_file_digest is not None and result.work_file_digest != work_file_digest:
        return (
            f"the result read a work file with checksum {result.work_file_digest}, "
            f"not {work_file_digest}"
        )
    outcome = result.result
    if not isinstance(outcome, (InferenceCompleted, InferenceFailure)):
        return None
    expected = work_digest(task.work)
    if outcome.work_digest != expected:
        return f"the result is for work {outcome.work_digest}, not {expected}"
    if isinstance(outcome, InferenceFailure):
        return None
    return _recordings_problem(outcome, task.work) or _location_problem(
        outcome, task.output_root, split_location
    )


def _recordings_problem(outcome: InferenceCompleted, work: InferenceWork) -> str | None:
    # The result's own validators already refuse a recording named twice.
    assigned = {(one.namespace, one.value) for one in work.recordings}
    reported = {(one.namespace, one.value) for one in (*outcome.coverage, *outcome.failed)}
    missing, extra = sorted(assigned - reported), sorted(reported - assigned)
    if not missing and not extra:
        return None
    return (
        f"the result must report each assigned recording once; it omits {missing} "
        f"and adds {extra}"
    )


def _location_problem(
    outcome: InferenceCompleted, output_root: str, split_location: SplitLocation
) -> str | None:
    for artifact in outcome.artifacts:
        if not _is_under(artifact.uri, output_root, split_location):
            return f"artifact {artifact.uri} is not under the task's output_root {output_root}"
    return None


def _is_under(uri: str, root: str, split_location: SplitLocation) -> bool:
    # A pure check cannot resolve links, so it compares the paths storage would read.
    # A segment that is empty, `.`, or `..` could name a place outside the root.
    root_place, uri_place = split_location(root), split_location(uri)
    if root_place is None or uri_place is None:
        return False
    (root_store, root_path), (uri_store, uri_path) = root_place, uri_place
    root_parts = tuple(root_path.removesuffix("/").split("/")) if root_path else ()
    uri_parts = tuple(uri_path.split("/"))
    if uri_store != root_store or uri_parts[: len(root_parts)] != root_parts:
        return False
    rest = uri_parts[len(root_parts):]
    return bool(rest) and all(
        segment not in ("", ".", "..") for segment in (*root_parts, *rest)
    )
