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
- `pin_settings` gets a setup's fields from the profile, and returns the fields that a
  new run stores. A run must always use the same settings, but some profile values
  refer to something that can change. For example, a Batch job definition name
  refers to the newest revision, and a new revision can be added at any time. For
  such a value, `pin_settings` returns the exact item that it refers to now, such as
  the ARN of that revision. It returns other values unchanged. If it cannot pin a
  value, it raises `ValueError`. The message names the field, never its value.

Robin reports an exception that is not `SubmissionRejected` by its type only, because its
text can hold a credential.

Each compute kind also has a setup class, a `ComputeSetupSettings`, which checks its
section of the profile.
"""

from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from robin_contracts.work import FiniteScalarMap, NonEmptyText

# What a location is for: an audio file, a file of the model, an embeddings file, or the
# storage root of a run.
LocationRole = Literal["recording", "model_file", "embeddings", "storage_root"]

# The form of a location: an absolute path, a `file://` URI, an `s3://` URI, or a
# `robin-models:` location of a built-in model's file.
LocationForm = Literal["path", "file", "s3", "bundled"]


class ComputeRequest(BaseModel, frozen=True, extra="forbid"):
    """What the task asks of the compute service: resource requests, limits, and a deadline.

    Names and values use the compute service's own forms, such as `cpu: "2"` or
    `memory: 8Gi`.
    """

    requests: Mapping[NonEmptyText, NonEmptyText] = {}
    limits: Mapping[NonEmptyText, NonEmptyText] = {}
    active_deadline_seconds: Annotated[int, Field(strict=True, ge=1)] | None = None


class ComputeSetupSettings(BaseModel):
    """One named compute setup in the profile: where its tasks run, its capacity, and
    its defaults. Each compute kind has a subclass, which names its own fields.

    `max_in_flight` is the most tasks of this setup name that run at the same time, for
    every user of the database. `request` is what each task asks of the compute
    service. `resources` goes into each work file for the worker's code. A model entry
    can change `request` and `resources` key by key. A field that the kind does not
    have is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The fields that say where the setup's tasks run. The storage section is also part
    # of the place.
    TARGET_FIELDS: ClassVar[tuple[str, ...]]
    # The forms of location that the kind's tasks can read, for each role.
    ACCEPTED_FORMS: ClassVar[Mapping[LocationRole, tuple[LocationForm, ...]]]
    # The declared fields that only the checks at submit use. A run does not keep them.
    SUBMIT_ONLY_FIELDS: ClassVar[tuple[str, ...]] = ()

    kind: str
    max_in_flight: int = Field(default=10, ge=1)
    request: ComputeRequest = ComputeRequest()
    resources: FiniteScalarMap = {}

    def __repr_args__(self):
        # Any field other than `kind` can hold a value that must not be shown, such as an
        # account number, so only the field names are shown.
        yield "kind", self.kind
        yield "fields", sorted(set(type(self).model_fields) - {"kind"})

    def check_new_run(self, request: ComputeRequest) -> None:
        """Raises `ValueError` if a new run cannot use this setup with `request`, the
        merged request of its model.

        Profile load does not make these checks, so that a bad model entry stops only
        the submits of that model. The message names the field, never its value.
        """

    def is_pinning_required(self) -> bool:
        """Returns True if a run cannot keep the setup's declared fields as the profile
        gives them, so that a new run needs the adapter's `pin_settings`."""
        return False


class ComputeTask(BaseModel, frozen=True):
    """The task that a compute call acts on.

    `settings` are the run's frozen compute settings. `request` is the request stored on
    the task when its slot was reserved. `compute_ref` is None until a submit is
    confirmed.

    `storage_env_vars` holds the environment variables that the worker needs to reach
    the storage, such as an S3 endpoint. The adapter adds them to the worker's
    environment.

    `run_id` and `setup` (the setup name) are for tags and labels, so that a compute
    service's cost reports can split cost by run and by setup. An adapter finds an
    execution by `task_id`, never by these.
    """

    task_id: UUID
    run_id: UUID
    setup: NonEmptyText
    work_uri: NonEmptyText
    settings: Mapping[str, JsonValue]
    request: ComputeRequest
    compute_ref: str | None = None
    storage_env_vars: Mapping[str, str] = {}


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
    def pin_settings(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]: ...
