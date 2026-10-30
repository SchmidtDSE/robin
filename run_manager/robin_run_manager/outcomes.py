"""Task updates from the compute adapter, outcome recording, and release.

An update asks the compute adapter about a `submitted` or `running` task, and moves the
task to the state that the answer shows. When the execution has ended, the task becomes
`finalizing` and frees its slot. Outcome recording then reads the task's `result.json` outside any transaction, checks it, and
records the task, its recordings, and its output files in one transaction. Release then
tells the compute adapter that it can free what it kept for the task's execution.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Json
from pydantic import ValidationError

from robin_contracts.ports.compute import ComputeAdapter, ComputeRequest, ComputeStatus, ComputeTask
from robin_contracts.ports.storage import TaskStorage
from robin_contracts.results import InferenceCompleted, InferenceFailure
from robin_contracts.tasks import (
    InferenceTaskResult,
    SplitLocation,
    TaskRefusal,
    task_result_problem,
)
from robin_run_manager.database import Database, lock_task, transaction, transaction_on
from robin_run_manager.errors import (
    INVALID_RESULT,
    NO_RESULT,
    WORKER_ERROR,
    ComputeUnavailable,
    invoke_adapter,
)
from robin_run_manager.launch import MATCHES_SETUP_WITH_SUPPORTED_STORAGE, get_compute_status
from robin_run_manager.task_files import WorkFile, build_reserved_work_file, read_result

_NO_RESULT_MESSAGE = (
    "The task's execution ended, and it wrote no result file. The worker may have stopped "
    "before its end."
)
_INVALID_RESULT_MESSAGE = (
    "The task's result file cannot be recorded, so none of its outputs are published. The "
    "detail says why."
)
_REFUSAL_MESSAGE = "The worker refused the task before inference started. The detail says why."
_WORKER_ERROR_MESSAGE = (
    "The worker stopped with an error that it did not expect. The detail holds the error."
)

# The states that a task can have after `update_task_from_compute`.
TaskStateAfterUpdate = Literal["submitted", "running", "finalizing"]

# The task states that are terminal. A task in one of them that was reserved is released.
_TERMINAL_STATES = ("succeeded", "failed", "cancelled")

# The column that records when a task reached each of these states.
_TIME_COLUMN_FOR_STATE = {"running": "started_at", "finalizing": "ended_at"}


@dataclass(frozen=True)
class _TaskToUpdate:
    """A task that Robin updates from the compute adapter: its state, and the task as
    the compute adapter gets it."""

    state: Literal["submitted", "running"]
    compute_task: ComputeTask


@dataclass(frozen=True)
class _Succeeded:
    """A task whose result is a completed work: its recordings are recorded from it."""

    completed: InferenceCompleted
    robin_release: str


@dataclass(frozen=True)
class _Failed:
    """A task that failed as a whole. `robin_release` is None if Robin cannot trust the
    result, or if there is none."""

    failure: Mapping[str, Any]
    robin_release: str | None = None


# What to record for a task.
_Outcome = _Succeeded | _Failed


def update_task_from_compute(
    database: Database, task_id: UUID, *, setup: Mapping[str, Any], compute: ComputeAdapter
) -> TaskStateAfterUpdate | Literal["skipped"]:
    """Ask the compute adapter about a `submitted` or `running` task, and give its new state.

    `setup` comes from `get_setup_query_parameters`. A started execution makes the task
    `running`. An ended execution, or none, makes it `finalizing`, which frees its slot.
    Gives `skipped` if another pass holds the task, or if it is not a `submitted` or
    `running` task of this setup, also when it changed during the update. Raises
    `ComputeUnavailable` if the adapter fails; the task keeps its state.
    """
    with lock_task(database, task_id) as connection:
        if connection is None:
            return "skipped"
        with transaction_on(connection, read_only=True):
            task = _get_task_to_update(connection, task_id, setup)
        if task is None:
            return "skipped"
        status = get_compute_status(compute, task.compute_task)
        state = _get_next_task_state(task.state, status)
        if state == task.state:
            return state
        with transaction_on(connection):
            changed = _set_next_task_state(connection, task, state, status.compute_ref)
    return state if changed else "skipped"


def store_task_outcome(
    database: Database,
    task_id: UUID,
    *,
    setup: Mapping[str, Any],
    storage: TaskStorage,
    split_location: SplitLocation,
) -> Literal["recorded", "already_recorded", "skipped"]:
    """Read and check the result of a `finalizing` task, and record its outcome.

    `setup` comes from `get_setup_query_parameters`, and `split_location` from the storage
    that the worker wrote with. A missing result fails the task with `no_result`, and a
    result that cannot be recorded fails it with `invalid_result`. Gives
    `already_recorded` if the outcome is recorded, and `skipped` if the task is not a
    `finalizing` task of this setup. Raises `StorageUnavailable` if the result cannot be
    read; the task stays `finalizing`. Raises `WorkFileConflict` if the work file that
    Robin builds now is not the one it reserved.
    """
    with transaction(database, read_only=True) as connection:
        task = _get_task_to_store(connection, task_id, setup)
        reason = _get_reason_not_to_store(task)
        if reason is not None:
            return reason
        work_file = build_reserved_work_file(connection, task_id, task["work_file_digest"])
    outcome = _get_outcome(read_result(storage, work_file), work_file, split_location)
    with transaction(database) as connection:
        task = _lock_task_to_store(connection, task_id)
        # Another pass can record the outcome while this pass reads the result.
        reason = _get_reason_not_to_store(task)
        if reason is not None:
            return reason
        _write_task_outcome(connection, task_id, task["run_id"], outcome)
    return "recorded"


def get_tasks_to_release(
    database: Database, *, actor: str, setup: Mapping[str, Any]
) -> list[UUID]:
    """The actor's tasks of this setup that ended and are not released, oldest first.

    `setup` comes from `get_setup_query_parameters`. A task that ended before it was reserved
    has no execution, so it is not released.
    """
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT t.task_id FROM tasks t JOIN runs r USING (run_id) "
            "WHERE t.state = ANY(%(states)s) AND t.reserved_at IS NOT NULL "
            "AND t.released_at IS NULL AND r.submitted_by = %(actor)s "
            f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE} ORDER BY t.created_at, t.task_id",
            {**setup, "states": list(_TERMINAL_STATES), "actor": actor},
        ).fetchall()
    return [row["task_id"] for row in rows]


def release_task(
    database: Database, task_id: UUID, *, setup: Mapping[str, Any], compute: ComputeAdapter
) -> Literal["released", "skipped"]:
    """Release a task that ended, and record the time of the release.

    `setup` comes from `get_setup_query_parameters`. Gives `skipped` if another pass holds the
    task, or if it is not a task of this setup that ended, was reserved, and is not
    released. Raises `ComputeUnavailable` if the adapter fails; the task does not
    change, and a later pass releases it again.
    """
    with lock_task(database, task_id) as connection:
        if connection is None:
            return "skipped"
        with transaction_on(connection, read_only=True):
            compute_task = _get_task_to_release(connection, task_id, setup)
        if compute_task is None:
            return "skipped"
        invoke_adapter(
            partial(compute.release, compute_task),
            partial(_get_release_failure, task_id),
        )
        with transaction_on(connection):
            connection.execute(
                "UPDATE tasks SET released_at = now() "
                "WHERE task_id = %s AND released_at IS NULL",
                (task_id,),
            )
    return "released"


def _get_task_to_update(
    connection: psycopg.Connection, task_id: UUID, setup: Mapping[str, Any]
) -> _TaskToUpdate | None:
    """The state of a `submitted` or `running` task of this setup, and the task as the
    compute adapter gets it. None for any other task."""
    row = connection.execute(
        "SELECT t.state, t.compute_ref, t.work_uri, t.compute_request, r.execution_settings "
        "FROM tasks t JOIN runs r USING (run_id) "
        "WHERE t.task_id = %(task_id)s AND t.state IN ('submitted', 'running') "
        f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE}",
        {**setup, "task_id": task_id},
    ).fetchone()
    if row is None:
        return None
    return _TaskToUpdate(state=row["state"], compute_task=_get_compute_task(task_id, row))


def _set_next_task_state(
    connection: psycopg.Connection,
    task: _TaskToUpdate,
    state: TaskStateAfterUpdate,
    compute_ref: str | None,
) -> bool:
    """Move the task to `state`. False if its state changed after it was read.

    A known `compute_ref` is kept; an empty one gets the adapter's value.
    """
    return connection.execute(
        f"UPDATE tasks SET state = %s, {_TIME_COLUMN_FOR_STATE[state]} = now(), "
        "compute_ref = coalesce(compute_ref, %s) WHERE task_id = %s AND state = %s",
        (state, compute_ref, task.compute_task.task_id, task.state),
    ).rowcount == 1


def _get_task_to_release(
    connection: psycopg.Connection, task_id: UUID, setup: Mapping[str, Any]
) -> ComputeTask | None:
    """The task as the compute adapter gets it, if it is a task of this setup that ended,
    was reserved, and is not released. None for any other task."""
    row = connection.execute(
        "SELECT t.compute_ref, t.work_uri, t.compute_request, r.execution_settings "
        "FROM tasks t JOIN runs r USING (run_id) "
        "WHERE t.task_id = %(task_id)s AND t.state = ANY(%(states)s) "
        "AND t.reserved_at IS NOT NULL AND t.released_at IS NULL "
        f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE}",
        {**setup, "task_id": task_id, "states": list(_TERMINAL_STATES)},
    ).fetchone()
    return None if row is None else _get_compute_task(task_id, row)


def _get_release_failure(task_id: UUID, error_type: str) -> ComputeUnavailable:
    return ComputeUnavailable(
        f"The compute adapter failed with {error_type} when Robin released task {task_id}. "
        "Its message is not shown, because it can hold a credential. The task's outcome "
        "does not change; a later pass releases the task again.",
        details={"task_id": str(task_id), "error_type": error_type},
    )


def _get_compute_task(task_id: UUID, row: Mapping[str, Any]) -> ComputeTask:
    return ComputeTask(
        task_id=task_id,
        work_uri=row["work_uri"],
        settings=row["execution_settings"]["compute"],
        request=ComputeRequest.model_validate(row["compute_request"]),
        compute_ref=row["compute_ref"],
    )


def _get_next_task_state(
    state: Literal["submitted", "running"], status: ComputeStatus
) -> TaskStateAfterUpdate:
    """The task state that the compute adapter's `status` shows, for a task in `state`.

    An ended execution, or none, gives `finalizing`, and a started one gives `running`.
    A waiting execution keeps `state`, so a task never moves back.
    """
    if status.state in ("ended", "not_found"):
        return "finalizing"
    if status.state == "running":
        return "running"
    return state


def _get_task_to_store(
    connection: psycopg.Connection, task_id: UUID, setup: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The task's state, the time its outcome was recorded, and its work-file digest.
    None if the task's run is not of this setup."""
    return connection.execute(
        "SELECT t.state, t.outcome_recorded_at, t.work_file_digest "
        "FROM tasks t JOIN runs r USING (run_id) "
        f"WHERE t.task_id = %(task_id)s AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE}",
        {**setup, "task_id": task_id},
    ).fetchone()


def _lock_task_to_store(connection: psycopg.Connection, task_id: UUID) -> dict[str, Any]:
    """Lock the task and its run until the transaction ends, and give the task's state,
    the time its outcome was recorded, and its run.

    The run is locked too, so that a cancel or a withdrawal waits for this record.
    """
    return connection.execute(
        "SELECT t.state, t.outcome_recorded_at, t.run_id "
        "FROM tasks t JOIN runs r USING (run_id) WHERE t.task_id = %s FOR UPDATE OF t, r",
        (task_id,),
    ).fetchone()


def _get_reason_not_to_store(
    task: Mapping[str, Any] | None,
) -> Literal["already_recorded", "skipped"] | None:
    """Why the outcome of `task` must not be recorded now, or None if it must be.

    `task` is None for a task that this setup does not handle.
    """
    if task is None:
        return "skipped"
    if task["outcome_recorded_at"] is not None:
        return "already_recorded"
    if task["state"] != "finalizing":
        return "skipped"
    return None


def _get_outcome(
    data: bytes | None, work_file: WorkFile, split_location: SplitLocation
) -> _Outcome:
    """What to record for the task, from the bytes of its result file."""
    if data is None:
        return _Failed(failure=_get_failure(NO_RESULT, _NO_RESULT_MESSAGE))
    result = _parse_result(data, work_file, split_location)
    if isinstance(result, str):
        return _Failed(failure=_get_failure(INVALID_RESULT, _INVALID_RESULT_MESSAGE, result))
    return _get_reported_outcome(result)


def _parse_result(
    data: bytes, work_file: WorkFile, split_location: SplitLocation
) -> InferenceTaskResult | str:
    """Parse the result file, and check it against the task's work file.

    Gives the result, or the text that says why it cannot be recorded.
    """
    try:
        result = InferenceTaskResult.model_validate_json(data)
    except ValidationError as error:
        return str(error)
    problem = task_result_problem(
        result, task=work_file.task, work_file_digest=work_file.digest,
        split_location=split_location,
    )
    return result if problem is None else problem


def _get_reported_outcome(result: InferenceTaskResult) -> _Outcome:
    """What to record for a result that passed its checks: the outcome that the worker
    reported."""
    outcome = result.result
    if isinstance(outcome, InferenceCompleted):
        return _Succeeded(completed=outcome, robin_release=result.robin_release)
    if isinstance(outcome, InferenceFailure):
        report = outcome.failure
        message = (
            f"The work failed at stage {report.stage} before its first recording started. "
            "The detail says why."
        )
        failure = _get_failure(report.code, message, report.detail)
    elif isinstance(outcome, TaskRefusal):
        failure = _get_failure(outcome.code, _REFUSAL_MESSAGE, outcome.detail)
    else:
        failure = _get_failure(WORKER_ERROR, _WORKER_ERROR_MESSAGE, outcome.detail)
    return _Failed(failure=failure, robin_release=result.robin_release)


def _get_failure(reason: str, message: str, detail: str | None = None) -> dict[str, Any]:
    """A task's failure: its published reason, Robin's message, and the specific text."""
    return {"reason": reason, "message": message, "detail": detail}


def _write_task_outcome(
    connection: psycopg.Connection, task_id: UUID, run_id: UUID, outcome: _Outcome
) -> None:
    """Record the task's outcome, each recording's outcome, and the output files."""
    if isinstance(outcome, _Failed):
        _update_task_row(connection, task_id, "failed", outcome.failure, outcome.robin_release)
        _fail_recordings_with_task(connection, task_id, outcome.failure)
    else:
        _update_task_row(connection, task_id, "succeeded", None, outcome.robin_release)
        _write_recording_outcomes(connection, task_id, run_id, outcome.completed)


def _update_task_row(
    connection: psycopg.Connection,
    task_id: UUID,
    state: Literal["succeeded", "failed"],
    failure: Mapping[str, Any] | None,
    robin_release: str | None,
) -> None:
    """Set the task's state, failure, and release, and the time its outcome was recorded."""
    connection.execute(
        "UPDATE tasks SET state = %s, failure = %s, robin_release = %s, "
        "outcome_recorded_at = now() WHERE task_id = %s",
        (state, None if failure is None else Json(failure), robin_release, task_id),
    )


def _fail_recordings_with_task(
    connection: psycopg.Connection, task_id: UUID, failure: Mapping[str, Any]
) -> None:
    """Fail each pending recording of a failed task with the task's reason. This
    releases their claims at commit."""
    connection.execute(
        "UPDATE recording_outcomes SET state = 'failed', failure = %s "
        "WHERE task_id = %s AND state = 'pending'",
        (
            Json({"scope": "task", "code": failure["reason"], "message": failure["message"]}),
            task_id,
        ),
    )


def _write_recording_outcomes(
    connection: psycopg.Connection, task_id: UUID, run_id: UUID, completed: InferenceCompleted
) -> None:
    """Record the outcome of each recording of a completed result: cover each finished
    recording, fail each failed one, and write one `output_files` row for each file."""
    with connection.cursor() as cursor:
        cursor.executemany(
            "UPDATE recording_outcomes SET state = 'covered' "
            "WHERE task_id = %s AND namespace = %s AND value = %s AND state = 'pending'",
            [(task_id, covered.namespace, covered.value) for covered in completed.coverage],
        )
        cursor.executemany(
            "UPDATE recording_outcomes SET state = 'failed', failure = %s "
            "WHERE task_id = %s AND namespace = %s AND value = %s AND state = 'pending'",
            [
                (
                    Json({
                        "scope": "recording",
                        "code": failed.failure.code,
                        "stage": failed.failure.stage,
                        "message": failed.failure.detail,
                    }),
                    task_id,
                    failed.namespace,
                    failed.value,
                )
                for failed in completed.failed
            ],
        )
        cursor.executemany(
            "INSERT INTO output_files (run_id, task_id, namespace, value, kind, contract_id, "
            "uri, checksum) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    run_id, task_id, artifact.namespace, artifact.value, artifact.kind,
                    artifact.contract_id, artifact.uri, artifact.checksum,
                )
                for artifact in completed.artifacts
            ],
        )
