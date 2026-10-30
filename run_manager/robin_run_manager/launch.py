"""Launch: reserve free slots for queued tasks, then write each work file and submit it.

A reservation is one short transaction. It marks each task `submitting`, so the task
holds a slot from then on, and it stores the checksum of the task's work file. The
work file is written and the task submitted outside any transaction. If the controller
stops between these steps, the task stays `submitting` and keeps its slot, and a later
pass recovers it with `recover_submission`.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Json, Jsonb

from robin_contracts.ports.compute import (
    ComputeAdapter,
    ComputeRequest,
    ComputeStatus,
    ComputeTask,
    SubmissionRejected,
)
from robin_contracts.ports.storage import TaskStorage
from robin_contracts.tasks import ModelOutputVersion, SplitLocation, output_root_problem
from robin_run_manager.database import (
    LAUNCH_LOCK,
    Database,
    lock_task,
    transaction,
    transaction_on,
)
from robin_run_manager.errors import (
    SUBMISSION_REJECTED,
    OUTPUT_VERSION_NOT_CURRENT,
    ComputeUnavailable,
    OperationError,
    OutcomeUncertain,
    StorageUnavailable,
    WorkFileConflict,
    invoke_adapter,
)
from robin_run_manager.profile import Profile, get_setup_target
from robin_run_manager.status import ACTIVE_TASK_STATES
from robin_run_manager.task_files import (
    WorkFile,
    build_reserved_work_file,
    build_work_file,
    read_result,
)
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS, get_outdated_versions

_IN_FLIGHT_STATES = ("submitting", "submitted", "running")

_FAILURE_MESSAGES = {
    OUTPUT_VERSION_NOT_CURRENT: "The run's output versions are not the versions of the "
    "installed release, so the task was not started. Submit a new run.",
    SUBMISSION_REJECTED: "The compute service refused the task, so it did not run.",
}

# The time columns that each new state of a recovered task sets.
_RECOVERED_TIMES = {
    "submitted": "submitted_at = now()",
    "running": "submitted_at = now(), started_at = now()",
    "finalizing": "ended_at = now()",
}


# What `submit_reserved_task` and `recover_submission` give.
SubmissionOutcome = Literal[
    "submitted", "running", "finalizing", "kept", "resubmitted", "rejected", "skipped"
]


@dataclass(frozen=True)
class TaskProblem:
    """An error that stopped one step for one task. The other tasks went on.

    The error is an `OperationError`, or an exception that Robin did not expect.
    """

    task_id: UUID
    error: Exception


@dataclass(frozen=True)
class Submission:
    """What a submit needs: the rebuilt work file, and the task as the compute call gets it."""

    work_file: WorkFile
    compute_task: ComputeTask


@dataclass(frozen=True)
class Reservation:
    """The result of `reserve_tasks`.

    `task_ids` are the reserved tasks, oldest first. `outdated_tasks` counts the queued
    tasks that failed because their output versions are not current. `problems` are the
    tasks that could not be reserved, which stay `queued`.
    """

    task_ids: tuple[UUID, ...]
    outdated_tasks: int
    problems: tuple[TaskProblem, ...]


# The conditions that a run `r` was frozen for this setup. Every step that uses the
# compute adapter or the task storage selects its tasks with them, with the parameters
# that `get_setup_query_parameters` gives. A query that uses them names the runs table
# `r`.
#
# A slot belongs to one setup name. The run's frozen compute must also hold the setup's
# target fields as the profile gives them, and its storage must be the profile's. Other
# frozen compute fields, such as a job definition revision, can differ.
MATCHES_SETUP_TARGET = (
    "(r.execution_settings ->> 'setup') = %(setup)s "
    "AND (r.execution_settings::jsonb -> 'compute') @> %(target_compute)s "
    "AND (r.execution_settings::jsonb -> 'storage') = %(storage)s"
)
# The same setup, and a storage kind that this pass has a task storage for.
MATCHES_SETUP_WITH_SUPPORTED_STORAGE = (
    f"{MATCHES_SETUP_TARGET} "
    "AND r.execution_settings -> 'storage' ->> 'kind' = ANY(%(storage_kinds)s)"
)
# A run `r` that no setup of the pass handles: the profile has no setup of its name
# (`setups`), or its storage kind has no task storage (`storage_kinds`).
_ELSEWHERE = (
    "NOT ((r.execution_settings ->> 'setup') = ANY(%(setups)s) "
    "AND r.execution_settings -> 'storage' ->> 'kind' = ANY(%(storage_kinds)s))"
)


# A run `r` that can launch: ready, and not cancelled or withdrawn.
_LAUNCHABLE_RUN = (
    "r.state = 'ready' AND r.cancelled_at IS NULL AND r.withdrawn_at IS NULL"
)


def get_setup_query_parameters(
    profile: Profile, setup: str, storage_kinds: Collection[str]
) -> dict[str, Any]:
    """The parameters of `MATCHES_SETUP_TARGET` and `MATCHES_SETUP_WITH_SUPPORTED_STORAGE`
    for the profile's setup `setup`.

    `storage_kinds` are the kinds that have a task storage. The target is sent as JSON,
    as a run stores it, so equal values compare equal. Raises `NotConfigured` if the
    profile has no `compute` section, and `KeyError` if it has no setup of that name.
    """
    target_compute, storage = get_setup_target(profile, setup)
    return {
        "setup": setup,
        "target_compute": Jsonb(target_compute),
        "storage": Jsonb(storage),
        "storage_kinds": list(storage_kinds),
    }


def reserve_tasks(
    database: Database,
    *,
    actor: str,
    profile: Profile,
    setup: str,
    storage_kinds: Collection[str],
    release_versions: Sequence[ModelOutputVersion] = RELEASE_OUTPUT_VERSIONS,
) -> Reservation:
    """Reserve a free slot for each of the actor's oldest queued tasks of the profile's
    setup `setup`, and build the work file of each.

    First, the actor's queued tasks of this setup whose output versions are not current
    in `release_versions` fail. Every task in `submitting`, `submitted`, or `running`
    holds a slot of its setup, whoever submitted it, and the setup's `max_in_flight`,
    read from the profile now, caps its slots. A task that cannot be reserved because of
    an error stays `queued`, and the next queued task takes the slot. Raises
    `NotConfigured` if the profile has no `compute` section.
    """
    # Raises NotConfigured before the transaction starts.
    parameters = get_setup_query_parameters(profile, setup, storage_kinds)
    assert profile.compute is not None
    max_in_flight = profile.compute.setups[setup].max_in_flight
    with transaction(database) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (LAUNCH_LOCK,))
        outdated_tasks = _fail_outdated_tasks(connection, actor, parameters, release_versions)
        in_flight = connection.execute(
            "SELECT count(*) AS n FROM tasks t JOIN runs r USING (run_id) "
            f"WHERE t.state = ANY(%(states)s) AND {MATCHES_SETUP_TARGET}",
            {**parameters, "states": list(_IN_FLIGHT_STATES)},
        ).fetchone()["n"]
        here = _get_queued_tasks(connection, actor, parameters)
        free = max(max_in_flight - in_flight, 0)
        reserved, problems = [], []
        for task_id in here:
            if len(reserved) == free:
                break
            try:
                # A savepoint: a failure undoes this task's changes only.
                with connection.transaction():
                    if _reserve_task(connection, task_id):
                        reserved.append(task_id)
            # Also an error that Robin does not expect, so that a defect in one task
            # does not undo the reservations of the others.
            except Exception as error:
                problems.append(TaskProblem(task_id=task_id, error=error))
    return Reservation(
        task_ids=tuple(reserved),
        outdated_tasks=outdated_tasks,
        problems=tuple(problems),
    )


def submit_reserved_task(
    database: Database,
    task_id: UUID,
    *,
    setup: Mapping[str, Any],
    compute: ComputeAdapter,
    storage: TaskStorage,
    split_location: SplitLocation,
) -> SubmissionOutcome:
    """Write the work file of a `submitting` task, submit it, and record the submit.

    `setup` comes from `get_setup_query_parameters`. Gives `skipped` if another pass holds the
    task, or if it is not a `submitting` task of this setup, also when it changed during
    the submit. Gives `rejected`, and fails the task, if the compute service refused it.

    Another pass can recover the task before this submit, so this submit first checks as
    `recover_submission` does, and gives the same values if it finds an execution or a
    result.

    Raises `OutcomeUncertain` if an execution may have started; the task keeps its slot.
    Raises `WorkFileConflict` or `StorageUnavailable` if the work file cannot be written;
    then nothing is submitted. Raises `ComputeUnavailable` if `describe` fails.
    """
    return _settle_submission(
        database, task_id, setup=setup, compute=compute, storage=storage,
        split_location=split_location, submitted_as="submitted",
    )


def get_tasks_in_state(
    database: Database, state: str, *, actor: str, setup: Mapping[str, Any]
) -> list[UUID]:
    """The actor's tasks in `state` whose runs are of this setup, oldest first.

    `setup` comes from `get_setup_query_parameters`.
    """
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT t.task_id FROM tasks t JOIN runs r USING (run_id) "
            "WHERE t.state = %(state)s AND r.submitted_by = %(actor)s "
            f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE} "
            "ORDER BY t.created_at, t.task_id",
            {**setup, "state": state, "actor": actor},
        ).fetchall()
    return [row["task_id"] for row in rows]


def get_queued_tasks_elsewhere(
    database: Database, *, actor: str, setups: Collection[str], storage_kinds: Collection[str]
) -> list[UUID]:
    """The actor's queued tasks of launchable runs whose setup name is not in `setups`,
    or whose storage kind is not in `storage_kinds`."""
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT t.task_id FROM tasks t JOIN runs r USING (run_id) "
            f"WHERE t.state = 'queued' AND r.submitted_by = %(actor)s AND {_LAUNCHABLE_RUN} "
            f"AND {_ELSEWHERE} ORDER BY t.created_at, t.task_id",
            {"setups": list(setups), "storage_kinds": list(storage_kinds), "actor": actor},
        ).fetchall()
    return [row["task_id"] for row in rows]


def get_active_tasks_elsewhere(
    database: Database, *, actor: str, setups: Collection[str], storage_kinds: Collection[str]
) -> list[UUID]:
    """The actor's started tasks whose setup name is not in `setups`, or whose storage
    kind is not in `storage_kinds`.

    A started task still needs a step: it is `submitting`, `submitted`, `running`, or
    `finalizing`, or it ended after its reservation and is not released.
    """
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT t.task_id FROM tasks t JOIN runs r USING (run_id) "
            "WHERE (t.state = ANY(%(states)s) "
            "OR (t.reserved_at IS NOT NULL AND t.released_at IS NULL)) "
            f"AND r.submitted_by = %(actor)s AND {_ELSEWHERE} "
            "ORDER BY t.created_at, t.task_id",
            {
                "setups": list(setups), "storage_kinds": list(storage_kinds),
                "states": list(ACTIVE_TASK_STATES), "actor": actor,
            },
        ).fetchall()
    return [row["task_id"] for row in rows]


def recover_submission(
    database: Database,
    task_id: UUID,
    *,
    setup: Mapping[str, Any],
    compute: ComputeAdapter,
    storage: TaskStorage,
    split_location: SplitLocation,
) -> SubmissionOutcome:
    """Settle a `submitting` task whose submit may have started an execution.

    Asks `describe`, then looks for `result.json`. If it finds neither, it submits the
    task again. Otherwise it moves the task to the state that they show, or gives `kept`
    while a result waits for its execution to end. Raises as `submit_reserved_task`
    does.
    """
    return _settle_submission(
        database, task_id, setup=setup, compute=compute, storage=storage,
        split_location=split_location, submitted_as="resubmitted",
    )


def _settle_submission(
    database: Database,
    task_id: UUID,
    *,
    setup: Mapping[str, Any],
    compute: ComputeAdapter,
    storage: TaskStorage,
    split_location: SplitLocation,
    submitted_as: Literal["submitted", "resubmitted"],
) -> SubmissionOutcome:
    """Submit a `submitting` task if it has no execution and no result.

    `submitted_as` is the value given when this call submits the task, so that a caller
    can tell a first submit from a repeated one.
    """
    with lock_task(database, task_id) as connection:
        if connection is None:
            return "skipped"
        with transaction_on(connection, read_only=True):
            submission = _get_submission(connection, task_id, setup)
        if submission is None:
            return "skipped"
        # Describe first. An execution that writes its result and ends after the result
        # check would otherwise look like one that never started.
        status = get_compute_status(compute, submission.compute_task)
        has_result = read_result(storage, submission.work_file) is not None
        if status.state == "not_found" and not has_result:
            outcome = _launch(
                connection, submission, compute=compute, storage=storage,
                split_location=split_location,
            )
            return submitted_as if outcome == "submitted" else outcome
        state = _get_recovered_state(status, has_result=has_result)
        with transaction_on(connection):
            if state is None:
                # Later calls need the reference, to find what is left of the execution.
                _record_compute_ref(connection, task_id, status.compute_ref)
                return "kept"
            changed = _record_recovered(connection, task_id, state, status.compute_ref)
    return state if changed else "skipped"


def _launch(
    connection: psycopg.Connection,
    submission: Submission,
    *,
    compute: ComputeAdapter,
    storage: TaskStorage,
    split_location: SplitLocation,
) -> Literal["submitted", "rejected", "skipped"]:
    """Write the work file, submit the task, and record the submit, under the task lock
    that `connection` holds."""
    task_id = submission.compute_task.task_id
    _check_output_root(submission.work_file, split_location)
    _write_work_file(storage, submission.work_file)
    submitted = _submit(compute, submission.compute_task)
    with transaction_on(connection):
        if isinstance(submitted, SubmissionRejected):
            changed = _fail_task(connection, task_id, SUBMISSION_REJECTED, str(submitted))
        else:
            changed = _record_submit(connection, task_id, submitted)
    if not changed:
        # The task changed during the submit, so this pass leaves it.
        return "skipped"
    return "rejected" if isinstance(submitted, SubmissionRejected) else "submitted"


def _get_recovered_state(
    status: ComputeStatus, *, has_result: bool
) -> Literal["submitted", "running", "finalizing"] | None:
    """The state of a recovered task that is not submitted again. None if it stays
    `submitting`."""
    if status.state in ("ended", "not_found"):
        return "finalizing"
    if has_result:
        # A result never frees a slot on its own: the slot is kept until the execution ends.
        return None
    return "running" if status.state == "running" else "submitted"


def _record_recovered(
    connection: psycopg.Connection, task_id: UUID, state: str, compute_ref: str | None
) -> bool:
    """Move a recovered task to `state`. False if it is no longer `submitting`.

    An empty `compute_ref` keeps the stored one.
    """
    return connection.execute(
        f"UPDATE tasks SET state = %s, compute_ref = coalesce(%s, compute_ref), "
        f"{_RECOVERED_TIMES[state]} "
        "WHERE task_id = %s AND state = 'submitting'",
        (state, compute_ref, task_id),
    ).rowcount == 1


def _record_compute_ref(
    connection: psycopg.Connection, task_id: UUID, compute_ref: str | None
) -> None:
    """Store the `compute_ref` of a task that stays `submitting`. An empty one changes
    nothing."""
    if compute_ref is not None:
        connection.execute(
            "UPDATE tasks SET compute_ref = %s WHERE task_id = %s AND state = 'submitting'",
            (compute_ref, task_id),
        )


def get_compute_status(compute: ComputeAdapter, compute_task: ComputeTask) -> ComputeStatus:
    """The state of the task's execution. Raises `ComputeUnavailable` if the adapter fails."""
    return invoke_adapter(
        partial(compute.describe, compute_task),
        partial(_get_describe_failure, compute_task.task_id),
    )


def _submit(compute: ComputeAdapter, compute_task: ComputeTask) -> str | SubmissionRejected:
    """Submit the task. Gives its `compute_ref`, or the `SubmissionRejected` if the compute
    service refused it.

    Raises `OutcomeUncertain` for any other error, because the task may have started.
    """
    try:
        return invoke_adapter(
            partial(compute.submit, compute_task),
            partial(_get_uncertain_submit, compute_task.task_id),
            passing=(SubmissionRejected,),
        )
    except SubmissionRejected as rejected:
        return rejected


def _record_submit(connection: psycopg.Connection, task_id: UUID, compute_ref: str) -> bool:
    """Mark the task `submitted`. False if it is no longer `submitting`."""
    return connection.execute(
        "UPDATE tasks SET state = 'submitted', compute_ref = %s, submitted_at = now() "
        "WHERE task_id = %s AND state = 'submitting'",
        (compute_ref, task_id),
    ).rowcount == 1


def _fail_task(connection: psycopg.Connection, task_id: UUID, reason: str, detail: str) -> bool:
    """Fail one task with `reason`. False if it is no longer `queued` or `submitting`."""
    return bool(_fail_tasks(connection, [task_id], reason, detail=detail))


def _fail_outdated_tasks(
    connection: psycopg.Connection,
    actor: str,
    setup: Mapping[str, Any],
    release_versions: Sequence[ModelOutputVersion],
) -> int:
    """Fail the actor's queued tasks of this setup whose run's output versions are not
    current in `release_versions`. Returns how many tasks failed."""
    runs = connection.execute(
        "SELECT r.run_id, i.output_versions FROM runs r JOIN inference_runs i USING (run_id) "
        f"WHERE r.submitted_by = %(actor)s AND {_LAUNCHABLE_RUN} "
        f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE} AND EXISTS ("
        "SELECT 1 FROM tasks t WHERE t.run_id = r.run_id AND t.state = 'queued')",
        {**setup, "actor": actor},
    ).fetchall()
    outdated_runs = [
        row["run_id"] for row in runs
        if get_outdated_versions(row["output_versions"], release_versions)
    ]
    if not outdated_runs:
        return 0
    tasks = connection.execute(
        "SELECT task_id FROM tasks WHERE run_id = ANY(%s) AND state = 'queued'",
        (outdated_runs,),
    ).fetchall()
    failed = _fail_tasks(
        connection, [row["task_id"] for row in tasks], OUTPUT_VERSION_NOT_CURRENT
    )
    return len(failed)


def _get_queued_tasks(
    connection: psycopg.Connection, actor: str, setup: dict[str, Any]
) -> list[UUID]:
    """The actor's queued tasks of runs of this setup that can launch, oldest first."""
    rows = connection.execute(
        "SELECT t.task_id FROM tasks t JOIN runs r USING (run_id) "
        f"WHERE t.state = 'queued' AND r.submitted_by = %(actor)s AND {_LAUNCHABLE_RUN} "
        f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE} ORDER BY t.created_at, t.task_id",
        {**setup, "actor": actor},
    ).fetchall()
    return [row["task_id"] for row in rows]


def _reserve_task(connection: psycopg.Connection, task_id: UUID) -> bool:
    """Lock the task and its run, check them again, and reserve the task's slot.

    The task takes the request and resources that its run froze at submit.
    """
    row = connection.execute(
        f"SELECT t.state, {_LAUNCHABLE_RUN} AS launchable "
        "FROM tasks t JOIN runs r USING (run_id) "
        "WHERE t.task_id = %s FOR UPDATE OF t, r",
        (task_id,),
    ).fetchone()
    if row["state"] != "queued" or not row["launchable"]:
        return False
    connection.execute(
        "UPDATE tasks SET compute_request = r.execution_settings -> 'request', "
        "work_resources = r.execution_settings -> 'resources' "
        "FROM runs r WHERE r.run_id = tasks.run_id AND tasks.task_id = %s",
        (task_id,),
    )
    work_file = build_work_file(connection, task_id)
    connection.execute(
        "UPDATE tasks SET state = 'submitting', reserved_at = now(), work_uri = %s, "
        "work_file_digest = %s WHERE task_id = %s",
        (work_file.work_uri, work_file.digest, task_id),
    )
    return True


def _get_submission(
    connection: psycopg.Connection, task_id: UUID, setup: Mapping[str, Any]
) -> Submission | None:
    """What the submit of a `submitting` task of this setup needs. None if the task is
    not `submitting`, or its run is not of this setup.

    Raises `WorkFileConflict` if the rebuilt work file is not the one reserved.
    """
    row = connection.execute(
        "SELECT t.work_file_digest, t.compute_request, t.compute_ref, r.execution_settings "
        "FROM tasks t JOIN runs r USING (run_id) "
        "WHERE t.task_id = %(task_id)s AND t.state = 'submitting' "
        f"AND {MATCHES_SETUP_WITH_SUPPORTED_STORAGE}",
        {**setup, "task_id": task_id},
    ).fetchone()
    if row is None:
        return None
    work_file = build_reserved_work_file(connection, task_id, row["work_file_digest"])
    compute_task = ComputeTask(
        task_id=task_id,
        work_uri=work_file.work_uri,
        settings=row["execution_settings"]["compute"],
        request=ComputeRequest.model_validate(row["compute_request"]),
        compute_ref=row["compute_ref"],
    )
    return Submission(work_file=work_file, compute_task=compute_task)


def _check_output_root(work_file: WorkFile, split_location: SplitLocation) -> None:
    if output_root_problem(work_file.task.output_root, split_location=split_location):
        raise StorageUnavailable(
            f"The storage root of task {work_file.task.task_id}'s run cannot hold the "
            "task's outputs, so Robin wrote and submitted nothing. The profile refuses "
            "such a root, so this is a defect, and a later pass gives the same result.",
            details={"task_id": str(work_file.task.task_id)},
        )


def _write_work_file(storage: TaskStorage, work_file: WorkFile) -> None:
    """Write the work file once. An existing file with the same bytes is accepted."""
    task_id = work_file.task.task_id
    try:
        invoke_adapter(
            partial(storage.create, work_file.work_uri, work_file.data),
            partial(_get_storage_failure, task_id, f"wrote task {task_id}'s work file"),
            passing=(FileExistsError,),
        )
    except FileExistsError:
        pass
    else:
        return
    # Checked outside the handler, so that an error of the check has no link to the
    # FileExistsError.
    _check_existing_work_file(storage, work_file)


def _check_existing_work_file(storage: TaskStorage, work_file: WorkFile) -> None:
    """Accept the existing work file if it has the same bytes. Raises
    `WorkFileConflict` if it does not."""
    task_id = work_file.task.task_id
    existing = invoke_adapter(
        partial(storage.read, work_file.work_uri),
        partial(_get_storage_failure, task_id, f"read task {task_id}'s work file"),
    )
    if existing != work_file.data:
        raise WorkFileConflict(
            f"Another file is at the place of task {task_id}'s work file, so Robin wrote "
            "and submitted nothing. Report this defect.",
            details={"task_id": str(task_id)},
        )


def _get_describe_failure(task_id: UUID, error_type: str) -> ComputeUnavailable:
    return ComputeUnavailable(
        f"The compute adapter failed with {error_type} when Robin checked task {task_id}. "
        "Its message is not shown, because it can hold a credential. Robin changed "
        "nothing; a later pass checks the task again.",
        details={"task_id": str(task_id), "error_type": error_type},
    )


def _get_uncertain_submit(task_id: UUID, error_type: str) -> OutcomeUncertain:
    return OutcomeUncertain(
        f"The compute adapter failed with {error_type} when Robin submitted task "
        f"{task_id}. Its message is not shown, because it can hold a credential. The "
        "task may have started, so it keeps its slot; a later pass checks it.",
        details={"task_id": str(task_id), "error_type": error_type},
    )


def _get_storage_failure(task_id: UUID, action: str, error_type: str) -> StorageUnavailable:
    """`action` says what Robin did, such as "wrote task 1's work file"."""
    return StorageUnavailable(
        f"The task storage failed with {error_type} when Robin {action}. Its message is "
        "not shown, because it can hold a credential. Robin submitted nothing; a later "
        "pass tries again.",
        details={"task_id": str(task_id), "error_type": error_type},
    )


def _fail_tasks(
    connection: psycopg.Connection,
    task_ids: Sequence[UUID],
    reason: str,
    *,
    detail: str | None = None,
) -> list[UUID]:
    """Fail each task that is not launched yet with `reason`, and fail its pending
    recordings, which releases their claims. Returns the tasks that failed."""
    if not task_ids:
        return []
    message = _FAILURE_MESSAGES[reason]
    failed = connection.execute(
        "UPDATE tasks SET state = 'failed', failure = %s "
        "WHERE task_id = ANY(%s) AND state IN ('queued', 'submitting') RETURNING task_id",
        (Json({"reason": reason, "message": message, "detail": detail}), list(task_ids)),
    ).fetchall()
    connection.execute(
        "UPDATE recording_outcomes SET state = 'failed', failure = %s "
        "WHERE task_id = ANY(%s) AND state = 'pending'",
        (
            Json({"scope": "task", "code": reason, "message": message}),
            [row["task_id"] for row in failed],
        ),
    )
    return [row["task_id"] for row in failed]
