"""One reconcile pass: settle each task and run that has work to do, and report what happened.

A pass does its steps in this order, and each step goes on past a failure in one task or
run:

1. `recovery`: settle each `submitting` task, whose submit may have started an execution.
2. `update`: ask the compute adapter about each `submitted` or `running` task.
3. `outcome`: record the outcome of each `finalizing` task.
4. `release`: release each task that ended and is not released.
5. `preparation`: prepare the oldest runs that wait for it, of any actor.
6. `launch`: reserve free slots for queued tasks, and submit each reserved task.

Steps 1 to 4 run for each setup of the profile, in name order, then step 5 runs once,
then step 6 runs for each setup. All steps except preparation act only on the actor's
runs of that setup. A setup whose compute kind has no adapter in this version is a
failure of step `setup`, and the pass goes on with the other setups. So is a setup
whose name has runs with work that were submitted with another target.
"""

import sys
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from robin_contracts.ports.compute import ComputeAdapter
from robin_contracts.ports.storage import TaskStorage
from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.compose import Services
from robin_run_manager.database import transaction
from robin_run_manager.errors import InternalError, NotConfigured, OperationError
from robin_run_manager.launch import (
    SubmissionOutcome,
    get_active_tasks_elsewhere,
    get_queued_tasks_elsewhere,
    get_setup_query_parameters,
    get_tasks_in_state,
    recover_submission,
    reserve_tasks,
    submit_reserved_task,
)
from robin_run_manager.outcomes import (
    get_tasks_to_release,
    release_task,
    store_task_outcome,
    update_task_from_compute,
)
from robin_run_manager.preparation import get_runs_to_prepare, prepare_run
from robin_run_manager.profile import get_compute_section, get_setup_target
from robin_run_manager.setup_targets import refuse_conflicting_setup_target
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS

# The most runs that one pass prepares. The others wait for a later pass.
PREPARATION_LIMIT = 10


@dataclass(frozen=True)
class StepFailure:
    """A step that failed for one task or run, or for all of a step's tasks if both IDs
    are None. `error` keeps the class of the failure, which gives the exit code."""

    step: str
    run_id: UUID | None
    task_id: UUID | None
    error: OperationError

    def body(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "run_id": None if self.run_id is None else str(self.run_id),
            "task_id": None if self.task_id is None else str(self.task_id),
            **self.error.body(),
        }


@dataclass
class ReconcileReport:
    """What one pass did.

    Each count except the elsewhere counts is the sum over the pass's setups. `launched`
    counts the tasks that this pass submitted, also again after an uncertain submit.
    `tasks_queued_elsewhere` counts the actor's queued tasks whose setup name the profile
    does not have, or whose storage kind has no task storage, and `tasks_active_elsewhere`
    the actor's started tasks of that kind that still need a step, also a release. They
    are counted once for the whole profile, and they are not failures. `stops_requested`
    is always 0 in this version.
    """

    prepared: int = 0
    launched: int = 0
    outcomes_recorded: int = 0
    released: int = 0
    stops_requested: int = 0
    tasks_queued_elsewhere: int = 0
    tasks_active_elsewhere: int = 0
    failures: list[StepFailure] = field(default_factory=list)

    def body(self) -> dict[str, Any]:
        return {
            "prepared": self.prepared,
            "launched": self.launched,
            "outcomes_recorded": self.outcomes_recorded,
            "released": self.released,
            "stops_requested": self.stops_requested,
            "tasks_queued_elsewhere": self.tasks_queued_elsewhere,
            "tasks_active_elsewhere": self.tasks_active_elsewhere,
            "failures": [failure.body() for failure in self.failures],
        }


@dataclass(frozen=True)
class _Pass:
    """What each step of one pass uses for one setup. The setups of a pass share its
    report."""

    services: Services
    actor: str
    setup_name: str
    compute: ComputeAdapter
    # The setup's query parameters, for all of this pass's task storage kinds.
    query_parameters: Mapping[str, Any]
    # Each task storage, with the setup's query parameters for its kind only.
    storages: tuple[tuple[TaskStorage, Mapping[str, Any]], ...]
    release_versions: Sequence[ModelOutputVersion]
    report: ReconcileReport
    should_stop: Callable[[], bool]


def reconcile_once(
    services: Services,
    *,
    actor: str,
    release_versions: Sequence[ModelOutputVersion] = RELEASE_OUTPUT_VERSIONS,
    should_stop: Callable[[], bool] = lambda: False,
) -> ReconcileReport:
    """Do one pass of the steps in the module's list, and give its report.

    A failure in one task or run is added to the report, and the pass goes on. Before
    each setup's target check, each step, and the operation on each task or run, the
    pass calls `should_stop`; if it gives True, the pass starts no more operations. Raises
    `NotConfigured` if the profile has no `compute` section, or if no setup of the
    profile has compute that this version can use.
    """
    report = ReconcileReport()
    passes = _get_passes(services, actor, release_versions, report, should_stop)
    checked_passes: list[_Pass] = []
    for one_pass in passes:
        # A check reads the database, so a stop is not held up by every setup's check.
        if should_stop():
            return report
        if _check_target(one_pass):
            checked_passes.append(one_pass)
    for step, one_pass in _get_steps_in_order(passes, checked_passes):
        if should_stop():
            break
        step(one_pass)
    return report


def _get_passes(
    services: Services,
    actor: str,
    release_versions: Sequence[ModelOutputVersion],
    report: ReconcileReport,
    should_stop: Callable[[], bool],
) -> list[_Pass]:
    """A pass for each setup of the profile whose kind has an adapter, in name order.

    A setup whose kind has none is added to the report as a `setup` failure. Raises
    `NotConfigured` if the profile has no `compute` section, or no setup has an adapter.
    """
    profile = services.profile
    setups = get_compute_section(profile).setups
    passes: list[_Pass] = []
    for name in sorted(setups):
        kind = setups[name].kind
        if kind not in services.compute:
            report.failures.append(StepFailure(
                step="setup", run_id=None, task_id=None,
                error=NotConfigured(
                    f"This version of Robin cannot use {kind} compute, so the pass did "
                    f"nothing for setup {name}.",
                    details={"setup": name},
                ),
            ))
            continue
        passes.append(_Pass(
            services=services,
            actor=actor,
            setup_name=name,
            compute=services.compute[kind],
            query_parameters=get_setup_query_parameters(profile, name, services.task_storage),
            storages=tuple(
                (storage, get_setup_query_parameters(profile, name, {storage_kind}))
                for storage_kind, storage in services.task_storage.items()
            ),
            release_versions=release_versions,
            report=report,
            should_stop=should_stop,
        ))
    if not passes:
        kinds = ", ".join(sorted({setup.kind for setup in setups.values()}))
        raise NotConfigured(
            f"This version of Robin cannot use {kinds} compute, so it cannot reconcile. "
            "Use a profile with a local setup."
        )
    return passes


def _get_steps_in_order(
    passes: list[_Pass], checked_passes: list[_Pass]
) -> Iterator[tuple[Callable[[_Pass], None], _Pass]]:
    """Each step of the pass, with the setup's pass that it runs for, in the order of
    the module's list.

    Preparation and the elsewhere counts are not of one setup, so each runs once, with
    the first pass for its services and report, also when every setup failed its check.
    """
    for one_pass in checked_passes:
        yield _recover_submissions, one_pass
        yield _update_tasks, one_pass
        yield _store_outcomes, one_pass
        yield _release_tasks, one_pass
    yield _prepare_runs, passes[0]
    for one_pass in checked_passes:
        yield _launch_tasks, one_pass
    yield _count_tasks_elsewhere, passes[0]


def _check_target(one_pass: _Pass) -> bool:
    """Give True if no run of the setup name has work and another target. Otherwise add
    a `setup` failure, and give False: the pass skips that setup's steps.

    The check takes no lock. A run of another target that is submitted after it is safe,
    because each step selects only the runs of the setup's target.
    """
    target_compute, storage = get_setup_target(
        one_pass.services.profile, one_pass.setup_name
    )
    checked = False
    with _add_failures(one_pass, "setup"):
        with transaction(one_pass.services.database, read_only=True) as connection:
            refuse_conflicting_setup_target(
                connection, one_pass.setup_name, target_compute, storage
            )
        checked = True
    return checked


def _recover_submissions(one_pass: _Pass) -> None:
    database = one_pass.services.database
    for storage, query_parameters in one_pass.storages:
        task_ids: list[UUID] = []
        with _add_failures(one_pass, "recovery"):
            task_ids = get_tasks_in_state(
                database, "submitting", actor=one_pass.actor, setup=query_parameters
            )
        for task_id in _until_stopped(one_pass, task_ids):
            with _add_failures(one_pass, "recovery", task_id=task_id):
                outcome = recover_submission(
                    database, task_id, setup=query_parameters, compute=one_pass.compute,
                    storage=storage, split_location=one_pass.services.split_location,
                )
                if outcome == "resubmitted":
                    one_pass.report.launched += 1


def _update_tasks(one_pass: _Pass) -> None:
    database = one_pass.services.database
    for state in ("submitted", "running"):
        task_ids: list[UUID] = []
        with _add_failures(one_pass, "update"):
            task_ids = get_tasks_in_state(
                database, state, actor=one_pass.actor, setup=one_pass.query_parameters
            )
        for task_id in _until_stopped(one_pass, task_ids):
            with _add_failures(one_pass, "update", task_id=task_id):
                update_task_from_compute(
                    database, task_id, setup=one_pass.query_parameters, compute=one_pass.compute
                )


def _store_outcomes(one_pass: _Pass) -> None:
    database = one_pass.services.database
    for storage, query_parameters in one_pass.storages:
        task_ids: list[UUID] = []
        with _add_failures(one_pass, "outcome"):
            task_ids = get_tasks_in_state(
                database, "finalizing", actor=one_pass.actor, setup=query_parameters
            )
        for task_id in _until_stopped(one_pass, task_ids):
            with _add_failures(one_pass, "outcome", task_id=task_id):
                outcome = store_task_outcome(
                    database, task_id, setup=query_parameters, storage=storage,
                    split_location=one_pass.services.split_location,
                )
                if outcome == "recorded":
                    one_pass.report.outcomes_recorded += 1


def _release_tasks(one_pass: _Pass) -> None:
    database = one_pass.services.database
    task_ids: list[UUID] = []
    with _add_failures(one_pass, "release"):
        task_ids = get_tasks_to_release(
            database, actor=one_pass.actor, setup=one_pass.query_parameters
        )
    for task_id in _until_stopped(one_pass, task_ids):
        with _add_failures(one_pass, "release", task_id=task_id):
            if release_task(
                database, task_id, setup=one_pass.query_parameters, compute=one_pass.compute
            ) == "released":
                one_pass.report.released += 1


def _prepare_runs(one_pass: _Pass) -> None:
    database = one_pass.services.database
    preparation = one_pass.services.profile.preparation
    run_ids: list[UUID] = []
    with _add_failures(one_pass, "preparation"):
        run_ids = get_runs_to_prepare(database)[:PREPARATION_LIMIT]
    for run_id in _until_stopped(one_pass, run_ids):
        with _add_failures(one_pass, "preparation", run_id=run_id):
            # A run that gives `try_again` waits for the next pass.
            prepared = prepare_run(
                database, run_id, recordings_per_task=preparation.recordings_per_task,
                release_versions=one_pass.release_versions,
            )
            if prepared.state == "ready":
                one_pass.report.prepared += 1


def _launch_tasks(one_pass: _Pass) -> None:
    database = one_pass.services.database
    report = one_pass.report
    with _add_failures(one_pass, "launch"):
        reservation = reserve_tasks(
            database, actor=one_pass.actor, profile=one_pass.services.profile,
            setup=one_pass.setup_name, storage_kinds=one_pass.services.task_storage,
            release_versions=one_pass.release_versions,
        )
        for problem in reservation.problems:
            _add_failure(one_pass, "launch", problem.error, task_id=problem.task_id)
        # A reserved task that this pass does not submit waits for a later recovery step.
        for task_id in _until_stopped(one_pass, reservation.task_ids):
            with _add_failures(one_pass, "launch", task_id=task_id):
                if _submit_with_its_storage(one_pass, task_id) == "submitted":
                    report.launched += 1


def _count_tasks_elsewhere(one_pass: _Pass) -> None:
    """Count the actor's queued and started tasks that no setup of the pass handles.

    The counts do not depend on the setups' steps, so they are given also when every
    setup failed its check.
    """
    services = one_pass.services
    elsewhere = {
        "actor": one_pass.actor,
        "setups": get_compute_section(services.profile).setups,
        "storage_kinds": services.task_storage,
    }
    with _add_failures(one_pass, "launch"):
        one_pass.report.tasks_queued_elsewhere = len(
            get_queued_tasks_elsewhere(services.database, **elsewhere)
        )
        one_pass.report.tasks_active_elsewhere = len(
            get_active_tasks_elsewhere(services.database, **elsewhere)
        )


def _until_stopped(one_pass: _Pass, ids: Iterable[UUID]) -> Iterator[UUID]:
    """The IDs in turn, until the pass is asked to stop."""
    for one_id in ids:
        if one_pass.should_stop():
            return
        yield one_id


def _submit_with_its_storage(one_pass: _Pass, task_id: UUID) -> SubmissionOutcome:
    """Submit a reserved task with the task storage of its run's storage kind.

    The submit skips a task whose run is not of the setup that it is given, so each
    storage is tried in turn until one does not skip it.
    """
    outcome: SubmissionOutcome = "skipped"
    for storage, query_parameters in one_pass.storages:
        outcome = submit_reserved_task(
            one_pass.services.database, task_id, setup=query_parameters, compute=one_pass.compute,
            storage=storage, split_location=one_pass.services.split_location,
        )
        if outcome != "skipped":
            break
    return outcome


@contextmanager
def _add_failures(
    one_pass: _Pass, step: str, *, run_id: UUID | None = None, task_id: UUID | None = None
) -> Iterator[None]:
    """Use as `with _add_failures(...):`. If the code in the `with` block raises an
    exception, add it to the pass's report as a failure of `step`, and do not raise it.
    The pass then goes on with the code after the `with` block.

    The code in the block after the exception does not run. So a variable that the block
    sets must also be set before the block, for example to an empty list.
    """
    try:
        yield
    except Exception as error:
        _add_failure(one_pass, step, error, run_id=run_id, task_id=task_id)


def _add_failure(
    one_pass: _Pass,
    step: str,
    error: Exception,
    *,
    run_id: UUID | None = None,
    task_id: UUID | None = None,
) -> None:
    """Add the error to the pass's report.

    An exception that Robin does not expect becomes an `InternalError`, whose stack goes
    to stderr without the exception's text.
    """
    if isinstance(error, OperationError):
        one_pass.report.failures.append(
            _get_failure(one_pass, step, task_id, error, run_id=run_id)
        )
        return
    request_id = uuid.uuid4().hex
    failure = _get_failure(
        one_pass, step, task_id, _get_internal_error(step, error, request_id), run_id=run_id
    )
    print(
        describe_exception(
            error, request_id=request_id, step=step,
            run_id=failure.run_id, task_id=failure.task_id,
        ),
        file=sys.stderr,
    )
    one_pass.report.failures.append(failure)


def _get_failure(
    one_pass: _Pass,
    step: str,
    task_id: UUID | None,
    error: OperationError,
    *,
    run_id: UUID | None = None,
) -> StepFailure:
    if run_id is None and task_id is not None:
        run_id = _get_run_id(one_pass, task_id)
    return StepFailure(step=step, run_id=run_id, task_id=task_id, error=error)


def _get_run_id(one_pass: _Pass, task_id: UUID) -> UUID | None:
    """The run of the task. None if the database cannot give it now."""
    try:
        with transaction(one_pass.services.database, read_only=True) as connection:
            row = connection.execute(
                "SELECT run_id FROM tasks WHERE task_id = %s", (task_id,)
            ).fetchone()
    except OperationError:
        return None
    return None if row is None else row["run_id"]


def _get_internal_error(step: str, error: Exception, request_id: str) -> InternalError:
    return InternalError(
        f"The {step} step failed with {type(error).__name__}, an error that Robin did not "
        "expect. This is a defect. The stack is on stderr, with the request ID.",
        details={"request_id": request_id},
    )


def describe_exception(
    error: BaseException,
    *,
    request_id: str,
    step: str,
    run_id: UUID | None,
    task_id: UUID | None,
) -> str:
    """The stderr block of an exception that Robin did not expect.

    It gives the request ID, the step, the run and task IDs, the exception's type, and
    its stack as `file:line function` lines. It gives no message, no source line, and
    no exception that this one came from, because their text can hold a credential.
    """
    lines = [
        f"robin: internal error, request {request_id}: step {step}, run {run_id or '-'}, "
        f"task {task_id or '-'}",
        f"{type(error).__module__}.{type(error).__qualname__}",
    ]
    for frame in traceback.extract_tb(error.__traceback__):
        lines.append(f"  {frame.filename}:{frame.lineno} {frame.name}")
    return "\n".join(lines)
