"""One reconcile pass: each step in order, its report, and the notice that a submit sends."""

import json
from datetime import UTC, datetime
import threading
from collections import Counter
from pathlib import Path

import psycopg
import pytest

from robin_adapters.storage.local import LocalTaskStorage
from robin_adapters.storage.location import split_location
from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import ComputeStatus
from robin_run_manager.compose import Services, compose
from robin_run_manager.database import NOTIFY_CHANNEL
from robin_run_manager.errors import (
    ComputeUnavailable,
    InternalError,
    NotConfigured,
    OutcomeUncertain,
    Unavailable,
    WorkFileConflict,
)
from robin_run_manager.launch import reserve_tasks
from robin_run_manager.preparation import get_runs_to_prepare, prepare_run
from robin_run_manager.profile import Profile, get_execution_settings
from robin_run_manager import launch, reconcile
from robin_run_manager.reconcile import reconcile_once
from robin_run_manager.task_files import WORK_FILE_NAME, get_task_folder
from run_fixtures import (
    KUBERNETES_SETTINGS,
    add_task,
    build_completed,
    build_task_result,
    get_current_files,
    get_run_row,
    get_task,
    get_tasks,
    get_work_file,
    register_fixture_models,
    submit,
    write_task_result,
)

from compute_doubles import ScriptedCompute

SECRETS = ("token=abc123", "password=xyz")


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


@pytest.fixture
def profile(database_settings, tmp_path):
    """A profile with local compute, local storage under the test's folder, and one
    recording in each task."""
    return Profile.model_validate({
        "database": database_settings,
        "compute": {"kind": "local"},
        "storage": {"kind": "local", "root": str(tmp_path / "storage")},
        "preparation": {"recordings_per_task": 1},
    })


def get_services(database, profile, compute, split=split_location) -> Services:
    return Services(
        profile=profile,
        database=database,
        compute=compute,
        task_storage={"local": LocalTaskStorage()},
        split_location=split,
    )


def submit_here(database, profile, values, *, actor="ana"):
    """A run of `values`, frozen with the profile's settings."""
    return submit(
        database, values, actor=actor, execution_settings=get_execution_settings(profile)
    )


def write_results(database, task_ids) -> None:
    for task_id in task_ids:
        work_file = get_work_file(database, task_id)
        write_task_result(work_file, build_task_result(work_file, build_completed(work_file)))


def get_secret_error() -> RuntimeError:
    """An error whose text, and whose cause's text, hold a secret."""
    error = RuntimeError(SECRETS[0])
    error.__cause__ = ValueError(SECRETS[1])
    return error


def split_location_with_a_defect(uri):
    raise get_secret_error()


def test_one_pass_prepares_launches_and_later_records(registered, profile):
    run_id = submit_here(registered, profile, ["a", "b"])
    compute = ScriptedCompute()
    services = get_services(registered, profile, compute)

    first = reconcile_once(services, actor="ana")

    assert first.body() == {
        "prepared": 1, "launched": 2, "outcomes_recorded": 0, "released": 0,
        "stops_requested": 0, "tasks_queued_elsewhere": 0, "tasks_active_elsewhere": 0,
        "failures": [],
    }
    task_ids = [task["task_id"] for task in get_tasks(registered, run_id)]
    assert {get_task(registered, task_id)["state"] for task_id in task_ids} == {"submitted"}
    assert len(compute.get_calls("submit")) == 2

    compute.script("describe", ComputeStatus(state="ended"), ComputeStatus(state="ended"))
    write_results(registered, task_ids)
    second = reconcile_once(services, actor="ana")

    assert (second.prepared, second.launched) == (0, 0)
    assert (second.outcomes_recorded, second.released, second.failures) == (2, 2, [])
    assert {get_task(registered, task_id)["state"] for task_id in task_ids} == {"succeeded"}
    assert len(compute.get_calls("release")) == 2
    assert [(row["value"], row["kind"], row["run_id"]) for row in get_current_files(registered)] == [
        ("a", "scores", run_id), ("b", "scores", run_id),
    ]


def test_a_reserved_task_that_was_not_submitted_is_submitted_and_counted(registered, profile):
    run_id = submit_here(registered, profile, ["a"])
    prepare_run(registered, run_id, recordings_per_task=1)
    # An earlier pass reserved the task and stopped before it submitted it.
    reserve_tasks(registered, actor="ana", profile=profile, storage_kinds={"local"})
    compute = ScriptedCompute()

    report = reconcile_once(get_services(registered, profile, compute), actor="ana")

    assert (report.launched, report.failures) == (1, [])
    [task] = get_tasks(registered, run_id)
    # The update step of the same pass then finds the execution running.
    assert get_task(registered, task["task_id"])["state"] == "running"
    assert len(compute.get_calls("submit")) == 1


def test_a_defect_in_one_reservation_does_not_stop_the_others(registered, profile, monkeypatch):
    run_id = submit_here(registered, profile, ["a", "b"])
    prepare_run(registered, run_id, recordings_per_task=1)
    first, second = [task["task_id"] for task in get_tasks(registered, run_id)]

    def build_work_file(connection, task_id):
        if task_id == first:
            raise RuntimeError(SECRETS[0])
        return real_build_work_file(connection, task_id)

    real_build_work_file = launch.build_work_file
    monkeypatch.setattr(launch, "build_work_file", build_work_file)

    report = reconcile_once(get_services(registered, profile, ScriptedCompute()), actor="ana")

    [failure] = report.failures
    assert (failure.step, failure.run_id, failure.task_id) == ("launch", run_id, first)
    assert isinstance(failure.error, InternalError)
    assert SECRETS[0] not in json.dumps(report.body())
    assert report.launched == 1
    assert get_task(registered, first)["state"] == "queued"
    assert get_task(registered, second)["state"] == "submitted"


def test_a_failure_in_one_task_does_not_stop_the_others(registered, profile):
    run_id = submit_here(registered, profile, ["a", "b"])
    compute = ScriptedCompute()
    services = get_services(registered, profile, compute)
    reconcile_once(services, actor="ana")
    first, second = [task["task_id"] for task in get_tasks(registered, run_id)]
    write_results(registered, [first, second])
    # The tasks are asked about oldest first.
    compute.script("describe", RuntimeError("lost"), ComputeStatus(state="ended"))

    report = reconcile_once(services, actor="ana")

    [failure] = report.failures
    assert (failure.step, failure.run_id, failure.task_id) == ("update", run_id, first)
    assert isinstance(failure.error, ComputeUnavailable)
    assert report.outcomes_recorded == 1
    assert get_task(registered, first)["state"] == "submitted"
    assert get_task(registered, second)["state"] == "succeeded"


def test_a_step_whose_list_fails_does_not_stop_the_later_steps(registered, profile, monkeypatch):
    def get_no_tasks(*arguments, **options):
        raise Unavailable("The database cannot be reached.")

    monkeypatch.setattr(reconcile, "get_tasks_to_release", get_no_tasks)
    submit_here(registered, profile, ["a"])

    report = reconcile_once(get_services(registered, profile, ScriptedCompute()), actor="ana")

    [failure] = report.failures
    assert (failure.step, failure.run_id, failure.task_id) == ("release", None, None)
    assert isinstance(failure.error, Unavailable)
    assert (report.prepared, report.launched) == (1, 1)


def add_conflicting_work_file(database, profile, run_id) -> None:
    """Prepare the run, and put other bytes where its task's work file goes."""
    prepare_run(database, run_id, recordings_per_task=1)
    [task] = get_tasks(database, run_id)
    folder = get_local_path(get_task_folder(profile.storage.root, run_id, task["task_id"]))
    folder.mkdir(parents=True)
    (folder / WORK_FILE_NAME).write_bytes(b"{}")


@pytest.mark.parametrize(
    ("case", "error_class", "reason"),
    [
        ("uncertain", OutcomeUncertain, "outcome_uncertain"),
        ("unavailable", ComputeUnavailable, "compute_unavailable"),
        ("refused", WorkFileConflict, "work_file_conflict"),
        ("unexpected", InternalError, "internal_error"),
    ],
)
def test_the_report_keeps_the_class_of_each_failure(registered, profile, case, error_class, reason):
    run_id = submit_here(registered, profile, ["a"])
    compute = ScriptedCompute()
    split = split_location
    if case == "uncertain":
        compute.script("submit", RuntimeError("lost"))
    elif case == "unavailable":
        compute.script("describe", RuntimeError("lost"))
    elif case == "refused":
        add_conflicting_work_file(registered, profile, run_id)
    else:
        split = split_location_with_a_defect

    report = reconcile_once(get_services(registered, profile, compute, split), actor="ana")

    [failure] = report.failures
    [task] = get_tasks(registered, run_id)
    assert type(failure.error) is error_class
    assert (failure.step, failure.run_id, failure.task_id) == ("launch", run_id, task["task_id"])
    [body] = report.body()["failures"]
    assert body["reason"] == reason
    assert get_task(registered, task["task_id"])["state"] == "submitting"


@pytest.mark.parametrize("source", ["compute", "unexpected"])
def test_an_exception_text_is_not_in_the_report(registered, profile, capsys, source):
    run_id = submit_here(registered, profile, ["a"])
    compute = ScriptedCompute()
    split = split_location
    if source == "compute":
        compute.script("describe", get_secret_error())
    else:
        split = split_location_with_a_defect

    report = reconcile_once(get_services(registered, profile, compute, split), actor="ana")

    [failure] = report.failures
    text = json.dumps(report.body())
    stderr = capsys.readouterr().err
    for secret in SECRETS:
        assert secret not in text
        assert secret not in stderr
    assert "RuntimeError" in failure.error.message
    if source == "unexpected":
        request_id = failure.error.details["request_id"]
        assert request_id in stderr
        assert "RuntimeError" in stderr
        assert str(run_id) in stderr
        # The stack names the function that raised, by file and line.
        assert "split_location_with_a_defect" in stderr
        assert f"{Path(__file__).name}:" in stderr
    else:
        assert stderr == ""


def test_two_passes_at_once_record_each_outcome_once(registered, profile):
    run_ids = [submit_here(registered, profile, [f"{n}-a", f"{n}-b"]) for n in range(3)]
    compute = ScriptedCompute()
    services = get_services(registered, profile, compute)

    def run_two_passes() -> list:
        start = threading.Barrier(2)
        reports = []

        def run():
            start.wait()
            reports.append(reconcile_once(services, actor="ana"))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return reports

    launches = run_two_passes()
    task_ids = [task["task_id"] for run_id in run_ids for task in get_tasks(registered, run_id)]
    assert len(task_ids) == 6
    assert [report.failures for report in launches] == [[], []]
    submits = Counter(task.task_id for task in compute.get_calls("submit"))
    assert submits == Counter({task_id: 1 for task_id in task_ids})

    compute.script("describe", *[ComputeStatus(state="ended")] * 6)
    write_results(registered, task_ids)
    records = run_two_passes()

    assert [report.failures for report in records] == [[], []]
    assert sum(report.outcomes_recorded for report in records) == 6
    assert sum(report.released for report in records) == 6
    releases = Counter(task.task_id for task in compute.get_calls("release"))
    assert releases == Counter({task_id: 1 for task_id in task_ids})
    assert len(get_current_files(registered)) == 6


def test_preparation_takes_at_most_10_runs_in_a_pass(registered, profile):
    run_ids = [submit_here(registered, profile, [f"{n}-a"]) for n in range(11)]

    report = reconcile_once(get_services(registered, profile, ScriptedCompute()), actor="ana")

    assert report.prepared == 10
    assert get_runs_to_prepare(registered) == [run_ids[-1]]
    assert {get_run_row(registered, run_id)["state"] for run_id in run_ids[:10]} == {"ready"}


ACTIVE_STATES = ("submitting", "submitted", "running", "finalizing")


def test_tasks_of_other_compute_are_counted_in_each_state(registered, profile):
    # Preparation in the pass gives this run one queued task.
    submit(registered, ["q"], execution_settings=KUBERNETES_SETTINGS)
    active = submit(registered, ["x"], execution_settings=KUBERNETES_SETTINGS)
    prepare_run(registered, active, recordings_per_task=1)
    for state in ACTIVE_STATES:
        add_task(registered, active, state)
    # Each ended task that was reserved needs a release, until it is released.
    for state in ("succeeded", "failed", "cancelled"):
        add_task(registered, active, state, reserved_at=datetime.now(UTC))
    add_task(registered, active, "succeeded", released_at=datetime.now(UTC))
    add_task(registered, active, "failed")
    # Another actor's task is not counted.
    other = submit(registered, ["y"], actor="bo", execution_settings=KUBERNETES_SETTINGS)
    add_task(registered, other, "running")
    compute = ScriptedCompute()

    report = reconcile_once(get_services(registered, profile, compute), actor="ana")

    assert (report.tasks_queued_elsewhere, report.tasks_active_elsewhere) == (2, 7)
    assert (report.launched, report.failures) == (0, [])
    assert compute.calls == []


def test_a_pass_refuses_with_no_compute(registered, profile):
    services = Services(profile=profile, database=registered, compute=None)

    with pytest.raises(NotConfigured):
        reconcile_once(services, actor="ana")


class StoppingCompute(ScriptedCompute):
    """Asks the pass to stop during each submit."""

    def __init__(self, stop) -> None:
        super().__init__()
        self._stop = stop

    def submit(self, task):
        self._stop.set()
        return super().submit(task)


def test_a_stop_during_a_run_operation_starts_no_other_operation(
    registered, profile, monkeypatch
):
    run_ids = [submit_here(registered, profile, [value]) for value in ("a", "b")]
    stop = threading.Event()
    compute = ScriptedCompute()

    def prepare_and_stop(*args, **kwargs):
        stop.set()
        return real_prepare_run(*args, **kwargs)

    real_prepare_run = reconcile.prepare_run
    monkeypatch.setattr(reconcile, "prepare_run", prepare_and_stop)

    report = reconcile_once(
        get_services(registered, profile, compute), actor="ana", should_stop=stop.is_set
    )

    assert (report.prepared, report.launched) == (1, 0)
    assert [get_run_row(registered, run_id)["state"] for run_id in run_ids] == [
        "ready", "preparing"
    ]
    assert compute.calls == []


def test_a_stop_during_a_task_operation_starts_no_other_operation(registered, profile):
    run_id = submit_here(registered, profile, ["a", "b"])
    prepare_run(registered, run_id, recordings_per_task=1)
    stop = threading.Event()
    compute = StoppingCompute(stop)

    report = reconcile_once(
        get_services(registered, profile, compute), actor="ana", should_stop=stop.is_set
    )

    assert report.launched == 1
    assert len(compute.get_calls("submit")) == 1
    # The other reserved task waits for the recovery step of a later pass.
    assert Counter(task["state"] for task in get_tasks(registered, run_id)) == Counter(
        {"submitted": 1, "submitting": 1}
    )


def connect(settings) -> psycopg.Connection:
    return psycopg.connect(
        host=settings.host, port=settings.port, dbname=settings.dbname, user=settings.user,
        password=settings.password.get_secret_value(), autocommit=True,
    )


def test_submit_sends_a_notice_at_commit(registered, database_settings, profile):
    with connect(database_settings) as listener, connect(database_settings) as blocker:
        listener.execute(f"LISTEN {NOTIFY_CHANNEL}")
        # The submit's insert into runs waits for this lock, after it sent its notice.
        blocker.execute("BEGIN")
        blocker.execute("LOCK TABLE runs IN SHARE MODE")
        thread = threading.Thread(target=submit_here, args=(registered, profile, ["a"]))
        thread.start()
        thread.join(timeout=1)
        assert thread.is_alive(), "the submit did not wait for the lock"

        assert list(listener.notifies(timeout=0.5)) == []

        blocker.execute("COMMIT")
        thread.join()
        notices = list(listener.notifies(timeout=5, stop_after=1))

    assert [notice.channel for notice in notices] == [NOTIFY_CHANNEL]


def test_compose_gives_no_compute_for_kubernetes(database_settings):
    profile = Profile.model_validate({
        "database": database_settings,
        "compute": {**KUBERNETES_SETTINGS["compute"]},
        "storage": KUBERNETES_SETTINGS["storage"],
    })

    services = compose(profile)
    try:
        assert services.compute is None
    finally:
        services.database.close()


def test_compose_gives_local_compute_and_task_storage(database_settings, tmp_path):
    from robin_adapters.compute.local import LocalCompute

    profile = Profile.model_validate({
        "database": database_settings,
        "compute": {"kind": "local"},
        "storage": {"kind": "local", "root": str(tmp_path)},
    })

    services = compose(profile)
    try:
        assert isinstance(services.compute, LocalCompute)
        assert list(services.task_storage) == ["local"]
        assert isinstance(services.task_storage["local"], LocalTaskStorage)
        assert services.split_location("file:///data/a.parquet") == ("file://", "data/a.parquet")
    finally:
        services.database.close()
