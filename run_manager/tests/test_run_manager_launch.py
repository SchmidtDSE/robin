"""Launch: reserve free slots for queued tasks, then write each work file and submit it."""

import threading
import time

import pytest

from robin_adapters.storage.local import LocalTaskStorage
from robin_adapters.storage.location import split_location
from robin_contracts.canonical import checksum_bytes
from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import ComputeRequest, ComputeStatus, SubmissionRejected
from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager import launch
from robin_run_manager.database import lock_task, open_database, transaction
from robin_run_manager.errors import (
    ComputeUnavailable,
    OutcomeUncertain,
    Refused,
    StorageUnavailable,
    WorkFileConflict,
)
from robin_run_manager.launch import (
    Reservation,
    TaskProblem,
    get_setup_parameters,
    get_tasks_in_state,
    recover_submission,
    reserve_tasks,
    submit_reserved_task,
)
from robin_run_manager.preparation import prepare_run
from robin_run_manager.profile import Profile
from robin_run_manager.task_files import build_work_file
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS
from run_fixtures import (
    KUBERNETES_SETTINGS,
    LOCAL_SETTINGS,
    LOCAL_SETUP,
    add_queued_tasks,
    cancel,
    count_in_flight,
    get_task,
    get_tasks,
    register_fixture_models,
    set_execution_settings,
    submit,
    withdraw,
)

from compute_doubles import ACCEPT_THEN_TIMEOUT, ScriptedCompute

NEWER_OWL = tuple(
    ModelOutputVersion(model=entry.model, version=entry.version + 1)
    if entry.model == "model:owl/v4" else entry
    for entry in RELEASE_OUTPUT_VERSIONS
)
S3_STORAGE_SETTINGS = {**LOCAL_SETTINGS, "storage": {"kind": "s3", "root": "s3://robin/"}}


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


@pytest.fixture
def root(tmp_path):
    """The storage root that the test's runs freeze."""
    return str(tmp_path / "storage")


@pytest.fixture
def make_profile(database_settings, root):
    def make(max_in_flight=10, **compute):
        return Profile.model_validate({
            "database": database_settings.model_dump(),
            "compute": {"kind": "local", "max_in_flight": max_in_flight, **compute},
            "storage": {"kind": "local", "root": root},
        })

    return make


def reserve(database, profile, actor="ana", **options) -> Reservation:
    return reserve_tasks(
        database, actor=actor, profile=profile, storage_kinds={"local"}, **options
    )


def submit_task(database, task_id, compute, storage=None):
    return submit_reserved_task(
        database, task_id, setup=LOCAL_SETUP, compute=compute,
        storage=storage or LocalTaskStorage(), split_location=split_location,
    )


def get_recording_failures(database, task_id) -> list[tuple]:
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT state, failure FROM recording_outcomes WHERE task_id = %s ORDER BY value",
            (task_id,),
        ).fetchall()
    return [(row["state"], row["failure"]["scope"], row["failure"]["code"]) for row in rows]


@pytest.mark.parametrize("repeat", range(5))
def test_concurrent_reservations_never_exceed_max_in_flight(
    registered, database_settings, make_profile, root, repeat
):
    add_queued_tasks(registered, [f"r{index}" for index in range(8)], root=root)
    profile = make_profile(max_in_flight=3)
    start = threading.Barrier(4)
    reserved, errors = [], []

    def run():
        database = open_database(database_settings)
        try:
            start.wait()
            reserved.extend(reserve(database, profile).task_ids)
        except Exception as error:  # noqa: BLE001 - the test reports any error
            errors.append(error)
        finally:
            database.close()

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == []
    assert len(reserved) == len(set(reserved)) == 3
    assert count_in_flight(registered) == 3


def test_a_reservation_counts_the_slots_that_another_reservation_takes(
    registered, database_settings, make_profile, root, monkeypatch
):
    add_queued_tasks(registered, [f"r{index}" for index in range(8)], root=root)
    profile = make_profile(max_in_flight=3)
    paused, other_done = threading.Event(), threading.Event()
    real = launch._get_queued_tasks

    def pause_once(*arguments):
        # The first reservation pauses after it counts the slots. With the launch lock,
        # the other reservation cannot run until it commits, so the wait times out.
        if not paused.is_set():
            paused.set()
            other_done.wait(2)
        return real(*arguments)

    monkeypatch.setattr(launch, "_get_queued_tasks", pause_once)
    first = []
    database = open_database(database_settings)
    thread = threading.Thread(target=lambda: first.append(reserve(database, profile)))
    thread.start()
    try:
        assert paused.wait(30)
        second = reserve(registered, profile)
        other_done.set()
        thread.join(30)
    finally:
        database.close()

    assert len(first[0].task_ids) + len(second.task_ids) == 3
    assert count_in_flight(registered) == 3


def test_a_slot_is_counted_for_every_submitter(registered, make_profile, root):
    add_queued_tasks(registered, ["a"], root=root, actor="bob")
    add_queued_tasks(registered, ["b", "c"], root=root, actor="ana")
    assert len(reserve(registered, make_profile(max_in_flight=1), actor="bob").task_ids) == 1

    assert reserve(registered, make_profile(max_in_flight=1)).task_ids == ()
    assert len(reserve(registered, make_profile(max_in_flight=2)).task_ids) == 1


def test_tasks_of_another_compute_setup_do_not_take_this_setups_slots(
    registered, make_profile, root
):
    other_run, [other_task] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile(max_in_flight=1))
    # The reserved task now belongs to a run of another compute setup.
    set_execution_settings(registered, other_run, KUBERNETES_SETTINGS)
    _, [task_id] = add_queued_tasks(registered, ["b"], root=root)

    assert reserve(registered, make_profile(max_in_flight=1)).task_ids == (task_id,)
    assert get_task(registered, other_task)["state"] == "submitting"


def test_the_oldest_queued_task_is_reserved_first(registered, make_profile, root):
    settings = {**LOCAL_SETTINGS, "storage": {"kind": "local", "root": root}}
    older_run = submit(registered, ["a"], execution_settings=settings)
    newer_run = submit(registered, ["b"], execution_settings=settings)
    # The newer run is prepared first, so its task is the oldest.
    prepare_run(registered, newer_run, recordings_per_task=1)
    prepare_run(registered, older_run, recordings_per_task=1)
    [oldest] = [task["task_id"] for task in get_tasks(registered, newer_run)]

    assert reserve(registered, make_profile(max_in_flight=1)).task_ids == (oldest,)


def test_only_the_actors_runs_are_launched(registered, make_profile, root):
    _, bobs = add_queued_tasks(registered, ["a"], root=root, actor="bob")
    _, anas = add_queued_tasks(registered, ["b"], root=root, actor="ana")

    reservation = reserve(registered, make_profile())

    assert reservation == Reservation(
        task_ids=tuple(anas), outdated_tasks=0, tasks_queued_elsewhere=0, problems=()
    )
    assert get_task(registered, bobs[0])["state"] == "queued"


def test_settings_that_json_changes_still_match_the_frozen_settings(
    registered, make_profile, root, monkeypatch
):
    # A tuple in the profile is stored as a JSON list on the run.
    real = launch.get_execution_settings

    def with_tuple(profile):
        settings = real(profile)
        return {**settings, "compute": {**settings["compute"], "zones": ("a", "b")}}

    monkeypatch.setattr(launch, "get_execution_settings", with_tuple)
    run_id, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    frozen = {
        "compute": {"kind": "local", "zones": ["a", "b"]},
        "storage": {"kind": "local", "root": root},
    }
    set_execution_settings(registered, run_id, frozen)

    assert reserve(registered, make_profile()).task_ids == (task_id,)


@pytest.mark.parametrize("settings", [KUBERNETES_SETTINGS, S3_STORAGE_SETTINGS])
def test_a_run_frozen_for_other_compute_stays_queued_and_is_counted(
    registered, make_profile, root, settings
):
    elsewhere, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    set_execution_settings(registered, elsewhere, settings)
    _, here = add_queued_tasks(registered, ["b"], root=root)

    reservation = reserve(registered, make_profile())

    assert reservation == Reservation(
        task_ids=tuple(here), outdated_tasks=0, tasks_queued_elsewhere=1, problems=()
    )
    assert get_task(registered, task_id)["state"] == "queued"


@pytest.mark.parametrize("stop", [cancel, withdraw])
def test_cancelled_and_withdrawn_runs_are_not_launched(registered, make_profile, root, stop):
    run_id, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    stop(registered, run_id)

    assert reserve(registered, make_profile()) == Reservation(
        task_ids=(), outdated_tasks=0, tasks_queued_elsewhere=0, problems=()
    )
    assert get_task(registered, task_id)["state"] == "queued"


def test_a_run_cancelled_after_the_queue_is_read_is_not_launched(
    registered, make_profile, root, monkeypatch
):
    run_id, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    real = launch._get_queued_tasks

    def cancel_after_reading(*arguments):
        queued = real(*arguments)
        cancel(registered, run_id)
        return queued

    monkeypatch.setattr(launch, "_get_queued_tasks", cancel_after_reading)

    assert reserve(registered, make_profile()).task_ids == ()
    assert get_task(registered, task_id)["state"] == "queued"


def test_outdated_queued_work_fails_and_releases_its_claims(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a", "b"], root=root, size=2)

    reservation = reserve(registered, make_profile(), release_versions=NEWER_OWL)

    assert reservation == Reservation(
        task_ids=(), outdated_tasks=1, tasks_queued_elsewhere=0, problems=()
    )
    task = get_task(registered, task_id)
    assert (task["state"], task["failure"]["reason"]) == ("failed", "output_version_not_current")
    assert get_recording_failures(registered, task_id) == [
        ("failed", "task", "output_version_not_current"),
    ] * 2
    second, _ = add_queued_tasks(registered, ["a", "b"], root=root, size=2)
    with transaction(registered, read_only=True) as connection:
        states = connection.execute(
            "SELECT state FROM recording_outcomes WHERE run_id = %s", (second,)
        ).fetchall()
    assert [row["state"] for row in states] == ["pending", "pending"]


@pytest.mark.parametrize("settings", [KUBERNETES_SETTINGS, S3_STORAGE_SETTINGS])
def test_outdated_work_of_another_setup_stays_queued_and_is_counted(
    registered, make_profile, root, settings
):
    elsewhere, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    set_execution_settings(registered, elsewhere, settings)

    reservation = reserve(registered, make_profile(), release_versions=NEWER_OWL)

    assert reservation == Reservation(
        task_ids=(), outdated_tasks=0, tasks_queued_elsewhere=1, problems=()
    )
    assert get_task(registered, task_id)["state"] == "queued"
    with transaction(registered, read_only=True) as connection:
        [recording] = connection.execute(
            "SELECT state FROM recording_outcomes WHERE task_id = %s", (task_id,)
        ).fetchall()
    assert recording["state"] == "pending"


def test_submit_writes_the_work_file_and_records_the_reference(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    assert submit_task(registered, task_id, compute) == "submitted"

    task = get_task(registered, task_id)
    with transaction(registered, read_only=True) as connection:
        expected = build_work_file(connection, task_id)
    assert get_local_path(task["work_uri"]).read_bytes() == expected.data
    assert task["work_uri"] == expected.work_uri
    assert (task["state"], task["compute_ref"]) == ("submitted", compute.executions[task_id])
    assert task["submitted_at"] is not None
    [call] = compute.get_calls("submit")
    assert (call.task_id, call.work_uri, call.settings) == (
        task_id, expected.work_uri, {"kind": "local"},
    )
    assert call.compute_ref is None


def test_the_reservation_stores_the_digest_of_the_work_file(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    reserved = get_task(registered, task_id)
    assert (reserved["state"], reserved["reserved_at"] is not None) == ("submitting", True)

    submit_task(registered, task_id, ScriptedCompute())

    written = get_local_path(reserved["work_uri"]).read_bytes()
    assert reserved["work_file_digest"] == checksum_bytes(written)


def test_a_crash_before_the_reservation_commit_leaves_the_task_queued(
    registered, make_profile, root, monkeypatch
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)

    def crash(connection, task_id):
        # The operator stops the controller. A reservation keeps an Exception with its
        # task, but not this.
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(launch, "build_work_file", crash)
        with pytest.raises(KeyboardInterrupt):
            reserve(registered, make_profile(max_in_flight=1))

    task = get_task(registered, task_id)
    assert (task["state"], task["work_file_digest"], task["reserved_at"]) == ("queued", None, None)
    assert count_in_flight(registered) == 0
    assert not (get_local_path(root) / "runs").exists()
    assert reserve(registered, make_profile(max_in_flight=1)).task_ids == (task_id,)


def test_a_task_whose_work_file_cannot_be_built_does_not_stop_the_others(
    registered, make_profile, root, monkeypatch
):
    _, [broken, other] = add_queued_tasks(registered, ["a", "b"], root=root)
    problem = Refused("unknown_model", "Model owl/v4 is not registered.")
    real = launch.build_work_file

    def fail_for_one_task(connection, task_id):
        if task_id == broken:
            raise problem
        return real(connection, task_id)

    monkeypatch.setattr(launch, "build_work_file", fail_for_one_task)

    reservation = reserve(registered, make_profile(max_in_flight=1))

    assert reservation.task_ids == (other,)
    assert reservation.problems == (TaskProblem(task_id=broken, error=problem),)
    task = get_task(registered, broken)
    assert (task["state"], task["compute_request"], task["work_file_digest"]) == (
        "queued", None, None,
    )


def test_the_task_records_the_request_for_its_model(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    profile = make_profile(
        request={"requests": {"cpu": "2"}},
        resources={"batch_size": 32},
        models={"owl/v4": {"request": {"requests": {"cpu": "4"}, "limits": {"memory": "8Gi"}}}},
    )

    reserve(registered, profile)
    compute = ScriptedCompute()
    submit_task(registered, task_id, compute)

    task = get_task(registered, task_id)
    request = ComputeRequest(requests={"cpu": "4"}, limits={"memory": "8Gi"})
    assert task["compute_request"] == request.model_dump(mode="json")
    assert task["work_resources"] == {"batch_size": 32}
    assert compute.get_calls("submit")[0].request == request


def test_a_rejected_submit_fails_the_task_and_frees_its_slot(registered, make_profile, root):
    _, [first, second] = add_queued_tasks(registered, ["a", "b"], root=root)
    profile = make_profile(max_in_flight=1)
    reserve(registered, profile)
    compute = ScriptedCompute()
    compute.script("submit", SubmissionRejected("robin-worker is not on the PATH"))

    assert submit_task(registered, first, compute) == "rejected"

    task = get_task(registered, first)
    assert (task["state"], task["failure"]["reason"]) == ("failed", "submission_rejected")
    assert get_recording_failures(registered, first) == [("failed", "task", "submission_rejected")]
    assert reserve(registered, profile).task_ids == (second,)


def test_an_uncertain_submit_keeps_the_slot(registered, make_profile, root):
    _, tasks = add_queued_tasks(registered, ["a", "b", "c"], root=root)
    reserve(registered, make_profile(max_in_flight=1))
    compute = ScriptedCompute()
    compute.script("submit", ACCEPT_THEN_TIMEOUT)

    with pytest.raises(OutcomeUncertain) as raised:
        submit_task(registered, tasks[0], compute)

    assert raised.value.details["error_type"] == "TimeoutError"
    assert "the reply was lost" not in raised.value.message
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    task = get_task(registered, tasks[0])
    assert (task["state"], task["compute_ref"]) == ("submitting", None)
    assert reserve(registered, make_profile(max_in_flight=2)).task_ids == (tasks[1],)


def test_a_storage_root_that_cannot_be_written_keeps_the_slot_and_submits_nothing(
    registered, make_profile, tmp_path
):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    root = str(blocker / "storage")
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    with pytest.raises(StorageUnavailable) as raised:
        submit_task(registered, task_id, compute)

    assert "a file, not a folder" not in raised.value.message
    assert get_task(registered, task_id)["state"] == "submitting"
    assert compute.get_calls("submit") == []
    blocker.unlink()
    assert submit_task(registered, task_id, compute) == "submitted"
    assert len(compute.get_calls("submit")) == 1


def test_a_task_of_another_setup_is_not_submitted(registered, make_profile, root):
    run_id, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    set_execution_settings(registered, run_id, KUBERNETES_SETTINGS)
    compute = ScriptedCompute()

    assert submit_task(registered, task_id, compute) == "skipped"

    assert compute.calls == []
    assert get_task(registered, task_id)["state"] == "submitting"


def test_an_existing_work_file_with_other_bytes_is_not_submitted(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    LocalTaskStorage().create(get_task(registered, task_id)["work_uri"], b"other bytes")
    compute = ScriptedCompute()

    with pytest.raises(WorkFileConflict):
        submit_task(registered, task_id, compute)

    assert get_task(registered, task_id)["state"] == "submitting"
    assert compute.get_calls("submit") == []


class WorkFileReadFails(LocalTaskStorage):
    def read(self, uri):
        if uri.endswith("work.json"):
            raise OSError("token=abc123")
        return super().read(uri)


def test_a_failed_read_of_an_existing_work_file_has_no_link_to_any_error(
    registered, make_profile, root
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    LocalTaskStorage().create(get_task(registered, task_id)["work_uri"], b"other bytes")
    compute = ScriptedCompute()

    with pytest.raises(StorageUnavailable) as raised:
        submit_task(registered, task_id, compute, storage=WorkFileReadFails())

    assert "token=abc123" not in raised.value.message
    assert raised.value.__context__ is None
    assert compute.get_calls("submit") == []


def test_a_task_that_another_pass_holds_is_skipped(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    with lock_task(registered, task_id) as connection:
        assert connection is not None
        assert submit_task(registered, task_id, compute) == "skipped"

    assert compute.calls == []
    assert submit_task(registered, task_id, compute) == "submitted"


def test_a_submit_needs_only_one_connection_of_the_pool(
    registered, database_settings, make_profile, root
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    one_connection = database_settings.model_copy(
        update={"pool_min_size": 1, "pool_max_size": 1, "pool_timeout_s": 2}
    )
    database = open_database(one_connection)
    try:
        assert submit_task(database, task_id, ScriptedCompute()) == "submitted"
    finally:
        database.close()


class ChangesTheTaskDuringSubmit(ScriptedCompute):
    """Accepts the task, and moves it to `cancelled` before the submit is recorded."""

    def __init__(self, database) -> None:
        super().__init__()
        self.database = database

    def submit(self, task):
        with transaction(self.database) as connection:
            connection.execute(
                "UPDATE tasks SET state = 'cancelled' WHERE task_id = %s", (task.task_id,)
            )
        return super().submit(task)


@pytest.mark.parametrize("rejected", [False, True])
def test_a_task_that_changed_during_the_submit_is_not_recorded(
    registered, make_profile, root, rejected
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ChangesTheTaskDuringSubmit(registered)
    if rejected:
        compute.script("submit", SubmissionRejected("no worker"))

    assert submit_task(registered, task_id, compute) == "skipped"

    task = get_task(registered, task_id)
    assert (task["state"], task["compute_ref"], task["submitted_at"]) == ("cancelled", None, None)


def recover(database, task_id, compute, storage=None, setup=LOCAL_SETUP):
    return recover_submission(
        database, task_id, setup=setup, compute=compute,
        storage=storage or LocalTaskStorage(), split_location=split_location,
    )


def write_result(database, task_id) -> None:
    """Write a `result.json` in the task's folder, as a worker does at its end."""
    work_file = get_local_path(get_task(database, task_id)["work_uri"])
    work_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.with_name("result.json").write_bytes(b"{}")


def get_submitting(database, actor="ana", setup=LOCAL_SETUP):
    return get_tasks_in_state(database, "submitting", actor=actor, setup=setup)


def test_the_tasks_in_a_state_are_the_actors_tasks_of_this_setup_oldest_first(
    registered, make_profile, root
):
    _, anas = add_queued_tasks(registered, ["a", "b"], root=root)
    add_queued_tasks(registered, ["c"], root=root, actor="bob")
    elsewhere, _ = add_queued_tasks(registered, ["d"], root=root)
    reserve(registered, make_profile())
    reserve(registered, make_profile(), actor="bob")
    set_execution_settings(registered, elsewhere, KUBERNETES_SETTINGS)

    assert get_submitting(registered) == anas
    assert get_tasks_in_state(registered, "queued", actor="ana", setup=LOCAL_SETUP) == []


def test_a_submitting_task_of_other_compute_is_not_recovered(registered, make_profile, root):
    run_id, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    set_execution_settings(registered, run_id, KUBERNETES_SETTINGS)
    compute = ScriptedCompute()

    assert recover(registered, task_id, compute) == "skipped"

    assert compute.calls == []
    assert get_task(registered, task_id)["state"] == "submitting"


@pytest.mark.parametrize(("found", "recovered"), [("waiting", "submitted"), ("running", "running")])
def test_accept_then_timeout_then_the_next_pass_starts_one_execution(
    registered, make_profile, root, found, recovered
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    compute.script("submit", ACCEPT_THEN_TIMEOUT)
    with pytest.raises(OutcomeUncertain):
        submit_task(registered, task_id, compute)
    compute.script("describe", ComputeStatus(state=found, compute_ref="ref-1"))

    assert recover(registered, task_id, compute) == recovered

    task = get_task(registered, task_id)
    assert (task["state"], task["compute_ref"]) == (recovered, "ref-1")
    assert task["submitted_at"] is not None
    assert (task["started_at"] is not None) == (recovered == "running")
    assert len(compute.get_calls("submit")) == 1
    assert len(compute.executions) == 1


def assert_one_execution(database, task_id, compute, state="submitted") -> None:
    task = get_task(database, task_id)
    assert (task["state"], task["compute_ref"]) == (state, compute.executions[task_id])
    assert len(compute.executions) == 1
    assert checksum_bytes(get_local_path(task["work_uri"]).read_bytes()) == (
        task["work_file_digest"]
    )


def test_a_crash_after_the_reservation_converges(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    assert recover(registered, task_id, compute) == "resubmitted"

    assert_one_execution(registered, task_id, compute)


def test_a_crash_after_the_work_file_write_converges(registered, make_profile, root, monkeypatch):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    def crash(compute, compute_task):
        raise RuntimeError("the controller stopped")

    with monkeypatch.context() as patch:
        patch.setattr(launch, "_submit", crash)
        with pytest.raises(RuntimeError):
            submit_task(registered, task_id, compute)
    assert get_local_path(get_task(registered, task_id)["work_uri"]).exists()

    assert recover(registered, task_id, compute) == "resubmitted"

    assert_one_execution(registered, task_id, compute)


def test_a_crash_after_the_submit_converges(registered, make_profile, root, monkeypatch):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    def crash(connection, task_id, compute_ref):
        raise RuntimeError("the controller stopped")

    with monkeypatch.context() as patch:
        patch.setattr(launch, "_record_submit", crash)
        with pytest.raises(RuntimeError):
            submit_task(registered, task_id, compute)

    assert recover(registered, task_id, compute) == "running"

    assert_one_execution(registered, task_id, compute, state="running")
    assert len(compute.get_calls("submit")) == 1


def test_a_changed_profile_does_not_change_the_rebuilt_work_file(
    registered, make_profile, root, tmp_path
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile(resources={"batch_size": 32}))
    changed = make_profile(resources={"batch_size": 64}).model_copy(
        update={"storage": make_profile().storage.model_copy(update={"root": str(tmp_path / "new")})}
    )
    compute = ScriptedCompute()

    assert recover(registered, task_id, compute, setup=get_setup_parameters(changed, {"local"})) == (
        "resubmitted"
    )

    assert_one_execution(registered, task_id, compute)
    [call] = compute.get_calls("submit")
    assert call.work_uri.startswith(get_task(registered, task_id)["work_uri"].rsplit("/", 1)[0])
    assert root in get_local_path(call.work_uri).as_posix()
    assert b'"batch_size":32' in get_local_path(call.work_uri).read_bytes()


def test_a_result_found_before_a_repeated_submit_prevents_it(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile(max_in_flight=1))
    write_result(registered, task_id)
    compute = ScriptedCompute()
    compute.script("describe", ComputeStatus(state="running", compute_ref="ref-9"))

    assert recover(registered, task_id, compute) == "kept"

    assert compute.get_calls("submit") == []
    assert get_task(registered, task_id)["state"] == "submitting"
    assert count_in_flight(registered) == 1


def test_a_kept_task_keeps_the_compute_ref_that_describe_gave_for_later_calls(
    registered, make_profile, root
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    write_result(registered, task_id)
    compute = ScriptedCompute()
    compute.script(
        "describe",
        ComputeStatus(state="running", compute_ref="ref-9"),
        ComputeStatus(state="not_found"),
    )

    assert recover(registered, task_id, compute) == "kept"
    assert get_task(registered, task_id)["compute_ref"] == "ref-9"

    assert recover(registered, task_id, compute) == "finalizing"

    assert [call.compute_ref for call in compute.get_calls("describe")] == [None, "ref-9"]
    assert get_task(registered, task_id)["compute_ref"] == "ref-9"


def test_a_repeated_submit_gets_the_stored_compute_ref(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    set_compute_ref(registered, task_id, "ref-9")
    compute = ScriptedCompute()

    assert recover(registered, task_id, compute) == "resubmitted"

    [call] = compute.get_calls("submit")
    assert call.compute_ref == "ref-9"


def set_compute_ref(database, task_id, compute_ref) -> None:
    with transaction(database) as connection:
        connection.execute(
            "UPDATE tasks SET compute_ref = %s WHERE task_id = %s", (compute_ref, task_id)
        )


@pytest.mark.parametrize("found", ["ended", "not_found"])
def test_a_result_and_an_ended_execution_make_the_task_finalizing(
    registered, make_profile, root, found
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    write_result(registered, task_id)
    compute = ScriptedCompute()
    compute.script("describe", ComputeStatus(state=found))

    assert recover(registered, task_id, compute) == "finalizing"

    task = get_task(registered, task_id)
    assert (task["state"], task["ended_at"] is not None) == ("finalizing", True)
    assert compute.get_calls("submit") == []
    assert count_in_flight(registered) == 0


def test_a_delayed_first_submit_does_not_start_a_task_that_has_a_result(
    registered, make_profile, root
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    # Before the reserving pass submits, another pass recovers the task, and its submit
    # is uncertain. The execution then writes its result and ends.
    compute.script("submit", ACCEPT_THEN_TIMEOUT)
    with pytest.raises(OutcomeUncertain):
        recover(registered, task_id, compute)
    write_result(registered, task_id)
    del compute.executions[task_id]

    assert submit_task(registered, task_id, compute) == "finalizing"

    assert len(compute.get_calls("submit")) == 1
    assert get_task(registered, task_id)["state"] == "finalizing"


def test_an_execution_that_ended_with_no_result_is_finalizing(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    compute.script("describe", ComputeStatus(state="ended", compute_ref="ref-3"))

    assert recover(registered, task_id, compute) == "finalizing"

    task = get_task(registered, task_id)
    assert (task["state"], task["compute_ref"]) == ("finalizing", "ref-3")
    assert compute.get_calls("submit") == []


class WritesTheResultDuringDescribe(ScriptedCompute):
    """The worker writes its result and ends while `describe` runs."""

    def __init__(self, database) -> None:
        super().__init__()
        self.database = database

    def describe(self, task):
        write_result(self.database, task.task_id)
        return ComputeStatus(state="not_found")


def test_a_result_written_while_describe_runs_prevents_a_repeated_submit(
    registered, make_profile, root
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = WritesTheResultDuringDescribe(registered)

    assert recover(registered, task_id, compute) == "finalizing"

    assert compute.get_calls("submit") == []


def test_not_found_and_no_result_resubmit_the_same_task(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    compute.script("submit", ConnectionError("the service did not answer"))
    with pytest.raises(OutcomeUncertain):
        submit_task(registered, task_id, compute)

    assert recover(registered, task_id, compute) == "resubmitted"

    first, again = compute.get_calls("submit")
    assert first == again
    assert_one_execution(registered, task_id, compute)


def test_a_rejected_resubmit_fails_the_task(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    compute.script("submit", SubmissionRejected("robin-worker is not on the PATH"))

    assert recover(registered, task_id, compute) == "rejected"

    task = get_task(registered, task_id)
    assert (task["state"], task["failure"]["reason"]) == ("failed", "submission_rejected")


def test_an_uncertain_resubmit_keeps_the_slot(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    compute.script("submit", ACCEPT_THEN_TIMEOUT)

    with pytest.raises(OutcomeUncertain) as raised:
        recover(registered, task_id, compute)

    assert raised.value.__context__ is None
    assert get_task(registered, task_id)["state"] == "submitting"


class ReadFails(LocalTaskStorage):
    def read(self, uri):
        raise OSError("token=abc123")


def test_a_storage_read_failure_in_the_result_check_submits_nothing(
    registered, make_profile, root
):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()

    with pytest.raises(StorageUnavailable) as raised:
        recover(registered, task_id, compute, storage=ReadFails())

    assert "token=abc123" not in raised.value.message
    assert raised.value.__context__ is None
    assert compute.get_calls("submit") == []
    assert get_task(registered, task_id)["state"] == "submitting"


def test_a_describe_failure_submits_nothing(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    compute.script("describe", RuntimeError("token=abc123"))

    with pytest.raises(ComputeUnavailable) as raised:
        recover(registered, task_id, compute)

    assert raised.value.details["error_type"] == "RuntimeError"
    assert "token=abc123" not in raised.value.message
    assert raised.value.__context__ is None
    assert compute.get_calls("submit") == []
    assert get_task(registered, task_id)["state"] == "submitting"


def test_two_passes_at_once_submit_once(registered, database_settings, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    compute = ScriptedCompute()
    go_on = threading.Event()
    compute.script("submit", go_on)
    first = []
    database = open_database(database_settings)
    thread = threading.Thread(target=lambda: first.append(recover(database, task_id, compute)))
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not compute.get_calls("submit") and time.monotonic() < deadline:
            time.sleep(0.01)
        second = recover(registered, task_id, compute)
        go_on.set()
        thread.join(30)
    finally:
        database.close()

    assert (first, second) == (["resubmitted"], "skipped")
    assert len(compute.get_calls("submit")) == 1


def test_a_delayed_submit_cannot_start_a_released_task(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())
    [read_earlier] = get_submitting(registered)
    with transaction(registered) as connection:
        # Another pass records the outcome and releases the task.
        connection.execute(
            "UPDATE tasks SET state = 'succeeded', ended_at = now(), "
            "outcome_recorded_at = now(), released_at = now() WHERE task_id = %s",
            (task_id,),
        )
    compute = ScriptedCompute()

    assert recover(registered, read_earlier, compute) == "skipped"

    assert compute.calls == []


class CancelsTheTaskDuringDescribe(ScriptedCompute):
    """Gives a running execution, and moves the task to `cancelled` before it is recorded."""

    def __init__(self, database) -> None:
        super().__init__()
        self.database = database

    def describe(self, task):
        with transaction(self.database) as connection:
            connection.execute(
                "UPDATE tasks SET state = 'cancelled' WHERE task_id = %s", (task.task_id,)
            )
        return ComputeStatus(state="running", compute_ref="ref-1")


def test_a_task_that_changed_during_the_recovery_is_not_recorded(registered, make_profile, root):
    _, [task_id] = add_queued_tasks(registered, ["a"], root=root)
    reserve(registered, make_profile())

    assert recover(registered, task_id, CancelsTheTaskDuringDescribe(registered)) == "skipped"

    task = get_task(registered, task_id)
    assert (task["state"], task["compute_ref"], task["started_at"]) == ("cancelled", None, None)
