"""Release: after a task's outcome is recorded, tell the compute adapter to free what
the execution kept."""

import pytest

from robin_adapters.compute.local import LocalCompute
from robin_adapters.storage.local import LocalTaskStorage
from robin_adapters.storage.location import split_location
from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import ComputeStatus
from robin_run_manager.errors import ComputeUnavailable
from robin_run_manager.outcomes import get_tasks_to_release, store_task_outcome, release_task
from run_fixtures import (
    KUBERNETES_SETTINGS,
    LOCAL_SETTINGS,
    get_local_setup,
    get_task_setup,
    add_submitted_task,
    add_task,
    build_completed,
    build_task_result,
    update_task,
    get_outcomes,
    get_task,
    get_work_file,
    register_fixture_models,
    set_execution_settings,
    submit,
    write_task_result,
)

from compute_doubles import ScriptedCompute


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


@pytest.fixture
def root(tmp_path):
    """The storage root that the test's runs freeze."""
    return str(tmp_path / "storage")


# The storage root of the runs that `submit` makes.
SUBMIT_ROOT = LOCAL_SETTINGS["storage"]["root"]


def list_to_release(database, root, actor="ana"):
    return get_tasks_to_release(database, actor=actor, setup=get_local_setup(root))


def release(database, task_id, compute):
    return release_task(
        database, task_id, setup=get_task_setup(database, task_id), compute=compute
    )


def add_succeeded_task(database, root, *, actor="ana"):
    """A succeeded task whose outcome is recorded. Gives the task and its compute."""
    task_id, compute = add_submitted_task(database, root, actor=actor)
    compute.script("describe", ComputeStatus(state="ended"))
    assert update_task(database, task_id, compute) == "finalizing"
    work_file = get_work_file(database, task_id)
    write_task_result(work_file, build_task_result(work_file, build_completed(work_file)))
    assert store_task_outcome(
        database, task_id, setup=get_task_setup(database, task_id), storage=LocalTaskStorage(),
        split_location=split_location,
    ) == "recorded"
    compute.calls.clear()
    return task_id, compute


@pytest.mark.parametrize("state", ["submitting", "submitted", "running", "finalizing"])
def test_release_runs_only_after_a_terminal_outcome(registered, state):
    task_id = add_task(registered, submit(registered, ["a"]), state)
    compute = ScriptedCompute()

    assert list_to_release(registered, SUBMIT_ROOT) == []
    assert release(registered, task_id, compute) == "skipped"
    assert compute.calls == []
    assert get_task(registered, task_id)["released_at"] is None


def test_a_task_cancelled_while_queued_needs_no_release(registered):
    task_id = add_task(registered, submit(registered, ["a"]), "cancelled")
    compute = ScriptedCompute()

    assert list_to_release(registered, SUBMIT_ROOT) == []
    assert release(registered, task_id, compute) == "skipped"
    assert compute.calls == []


def test_a_repeated_release_succeeds_when_the_job_is_already_absent(registered, root):
    task_id, compute = add_succeeded_task(registered, root)
    assert list_to_release(registered, root) == [task_id]
    # An earlier pass released the job and stopped before it recorded the release.
    del compute.executions[task_id]

    assert release(registered, task_id, compute) == "released"

    [call] = compute.get_calls("release")
    assert (call.task_id, call.compute_ref) == (task_id, "ref-1")
    assert get_task(registered, task_id)["released_at"] is not None
    assert list_to_release(registered, root) == []
    assert release(registered, task_id, compute) == "skipped"
    assert len(compute.get_calls("release")) == 1


def test_a_failed_release_stays_pending_and_keeps_the_outcome(registered, root):
    task_id, compute = add_succeeded_task(registered, root)
    before = get_task(registered, task_id)
    outcomes = get_outcomes(registered, before["run_id"])
    compute.script("release", RuntimeError("token=abc123"))

    with pytest.raises(ComputeUnavailable) as raised:
        release(registered, task_id, compute)

    assert "token=abc123" not in raised.value.message
    assert raised.value.__context__ is None
    assert get_task(registered, task_id) == before
    assert get_outcomes(registered, before["run_id"]) == outcomes
    assert list_to_release(registered, root) == [task_id]

    assert release(registered, task_id, compute) == "released"
    after = get_task(registered, task_id)
    assert after["released_at"] is not None
    assert {**after, "released_at": None} == before


def get_files(folder) -> dict:
    """Each file under `folder`, with its bytes."""
    return {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def test_release_leaves_the_work_file_result_and_outputs(registered, root):
    task_id, _ = add_succeeded_task(registered, root)
    work_path = get_local_path(get_task(registered, task_id)["work_uri"])
    output = work_path.parent / "outputs" / "scores" / "a.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"rows")
    files = get_files(work_path.parent)
    assert {work_path.name, "result.json"} <= {path.name for path in files}

    assert release(registered, task_id, LocalCompute()) == "released"

    assert get_files(work_path.parent) == files


def test_only_the_submitters_tasks_are_released(registered, root):
    task_id, compute = add_succeeded_task(registered, root, actor="bo")

    assert list_to_release(registered, root, actor="ana") == []
    assert list_to_release(registered, root, actor="bo") == [task_id]


def test_a_task_of_other_compute_is_not_released(registered, root):
    task_id, compute = add_succeeded_task(registered, root)
    run_id = get_task(registered, task_id)["run_id"]
    set_execution_settings(registered, run_id, KUBERNETES_SETTINGS)

    assert list_to_release(registered, root) == []
    assert release(registered, task_id, compute) == "skipped"

    assert compute.calls == []
    assert get_task(registered, task_id)["released_at"] is None
