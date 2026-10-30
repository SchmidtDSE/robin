"""Task updates from the compute adapter, and outcome recording."""

import uuid

import pytest

from robin_adapters.storage.local import LocalTaskStorage
from robin_adapters.storage.location import split_location
from robin_contracts.ports.compute import ComputeStatus
from robin_contracts.results import FailureReport, InferenceFailure
from robin_contracts.tasks import TaskRefusal, WorkerError
from robin_contracts.work import work_digest
from robin_run_manager.database import transaction
from robin_run_manager.errors import ComputeUnavailable, StorageUnavailable
from robin_run_manager.outcomes import store_task_outcome
from robin_run_manager.preparation import prepare_run
from run_fixtures import (
    KUBERNETES_SETTINGS,
    LOCAL_SETTINGS,
    get_task_setup,
    ROBIN_RELEASE,
    add_finalizing_task,
    add_submitted_task,
    build_completed,
    build_task_result,
    update_task,
    count_in_flight,
    get_current_files,
    get_outcomes,
    get_task,
    get_work_file,
    register_fixture_models,
    set_execution_settings,
    submit,
    write_task_result,
)

REFUSAL_CODES = (
    "work_file_unreadable",
    "work_file_invalid",
    "unknown_schema_version",
    "model_identifier_mismatch",
    "output_version_mismatch",
)


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


@pytest.fixture
def root(tmp_path):
    """The storage root that the test's runs freeze."""
    return str(tmp_path / "storage")


def store_outcome(database, task_id, storage=None):
    return store_task_outcome(
        database, task_id, setup=get_task_setup(database, task_id),
        storage=storage or LocalTaskStorage(), split_location=split_location,
    )


def get_recordings(database, task_id) -> dict[str, tuple]:
    """Each recording of the task, by value: its state and its failure."""
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT value, state, failure FROM recording_outcomes WHERE task_id = %s",
            (task_id,),
        ).fetchall()
    return {row["value"]: (row["state"], row["failure"]) for row in rows}


def count_output_files(database, task_id) -> int:
    with transaction(database, read_only=True) as connection:
        return connection.execute(
            "SELECT count(*) AS n FROM output_files WHERE task_id = %s", (task_id,)
        ).fetchone()["n"]


def assert_task_failed(database, task_id, reason) -> dict:
    """Check that the task and each of its recordings failed with `reason`. Gives the
    task's failure."""
    task = get_task(database, task_id)
    assert task["state"] == "failed"
    assert task["failure"]["reason"] == reason
    assert task["outcome_recorded_at"] is not None
    for state, failure in get_recordings(database, task_id).values():
        assert (state, failure["scope"], failure["code"]) == ("failed", "task", reason)
    assert count_output_files(database, task_id) == 0
    return task["failure"]


class ReadFails(LocalTaskStorage):
    def read(self, uri):
        raise OSError("token=abc123")


# --- update_task_from_compute -------------------------------------------------

def test_ended_frees_the_slot_and_keeps_the_claims(registered, root):
    task_id, compute = add_submitted_task(registered, root)
    assert count_in_flight(registered) == 1
    compute.script("describe", ComputeStatus(state="ended"))

    assert update_task(registered, task_id, compute) == "finalizing"

    task = get_task(registered, task_id)
    assert (task["state"], task["ended_at"] is not None) == ("finalizing", True)
    assert count_in_flight(registered) == 0
    assert {state for state, _ in get_recordings(registered, task_id).values()} == {"pending"}


def test_a_started_execution_makes_the_task_running(registered, root):
    task_id, compute = add_submitted_task(registered, root)

    assert update_task(registered, task_id, compute) == "running"

    task = get_task(registered, task_id)
    assert (task["state"], task["started_at"] is not None) == ("running", True)
    assert count_in_flight(registered) == 1


def test_a_waiting_execution_keeps_the_task_submitted(registered, root):
    task_id, compute = add_submitted_task(registered, root)
    compute.script("describe", ComputeStatus(state="waiting"))

    assert update_task(registered, task_id, compute) == "submitted"

    assert get_task(registered, task_id)["state"] == "submitted"


def test_not_found_for_a_confirmed_submit_is_ended(registered, root):
    task_id, compute = add_submitted_task(registered, root)
    update_task(registered, task_id, compute)
    del compute.executions[task_id]

    assert update_task(registered, task_id, compute) == "finalizing"

    assert get_task(registered, task_id)["state"] == "finalizing"


def test_a_running_task_of_other_compute_is_not_described(registered, root):
    task_id, compute = add_submitted_task(registered, root)
    update_task(registered, task_id, compute)
    set_execution_settings(registered, get_task(registered, task_id)["run_id"], KUBERNETES_SETTINGS)
    compute.calls.clear()

    assert update_task(registered, task_id, compute) == "skipped"

    assert compute.calls == []
    assert get_task(registered, task_id)["state"] == "running"


def test_a_describe_failure_keeps_the_task_and_hides_the_message(registered, root):
    task_id, compute = add_submitted_task(registered, root)
    compute.script("describe", RuntimeError("token=abc123"))

    with pytest.raises(ComputeUnavailable) as raised:
        update_task(registered, task_id, compute)

    assert "token=abc123" not in raised.value.message
    assert raised.value.__context__ is None
    assert get_task(registered, task_id)["state"] == "submitted"


def test_a_result_while_the_execution_runs_records_nothing(registered, root):
    task_id, compute = add_submitted_task(registered, root)
    work_file = get_work_file(registered, task_id)
    write_task_result(work_file, build_task_result(work_file, build_completed(work_file)))

    assert update_task(registered, task_id, compute) == "running"
    assert store_outcome(registered, task_id) == "skipped"

    task = get_task(registered, task_id)
    assert (task["state"], task["outcome_recorded_at"]) == ("running", None)
    assert count_in_flight(registered) == 1
    assert count_output_files(registered, task_id) == 0


# --- store_task_outcome -------------------------------------------------------

def test_a_finalizing_task_of_other_storage_is_not_read(registered, root):
    task_id, _ = add_finalizing_task(registered, root)
    settings = {**LOCAL_SETTINGS, "storage": {"kind": "s3", "root": "s3://robin/"}}
    set_execution_settings(registered, get_task(registered, task_id)["run_id"], settings)

    assert store_outcome(registered, task_id, storage=ReadFails()) == "skipped"

    assert get_task(registered, task_id)["state"] == "finalizing"


def test_a_completed_result_covers_its_recordings_and_publishes_their_files(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    completed = build_completed(work_file)
    write_task_result(work_file, build_task_result(work_file, completed))

    assert store_outcome(registered, task_id) == "recorded"

    task = get_task(registered, task_id)
    assert (task["state"], task["failure"], task["robin_release"]) == (
        "succeeded", None, ROBIN_RELEASE,
    )
    assert task["outcome_recorded_at"] is not None
    assert get_recordings(registered, task_id) == {
        "a": ("covered", None), "b": ("covered", None),
    }
    current = get_current_files(registered)
    assert [(row["value"], row["kind"], row["uri"], row["checksum"]) for row in current] == [
        (one.value, one.kind, one.uri, one.checksum) for one in completed.artifacts
    ]
    assert {row["run_id"] for row in current} == {task["run_id"]}


def test_a_completed_result_keeps_each_recording_failure(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    write_task_result(
        work_file, build_task_result(work_file, build_completed(work_file, failed={"b"}))
    )

    assert store_outcome(registered, task_id) == "recorded"

    assert get_task(registered, task_id)["state"] == "succeeded"
    assert get_recordings(registered, task_id) == {
        "a": ("covered", None),
        "b": ("failed", {
            "scope": "recording", "code": "input_unavailable", "stage": "acquire_input",
            "message": "gone",
        }),
    }
    assert count_output_files(registered, task_id) == 1
    # The failed recording's claim is released, so a new run claims it.
    settings = {**LOCAL_SETTINGS, "storage": {"kind": "local", "root": root}}
    again = submit(registered, ["a", "b"], execution_settings=settings)
    prepare_run(registered, again, recordings_per_task=2)
    outcomes = get_outcomes(registered, again)
    assert (outcomes["a"]["skip_reason"], outcomes["b"]["state"]) == ("already_covered", "pending")


def test_a_result_where_every_recording_failed_is_succeeded(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    write_task_result(
        work_file, build_task_result(work_file, build_completed(work_file, failed={"a", "b"}))
    )

    assert store_outcome(registered, task_id) == "recorded"

    assert get_task(registered, task_id)["state"] == "succeeded"
    assert {state for state, _ in get_recordings(registered, task_id).values()} == {"failed"}
    assert count_output_files(registered, task_id) == 0


def test_a_missing_result_after_the_end_is_no_result(registered, root):
    task_id, _ = add_finalizing_task(registered, root)

    assert store_outcome(registered, task_id) == "recorded"

    assert_task_failed(registered, task_id, "no_result")
    assert get_task(registered, task_id)["robin_release"] is None


def test_a_storage_read_failure_leaves_the_task_finalizing(registered, root):
    task_id, _ = add_finalizing_task(registered, root)

    with pytest.raises(StorageUnavailable) as raised:
        store_outcome(registered, task_id, storage=ReadFails())

    assert "token=abc123" not in raised.value.message
    assert raised.value.__context__ is None
    task = get_task(registered, task_id)
    assert (task["state"], task["outcome_recorded_at"]) == ("finalizing", None)
    assert {state for state, _ in get_recordings(registered, task_id).values()} == {"pending"}


def test_a_result_that_is_not_json_is_invalid_result(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    write_task_result(work_file, b"{not json")

    assert store_outcome(registered, task_id) == "recorded"

    assert assert_task_failed(registered, task_id, "invalid_result")["detail"]


@pytest.mark.parametrize(
    "fields",
    [{"task_id": uuid.uuid4()}, {"work_file_digest": "sha256:" + "9" * 64}],
    ids=["task_id", "work_file_digest"],
)
def test_a_result_for_another_task_or_work_file_is_invalid_result(registered, root, fields):
    task_id, work_file = add_finalizing_task(registered, root)
    write_task_result(
        work_file, build_task_result(work_file, build_completed(work_file), **fields)
    )

    assert store_outcome(registered, task_id) == "recorded"

    assert_task_failed(registered, task_id, "invalid_result")
    assert get_current_files(registered) == []


def test_an_incomplete_result_is_invalid_result_and_publishes_nothing(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    completed = build_completed(work_file)
    [kept_artifact, _] = completed.artifacts
    [kept_coverage, _] = completed.coverage
    incomplete = completed.model_copy(
        update={"artifacts": (kept_artifact,), "coverage": (kept_coverage,)}
    )
    write_task_result(work_file, build_task_result(work_file, incomplete))

    assert store_outcome(registered, task_id) == "recorded"

    assert "omits" in assert_task_failed(registered, task_id, "invalid_result")["detail"]
    assert get_current_files(registered) == []


def test_an_artifact_outside_the_output_root_is_invalid_result(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    completed = build_completed(work_file)
    outside = completed.artifacts[0].model_copy(update={"uri": f"file://{root}/elsewhere.parquet"})
    moved = completed.model_copy(update={"artifacts": (outside, *completed.artifacts[1:])})
    write_task_result(work_file, build_task_result(work_file, moved))

    assert store_outcome(registered, task_id) == "recorded"

    assert_task_failed(registered, task_id, "invalid_result")


def test_an_inference_failure_fails_the_task_with_its_code(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    failure = InferenceFailure(
        schema_version="robin.inference-result/1",
        work_digest=work_digest(work_file.task.work),
        failure=FailureReport(
            code="model_file_unavailable", stage="acquire_model", detail="the weights are gone"
        ),
    )
    write_task_result(work_file, build_task_result(work_file, failure))

    assert store_outcome(registered, task_id) == "recorded"

    failure = assert_task_failed(registered, task_id, "model_file_unavailable")
    assert failure["detail"] == "the weights are gone"
    assert get_task(registered, task_id)["robin_release"] == ROBIN_RELEASE


@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_each_refusal_code_fails_the_task_with_that_code(registered, root, code):
    task_id, work_file = add_finalizing_task(registered, root)
    refusal = TaskRefusal(code=code, detail="the worker refused")
    write_task_result(work_file, build_task_result(work_file, refusal))

    assert store_outcome(registered, task_id) == "recorded"

    assert assert_task_failed(registered, task_id, code)["detail"] == "the worker refused"


def test_a_worker_error_keeps_its_detail_on_the_task(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    error = WorkerError(detail="MemoryError: out of memory")
    write_task_result(work_file, build_task_result(work_file, error))

    assert store_outcome(registered, task_id) == "recorded"

    failure = assert_task_failed(registered, task_id, "worker_error")
    assert failure["detail"] == "MemoryError: out of memory"


def get_rows(database, task_id) -> tuple:
    with transaction(database, read_only=True) as connection:
        task = connection.execute(
            "SELECT * FROM tasks WHERE task_id = %s", (task_id,)
        ).fetchone()
        outcomes = connection.execute(
            "SELECT * FROM recording_outcomes WHERE task_id = %s ORDER BY value", (task_id,)
        ).fetchall()
        files = connection.execute(
            "SELECT * FROM output_files WHERE task_id = %s ORDER BY file_id", (task_id,)
        ).fetchall()
    return task, outcomes, files


def test_recording_twice_gives_the_same_state(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    write_task_result(work_file, build_task_result(work_file, build_completed(work_file)))
    assert store_outcome(registered, task_id) == "recorded"
    before = get_rows(registered, task_id)
    # A later file does not change a recorded outcome.
    write_task_result(work_file, build_task_result(work_file, WorkerError(detail="later")))

    assert store_outcome(registered, task_id) == "already_recorded"

    assert get_rows(registered, task_id) == before


class RecordsDuringRead(LocalTaskStorage):
    """Another pass records the task's outcome while this pass reads the result."""

    def __init__(self, database) -> None:
        super().__init__()
        self.database = database

    def read(self, uri):
        data = super().read(uri)
        if uri.endswith("result.json"):
            task_id = uuid.UUID(uri.split("/")[-2])
            assert store_outcome(self.database, task_id) == "recorded"
        return data


def test_an_outcome_recorded_during_the_read_is_not_recorded_again(registered, root):
    task_id, work_file = add_finalizing_task(registered, root)
    write_task_result(work_file, build_task_result(work_file, build_completed(work_file)))

    assert store_outcome(registered, task_id, storage=RecordsDuringRead(registered)) == (
        "already_recorded"
    )

    assert get_task(registered, task_id)["state"] == "succeeded"
    assert count_output_files(registered, task_id) == 2
