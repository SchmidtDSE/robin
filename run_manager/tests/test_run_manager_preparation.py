"""Preparation: claim recordings, record which ones are skipped and why, and make tasks."""

import threading
import time

import pytest

from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.configuration import UnresolvedConfiguration
from robin_run_manager import preparation
from robin_run_manager.database import transaction
from robin_run_manager.preparation import Prepared, get_runs_to_prepare, prepare_run
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS
from run_fixtures import (
    HEAD_CONFIGURATION,
    KUBERNETES_SETTINGS,
    OWL_CONFIGURATION,
    PERCH_CONFIGURATION,
    add_finished_run,
    add_output_file,
    cancel,
    get_outcomes,
    get_run_row,
    get_tasks,
    register_fixture_models,
    set_outcome,
    submit,
    withdraw,
)

VALUES = [f"r{index:03}" for index in range(250)]
NEWER_OWL = tuple(
    ModelOutputVersion(model=entry.model, version=2) if entry.model == "model:owl/v4" else entry
    for entry in RELEASE_OUTPUT_VERSIONS
)


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


def prepare(database, run_id, size=100, **options) -> Prepared:
    return prepare_run(database, run_id, recordings_per_task=size, **options)


def get_states(database, run_id) -> dict[str, tuple]:
    return {
        value: (row["state"], row["skip_reason"])
        for value, row in get_outcomes(database, run_id).items()
    }


def test_preparation_claims_every_recording_and_makes_tasks(registered):
    run_id = submit(registered, list(reversed(VALUES)))

    prepared = prepare(registered, run_id)

    assert prepared == Prepared(state="ready", tasks=3, claimed=250, skipped=0)
    tasks = get_tasks(registered, run_id)
    assert [task["state"] for task in tasks] == ["queued"] * 3
    assert [task["values"] for task in tasks] == [VALUES[:100], VALUES[100:200], VALUES[200:]]
    rows = get_outcomes(registered, run_id)
    assert {row["state"] for row in rows.values()} == {"pending"}
    assert all(row["task_id"] is not None for row in rows.values())
    assert get_run_row(registered, run_id)["state"] == "ready"


def test_a_recording_covered_by_another_run_is_skipped_as_already_covered(registered):
    add_finished_run(registered, ["a"])
    run_id = submit(registered, ["a", "b"])

    prepared = prepare(registered, run_id)

    assert (prepared.claimed, prepared.skipped) == (1, 1)
    assert get_states(registered, run_id) == {
        "a": ("skipped", "already_covered"),
        "b": ("pending", None),
    }


def test_a_recording_pending_in_another_run_is_skipped_as_in_progress(registered):
    add_finished_run(registered, ["a"], state="pending")
    run_id = submit(registered, ["a", "b"])

    prepare(registered, run_id)

    assert get_states(registered, run_id)["a"] == ("skipped", "in_progress")


def test_a_recording_covered_at_other_versions_is_claimed(registered):
    add_finished_run(registered, ["a"])
    run_id = submit(registered, ["a"], release_versions=NEWER_OWL)

    prepare(registered, run_id, release_versions=NEWER_OWL)

    assert get_states(registered, run_id) == {"a": ("pending", None)}


def test_a_recording_covered_by_a_withdrawn_run_is_claimed(registered):
    withdraw(registered, add_finished_run(registered, ["a"]))
    run_id = submit(registered, ["a"])

    prepare(registered, run_id)

    assert get_states(registered, run_id) == {"a": ("pending", None)}


def test_a_run_with_nothing_to_claim_is_ready_with_no_tasks(registered):
    add_finished_run(registered, ["a", "b"])
    run_id = submit(registered, ["a", "b"])

    prepared = prepare(registered, run_id)

    assert prepared == Prepared(state="ready", tasks=0, claimed=0, skipped=2)
    assert get_tasks(registered, run_id) == []
    assert get_run_row(registered, run_id)["state"] == "ready"


def test_outdated_versions_fail_preparation_and_create_nothing(registered):
    run_id = submit(registered, ["a", "b"])

    prepared = prepare(registered, run_id, release_versions=NEWER_OWL)

    assert prepared.state == "preparation_failed"
    run = get_run_row(registered, run_id)
    assert run["state"] == "preparation_failed"
    assert run["preparation_failure"]["reason"] == "output_version_not_current"
    assert run["preparation_failure"]["details"] == {
        "output_versions": {"model:owl/v4": {"run": 1, "release": 2}}
    }
    assert "new run" in run["preparation_failure"]["message"]
    assert (get_outcomes(registered, run_id), get_tasks(registered, run_id)) == ({}, [])


def test_a_cancelled_preparing_run_keeps_its_list_and_gets_no_tasks(registered):
    run_id = submit(registered, ["a", "b"])
    cancel(registered, run_id)

    prepared = prepare(registered, run_id)

    assert prepared.state == "not_preparing"
    assert (get_outcomes(registered, run_id), get_tasks(registered, run_id)) == ({}, [])
    with transaction(registered, read_only=True) as connection:
        kept = connection.execute(
            "SELECT count(*) AS n FROM run_recordings WHERE run_id = %s", (run_id,)
        ).fetchone()["n"]
    assert kept == 2
    assert get_run_row(registered, run_id)["state"] == "preparing"


def test_a_crash_before_commit_leaves_nothing_and_the_next_pass_prepares_the_same_list(
    registered, monkeypatch
):
    clean = submit(registered, VALUES, OWL_CONFIGURATION)
    prepare(registered, clean, size=40)
    with transaction(registered) as connection:
        # Free the clean run's claims, so that the second run can claim the same list.
        connection.execute("UPDATE recording_outcomes SET state = 'released'")
    run_id = submit(registered, VALUES)

    def crash(*arguments, **options):
        raise RuntimeError("the controller stopped")

    with monkeypatch.context() as patched:
        patched.setattr(preparation, "_create_tasks", crash)
        with pytest.raises(RuntimeError):
            prepare(registered, run_id, size=40)
    assert get_outcomes(registered, run_id) == {}
    assert get_run_row(registered, run_id)["state"] == "preparing"

    prepared = prepare(registered, run_id, size=40)

    assert prepared.state == "ready"
    assert [task["values"] for task in get_tasks(registered, run_id)] == [
        task["values"] for task in get_tasks(registered, clean)
    ]


def test_a_second_preparation_of_a_ready_run_changes_nothing(registered):
    run_id = submit(registered, VALUES)
    prepare(registered, run_id)

    again = prepare(registered, run_id)

    assert again == Prepared(state="not_preparing")
    assert len(get_tasks(registered, run_id)) == 3


@pytest.mark.parametrize("repeat", range(20))
def test_concurrent_overlapping_preparations_claim_each_recording_once(registered, repeat):
    shared = [f"s{index:02}" for index in range(30)]
    first = submit(registered, ["x1", *shared])
    second = submit(registered, [*reversed(shared), "x2"])
    start = threading.Barrier(2)
    errors = []

    def run(run_id):
        start.wait()
        try:
            prepare(registered, run_id, size=7)
        except Exception as error:  # noqa: BLE001 - the test reports any error
            errors.append(error)

    threads = [threading.Thread(target=run, args=(run_id,)) for run_id in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    one, two = get_states(registered, first), get_states(registered, second)
    for value in shared:
        assert sorted([one[value], two[value]]) == [("pending", None), ("skipped", "in_progress")]
    assert (one["x1"], two["x2"]) == (("pending", None), ("pending", None))


def release_a_claim_during_preparation(registered, monkeypatch):
    """Make run A find, after its claim insert, that a claim it skipped was released.

    Returns A's run ID, B's run ID, A's outcome, B's error list, and A's call count.
    """
    holder = add_finished_run(registered, ["a"], state="pending")
    run_a = submit(registered, ["a", "b"])
    run_b = submit(registered, ["b", "a"])
    paused, go_on = threading.Event(), threading.Event()
    calls = {run_a: 0, run_b: 0}
    real = preparation._claim_recordings

    def claim_then_pause(connection, run):
        result = real(connection, run)
        calls[run.run_id] += 1
        if run.run_id == run_a and calls[run_a] == 1:
            paused.set()
            assert go_on.wait(30)
        return result

    monkeypatch.setattr(preparation, "_claim_recordings", claim_then_pause)
    results, errors = {}, []

    def run(run_id):
        try:
            results[run_id] = prepare(registered, run_id)
        except Exception as error:  # noqa: BLE001 - the test reports any error
            errors.append(error)

    thread_a = threading.Thread(target=run, args=(run_a,))
    thread_a.start()
    assert paused.wait(30)
    with transaction(registered) as connection:
        connection.execute(
            "UPDATE recording_outcomes SET state = 'released' WHERE run_id = %s", (holder,)
        )
    thread_b = threading.Thread(target=run, args=(run_b,))
    thread_b.start()
    wait_until_a_backend_waits_on_a_lock(registered)
    go_on.set()
    thread_a.join(30)
    thread_b.join(30)
    return run_a, run_b, results, errors, calls


def wait_until_a_backend_waits_on_a_lock(database) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with transaction(database, read_only=True) as connection:
            waiting = connection.execute(
                "SELECT count(*) AS n FROM pg_stat_activity "
                "WHERE datname = current_database() AND wait_event_type = 'Lock'"
            ).fetchone()["n"]
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError("B never waited on A's claim")


def test_a_claim_released_during_preparation_does_not_deadlock(registered, monkeypatch):
    run_a, run_b, results, errors, calls = release_a_claim_during_preparation(
        registered, monkeypatch
    )

    assert errors == []
    assert (results[run_a].state, results[run_b].state) == ("ready", "ready")
    assert get_states(registered, run_b) == {"a": ("pending", None), "b": ("pending", None)}
    assert get_states(registered, run_a) == {
        "a": ("skipped", "in_progress"),
        "b": ("skipped", "in_progress"),
    }
    assert calls[run_a] == 2


def test_a_run_that_keeps_finding_released_claims_returns_try_again(registered, monkeypatch):
    monkeypatch.setattr(preparation, "_ATTEMPTS", 1)

    run_a, run_b, results, errors, calls = release_a_claim_during_preparation(
        registered, monkeypatch
    )

    assert errors == []
    assert results[run_a] == Prepared(state="try_again")
    assert get_run_row(registered, run_a)["state"] == "preparing"
    assert get_outcomes(registered, run_a) == {}


def test_initial_tasks_name_the_submitter(registered):
    run_id = submit(registered, ["a"], actor="ben")

    prepare(registered, run_id)

    assert [task["created_by"] for task in get_tasks(registered, run_id)] == ["ben"]


def test_get_runs_to_prepare_leaves_out_cancelled_withdrawn_and_ready_runs(registered):
    older = submit(registered, ["a"])
    cancelled = submit(registered, ["b"])
    withdrawn = submit(registered, ["c"])
    ready = submit(registered, ["d"])
    newer = submit(registered, ["e"])
    cancel(registered, cancelled)
    withdraw(registered, withdrawn)
    prepare(registered, ready)

    assert get_runs_to_prepare(registered) == [older, newer]


def test_a_released_claim_holds_nothing(registered):
    holder = add_finished_run(registered, ["a"], state="pending")
    set_outcome(registered, holder, "a", "released")
    run_id = submit(registered, ["a"])

    prepare(registered, run_id)

    assert get_states(registered, run_id) == {"a": ("pending", None)}


# Head input pinning


def get_pins(database, run_id) -> dict[str, tuple]:
    return {
        value: (row["state"], row["skip_reason"], row["input_file_id"])
        for value, row in get_outcomes(database, run_id).items()
    }


def add_backbone_with_files(database, values, **options):
    """A finished backbone run whose recordings are covered, each with its embeddings file."""
    run_id = add_finished_run(database, values, PERCH_CONFIGURATION, **options)
    return run_id, {value: add_output_file(database, run_id, value) for value in values}


def add_prepared_head(database, values):
    run_id = submit(database, values, HEAD_CONFIGURATION)
    prepare(database, run_id)
    return run_id


def test_a_head_pins_the_backbones_embeddings_file(registered):
    _, files = add_backbone_with_files(registered, ["a", "b"])

    head = add_prepared_head(registered, ["a", "b"])

    assert get_pins(registered, head) == {
        "a": ("pending", None, files["a"]),
        "b": ("pending", None, files["b"]),
    }


def test_no_backbone_result_is_skipped_as_input_not_available(registered):
    add_backbone_with_files(registered, ["a"])

    head = add_prepared_head(registered, ["a", "b"])

    assert get_pins(registered, head)["b"] == ("skipped", "input_not_available", None)
    assert get_pins(registered, head)["a"][0] == "pending"


@pytest.mark.parametrize("state", ["pending", "failed"])
def test_a_pending_or_failed_backbone_outcome_is_input_not_available(registered, state):
    add_finished_run(registered, ["a"], PERCH_CONFIGURATION, state=state)

    head = add_prepared_head(registered, ["a"])

    assert get_pins(registered, head) == {"a": ("skipped", "input_not_available", None)}


def test_a_covered_backbone_with_no_embeddings_file_is_input_has_no_windows(registered):
    backbone = add_finished_run(registered, ["a"], PERCH_CONFIGURATION)
    add_output_file(registered, backbone, "a", kind="scores")

    head = add_prepared_head(registered, ["a"])

    assert get_pins(registered, head) == {"a": ("skipped", "input_has_no_windows", None)}


def test_a_withdrawn_backbone_run_is_ignored(registered):
    backbone, _ = add_backbone_with_files(registered, ["a"])
    with transaction(registered) as connection:
        connection.execute(
            "UPDATE runs SET withdrawn_at = now(), withdrawn_by = 'ana' WHERE run_id = %s",
            (backbone,),
        )

    head = add_prepared_head(registered, ["a"])

    assert get_pins(registered, head) == {"a": ("skipped", "input_not_available", None)}


def test_a_backbone_result_at_another_version_is_ignored(registered):
    newer_perch = tuple(
        ModelOutputVersion(model=entry.model, version=2)
        if entry.model == "model:perch/v8" else entry
        for entry in RELEASE_OUTPUT_VERSIONS
    )
    add_backbone_with_files(registered, ["a"], release_versions=newer_perch)

    head = add_prepared_head(registered, ["a"])

    assert get_pins(registered, head) == {"a": ("skipped", "input_not_available", None)}


def test_a_backbone_result_of_another_backbone_configuration_is_ignored(registered):
    other_perch = UnresolvedConfiguration.model_validate(
        PERCH_CONFIGURATION.model_dump() | {"settings": {"window_overlap": 1.0}}
    )
    # Stores the head's own backbone configuration, with no result for "a".
    submit(registered, ["unused"], PERCH_CONFIGURATION)
    other = add_finished_run(registered, ["a"], other_perch)
    add_output_file(registered, other, "a")

    head = add_prepared_head(registered, ["a"])

    assert get_pins(registered, head) == {"a": ("skipped", "input_not_available", None)}


def test_a_local_embeddings_file_under_a_kubernetes_run_is_skipped(registered):
    backbone = add_finished_run(registered, ["a", "b"], PERCH_CONFIGURATION)
    add_output_file(registered, backbone, "a", uri="file:///out/a.parquet")
    pinned = add_output_file(registered, backbone, "b")

    head = submit(
        registered, ["a", "b"], HEAD_CONFIGURATION, execution_settings=KUBERNETES_SETTINGS
    )
    prepare(registered, head)

    assert get_pins(registered, head) == {
        "a": ("skipped", "input_location_unsupported", None),
        "b": ("pending", None, pinned),
    }


def test_a_head_recording_with_no_input_holds_no_claim(registered):
    submit(registered, ["unused"], PERCH_CONFIGURATION)
    first = add_prepared_head(registered, ["a"])
    _, files = add_backbone_with_files(registered, ["a"])

    later = add_prepared_head(registered, ["a"])

    assert get_pins(registered, first) == {"a": ("skipped", "input_not_available", None)}
    assert get_pins(registered, later) == {"a": ("pending", None, files["a"])}


def test_an_audio_run_pins_nothing(registered):
    run_id = submit(registered, ["a", "b"])

    prepare(registered, run_id)

    assert {row[2] for row in get_pins(registered, run_id).values()} == {None}
