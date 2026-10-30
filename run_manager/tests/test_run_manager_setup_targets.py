"""The target check: a setup name keeps one target while its runs have work."""

from datetime import UTC, datetime

import pytest

from robin_run_manager.database import transaction
from robin_run_manager.errors import NotConfigured
from robin_run_manager.preparation import prepare_run
from robin_run_manager.setup_targets import refuse_conflicting_setup_target
from run_fixtures import (
    KUBERNETES_SETTINGS,
    LOCAL_SETTINGS,
    get_tasks,
    register_fixture_models,
    submit,
)


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


def add_queued_run(database, settings):
    """A prepared run with one queued task, frozen with `settings`."""
    run_id = submit(database, ["a"], execution_settings=settings)
    prepare_run(database, run_id, recordings_per_task=1)
    return run_id


def check(database, setup, target_compute, storage):
    with transaction(database, read_only=True) as connection:
        refuse_conflicting_setup_target(connection, setup, target_compute, storage)


def test_the_same_target_passes(registered):
    add_queued_run(registered, LOCAL_SETTINGS)

    check(registered, "local", {"kind": "local"}, LOCAL_SETTINGS["storage"])


def test_a_changed_target_with_a_queued_task_is_refused(registered):
    add_queued_run(registered, KUBERNETES_SETTINGS)
    target = {"kind": "kubernetes", "namespace": "ns-b"}

    with pytest.raises(NotConfigured) as raised:
        check(registered, "cluster", target, KUBERNETES_SETTINGS["storage"])

    assert raised.value.details == {"setup": "cluster"}
    assert "cluster" in raised.value.message
    for value in ("ns-b", "robin", "s3://", "s3.example"):
        assert value not in raised.value.message


def test_a_changed_target_after_all_work_is_released_passes(registered):
    run_id = add_queued_run(registered, KUBERNETES_SETTINGS)
    [task] = get_tasks(registered, run_id)
    now = datetime.now(UTC)
    with transaction(registered) as connection:
        connection.execute(
            "UPDATE tasks SET state = 'succeeded', reserved_at = %s, released_at = %s, "
            "work_file_digest = %s WHERE task_id = %s",
            (now, now, "sha256:" + "2" * 64, task["task_id"]),
        )

    check(
        registered, "cluster", {"kind": "kubernetes", "namespace": "ns-b"},
        KUBERNETES_SETTINGS["storage"],
    )


def test_a_changed_storage_root_is_a_changed_target(registered):
    add_queued_run(registered, LOCAL_SETTINGS)

    with pytest.raises(NotConfigured) as raised:
        check(registered, "local", {"kind": "local"}, {"kind": "local", "root": "/data/other"})

    assert "/data" not in raised.value.message
