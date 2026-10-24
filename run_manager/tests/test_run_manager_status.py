"""The run status: one function, with the rules in priority order."""

import pytest

from robin_run_manager.status import get_run_status

READY = {"state": "ready", "cancelled": False, "withdrawn": False}


@pytest.mark.parametrize(
    "run, tasks, recordings, status",
    [
        # One case for each row.
        ({"withdrawn": True}, {}, {}, "withdrawn"),
        ({"cancelled": True}, {}, {}, "cancelled"),
        ({"state": "preparing"}, {}, {}, "preparing"),
        ({"state": "preparation_failed"}, {}, {}, "preparation_failed"),
        ({}, {"running": 1}, {"pending": 5}, "running"),
        ({}, {"queued": 2}, {"pending": 5}, "queued"),
        ({}, {"succeeded": 1, "failed": 1}, {"covered": 3, "failed": 2}, "finished_with_failures"),
        ({}, {"succeeded": 2}, {"covered": 5}, "finished"),
        # Each row comes before the rows after it.
        ({"withdrawn": True, "cancelled": True}, {"running": 1}, {}, "withdrawn"),
        ({"cancelled": True, "state": "preparing"}, {"running": 1}, {}, "cancelled"),
        ({"state": "preparing"}, {"running": 1}, {}, "preparing"),
        ({"state": "preparation_failed"}, {"queued": 1}, {}, "preparation_failed"),
        ({}, {"submitting": 1, "queued": 1}, {"failed": 1}, "running"),
        ({}, {"submitted": 1}, {}, "running"),
        ({}, {"finalizing": 1}, {}, "running"),
        ({}, {"queued": 1}, {"failed": 1}, "queued"),
        ({}, {"failed": 1}, {"failed": 1, "skipped": 4}, "finished_with_failures"),
        ({}, {"cancelled": 1, "running": 0}, {"released": 2}, "finished"),
    ],
)
def test_each_row_of_the_status_table(run, tasks, recordings, status):
    assert get_run_status(**(READY | run), task_counts=tasks, recording_counts=recordings) == status


def test_a_ready_run_with_no_tasks_is_finished():
    assert get_run_status(**READY, task_counts={}, recording_counts={"skipped": 3}) == "finished"


def test_a_retried_failure_that_succeeded_is_finished():
    status = get_run_status(
        **READY, task_counts={"failed": 1, "succeeded": 1}, recording_counts={"covered": 4}
    )

    assert status == "finished"
