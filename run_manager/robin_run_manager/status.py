"""The status of a run, computed from its rows. No column stores it."""

from collections.abc import Mapping

# Task states in which a task has started and has not ended.
ACTIVE_TASK_STATES = ("submitting", "submitted", "running", "finalizing")


def get_run_status(
    *,
    state: str,
    cancelled: bool,
    withdrawn: bool,
    task_counts: Mapping[str, int],
    recording_counts: Mapping[str, int],
) -> str:
    """The status that people see. The first rule that applies gives it.

    `finished` means that the run's own tasks ended, not that each recording has a result.
    """
    if withdrawn:
        return "withdrawn"
    if cancelled:
        return "cancelled"
    if state in ("preparing", "preparation_failed"):
        return state
    if any(task_counts.get(active, 0) > 0 for active in ACTIVE_TASK_STATES):
        return "running"
    if task_counts.get("queued", 0) > 0:
        return "queued"
    if recording_counts.get("failed", 0) > 0:
        return "finished_with_failures"
    return "finished"
