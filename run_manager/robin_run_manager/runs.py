"""The run reads: list runs, show one run, and list a run's recordings and tasks."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg

from robin_run_manager.database import Database, transaction
from robin_run_manager.errors import INVALID_REQUEST, NOT_FOUND, Refused
from robin_run_manager.pages import DEFAULT_LIMIT, Page, start_page
from robin_run_manager.status import get_run_status

RECORDING_STATES = ("skipped", "pending", "covered", "failed", "released", "withdrawn")
SKIP_REASONS = ("already_covered", "in_progress", "input_not_available", "input_has_no_windows")

# The run's columns, and its task and recording counts by state, as JSON objects.
_RUN_COLUMNS = """
SELECT r.run_id, r.state, r.submitted_by, r.created_at, r.preparation_failure,
    r.cancelled_at, r.cancelled_by, r.withdrawn_at, r.withdrawn_by,
    i.configuration_digest, i.output_versions, i.observed_provenance,
    (SELECT coalesce(json_object_agg(state, n), '{}') FROM (
        SELECT state, count(*) AS n FROM tasks t WHERE t.run_id = r.run_id GROUP BY state
    ) s) AS task_counts,
    (SELECT coalesce(json_object_agg(state, n), '{}') FROM (
        SELECT state, count(*) AS n FROM recording_outcomes o WHERE o.run_id = r.run_id
        GROUP BY state
    ) s) AS recording_counts"""
_SKIP_COUNTS = """,
    (SELECT coalesce(json_object_agg(skip_reason, n), '{}') FROM (
        SELECT skip_reason, count(*) AS n FROM recording_outcomes o
        WHERE o.run_id = r.run_id AND o.state = 'skipped' GROUP BY skip_reason
    ) s) AS skip_counts"""
_RUN_FROM = """
FROM runs r JOIN inference_runs i USING (run_id)
"""
_RUN_SELECT = _RUN_COLUMNS + _RUN_FROM
# One query, so that the skip counts and the other counts come from one snapshot.
_RUN_WITH_SKIPS_SELECT = _RUN_COLUMNS + _SKIP_COUNTS + _RUN_FROM


def list_runs(
    database: Database,
    *,
    submitted_by: str | None = None,
    created_after: str | None = None,
    limit: int = DEFAULT_LIMIT,
    cursor: str | None = None,
) -> Page:
    """A page of runs, newest first. `created_after` is an ISO 8601 time with a zone."""
    after_time = None
    if created_after is not None:
        try:
            after_time = _parse_aware_time(created_after)
        except ValueError:
            raise Refused(
                INVALID_REQUEST,
                f"created_after {created_after!r} is not an ISO 8601 time with a time zone. "
                "Give one such as 2026-10-22T09:00:00+00:00.",
                details={"created_after": created_after},
            ) from None
    filters = {"submitted_by": submitted_by, "created_after": created_after}
    paging = start_page("runs", cursor, limit, filters)
    after = paging.parse_after_key(_parse_aware_time, UUID) or (None, None)
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            f"{_RUN_SELECT} WHERE (%(submitted_by)s::text IS NULL "
            "OR r.submitted_by = %(submitted_by)s) "
            "AND (%(created_after)s::timestamptz IS NULL OR r.created_at > %(created_after)s) "
            "AND (%(after_time)s::timestamptz IS NULL "
            "OR (r.created_at, r.run_id) < (%(after_time)s, %(after_id)s::uuid)) "
            "ORDER BY r.created_at DESC, r.run_id DESC LIMIT %(limit)s",
            {
                "submitted_by": submitted_by,
                "created_after": after_time,
                "after_time": after[0],
                "after_id": after[1],
                "limit": paging.limit + 1,
            },
        ).fetchall()
    return paging.make_page(
        rows,
        key=lambda row: json.dumps([row["created_at"].isoformat(), str(row["run_id"])]),
        item=_summarize_run,
    )


def show_run(database: Database, run_id: str) -> dict:
    """One run: its summary, versions, provenance, counts, and failures."""
    with transaction(database, read_only=True) as connection:
        row = _get_run_row(connection, run_id, select=_RUN_WITH_SKIPS_SELECT)
    failure = row["preparation_failure"]
    return _summarize_run(row) | {
        "output_versions": row["output_versions"],
        "observed_provenance": row["observed_provenance"],
        "cancelled_at": _format_time(row["cancelled_at"]),
        "cancelled_by": row["cancelled_by"],
        "withdrawn_at": _format_time(row["withdrawn_at"]),
        "withdrawn_by": row["withdrawn_by"],
        "task_counts": row["task_counts"],
        "recording_counts": row["recording_counts"],
        "skip_counts": row["skip_counts"],
        "failures": {
            "preparation": (
                None
                if failure is None
                else {"reason": failure["reason"], "message": failure["message"]}
            ),
        },
    }


def list_run_recordings(
    database: Database,
    run_id: str,
    *,
    state: str | None = None,
    skip_reason: str | None = None,
    limit: int = DEFAULT_LIMIT,
    cursor: str | None = None,
) -> Page:
    """A page of the run's recordings. `state` is None until preparation reaches one."""
    _check_choice("state", state, RECORDING_STATES)
    _check_choice("skip_reason", skip_reason, SKIP_REASONS)
    filters = {"run_id": run_id, "state": state, "skip_reason": skip_reason}
    paging = start_page("run_recordings", cursor, limit, filters)
    after = paging.parse_after_key(_parse_storable_text, _parse_storable_text) or (None, None)
    with transaction(database, read_only=True) as connection:
        found = _get_run_row(connection, run_id)["run_id"]
        rows = connection.execute(
            "SELECT r.namespace, r.value, r.audio_uri, r.duration_seconds, o.state, "
            "o.skip_reason, o.task_id, o.input_file_id "
            "FROM run_recordings r LEFT JOIN recording_outcomes o USING (run_id, namespace, value) "
            "WHERE r.run_id = %(run_id)s "
            "AND (%(state)s::text IS NULL OR o.state = %(state)s) "
            "AND (%(skip_reason)s::text IS NULL OR o.skip_reason = %(skip_reason)s) "
            "AND (%(after_namespace)s::text IS NULL "
            "OR (r.namespace, r.value) > (%(after_namespace)s, %(after_value)s::text)) "
            "ORDER BY r.namespace, r.value LIMIT %(limit)s",
            {
                "run_id": found,
                "state": state,
                "skip_reason": skip_reason,
                "after_namespace": after[0],
                "after_value": after[1],
                "limit": paging.limit + 1,
            },
        ).fetchall()
    return paging.make_page(
        rows,
        key=lambda row: json.dumps([row["namespace"], row["value"]]),
        item=lambda row: {**row, "task_id": _format_text(row["task_id"])},
    )


def list_run_tasks(
    database: Database, run_id: str, *, limit: int = DEFAULT_LIMIT, cursor: str | None = None
) -> Page:
    """A page of the run's tasks, oldest first, each with its number of recordings."""
    paging = start_page("run_tasks", cursor, limit, {"run_id": run_id})
    after = paging.parse_after_key(_parse_aware_time, UUID) or (None, None)
    with transaction(database, read_only=True) as connection:
        found = _get_run_row(connection, run_id)["run_id"]
        rows = connection.execute(
            "SELECT t.task_id, t.state, t.retry_of, t.created_by, t.created_at, "
            "(SELECT count(*) FROM recording_outcomes o "
            "WHERE o.run_id = t.run_id AND o.task_id = t.task_id) "
            "AS recording_count "
            "FROM tasks t WHERE t.run_id = %(run_id)s "
            "AND (%(after_time)s::timestamptz IS NULL "
            "OR (t.created_at, t.task_id) > (%(after_time)s, %(after_id)s::uuid)) "
            "ORDER BY t.created_at, t.task_id LIMIT %(limit)s",
            {
                "run_id": found,
                "after_time": after[0],
                "after_id": after[1],
                "limit": paging.limit + 1,
            },
        ).fetchall()
    return paging.make_page(
        rows,
        key=lambda row: json.dumps([row["created_at"].isoformat(), str(row["task_id"])]),
        item=lambda row: {
            "task_id": str(row["task_id"]),
            "state": row["state"],
            "recording_count": row["recording_count"],
            "retry_of": _format_text(row["retry_of"]),
            "created_by": row["created_by"],
            "created_at": row["created_at"].isoformat(),
        },
    )


def _get_run_row(connection: psycopg.Connection, run_id: str, *, select: str = _RUN_SELECT) -> dict:
    try:
        parsed = UUID(run_id)
    except ValueError:
        parsed = None
    row = None
    if parsed is not None:
        row = connection.execute(f"{select} WHERE r.run_id = %s", (parsed,)).fetchone()
    if row is None:
        raise Refused(NOT_FOUND, f"There is no run {run_id}.", details={"run_id": run_id})
    return row


def _summarize_run(row: dict) -> dict:
    status = get_run_status(
        state=row["state"],
        cancelled=row["cancelled_at"] is not None,
        withdrawn=row["withdrawn_at"] is not None,
        task_counts=row["task_counts"],
        recording_counts=row["recording_counts"],
    )
    return {
        "run_id": str(row["run_id"]),
        "status": status,
        "submitted_by": row["submitted_by"],
        "created_at": row["created_at"].isoformat(),
        "configuration_digest": row["configuration_digest"],
        "recording_count": row["observed_provenance"]["recording_count"],
    }


def _check_choice(name: str, value: str | None, choices: tuple[str, ...]) -> None:
    if value is not None and value not in choices:
        raise Refused(
            INVALID_REQUEST,
            f"{name} {value!r} is not valid. Use one of: {', '.join(choices)}.",
            details={name: value, "valid": list(choices)},
        )


def _parse_aware_time(text: str) -> datetime:
    """An ISO 8601 time. Raises `ValueError` if it has no time zone."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"{text!r} has no time zone")
    return parsed


def _parse_storable_text(text: str) -> str:
    """`text`, if PostgreSQL can store it. Raises `ValueError` for NUL or a lone surrogate."""
    text.encode("utf-8")
    if "\x00" in text:
        raise ValueError("text holds a NUL character")
    return text


def _format_time(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _format_text(value: Any) -> str | None:
    return None if value is None else str(value)
