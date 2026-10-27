"""Preparation: claim a run's recordings, record which ones are skipped and why, and make tasks.

One transaction does all of it. If it stops before commit, the run stays `preparing`.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Json, Jsonb

from robin_contracts.tasks import ModelOutputVersion, output_version_id
from robin_run_manager.configuration import BackboneInput, Configuration
from robin_run_manager.database import Database, transaction
from robin_run_manager.errors import OUTPUT_VERSION_NOT_CURRENT
from robin_run_manager.location_forms import get_unsupported_locations
from robin_run_manager.stored_configurations import get_stored_configuration
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS, get_outdated_versions

# How many times one call tries again after it finds a released claim.
_ATTEMPTS = 3

# The skip reason for a recording that another run holds, by the state of that run's claim.
_SKIP_REASONS = {"covered": "already_covered", "pending": "in_progress"}


@dataclass(frozen=True)
class Prepared:
    """The result of `prepare_run`. `not_preparing` and `try_again` change nothing.

    `try_again` means that another run released a claim during each attempt.
    """

    state: Literal["ready", "preparation_failed", "not_preparing", "try_again"]
    tasks: int = 0
    claimed: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class _Run:
    run_id: UUID
    submitted_by: str
    configuration_digest: str
    output_versions: dict[str, int]
    configuration: Configuration
    # The kind frozen at submit, so that the result does not depend on who prepares.
    compute_kind: str


def get_runs_to_prepare(database: Database) -> list[UUID]:
    """The runs that preparation must still do, oldest first.

    Cancelled and withdrawn runs are left out: they need no tasks.
    """
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT run_id FROM runs WHERE state = 'preparing' AND cancelled_at IS NULL "
            "AND withdrawn_at IS NULL ORDER BY created_at, run_id"
        ).fetchall()
    return [row["run_id"] for row in rows]


def prepare_run(
    database: Database,
    run_id: UUID,
    *,
    recordings_per_task: int,
    release_versions: Sequence[ModelOutputVersion] = RELEASE_OUTPUT_VERSIONS,
) -> Prepared:
    """Claim the run's free recordings, skip the others, make tasks, and mark it `ready`.

    If the run's output versions are not current in `release_versions`, the run becomes
    `preparation_failed`. Raises `Unavailable` if the database fails.
    """
    for _ in range(_ATTEMPTS):
        with transaction(database) as connection:
            prepared = _prepare_once(connection, run_id, recordings_per_task, release_versions)
            if prepared is None:
                # The rollback gives back this attempt's claims, so that the next
                # attempt holds no claim while it waits on another transaction.
                connection.rollback()
                continue
        return prepared
    return Prepared(state="try_again")


def _prepare_once(
    connection: psycopg.Connection,
    run_id: UUID,
    recordings_per_task: int,
    release_versions: Sequence[ModelOutputVersion],
) -> Prepared | None:
    """One attempt. None means that it must be rolled back and started again."""
    run = _lock_preparing_run(connection, run_id)
    if run is None:
        return Prepared(state="not_preparing")
    outdated = get_outdated_versions(run.output_versions, release_versions)
    if outdated:
        _fail_preparation(connection, run, outdated)
        return Prepared(state="preparation_failed")
    embeddings = {}
    if isinstance(run.configuration.input, BackboneInput):
        # The claim takes only the recordings that have no outcome yet, so a recording
        # that is skipped here is not claimed.
        embeddings = _skip_recordings_without_usable_embeddings(
            connection, run, run.configuration.input.backbone_configuration
        )
    _claim_recordings(connection, run)
    unclaimed = _get_unclaimed_recordings(connection, run)
    if any(row["holder"] is None for row in unclaimed):
        # A claim was released after this attempt claimed. Claiming it now could deadlock.
        return None
    _insert_skipped(
        connection,
        run,
        [(row["namespace"], row["value"], _SKIP_REASONS[row["holder"]]) for row in unclaimed],
    )
    _store_pinned_inputs(connection, run, embeddings)
    counts = connection.execute(
        "SELECT count(*) FILTER (WHERE state = 'pending') AS claimed, "
        "count(*) FILTER (WHERE state = 'skipped') AS skipped "
        "FROM recording_outcomes WHERE run_id = %s",
        (run.run_id,),
    ).fetchone()
    tasks = _create_tasks(connection, run, counts["claimed"], recordings_per_task)
    connection.execute("UPDATE runs SET state = 'ready' WHERE run_id = %s", (run.run_id,))
    return Prepared(
        state="ready", tasks=tasks, claimed=counts["claimed"], skipped=counts["skipped"]
    )


def _lock_preparing_run(connection: psycopg.Connection, run_id: UUID) -> _Run | None:
    row = connection.execute(
        "SELECT r.run_id, r.state, r.cancelled_at, r.withdrawn_at, r.submitted_by, "
        "r.execution_settings, i.configuration_digest, i.output_versions, c.configuration "
        "FROM runs r JOIN inference_runs i USING (run_id) "
        "JOIN configurations c ON c.digest = i.configuration_digest "
        "WHERE r.run_id = %s FOR UPDATE OF r",
        (run_id,),
    ).fetchone()
    if row is None or row["state"] != "preparing":
        return None
    if row["cancelled_at"] is not None or row["withdrawn_at"] is not None:
        return None
    return _Run(
        run_id=row["run_id"],
        submitted_by=row["submitted_by"],
        configuration_digest=row["configuration_digest"],
        output_versions=row["output_versions"],
        configuration=Configuration.model_validate(row["configuration"]),
        compute_kind=row["execution_settings"]["compute"]["kind"],
    )


def _fail_preparation(
    connection: psycopg.Connection, run: _Run, outdated: dict[str, dict]
) -> None:
    failure = {
        "reason": OUTPUT_VERSION_NOT_CURRENT,
        "message": "The run's output versions are not the versions of the installed "
        "release, so its work would give outdated results. Submit a new run.",
        "details": {"output_versions": outdated},
    }
    connection.execute(
        "UPDATE runs SET state = 'preparation_failed', preparation_failure = %s "
        "WHERE run_id = %s",
        (Json(failure), run.run_id),
    )


def _skip_recordings_without_usable_embeddings(
    connection: psycopg.Connection, run: _Run, backbone_digest: str
) -> dict[tuple[str, str], int]:
    """Skip each recording whose backbone embeddings file is missing or cannot be read.

    A file cannot be read if the run's compute kind does not accept its location form.
    Returns the embeddings file ID of each other recording, by `(namespace, value)`.
    """
    backbone = get_stored_configuration(connection, backbone_digest)
    backbone_id = output_version_id(backbone.model.card)
    rows = connection.execute(
        "SELECT r.namespace, r.value, c.run_id IS NOT NULL AS covered, f.file_id, f.uri "
        "FROM run_recordings r "
        "LEFT JOIN (recording_outcomes c JOIN runs cr "
        "ON cr.run_id = c.run_id AND cr.withdrawn_at IS NULL) "
        "ON c.namespace = r.namespace AND c.value = r.value AND c.state = 'covered' "
        "AND c.configuration_digest = %(backbone)s AND c.output_versions = %(versions)s "
        "LEFT JOIN output_files f ON f.run_id = c.run_id AND f.namespace = r.namespace "
        "AND f.value = r.value AND f.kind = 'embeddings' "
        "WHERE r.run_id = %(run_id)s",
        {
            "run_id": run.run_id,
            "backbone": backbone_digest,
            "versions": Jsonb({backbone_id: run.output_versions[backbone_id]}),
        },
    ).fetchall()
    skipped, pinned = [], {}
    for row in rows:
        if not row["covered"]:
            skipped.append((row["namespace"], row["value"], "input_not_available"))
        elif row["file_id"] is None:
            skipped.append((row["namespace"], row["value"], "input_has_no_windows"))
        elif get_unsupported_locations(run.compute_kind, [("embeddings", row["uri"])]):
            skipped.append((row["namespace"], row["value"], "input_location_unsupported"))
        else:
            pinned[(row["namespace"], row["value"])] = row["file_id"]
    _insert_skipped(connection, run, skipped)
    return pinned


def _store_pinned_inputs(
    connection: psycopg.Connection, run: _Run, pinned: dict[tuple[str, str], int]
) -> None:
    """Store the pinned file ID on each recording that the run claimed."""
    if not pinned:
        return
    namespaces, values = (list(column) for column in zip(*pinned))
    connection.execute(
        "UPDATE recording_outcomes o SET input_file_id = p.file_id "
        "FROM unnest(%(namespaces)s::text[], %(values)s::text[], %(files)s::bigint[]) "
        "AS p(namespace, value, file_id) "
        "WHERE o.run_id = %(run_id)s AND o.state = 'pending' "
        "AND o.namespace = p.namespace AND o.value = p.value",
        {
            "run_id": run.run_id,
            "namespaces": namespaces,
            "values": values,
            "files": list(pinned.values()),
        },
    )


def _claim_recordings(connection: psycopg.Connection, run: _Run) -> None:
    """Claim each recording that no other run holds.

    This is the only statement that waits on other runs. It takes the keys in sorted
    order, so two preparations cannot deadlock.
    """
    connection.execute(
        "INSERT INTO recording_outcomes (run_id, namespace, value, configuration_digest, "
        "output_versions, state) "
        "SELECT r.run_id, r.namespace, r.value, %(digest)s, %(versions)s, 'pending' "
        "FROM run_recordings r WHERE r.run_id = %(run_id)s AND NOT EXISTS ("
        "SELECT 1 FROM recording_outcomes o WHERE o.run_id = r.run_id "
        "AND o.namespace = r.namespace AND o.value = r.value) "
        "ORDER BY r.namespace, r.value "
        "ON CONFLICT (namespace, value, configuration_digest, output_versions) "
        "WHERE state IN ('pending', 'covered') DO NOTHING",
        {
            "run_id": run.run_id,
            "digest": run.configuration_digest,
            "versions": Jsonb(run.output_versions),
        },
    )


def _get_unclaimed_recordings(connection: psycopg.Connection, run: _Run) -> list[dict]:
    """Each recording that the claim did not get, and that has no outcome in the run yet.

    `holder` is the state of the other run's claim, or None if no run holds it now.
    """
    return connection.execute(
        "SELECT r.namespace, r.value, h.state AS holder "
        "FROM run_recordings r "
        "LEFT JOIN recording_outcomes h ON h.namespace = r.namespace AND h.value = r.value "
        "AND h.configuration_digest = %(digest)s AND h.output_versions = %(versions)s "
        "AND h.state IN ('pending', 'covered') AND h.run_id <> r.run_id "
        "WHERE r.run_id = %(run_id)s AND NOT EXISTS ("
        "SELECT 1 FROM recording_outcomes o WHERE o.run_id = r.run_id "
        "AND o.namespace = r.namespace AND o.value = r.value)",
        {
            "run_id": run.run_id,
            "digest": run.configuration_digest,
            "versions": Jsonb(run.output_versions),
        },
    ).fetchall()


def _insert_skipped(
    connection: psycopg.Connection, run: _Run, skipped: list[tuple[str, str, str]]
) -> None:
    """Store each `(namespace, value, skip_reason)` as a skipped outcome. It holds no claim."""
    if not skipped:
        return
    namespaces, values, reasons = (list(column) for column in zip(*skipped))
    connection.execute(
        "INSERT INTO recording_outcomes (run_id, namespace, value, configuration_digest, "
        "output_versions, state, skip_reason) "
        "SELECT %(run_id)s, s.namespace, s.value, %(digest)s, %(versions)s, 'skipped', s.reason "
        "FROM unnest(%(namespaces)s::text[], %(values)s::text[], %(reasons)s::text[]) "
        "AS s(namespace, value, reason)",
        {
            "run_id": run.run_id,
            "digest": run.configuration_digest,
            "versions": Jsonb(run.output_versions),
            "namespaces": namespaces,
            "values": values,
            "reasons": reasons,
        },
    )


def _create_tasks(
    connection: psycopg.Connection, run: _Run, claimed: int, recordings_per_task: int
) -> int:
    """Cut the `claimed` recordings into queued tasks, and return the number of tasks."""
    count = -(-claimed // recordings_per_task)
    if count == 0:
        return 0
    # All tasks of one transaction have the same created_at, and tasks are listed by
    # (created_at, task_id). Sorted IDs list them in the order of their recordings.
    task_ids = sorted(uuid.uuid4() for _ in range(count))
    connection.execute(
        "INSERT INTO tasks (task_id, run_id, state, created_by) "
        "SELECT task_id, %s, 'queued', %s FROM unnest(%s::uuid[]) AS t(task_id)",
        (run.run_id, run.submitted_by, task_ids),
    )
    connection.execute(
        "UPDATE recording_outcomes o SET task_id = (%(task_ids)s::uuid[])[n.position + 1] "
        "FROM (SELECT namespace, value, "
        "(row_number() OVER (ORDER BY namespace, value) - 1) / %(size)s AS position "
        "FROM recording_outcomes WHERE run_id = %(run_id)s AND state = 'pending') n "
        "WHERE o.run_id = %(run_id)s AND o.namespace = n.namespace AND o.value = n.value",
        {"task_ids": task_ids, "size": recordings_per_task, "run_id": run.run_id},
    )
    return count
