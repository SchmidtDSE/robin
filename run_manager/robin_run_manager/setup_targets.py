"""The target check: one setup name has one target while its runs have work.

A setup's target is its compute kind and the fields that name where its tasks run,
such as a namespace, and the storage section. Setup names are shared by every profile
that uses one database.
"""

from collections.abc import Mapping
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from robin_run_manager.database import SETUP_LOCK
from robin_run_manager.errors import NotConfigured
from robin_run_manager.launch import MATCHES_SETUP_TARGET

# A run `r` that still has work: it waits to be prepared, or it has a task that is
# queued, or a reserved task that is not released. A reserved task is not released
# while it is submitting, submitted, running, or finalizing, nor after it ends until a
# pass releases it.
_RUN_HAS_WORK = (
    "((r.state = 'preparing' AND r.cancelled_at IS NULL AND r.withdrawn_at IS NULL) "
    "OR EXISTS (SELECT 1 FROM tasks t WHERE t.run_id = r.run_id AND (t.state = 'queued' "
    "OR (t.reserved_at IS NOT NULL AND t.released_at IS NULL))))"
)


def lock_setup_name(connection: psycopg.Connection, setup: str) -> None:
    """Take the setup's lock until the transaction ends.

    A submit holds it from its target check until its run is committed, so a second
    submit of the same setup name waits, and then sees the first run.
    """
    connection.execute(
        "SELECT pg_advisory_xact_lock(%s, hashtext(%s::text))", (SETUP_LOCK, setup)
    )


def refuse_conflicting_setup_target(
    connection: psycopg.Connection,
    setup: str,
    target_compute: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> None:
    """Raise `NotConfigured` if a run of the setup name, of any user, still has work and
    was frozen with another target. A run whose work is all done does not count."""
    row = connection.execute(
        f"SELECT 1 FROM runs r WHERE (r.execution_settings ->> 'setup') = %(setup)s "
        f"AND NOT ({MATCHES_SETUP_TARGET}) AND {_RUN_HAS_WORK} LIMIT 1",
        {
            "setup": setup,
            "target_compute": Jsonb(target_compute),
            "storage": Jsonb(storage),
        },
    ).fetchone()
    if row is not None:
        raise NotConfigured(
            f"Setup {setup} has runs with work that were submitted with other compute or "
            "storage settings. Give the new settings a new setup name, or let the earlier "
            f"runs of {setup} finish first. Setup names are shared by every profile that "
            "uses this database.",
            details={"setup": setup},
        )
