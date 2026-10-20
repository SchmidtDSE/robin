"""Schema migrations: numbered SQL files, applied in order, each recorded once.

A file is `NNNN_name.sql`. The numbers start at 1 and have no gap. An applied file must
never change; a change to the schema is a new file.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

from robin_run_manager.database import MIGRATION_LOCK, Database, transaction
from robin_run_manager.errors import (
    MIGRATION_CHANGED,
    MIGRATION_UNKNOWN,
    SCHEMA_NOT_CURRENT,
    Refused,
)

_FILE_NAME = re.compile(r"(?P<version>\d{4})_[a-z0-9_]+\.sql")
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version integer PRIMARY KEY,
    name text NOT NULL,
    sha256 text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    sha256: str


def load_migrations(folder: Traversable) -> tuple[Migration, ...]:
    """Read the `.sql` files in `folder`, in order. Raise `ValueError` for a bad name or a gap."""
    found = []
    for entry in folder.iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = _FILE_NAME.fullmatch(entry.name)
        if match is None:
            raise ValueError(f"migration file {entry.name!r} is not named NNNN_name.sql")
        data = entry.read_bytes()
        found.append(
            Migration(
                version=int(match["version"]),
                name=entry.name.removesuffix(".sql"),
                sql=data.decode("utf-8"),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    found.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in found]
    if versions != list(range(1, len(found) + 1)):
        raise ValueError(f"migration numbers must be 1, 2, 3, ... with no gap; found {versions}")
    return tuple(found)


def packaged_migrations() -> tuple[Migration, ...]:
    return load_migrations(files("robin_run_manager") / "schema")


def migrate(
    database: Database, migrations: Sequence[Migration] | None = None
) -> tuple[Migration, ...]:
    """Apply the migrations that the database does not have, in one transaction.

    Return the migrations applied; a second call returns an empty tuple. An applied
    migration that changed, or one that this release does not have, is refused, and
    nothing is applied. A lock makes a second `migrate` wait for the first.
    """
    migrations = tuple(packaged_migrations() if migrations is None else migrations)
    with transaction(database) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK,))
        connection.execute(_CREATE_TABLE)
        applied = connection.execute(
            "SELECT version, name, sha256 FROM schema_migrations ORDER BY version"
        ).fetchall()
        _check_applied(applied, migrations)
        pending = migrations[len(applied):]
        for migration in pending:
            # With no parameters, psycopg sends the text as is, so one file can hold
            # several statements.
            connection.execute(migration.sql)
            connection.execute(
                "INSERT INTO schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.sha256),
            )
    return pending


def require_current_schema(database: Database) -> None:
    """Refuse with `schema_not_current` unless the database has exactly this release's migrations.

    The versions, the names, and the checksums must all be equal.
    """
    with transaction(database, read_only=True) as connection:
        table = connection.execute(
            # Unqualified, so that it finds the table where `migrate` made it: in the
            # first schema on the search_path.
            "SELECT to_regclass('schema_migrations') AS name"
        ).fetchone()
        applied = []
        if table["name"] is not None:
            applied = connection.execute(
                "SELECT version, name, sha256 FROM schema_migrations ORDER BY version"
            ).fetchall()
    found = [(row["version"], row["name"], row["sha256"]) for row in applied]
    expected = [(m.version, m.name, m.sha256) for m in packaged_migrations()]
    if found == expected:
        return
    if found == expected[: len(found)]:
        message = (
            f"The database has {len(found)} of this release's {len(expected)} migrations. "
            "Run `robin db migrate`."
        )
    else:
        message = (
            "The database's migrations are not this release's migrations. Use the release "
            "that migrated the database."
        )
    raise Refused(
        SCHEMA_NOT_CURRENT,
        message,
        details={"applied": len(found), "expected": len(expected)},
    )


def _check_applied(applied: list[dict], migrations: tuple[Migration, ...]) -> None:
    for row in applied:
        if row["version"] > len(migrations):
            raise Refused(
                MIGRATION_UNKNOWN,
                f"The database has migration {row['name']}, which this release does not "
                "have. Use the release that migrated the database.",
                details={"version": row["version"], "name": row["name"]},
            )
        expected = migrations[row["version"] - 1]
        if (row["name"], row["sha256"]) != (expected.name, expected.sha256):
            raise Refused(
                MIGRATION_CHANGED,
                f"Migration {expected.name} is not the file that was applied as "
                f"{row['name']}. An applied migration must not change; add a new one.",
                details={"version": row["version"]},
            )
