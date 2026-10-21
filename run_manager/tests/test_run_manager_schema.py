"""The real schema that this release's migrations make."""

from robin_run_manager.database import transaction
from robin_run_manager.migrations import migrate, require_current_schema


def table_names(database):
    with transaction(database, read_only=True) as connection:
        found = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {row["table_name"] for row in found}


def test_the_migrations_make_the_model_and_preset_tables(database):
    migrate(database)

    assert {"models", "configurations", "presets"} <= table_names(database)


def test_the_schema_is_current_after_migrate(database):
    migrate(database)

    require_current_schema(database)
