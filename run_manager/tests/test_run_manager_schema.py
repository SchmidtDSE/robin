"""The real schema that this release's migrations make."""

import pytest
from psycopg import errors
from psycopg.types.json import Json, Jsonb

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


RUN_TABLES = {
    "runs", "inference_runs", "run_recordings", "tasks", "output_files", "recording_outcomes",
}
VERSIONS = {"model:owl/v4": 1}
EXECUTION_SETTINGS = {
    "compute": {"kind": "local"},
    "storage": {"kind": "local", "root": "/data/robin"},
}


def insert_run(connection, versions=VERSIONS) -> str:
    """A run of one stored configuration, with recording ("ns", "a") in its list."""
    connection.execute(
        "INSERT INTO models (model_id, card, card_digest, files, registered_by) "
        "VALUES ('owl/v4', '{}', 'card', '{}', 'ana') ON CONFLICT DO NOTHING"
    )
    connection.execute(
        "INSERT INTO configurations (digest, model_id, configuration) "
        "VALUES ('config', 'owl/v4', '{}') ON CONFLICT DO NOTHING"
    )
    run_id = connection.execute(
        "INSERT INTO runs (kind, submitted_by, execution_settings) "
        "VALUES ('inference', 'ana', %s) RETURNING run_id",
        (Json(EXECUTION_SETTINGS),),
    ).fetchone()["run_id"]
    connection.execute(
        "INSERT INTO inference_runs (run_id, configuration_digest, output_versions, "
        "provenance) VALUES (%s, 'config', %s, '{}')",
        (run_id, Jsonb(versions)),
    )
    connection.execute(
        "INSERT INTO run_recordings (run_id, namespace, value, audio_uri) "
        "VALUES (%s, 'ns', 'a', 'file:///a.wav')",
        (run_id,),
    )
    return run_id


def insert_outcome(connection, run_id, state, versions=VERSIONS, skip_reason=None) -> None:
    connection.execute(
        "INSERT INTO recording_outcomes (run_id, namespace, value, configuration_digest, "
        "output_versions, state, skip_reason) VALUES (%s, 'ns', 'a', 'config', %s, %s, %s)",
        (run_id, Jsonb(versions), state, skip_reason),
    )


def test_the_migrations_make_the_run_tables(database):
    migrate(database)

    assert RUN_TABLES <= table_names(database)


def test_a_second_claim_at_the_same_versions_is_refused(migrated_database):
    with pytest.raises(errors.UniqueViolation):
        with transaction(migrated_database) as connection:
            insert_outcome(connection, insert_run(connection), "pending")
            insert_outcome(connection, insert_run(connection), "pending")


def test_a_claim_at_other_versions_is_allowed(migrated_database):
    with transaction(migrated_database) as connection:
        insert_outcome(connection, insert_run(connection), "pending")
        other = {"model:owl/v4": 2}
        insert_outcome(connection, insert_run(connection, other), "pending", other)


def test_skipped_and_failed_rows_do_not_hold_a_claim(migrated_database):
    with transaction(migrated_database) as connection:
        insert_outcome(connection, insert_run(connection), "pending")
        insert_outcome(connection, insert_run(connection), "skipped", skip_reason="in_progress")
        insert_outcome(connection, insert_run(connection), "failed")


def test_an_outcome_must_copy_its_runs_configuration_and_versions(migrated_database):
    with pytest.raises(errors.ForeignKeyViolation):
        with transaction(migrated_database) as connection:
            insert_outcome(connection, insert_run(connection), "pending", {"model:owl/v4": 2})


def test_a_skipped_row_needs_a_skip_reason(migrated_database):
    with pytest.raises(errors.CheckViolation):
        with transaction(migrated_database) as connection:
            insert_outcome(connection, insert_run(connection), "skipped")


@pytest.mark.parametrize(
    "state, failure",
    [("preparation_failed", None), ("ready", '{"reason": "x", "message": "y", "details": {}}')],
)
def test_a_preparation_failure_is_set_exactly_when_preparation_failed(
    migrated_database, state, failure
):
    with pytest.raises(errors.CheckViolation):
        with transaction(migrated_database) as connection:
            connection.execute(
                "INSERT INTO runs (kind, submitted_by, state, preparation_failure, "
                "execution_settings) VALUES ('inference', 'ana', %s, %s, %s)",
                (state, failure, Json(EXECUTION_SETTINGS)),
            )


def test_outcomes_can_be_found_by_task(migrated_database):
    with transaction(migrated_database, read_only=True) as connection:
        indexes = connection.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'recording_outcomes'"
        ).fetchall()

    assert any(row["indexdef"].endswith("(task_id)") for row in indexes)


def test_a_run_without_execution_settings_is_refused(migrated_database):
    with pytest.raises(errors.NotNullViolation):
        with transaction(migrated_database) as connection:
            connection.execute(
                "INSERT INTO runs (kind, submitted_by) VALUES ('inference', 'ana')"
            )


def test_a_skip_for_an_unsupported_input_location_is_allowed(migrated_database):
    with transaction(migrated_database) as connection:
        insert_outcome(
            connection, insert_run(connection), "skipped", skip_reason="input_location_unsupported"
        )
