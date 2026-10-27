"""Numbered SQL migrations: apply once, refuse edits, and check the schema."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import psycopg
import pytest

from robin_run_manager import migrations
from robin_run_manager.database import transaction
from robin_run_manager.errors import Refused
from robin_run_manager.migrations import load_migrations, migrate, require_current_schema


def folder_with(path, files):
    path.mkdir(parents=True)
    for name, text in files.items():
        (path / name).write_text(text)
    return load_migrations(path)


def table_names(database):
    with transaction(database) as connection:
        found = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {row["table_name"] for row in found}


def applied_versions(database):
    with transaction(database) as connection:
        found = connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row["version"] for row in found.fetchall()]


@pytest.fixture(autouse=True)
def one_packaged_migration(monkeypatch, tmp_path):
    """Use one test migration as this release's migrations.

    These tests check the migration rules, not the real schema.
    """
    packaged = folder_with(
        tmp_path / "packaged", {"0001_first.sql": "CREATE TABLE first (x integer);"}
    )
    monkeypatch.setattr(migrations, "packaged_migrations", lambda: packaged)


def test_the_schema_folder_holds_the_release_migrations():
    # Read directly: the fixture above replaces packaged_migrations.
    found = load_migrations(files("robin_run_manager") / "schema")

    assert [migration.name for migration in found] == [
        "0001_models_and_presets", "0002_runs", "0003_launch_and_outcomes",
    ]


def test_migrate_applies_every_migration_to_an_empty_database(database):
    applied = migrate(database)

    assert applied == migrations.packaged_migrations()
    assert {"schema_migrations", "first"} <= table_names(database)


def test_a_second_migrate_changes_nothing(database):
    migrate(database)

    assert migrate(database) == ()
    assert applied_versions(database) == [m.version for m in migrations.packaged_migrations()]


def test_new_migrations_apply_after_the_applied_ones(database, tmp_path):
    migrate(database, folder_with(tmp_path / "a", {"0001_a.sql": "CREATE TABLE a (x int);"}))

    later = folder_with(tmp_path / "b", {
        "0001_a.sql": "CREATE TABLE a (x int);",
        "0002_b.sql": "CREATE TABLE b (x int); CREATE TABLE c (x int);",
    })

    assert [m.version for m in migrate(database, later)] == [2]
    assert {"a", "b", "c"} <= table_names(database)


def test_an_edited_applied_migration_is_refused(database, tmp_path):
    migrate(database, folder_with(tmp_path / "a", {"0001_t.sql": "CREATE TABLE t (x int);"}))
    edited = folder_with(tmp_path / "b", {"0001_t.sql": "CREATE TABLE t (x bigint);"})

    with pytest.raises(Refused) as raised:
        migrate(database, edited)

    assert raised.value.reason == "migration_changed"


def test_a_database_with_a_migration_this_release_lacks_is_refused(database, tmp_path):
    migrate(database, folder_with(tmp_path / "a", {
        "0001_a.sql": "CREATE TABLE a (x int);",
        "0002_b.sql": "CREATE TABLE b (x int);",
    }))
    older = folder_with(tmp_path / "b", {"0001_a.sql": "CREATE TABLE a (x int);"})

    with pytest.raises(Refused) as raised:
        migrate(database, older)

    assert raised.value.reason == "migration_unknown"


def test_a_failing_migration_applies_nothing(database, tmp_path):
    broken = folder_with(tmp_path / "a", {
        "0001_a.sql": "CREATE TABLE a (x int);",
        "0002_b.sql": "CREATE TABLE b (x no_such_type);",
    })

    with pytest.raises(psycopg.Error):
        migrate(database, broken)

    assert "a" not in table_names(database)


@pytest.mark.parametrize(
    "files",
    [{"0002_a.sql": ""}, {"1_a.sql": ""}, {"0001_a.sql": "", "0003_c.sql": ""}],
)
def test_migration_files_must_be_named_and_numbered_without_gaps(tmp_path, files):
    with pytest.raises(ValueError):
        folder_with(tmp_path / "a", files)


def test_two_migrations_at_the_same_time_apply_each_file_once(database):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: migrate(database), range(2)))

    assert sorted(len(applied) for applied in results) == [0, len(migrations.packaged_migrations())]


def test_a_database_without_the_migrations_is_not_current(database):
    with pytest.raises(Refused) as raised:
        require_current_schema(database)

    assert raised.value.reason == "schema_not_current"
    assert "robin db migrate" in raised.value.message


def test_a_migrated_database_is_current(migrated_database):
    require_current_schema(migrated_database)


def test_an_edited_migration_with_the_same_count_is_not_current(database, tmp_path):
    edited = folder_with(tmp_path / "a", {
        f"{m.name}.sql": m.sql + "\n-- edited\n" for m in migrations.packaged_migrations()
    })
    migrate(database, edited)

    with pytest.raises(Refused) as raised:
        require_current_schema(database)

    assert raised.value.reason == "schema_not_current"
    assert "robin db migrate" not in raised.value.message


def test_other_migration_names_with_the_same_count_are_not_current(database, tmp_path):
    renamed = folder_with(tmp_path / "a", {
        f"{m.version:04d}_other_{m.version}.sql": m.sql for m in migrations.packaged_migrations()
    })
    migrate(database, renamed)

    with pytest.raises(Refused) as raised:
        require_current_schema(database)

    assert raised.value.reason == "schema_not_current"


def test_a_database_user_with_its_own_schema_is_current_after_migrate(database):
    # A schema with the user's name comes first in the default search_path, so the
    # migrations create their tables there and not in `public`.
    with transaction(database) as connection:
        connection.execute("CREATE SCHEMA robin AUTHORIZATION robin")

    migrate(database)

    require_current_schema(database)
