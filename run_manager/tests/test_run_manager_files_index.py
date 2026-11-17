import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from robin_adapters.storage.local.adapter import LocalTaskStorage
from robin_contracts.output_contracts import CURRENT_FILES_CONTRACT_ID
from robin_run_manager.database import lock_index_export, open_database, transaction
from robin_run_manager.errors import (
    EXPORT_IN_PROGRESS,
    OutcomeUncertain,
    Refused,
    StorageUnavailable,
)
from robin_run_manager.files_index import build_index, export_index, get_index_rows
from run_fixtures import (
    add_covered_result,
    get_current_files,
    register_fixture_models,
    submit,
    withdraw,
)

EXPORTED_AT = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
COLUMNS = [
    "namespace", "value", "model_id", "presets", "configuration_digest", "kind",
    "contract_id", "uri", "checksum", "size_bytes", "rows", "output_versions",
    "up_to_date", "run_id",
]


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


class CreateFails(LocalTaskStorage):
    def create(self, uri, data):
        raise RuntimeError("token=secret")


class ReplaceFails(LocalTaskStorage):
    def replace(self, uri, data):
        raise RuntimeError("token=secret")


class ReplacesThenFails(LocalTaskStorage):
    """A storage whose write is made, but whose response is lost."""

    def replace(self, uri, data):
        super().replace(uri, data)
        raise TimeoutError("token=secret")


def export(database, root: Path, exported_at=EXPORTED_AT, storage=None):
    return export_index(database, storage or LocalTaskStorage(), str(root), exported_at)


def read_rows(uri: str) -> list[dict]:
    return pq.read_table(uri).to_pylist()


def add_presets(database, digest: str, names) -> None:
    with transaction(database) as connection:
        for name in names:
            connection.execute(
                "INSERT INTO presets (name, configuration_digest, created_by) "
                "VALUES (%s, %s, 'ana')",
                (name, digest),
            )


def test_the_index_has_one_row_for_each_current_file(registered, tmp_path):
    run_id = submit(registered, ["a", "b"])
    add_covered_result(registered, run_id, "a", files=("embeddings", "scores"))
    add_covered_result(registered, run_id, "b", files=("embeddings", "scores"))

    rows = read_rows(export(registered, tmp_path).uri)

    current = get_current_files(registered)
    assert len(rows) == 4
    assert [(r["value"], r["kind"], r["uri"], r["checksum"], r["run_id"]) for r in rows] == [
        (c["value"], c["kind"], c["uri"], c["checksum"], str(c["run_id"])) for c in current
    ]
    assert {r["model_id"] for r in rows} == {"owl/v4"}


def test_the_index_has_the_contract_and_the_fixed_schema(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")

    table = pq.read_table(export(registered, tmp_path).uri)

    assert table.schema.metadata[b"robin.contract"] == CURRENT_FILES_CONTRACT_ID.encode()
    assert table.schema.names == COLUMNS
    assert table.schema.field("size_bytes").type == pa.int64()
    assert table.schema.field("rows").type == pa.int64()
    assert table.schema.field("presets").type == pa.list_(pa.string())
    assert table.schema.field("up_to_date").type == pa.bool_()


def test_an_empty_database_gives_an_index_with_no_rows(registered, tmp_path):
    result = export(registered, tmp_path)

    table = pq.read_table(result.uri)
    assert (result.rows, table.num_rows, table.schema.names) == (0, 0, COLUMNS)


def test_a_file_recorded_before_0004_has_null_size_and_rows(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")

    [row] = read_rows(export(registered, tmp_path).uri)

    assert (row["size_bytes"], row["rows"]) == (None, None)


def test_a_file_recorded_with_its_size_has_its_size_and_rows(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")
    with transaction(registered) as connection:
        connection.execute("UPDATE output_files SET size_bytes = 61829322, rows = 10800")

    [row] = read_rows(export(registered, tmp_path).uri)

    assert (row["size_bytes"], row["rows"]) == (61829322, 10800)


def test_presets_lists_every_name_of_the_configuration(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")
    [before] = read_rows(export(registered, tmp_path).uri)
    add_presets(registered, before["configuration_digest"], ["b-name", "a-name"])

    rows = read_rows(export(registered, tmp_path, EXPORTED_AT + timedelta(seconds=1)).uri)

    assert before["presets"] == []
    assert [row["presets"] for row in rows] == [["a-name", "b-name"]]


def test_the_output_versions_are_compact_json_with_sorted_keys(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")

    [row] = read_rows(export(registered, tmp_path).uri)

    [current] = get_current_files(registered)
    versions = current["output_versions"]
    assert row["output_versions"] == (
        "{" + ",".join(f'"{key}":{versions[key]}' for key in sorted(versions)) + "}"
    )


def test_a_withdrawn_run_has_no_row(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")
    withdraw(registered, run_id)

    assert read_rows(export(registered, tmp_path).uri) == []


def test_export_writes_the_dated_copy_and_the_current_file(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")

    result = export(registered, tmp_path)

    assert result.uri == f"{tmp_path}/index/current_files.parquet"
    assert result.dated_uri == (
        f"{tmp_path}/index/history/20261117T120000Z/current_files.parquet"
    )
    data = Path(result.uri).read_bytes()
    assert Path(result.dated_uri).read_bytes() == data
    assert result.checksum == "sha256:" + hashlib.sha256(data).hexdigest()
    assert result.rows == 1


def test_a_root_with_a_trailing_slash_gives_the_same_names(registered, tmp_path):
    result = export_index(registered, LocalTaskStorage(), f"{tmp_path}/", EXPORTED_AT)

    assert result.uri == f"{tmp_path}/index/current_files.parquet"


def test_a_second_export_replaces_the_current_file(registered, tmp_path):
    run_id = submit(registered, ["a", "b"])
    add_covered_result(registered, run_id, "a")
    first = export(registered, tmp_path)
    first_bytes = Path(first.dated_uri).read_bytes()
    add_covered_result(registered, run_id, "b")

    second = export(registered, tmp_path, EXPORTED_AT + timedelta(seconds=1))

    assert second.uri == first.uri
    assert pq.read_table(second.uri).num_rows == 2
    assert Path(first.dated_uri).read_bytes() == first_bytes


def test_a_dated_name_that_exists_replaces_nothing(registered, tmp_path):
    run_id = submit(registered, ["a", "b"])
    add_covered_result(registered, run_id, "a")
    first = export(registered, tmp_path)
    first_bytes = Path(first.uri).read_bytes()
    add_covered_result(registered, run_id, "b")

    with pytest.raises(FileExistsError):
        export(registered, tmp_path)

    assert Path(first.uri).read_bytes() == first_bytes


def test_a_failed_dated_write_replaces_nothing(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")
    first = export(registered, tmp_path)
    first_bytes = Path(first.uri).read_bytes()
    later = EXPORTED_AT + timedelta(seconds=1)

    with pytest.raises(StorageUnavailable) as raised:
        export(registered, tmp_path, later, storage=CreateFails())

    assert "secret" not in str(raised.value)
    assert raised.value.details["error_type"] == "RuntimeError"
    assert Path(first.uri).read_bytes() == first_bytes


def test_a_failed_replace_is_uncertain_and_keeps_the_dated_copy(registered, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")
    first = export(registered, tmp_path)
    first_bytes = Path(first.uri).read_bytes()
    later = EXPORTED_AT + timedelta(seconds=1)

    with pytest.raises(OutcomeUncertain) as raised:
        export(registered, tmp_path, later, storage=ReplaceFails())

    assert "secret" not in str(raised.value)
    assert raised.value.details["error_type"] == "RuntimeError"
    assert Path(first.uri).read_bytes() == first_bytes
    assert (tmp_path / "index/history/20261117T120001Z/current_files.parquet").is_file()


def test_a_replace_that_fails_after_it_writes_is_uncertain(registered, tmp_path):
    run_id = submit(registered, ["a", "b"])
    add_covered_result(registered, run_id, "a")
    first = export(registered, tmp_path)
    add_covered_result(registered, run_id, "b")
    later = EXPORTED_AT + timedelta(seconds=1)

    with pytest.raises(OutcomeUncertain) as raised:
        export(registered, tmp_path, later, storage=ReplacesThenFails())

    assert "secret" not in str(raised.value)
    assert raised.value.details["error_type"] == "TimeoutError"
    assert pq.read_table(first.uri).num_rows == 2


def test_an_export_works_with_a_pool_of_one_connection(registered, database_settings, tmp_path):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a")
    settings = database_settings.model_copy(
        update={"pool_min_size": 1, "pool_max_size": 1, "pool_timeout_s": 1}
    )
    one_connection = open_database(settings)
    try:
        result = export(one_connection, tmp_path)
    finally:
        one_connection.close()

    assert result.rows == 1


def test_an_export_while_another_runs_is_refused(registered, tmp_path):
    with lock_index_export(registered) as connection:
        assert connection is not None
        with pytest.raises(Refused) as raised:
            export(registered, tmp_path)

    assert raised.value.reason == EXPORT_IN_PROGRESS
    assert not (tmp_path / "index").exists()
    assert export(registered, tmp_path).rows == 0


def test_the_same_data_gives_the_same_bytes(registered):
    run_id = submit(registered, ["a", "b"])
    add_covered_result(registered, run_id, "a", files=("embeddings", "scores"))
    add_covered_result(registered, run_id, "b")

    with transaction(registered, read_only=True) as connection:
        first = build_index(get_index_rows(connection), EXPORTED_AT)
        second = build_index(get_index_rows(connection), EXPORTED_AT)

    assert first == second
