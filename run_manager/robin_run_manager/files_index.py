"""The current files index: one Parquet file that lists each current output file.

`robin files export` writes it to `<root>/index/current_files.parquet` in the profile's
storage, and keeps a dated copy under `<root>/index/history/`. Its rows are a copy of
the `current_files` view, with each file's model, preset names, size, and row count.
The file changes only when an export runs, and only one export runs at a time.

The contract `CURRENT_FILES_CONTRACT_ID` fixes the columns. A change to them is a new
contract ID and a new file name.
"""

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any

import psycopg
from robin_contracts.output_contracts import CURRENT_FILES_CONTRACT_ID
from robin_contracts.ports.storage import TaskStorage

from robin_run_manager.database import Database, lock_index_export, transaction_on
from robin_run_manager.errors import (
    EXPORT_IN_PROGRESS,
    OperationError,
    OutcomeUncertain,
    Refused,
    StorageUnavailable,
    invoke_adapter,
)

_FILE_NAME = "current_files.parquet"

_ROWS_QUERY = """
SELECT
    c.namespace,
    c.value,
    m.model_id,
    array(
        SELECT p.name FROM presets p
        WHERE p.configuration_digest = c.configuration_digest
        ORDER BY p.name
    ) AS presets,
    c.configuration_digest,
    c.kind,
    c.contract_id,
    c.uri,
    c.checksum,
    f.size_bytes,
    f.rows,
    c.output_versions,
    c.up_to_date,
    c.run_id
FROM current_files c
JOIN output_files f
    ON f.run_id = c.run_id AND f.namespace = c.namespace AND f.value = c.value
    AND f.kind = c.kind
JOIN configurations m ON m.digest = c.configuration_digest
ORDER BY c.namespace, c.value, m.model_id, c.configuration_digest, c.kind
"""


@dataclass(frozen=True)
class IndexExport:
    """What one export wrote: the current file, its dated copy, its row count, and the
    `sha256:` checksum of its bytes."""

    uri: str
    dated_uri: str
    rows: int
    checksum: str


def get_index_uris(root: str, exported_at: datetime) -> tuple[str, str]:
    """Returns the URI of the current index file under `root`, and the URI of the
    dated copy for `exported_at`."""
    index = f"{root.rstrip('/')}/index"
    stamp = exported_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{index}/{_FILE_NAME}", f"{index}/history/{stamp}/{_FILE_NAME}"


def get_index_rows(connection: psycopg.Connection) -> list[dict[str, Any]]:
    """Returns one row for each current file, in the order of the index. Each value has
    its database type, such as a dict for `output_versions` and a UUID for `run_id`."""
    return connection.execute(_ROWS_QUERY).fetchall()


def build_index(rows: Sequence[Mapping[str, Any]], exported_at: datetime) -> bytes:
    """Returns the Parquet bytes of the index for `rows`, from `get_index_rows`.

    `output_versions` becomes compact JSON with sorted keys, so that the same data gives
    the same bytes, and `run_id` becomes text.
    """
    # Only an export needs pyarrow, so other commands do not load it.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("namespace", pa.string()),
            ("value", pa.string()),
            ("model_id", pa.string()),
            ("presets", pa.list_(pa.string())),
            ("configuration_digest", pa.string()),
            ("kind", pa.string()),
            ("contract_id", pa.string()),
            ("uri", pa.string()),
            ("checksum", pa.string()),
            ("size_bytes", pa.int64()),
            ("rows", pa.int64()),
            ("output_versions", pa.string()),
            ("up_to_date", pa.bool_()),
            ("run_id", pa.string()),
        ],
        metadata={
            "robin.contract": CURRENT_FILES_CONTRACT_ID,
            "robin.exported_at": exported_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    index_rows = [
        row | {
            "output_versions": json.dumps(
                row["output_versions"], sort_keys=True, separators=(",", ":")
            ),
            "run_id": str(row["run_id"]),
        }
        for row in rows
    ]
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(index_rows, schema=schema), buffer)
    return buffer.getvalue()


def export_index(
    database: Database, storage: TaskStorage, root: str, exported_at: datetime
) -> IndexExport:
    """Write the index of the current files under `root`.

    It writes the dated copy first, then replaces the current file with the same bytes.
    It holds the export lock until both writes end, so a later export always reads newer
    data than an earlier one.

    Raises `Refused` with `export_in_progress` if another export runs, and writes
    nothing. Raises `FileExistsError` if a dated copy for `exported_at` exists, and
    `StorageUnavailable` if the dated copy cannot be written. In those cases the current
    file keeps its old bytes. Raises `OutcomeUncertain` if the replace of the current
    file fails, because the storage may have made the write before it failed.
    """
    uri, dated_uri = get_index_uris(root, exported_at)
    with lock_index_export(database) as connection:
        if connection is None:
            raise Refused(
                EXPORT_IN_PROGRESS,
                "Another export of the current files index is running. Robin wrote "
                "nothing. Run the command again when that export ends.",
            )
        with transaction_on(connection, read_only=True):
            rows = get_index_rows(connection)
        data = build_index(rows, exported_at)
        invoke_adapter(
            partial(storage.create, dated_uri, data),
            partial(_get_dated_copy_failure, dated_uri),
            passing=(FileExistsError,),
        )
        invoke_adapter(
            partial(storage.replace, uri, data),
            partial(_get_replace_failure, uri, dated_uri),
        )
    return IndexExport(
        uri=uri,
        dated_uri=dated_uri,
        rows=len(rows),
        checksum="sha256:" + hashlib.sha256(data).hexdigest(),
    )


def _get_dated_copy_failure(uri: str, error_type: str) -> OperationError:
    return StorageUnavailable(
        f"The task storage failed with {error_type} when Robin wrote the dated copy of "
        "the index. Its message is not shown, because it can hold a credential. Robin "
        "did not change the current index file. Run the command again.",
        details={"error_type": error_type, "uri": uri},
    )


def _get_replace_failure(uri: str, dated_uri: str, error_type: str) -> OperationError:
    return OutcomeUncertain(
        f"The task storage failed with {error_type} when Robin replaced the current "
        "index file. Its message is not shown, because it can hold a credential. The "
        "storage may have made the write, so the current index file holds the old index "
        f"or the new one. The new index is also at {dated_uri}. Run the command again.",
        details={"error_type": error_type, "uri": uri, "dated_uri": dated_uri},
    )
