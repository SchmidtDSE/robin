"""The score stream: one recording's rows, one per window and label, verified when it
is read back.

The header carries the run's provenance, so the file can be interpreted on its own. The
rows do not name their recording: the published path and the artifact record do.
Rows are written in bounded batches, so the whole artifact is never held in memory,
and the reader checks the whole file before returning any row.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import pyarrow as pa

from robin_contracts.canonical import checksum_file
from robin_contracts.output_contracts import ScoresContractId
from robin_contracts.protocols import ScoreRetention
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts.metadata import (
    CONTRACT_KEY,
    REGISTRY_KEYS,
    REQUIRED_KEYS,
    SCORE_FLOOR_KEY,
    SCORE_KEYS,
    SCORE_RETENTION_KEY,
    SCORE_TOP_K_KEY,
    decode_metadata,
    require_metadata_keys,
)
from robin_inference_engine.artifacts.staging import (
    StagedArtifact,
    invalid_schema,
    open_stream,
    read_batches,
    require_checksum,
    require_contract,
    unreadable,
)

CONTRACT_ID: ScoresContractId = get_args(ScoresContractId)[0]

RETENTIONS: tuple[str, ...] = get_args(ScoreRetention)

# Rows, not windows: one window is 51 rows for one model and 6,522 for another, so a
# window count bounds memory differently for every model it is used with. A window is
# never split across batches, so a batch can pass this by up to one window's rows.
SCORE_BATCH_ROWS = 8192

SCORES_SCHEMA = pa.schema(
    [
        pa.field("window_start_s", pa.float64(), nullable=False),
        pa.field("window_end_s", pa.float64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ]
)


class ScoresWriter:
    """Writes one recording's accepted windows to an Arrow stream, a batch at a time.

    It checks the header it is given, writes the windows it is given, and leaves
    the file on disk. The caller publishes the finished bytes and deletes the file.
    """

    def __init__(
        self, path: Path, *, recording: RecordingRef, metadata: Mapping[bytes, bytes]
    ) -> None:
        _require_writable_header(metadata)
        self._path = path
        self._recording = recording
        self._file: pa.OSFile | None = pa.OSFile(str(path), "wb")
        self._stream = pa.ipc.new_stream(
            self._file, SCORES_SCHEMA.with_metadata(dict(metadata))
        )
        self._pending = _empty_columns()
        self._pending_rows = 0
        self._rows = 0
        self._closed = False

    def write(self, window: AcceptedWindow) -> None:
        """Append one row per score in this window, and no rows if it has none."""
        if self._closed or self._stream is None:
            raise RuntimeError(f"{self._path.name} is closed; no window can be added")
        for score in window.scores:
            self._pending["window_start_s"].append(window.start)
            self._pending["window_end_s"].append(window.end)
            self._pending["label"].append(score.label)
            self._pending["score"].append(score.score)
        self._pending_rows += len(window.scores)
        # The buffer is flushed at the first window boundary at or past the bound,
        # so a window is never split across two batches.
        if self._pending_rows >= SCORE_BATCH_ROWS:
            self._flush()

    def close(self) -> StagedArtifact:
        """Finish the stream and stage the file. The file stays on disk."""
        if self._closed:
            raise RuntimeError(f"{self._path.name} has already been closed")
        if self._stream is None:
            # Releasing discards whatever was still pending, so the row count and
            # checksum staged here would describe a file that was never finished.
            raise RuntimeError(f"{self._path.name} was released before it was closed")
        self._flush()
        self._release()
        self._closed = True
        return StagedArtifact(
            kind="scores",
            contract_id=CONTRACT_ID,
            recording=self._recording,
            path=self._path,
            checksum=checksum_file(self._path),
            rows=self._rows,
        )

    def __enter__(self) -> "ScoresWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        # This closes the handles and nothing more. An unfinished file is left for
        # the caller to delete, and close() refuses to stage one.
        self._release()

    def _flush(self) -> None:
        if self._stream is None:
            raise RuntimeError(f"{self._path.name} has no open stream to write to")
        if not self._pending_rows:
            return
        self._stream.write_batch(
            pa.RecordBatch.from_arrays(
                [
                    pa.array(self._pending[field.name], type=field.type)
                    for field in SCORES_SCHEMA
                ],
                schema=SCORES_SCHEMA,
            )
        )
        self._rows += self._pending_rows
        self._pending = _empty_columns()
        self._pending_rows = 0

    def _release(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._file is not None:
            self._file.close()
            self._file = None


@dataclass(frozen=True, slots=True)
class ScoresStream:
    """The header of a verified artifact, and an iterator over its rows on disk."""

    metadata: Mapping[str, str]
    batches: Iterator[pa.RecordBatch]


@contextmanager
def read_scores(path: Path, *, expected_checksum: str) -> Iterator[ScoresStream]:
    """Verify an artifact's bytes, schema and header, then stream its rows.

    `batches` can be read once, and the file is closed when the block ends.
    Returning every row instead would hold the whole artifact in memory, which
    streaming avoids.
    """
    try:
        actual = checksum_file(path)
        handle = pa.OSFile(str(path), "rb")
    except OSError as exc:
        raise unreadable(path.name, exc) from exc
    with handle:
        require_checksum(path.name, actual=actual, expected=expected_checksum)
        with open_stream(handle, path.name) as reader:
            yield ScoresStream(
                metadata=_verified_header(reader.schema),
                batches=read_batches(reader, path.name),
            )


def _empty_columns() -> dict[str, list[object]]:
    return {name: [] for name in SCORES_SCHEMA.names}


def _verified_header(schema: pa.Schema) -> dict[str, str]:
    """Check the schema and the provenance header, and return the header as text."""
    _require_schema(schema)
    metadata = decode_metadata(schema.metadata)
    require_contract(metadata.get(CONTRACT_KEY), expected=CONTRACT_ID)
    _require_provenance(metadata)
    return metadata


def _require_schema(schema: pa.Schema) -> None:
    # Fields are matched by name, never by position, and compared on type and
    # nullability. Additional fields and reordered columns are accepted.
    for declared in SCORES_SCHEMA:
        positions = schema.get_all_field_indices(declared.name)
        if not positions:
            raise _invalid_schema(f"{declared.name!r} is not one of {schema.names}")
        if len(positions) > 1:
            # Arrow permits a repeated field name, and a lookup by that name then
            # has no single answer, so the artifact is refused rather than read
            # from whichever column comes first.
            raise _invalid_schema(
                f"{declared.name!r} names {len(positions)} of this artifact's columns"
            )
        actual = schema.field(positions[0])
        if actual.type != declared.type or actual.nullable != declared.nullable:
            raise _invalid_schema(
                f"{declared.name!r} is {actual.type} nullable={actual.nullable}, not "
                f"{declared.type} nullable={declared.nullable}"
            )


def _invalid_schema(detail: str) -> errors.EngineError:
    return invalid_schema(f"{CONTRACT_ID} does not accept this schema: {detail}")


def _require_provenance(metadata: Mapping[str, str]) -> None:
    """Refuse a header missing a required key, or declaring an unknown retention."""
    retention = metadata.get(SCORE_RETENTION_KEY)
    require_metadata_keys(metadata, _required_keys(retention), contract_id=CONTRACT_ID)
    if retention not in RETENTIONS:
        # The keys required above are chosen from this value, so a value outside
        # the declared set would match no branch and leave the retention's own
        # keys unchecked.
        raise invalid_schema(
            f"{CONTRACT_ID} declares the retention values {', '.join(RETENTIONS)}, "
            f"not {retention!r}"
        )


def _require_writable_header(metadata: Mapping[bytes, bytes]) -> None:
    """Refuse a header this contract's reader would reject, before a file is created.

    The engine builds this header itself, so a wrong one is a defect in the calling
    code rather than an untrusted artifact. This raises RuntimeError, like the
    writer's other refusals.
    """
    header = _decode_header(metadata)
    declared = header.get(CONTRACT_KEY)
    if declared != CONTRACT_ID:
        raise RuntimeError(
            f"a scores header must declare {CONTRACT_ID}, not {declared!r}"
        )
    retention = header.get(SCORE_RETENTION_KEY)
    missing = [key for key in _required_keys(retention) if key not in header]
    if missing:
        raise RuntimeError(f"a scores header carries no {', '.join(missing)}")
    if retention not in RETENTIONS:
        raise RuntimeError(
            f"a scores header must declare one of the retention values "
            f"{', '.join(RETENTIONS)}, not {retention!r}"
        )


def _decode_header(metadata: Mapping[bytes, bytes]) -> dict[str, str]:
    try:
        return {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in metadata.items()
        }
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"a scores header must be UTF-8 throughout: {exc}") from exc


def _required_keys(retention: str | None) -> tuple[str, ...]:
    """Every key a scores header carries, given the retention it declares."""
    conditional: tuple[str, ...] = ()
    if retention == "thresholded":
        conditional = (SCORE_FLOOR_KEY,)
    elif retention == "top_k":
        conditional = (SCORE_FLOOR_KEY, SCORE_TOP_K_KEY)
    # A score is a position in a label vocabulary, so a scores artifact must name the
    # registry that defines it. The pair is required here, not optional.
    return (*REQUIRED_KEYS, *REGISTRY_KEYS, *SCORE_KEYS, *conditional)
