"""Read and write Arrow streams containing one recording's raw embeddings, one per window.

The rows do not name their recording: the published path and the artifact record do.

The reader verifies the file checksum, schema and required metadata before yielding
an iterator. Batches are decoded as the caller iterates; malformed batches raise a
typed artifact error at that point.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np
import pyarrow as pa

from robin_contracts.canonical import checksum_file
from robin_contracts.output_contracts import EmbeddingsContractId
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts.metadata import (
    CONTRACT_KEY,
    EMBEDDING_DIM_KEY,
    EMBEDDING_KEYS,
    EMBEDDING_STORAGE_DTYPE_KEY,
    REGISTRY_KEYS,
    REQUIRED_KEYS,
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

CONTRACT_ID: EmbeddingsContractId = get_args(EmbeddingsContractId)[0]

# In bytes, not rows: a row is a whole vector, so rows bound memory differently per model.
# ~ 1 megabyte
EMBEDDING_BATCH_BYTES = 1 << 20

_ARROW_VALUE_TYPE: dict[str, pa.DataType] = {
    "float32": pa.float32(),
    "float16": pa.float16(),
}

_NUMPY_VALUE_TYPE: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
}

_DTYPE_NAME: dict[pa.DataType, str] = {
    value: name for name, value in _ARROW_VALUE_TYPE.items()
}

_EMBEDDING_FIELD = "embedding"


def embeddings_schema(dim: int, storage_dtype: str) -> pa.Schema:
    """The three declared fields, with the width and value type in the list type."""
    return pa.schema(
        [
            pa.field("window_start_s", pa.float64(), nullable=False),
            pa.field("window_end_s", pa.float64(), nullable=False),
            pa.field(
                _EMBEDDING_FIELD,
                pa.list_(_ARROW_VALUE_TYPE[storage_dtype], dim),
                nullable=False,
            ),
        ]
    )


class EmbeddingsWriter:
    """Writes one vector per accepted window of one recording to an Arrow stream, a
    batch at a time.

    The caller supplies the width, so a stream holding no window is still correctly typed.
    """

    def __init__(
        self,
        path: Path,
        *,
        recording: RecordingRef,
        dim: int,
        storage_dtype: str,
        metadata: Mapping[bytes, bytes],
    ) -> None:
        _require_writable_header(metadata, dim=dim, storage_dtype=storage_dtype)
        self._schema = embeddings_schema(dim, storage_dtype)
        self._dim = dim
        self._storage_dtype = storage_dtype
        self._value_dtype = _NUMPY_VALUE_TYPE[storage_dtype]
        self._batch_rows = max(1, EMBEDDING_BATCH_BYTES // (dim * self._value_dtype.itemsize))
        self._path = path
        self._recording = recording
        self._file: pa.OSFile | None = pa.OSFile(str(path), "wb")
        self._stream = pa.ipc.new_stream(
            self._file, self._schema.with_metadata(dict(metadata))
        )
        self._pending = self._empty_columns()
        self._pending_rows = 0
        self._rows = 0
        self._closed = False

    def write(self, window: AcceptedWindow) -> None:
        """Append this window's vector, and nothing at all if it carries none."""
        if self._closed or self._stream is None:
            raise RuntimeError(f"{self._path.name} is closed; no window can be added")
        if window.embedding is None:
            return
        self._require_declared_width(window)
        stored = self._narrowed(window)
        self._pending["window_start_s"].append(window.start)
        self._pending["window_end_s"].append(window.end)
        self._pending[_EMBEDDING_FIELD].append(stored)
        self._pending_rows += 1
        if self._pending_rows >= self._batch_rows:
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
            kind="embeddings",
            contract_id=CONTRACT_ID,
            recording=self._recording,
            path=self._path,
            checksum=checksum_file(self._path),
            rows=self._rows,
        )

    def __enter__(self) -> "EmbeddingsWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        # This closes the handles and nothing more. An unfinished file is left for
        # the caller to delete, and close() refuses to stage one.
        self._release()

    def _release(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _require_declared_width(self, window: AcceptedWindow) -> None:
        # The acceptance boundary refused every other width, so one arriving here is
        # an engine defect rather than an untrusted input.
        if window.embedding.size != self._dim:
            raise RuntimeError(
                f"{self._path.name} stores {self._dim}-wide vectors; this window's "
                f"is {window.embedding.size} wide"
            )

    def _narrowed(self, window: AcceptedWindow) -> np.ndarray:
        """The vector at its stored width, refused if narrowing made it infinite."""
        with np.errstate(over="ignore"):  # detected and refused below
            stored = window.embedding.astype(self._value_dtype)
        # Underflow to zero is the precision loss the narrow width was asked for, not
        # this failure; only a value that was finite and is not any more is.
        lost = np.isfinite(window.embedding) & ~np.isfinite(stored)
        if not lost.any():
            return stored
        value = float(window.embedding[lost.argmax()])
        raise errors.EngineError(
            errors.EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE,
            errors.WRITE_ARTIFACT,
            f"{value} is outside the range {self._storage_dtype} can store",
            recording=window.recording,
            window_start_s=window.start,
        )

    def _flush(self) -> None:
        if self._stream is None:
            raise RuntimeError(f"{self._path.name} has no open stream to write to")
        if not self._pending_rows:
            return
        self._stream.write_batch(
            pa.RecordBatch.from_arrays(
                [self._column(field) for field in self._schema], schema=self._schema
            )
        )
        self._rows += self._pending_rows
        self._pending = self._empty_columns()
        self._pending_rows = 0

    def _empty_columns(self) -> dict[str, list[object]]:
        return {name: [] for name in self._schema.names}

    def _column(self, field: pa.Field) -> pa.Array:
        if field.name != _EMBEDDING_FIELD:
            return pa.array(self._pending[field.name], type=field.type)
        # The vectors are already NumPy, and a fixed-size list is a width over a
        # flat child, so no Python list is built per window.
        values = np.concatenate(self._pending[_EMBEDDING_FIELD])
        return pa.FixedSizeListArray.from_arrays(pa.array(values), self._dim)


@dataclass(frozen=True, slots=True)
class EmbeddingsStream:
    """The header of a verified artifact, and an iterator over its rows on disk."""

    metadata: Mapping[str, str]
    batches: Iterator[pa.RecordBatch]


@contextmanager
def read_embeddings(path: Path, *, expected_checksum: str) -> Iterator[EmbeddingsStream]:
    """Verify an artifact's bytes, schema and header, then stream its rows.

    `batches` can be read once, and the file is closed when the block ends.
    """
    try:
        actual = checksum_file(path)
        handle = pa.OSFile(str(path), "rb")
    except OSError as exc:
        raise unreadable(path.name, exc) from exc
    with handle:
        require_checksum(path.name, actual=actual, expected=expected_checksum)
        with open_stream(handle, path.name) as reader:
            yield EmbeddingsStream(
                metadata=_verified_header(reader.schema),
                batches=read_batches(reader, path.name),
            )


def _verified_header(schema: pa.Schema) -> dict[str, str]:
    """Check the schema and the provenance header, and return the header as text."""
    metadata = decode_metadata(schema.metadata)
    require_contract(metadata.get(CONTRACT_KEY), expected=CONTRACT_ID)
    require_metadata_keys(metadata, _required_keys(metadata), contract_id=CONTRACT_ID)
    _require_schema(schema, metadata)
    return metadata


def _required_keys(header: Mapping[str, str]) -> tuple[str, ...]:
    """Every key an embeddings header carries, given whether it bound a registry."""
    # These rows carry no labels, so a work that bound no registry writes neither
    # key; one alone is a producer defect, so the present one requires the other.
    bound = any(key in header for key in REGISTRY_KEYS)
    return (*REQUIRED_KEYS, *EMBEDDING_KEYS, *(REGISTRY_KEYS if bound else ()))


def _require_schema(schema: pa.Schema, header: Mapping[str, str]) -> None:
    """Refuse a schema that is not the one this artifact's own header describes."""
    # The width and value type belong to the artifact rather than to the contract, so
    # a disagreement between the two is the artifact contradicting itself.
    dim = _declared_dim(header)
    storage_dtype = _declared_storage_dtype(header)
    for field in embeddings_schema(dim, storage_dtype):
        # Matched by name; additional and reordered columns are accepted.
        actual = _matching_field(schema, field.name)
        if actual.nullable != field.nullable:
            raise _invalid_schema(
                f"{field.name!r} is nullable={actual.nullable}, not "
                f"nullable={field.nullable}"
            )
        if actual.type == field.type:
            continue
        if field.name == _EMBEDDING_FIELD:
            raise _invalid_schema(_embedding_detail(actual.type, dim, storage_dtype))
        raise _invalid_schema(f"{field.name!r} is {actual.type}, not {field.type}")


def _matching_field(schema: pa.Schema, name: str) -> pa.Field:
    """The one column with this name. Arrow permits a repeat; this contract does not."""
    positions = schema.get_all_field_indices(name)
    if not positions:
        raise _invalid_schema(f"{name!r} is not one of {schema.names}")
    if len(positions) > 1:
        # A lookup by a repeated name has no single answer.
        raise _invalid_schema(
            f"{name!r} names {len(positions)} of this artifact's columns"
        )
    return schema.field(positions[0])


def _embedding_detail(actual: pa.DataType, dim: int, storage_dtype: str) -> str:
    declared = f"the fixed-size list of {dim} {storage_dtype} values this header declares"
    if not pa.types.is_fixed_size_list(actual):
        return f"{_EMBEDDING_FIELD!r} is {actual}, not {declared}"
    held = _DTYPE_NAME.get(actual.value_type, str(actual.value_type))
    return (
        f"{_EMBEDDING_FIELD!r} holds {actual.list_size} {held} values per row, "
        f"not {declared}"
    )


def _declared_dim(header: Mapping[str, str]) -> int:
    raw = header[EMBEDDING_DIM_KEY]
    # Both checks are needed: isdigit() accepts digits int() cannot convert, such as
    # "²", and int() refuses a digit string longer than its conversion limit.
    try:
        width = int(raw) if raw.isdigit() else 0
    except ValueError:
        width = 0
    # Arrow stores fixed-size list dimensions as signed 32-bit integers.
    if not 0 < width <= np.iinfo(np.int32).max:
        raise _invalid_schema(
            f"{EMBEDDING_DIM_KEY} must be a positive 32-bit integer, got {raw!r}"
        )
    return width


def _declared_storage_dtype(header: Mapping[str, str]) -> str:
    declared = header[EMBEDDING_STORAGE_DTYPE_KEY]
    if declared not in _ARROW_VALUE_TYPE:
        # The comparison schema is chosen from this value, so an unknown one checks nothing.
        raise _invalid_schema(
            f"{CONTRACT_ID} stores {' or '.join(_ARROW_VALUE_TYPE)} values, "
            f"not {declared!r}"
        )
    return declared


def _invalid_schema(detail: str) -> errors.EngineError:
    return invalid_schema(f"{CONTRACT_ID} does not accept this schema: {detail}")


def _require_writable_header(
    metadata: Mapping[bytes, bytes], *, dim: int, storage_dtype: str
) -> None:
    """Refuse a header this contract's reader would reject, before a file is created.

    The engine builds this header itself, so a wrong one is a defect in the caller.
    """
    header = _decode_header(metadata)
    declared = header.get(CONTRACT_KEY)
    if declared != CONTRACT_ID:
        raise RuntimeError(
            f"an embeddings header must declare {CONTRACT_ID}, not {declared!r}"
        )
    missing = [key for key in _required_keys(header) if key not in header]
    if missing:
        raise RuntimeError(f"an embeddings header carries no {', '.join(missing)}")
    _require_the_header_describes_the_stream(header, dim=dim, storage_dtype=storage_dtype)
    try:
        _declared_dim(header)
    except errors.EngineError as exc:
        raise RuntimeError(exc.detail) from exc


def _require_the_header_describes_the_stream(
    header: Mapping[str, str], *, dim: int, storage_dtype: str
) -> None:
    """Refuse a header whose declared width or value type is not the stream's."""
    for key, value in (
        (EMBEDDING_DIM_KEY, str(dim)),
        (EMBEDDING_STORAGE_DTYPE_KEY, storage_dtype),
    ):
        if header[key] != value:
            raise RuntimeError(
                f"an embeddings header declares {key} {header[key]!r} for a stream "
                f"written as {value!r}"
            )


def _decode_header(metadata: Mapping[bytes, bytes]) -> dict[str, str]:
    try:
        return {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in metadata.items()
        }
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"an embeddings header must be UTF-8 throughout: {exc}"
        ) from exc
