"""A finished artifact file, and the checks every reader holds one to."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robin_contracts.results import ArtifactContractId, ArtifactKind
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    """A finished artifact on local disk, holding one recording's rows, before a writer
    port publishes it.

    It has no uri and no size: the writer places it by its kind and recording, and the
    size is measured from the finished file once it is published.
    """

    kind: ArtifactKind
    contract_id: ArtifactContractId
    recording: RecordingRef
    path: Path
    checksum: str
    rows: int


def malformed(detail: str) -> errors.EngineError:
    """Bytes that cannot be read as this contract's container at all."""
    return errors.EngineError(
        errors.ARTIFACT_MALFORMED, errors.READ_INPUT_ARTIFACT, detail
    )


def invalid_schema(detail: str) -> errors.EngineError:
    """Bytes that are read, but do not match what the contract declares."""
    return errors.EngineError(
        errors.ARTIFACT_SCHEMA_INVALID, errors.READ_INPUT_ARTIFACT, detail
    )


def unreadable(name: str, exc: OSError) -> errors.EngineError:
    """A file this reader could not open or read at all."""
    return errors.EngineError(
        errors.ARTIFACT_UNREADABLE,
        errors.READ_INPUT_ARTIFACT,
        f"{name} could not be read: {exc}",
    )


def open_parquet(handle: pa.NativeFile, name: str) -> pq.ParquetFile:
    """Open a Parquet file, refusing bytes that are not one."""
    try:
        return pq.ParquetFile(handle)
    # pyarrow raises OSError for a footer that does not decode.
    except (pa.ArrowInvalid, OSError) as exc:
        raise malformed(f"{name} does not open as Parquet: {exc}") from exc


def read_batches(
    file: pq.ParquetFile, name: str, *, batch_rows: int
) -> Iterator[pa.RecordBatch]:
    """Yield at most `batch_rows` rows at a time as they are decoded, refusing a batch
    that cannot be read or that holds a null in a field its schema declares not nullable."""
    batches = file.iter_batches(batch_size=batch_rows)
    while True:
        try:
            batch = next(batches)
        except StopIteration:
            return
        except (pa.ArrowInvalid, OSError) as exc:
            raise malformed(f"{name} has an unreadable Parquet row group: {exc}") from exc
        _refuse_a_null(batch, name)
        yield batch


def _refuse_a_null(batch: pa.RecordBatch, name: str) -> None:
    # Each reader refuses a nullable schema before any row is read, so a null here
    # means the file breaks its own schema.
    for field, column in zip(batch.schema, batch.columns):
        if not field.nullable and column.null_count:
            raise malformed(
                f"{name} has {column.null_count} null values in {field.name!r}, "
                f"which its schema declares not nullable"
            )


def require_checksum(name: str, *, actual: str, expected: str) -> None:
    """Refuse an artifact whose bytes do not hash to the checksum it was named by.

    Each reader hashes the way its own artifact allows, streamed or in memory, and
    hands the result here.
    """
    if actual != expected:
        raise errors.EngineError(
            errors.ARTIFACT_CHECKSUM_MISMATCH,
            errors.READ_INPUT_ARTIFACT,
            f"{name} was read as {actual} but was named as {expected}",
        )


def require_contract(declared: object, *, expected: ArtifactContractId) -> None:
    """Refuse an artifact that declares a contract other than the one being read."""
    if declared != expected:
        raise errors.EngineError(
            errors.ARTIFACT_CONTRACT_UNEXPECTED,
            errors.READ_INPUT_ARTIFACT,
            f"expected contract {expected} but the artifact declares {declared!r}",
        )


def checksum_bytes(payload: bytes) -> str:
    """The `sha256:` checksum of bytes already in memory."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()
