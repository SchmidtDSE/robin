"""The record of which recording each processing index refers to.

Rows in the window artifacts carry only an integer index. This file gives each index
its recording, archive and audio version, so the artifacts can be read on their own.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes, is_sha256_v1
from robin_contracts.output_contracts import RecordingMapContractId
from robin_contracts.work import InferenceWork, RecordingRef, work_digest
from robin_inference_engine import errors
from robin_inference_engine.artifacts.staging import (
    StagedArtifact,
    checksum_bytes,
    invalid_schema,
    malformed,
    require_checksum,
    require_contract,
    unreadable,
)

CONTRACT_ID: RecordingMapContractId = get_args(RecordingMapContractId)[0]

# The only keys this contract declares; a reader refuses any other.
DECLARED_KEYS = frozenset({"schema_version", "work_digest", "recordings"})


@dataclass(frozen=True, slots=True)
class RecordingMap:
    """A recording map: the digest of the work it was written for, and its recordings.

    Construction refuses a repeated index and a repeated (namespace, value), so a
    lookup by index has exactly one answer however the map was built.
    """

    work_digest: str
    recordings: tuple[RecordingRef, ...]
    _by_index: dict[int, RecordingRef] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_distinct_indices(self.recordings)
        _require_distinct_identities(self.recordings)
        object.__setattr__(
            self, "_by_index", {one.index: one for one in self.recordings}
        )

    def by_index(self, index: int) -> RecordingRef:
        """The recording this index refers to, raising KeyError where there is none."""
        # Untyped: the caller, not the map, knows which stage is resolving the index.
        return self._by_index[index]


def write_recording_map(work: InferenceWork, *, path: Path) -> StagedArtifact:
    """Write this work's map and stage it."""
    payload = canonical_json_bytes(
        {
            "schema_version": CONTRACT_ID,
            "work_digest": work_digest(work),
            "recordings": [one.model_dump(mode="json") for one in work.recordings],
        }
    )
    path.write_bytes(payload)
    # The checksum covers the bytes the writer produced, not a re-read of the file, so
    # an incomplete write fails the checksum rather than matching it.
    return StagedArtifact(
        kind="recording_map",
        contract_id=CONTRACT_ID,
        path=path,
        checksum=checksum_bytes(payload),
        rows=len(work.recordings),
    )


def read_recording_map(path: Path, *, expected_checksum: str) -> RecordingMap:
    """Verify a map's bytes and shape, then return it for index lookups."""
    payload = _read_bytes(path)
    require_checksum(
        path.name, actual=checksum_bytes(payload), expected=expected_checksum
    )
    document = _parse(path, payload)
    require_contract(document.get("schema_version"), expected=CONTRACT_ID)
    _require_declared_keys(document)
    digest = _require_work_digest(document)
    recordings = _validate_recordings(document)
    return RecordingMap(work_digest=digest, recordings=recordings)


def _require_declared_keys(document: dict[str, Any]) -> None:
    undeclared = sorted(set(document) - DECLARED_KEYS)
    if undeclared:
        raise invalid_schema(
            f"the map carries keys {CONTRACT_ID} does not declare: "
            f"{', '.join(undeclared)}"
        )


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise unreadable(path.name, exc) from exc


def _parse(path: Path, payload: bytes) -> dict[str, Any]:
    # json.loads decodes before it parses, so bytes that are not UTF-8 fail here.
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise malformed(f"{path.name} does not parse as JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise malformed(f"{path.name} holds a {type(document).__name__}, not an object")
    return document


def _require_work_digest(document: dict[str, Any]) -> str:
    value = document.get("work_digest")
    if value is None:
        raise invalid_schema("the map carries no work_digest")
    if not isinstance(value, str):
        raise invalid_schema(f"work_digest holds {type(value).__name__}, not text")
    if not is_sha256_v1(value):
        raise invalid_schema(f"work_digest is not a canonical digest: {value!r}")
    return value


def _validate_recordings(document: dict[str, Any]) -> tuple[RecordingRef, ...]:
    rows = document.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise invalid_schema("a map names at least one recording")
    validated = []
    for position, row in enumerate(rows):
        try:
            validated.append(RecordingRef.model_validate(row))
        except ValidationError as exc:
            raise invalid_schema(
                f"recording at position {position} is not a valid reference: {exc}"
            ) from exc
    return tuple(validated)


def _require_distinct_indices(recordings: tuple[RecordingRef, ...]) -> None:
    seen: set[int] = set()
    for one in recordings:
        if one.index in seen:
            raise invalid_schema(f"index {one.index} names two recordings")
        seen.add(one.index)


def _require_distinct_identities(recordings: tuple[RecordingRef, ...]) -> None:
    # Two archives that give a recording the same number are naming two different
    # recordings. A repeat within one namespace is an error, not something to merge.
    seen: set[tuple[str, str]] = set()
    for one in recordings:
        identity = (one.namespace, one.value)
        if identity in seen:
            raise invalid_schema(
                f"({one.namespace}, {one.value}) names two recordings in one map"
            )
        seen.add(identity)
