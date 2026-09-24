"""What a caller hands the engine: what to run, over what, with what asked for.

`frozen=True` does not reach inside `settings` and `resources`. Mutating `work.settings`
changes the `work_digest`; mutating `work.resources` does not, because the digest
excludes it.
"""

import math
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from robin_contracts.canonical import is_sha256_v1, sha256_v1
from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.output_contracts import (
    EmbeddingsContractId,
    OutputRequest,
    WorkContractId,
)
from robin_contracts.protocols import JsonScalar

# `sha256:` hashes file bytes, `sha256:v1:` hashes canonical JSON, so a digest of one
# kind never matches one of the other. Strip the label where a path needs bare hex.
_BYTES_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _bytes_digest(value: str) -> str:
    if not _BYTES_DIGEST.fullmatch(value):
        raise ValueError(f"expected 'sha256:' and 64 hex characters, got {value!r}")
    return value


def _canonical_digest(value: str) -> str:
    if not is_sha256_v1(value):
        raise ValueError(f"expected 'sha256:v1:' and 64 hex characters, got {value!r}")
    return value


def _non_empty(value: str) -> str:
    if not value:
        raise ValueError("this field must be non-empty")
    return value


def _positive_duration(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"a duration must be finite and positive, got {value}")
    return value


BytesDigest = Annotated[str, AfterValidator(_bytes_digest)]
CanonicalDigest = Annotated[str, AfterValidator(_canonical_digest)]
NonEmptyText = Annotated[str, AfterValidator(_non_empty)]


# The role of the pinned file that holds the label registry.
REGISTRY_ROLE = "taxa_registry"


class RecordingId(BaseModel, frozen=True, extra="forbid"):
    """A recording's identity: its archive, and its identifier within that archive."""

    namespace: NonEmptyText
    value: NonEmptyText

    def __str__(self) -> str:
        # Quoted as a pair: any separator could also appear inside a value.
        return f"({self.namespace!r}, {self.value!r})"


class RecordingRef(BaseModel, frozen=True, extra="forbid"):
    """One recording: its identity, and where its audio is."""

    namespace: NonEmptyText
    value: NonEmptyText
    audio_uri: NonEmptyText
    audio_digest: BytesDigest | None = None
    duration_seconds: Annotated[float, AfterValidator(_positive_duration)] | None = None

    @property
    def id(self) -> RecordingId:
        """This recording's identity, as results and the engine name it."""
        return RecordingId(namespace=self.namespace, value=self.value)


class PinnedFile(BaseModel, frozen=True, extra="forbid"):
    """One file this work pins by content: where it is, and what it hashes to."""

    uri: NonEmptyText
    digest: BytesDigest
    size_bytes: Annotated[int, Field(ge=0)]


class PinnedModel(BaseModel, frozen=True, extra="forbid"):
    """The exact model this work runs: its card, and its files keyed by role.

    `frozen=True` does not reach inside `files`, as it does not for a work's `settings`.
    """

    card: ModelCard | HeadCard
    files: Mapping[NonEmptyText, PinnedFile]
    registry_fingerprint: BytesDigest | None = None


class AudioInput(BaseModel, frozen=True, extra="forbid"):
    """Inference runs over the recordings' own audio."""

    kind: Literal["audio"] = "audio"


class EmbeddingArtifactInput(BaseModel, frozen=True, extra="forbid"):
    """Inference runs over an embedding artifact some earlier work produced."""

    kind: Literal["embedding_artifact"] = "embedding_artifact"
    contract_id: EmbeddingsContractId
    uri: NonEmptyText
    checksum: BytesDigest


class InferenceWork(BaseModel, frozen=True, extra="forbid"):
    """One scientific claim about what is to be computed.

    It carries no run id, attempt, actor, purpose, schedule, deployment placement,
    batch index or output location: a local script and a worker construct the identical
    work, and the caller owns placement by constructing the writer it wants.
    """

    schema_version: WorkContractId
    recordings: tuple[RecordingRef, ...]
    model: PinnedModel
    input: Annotated[AudioInput | EmbeddingArtifactInput, Field(discriminator="kind")]
    settings: Mapping[str, JsonScalar]
    resources: Mapping[str, JsonScalar]
    outputs: tuple[OutputRequest, ...]

    @field_validator("settings", "resources")
    @classmethod
    def _values_have_a_canonical_encoding(
        cls, value: Mapping[str, JsonScalar]
    ) -> Mapping[str, JsonScalar]:
        # Without this a caller can build a work whose own work_digest raises.
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"{key!r} is {item}, which has no canonical encoding")
        return value

    @model_validator(mode="after")
    def _recordings_are_identified_once_each(self) -> "InferenceWork":
        if not self.recordings:
            raise ValueError("a work must name at least one recording")
        identities = [recording.id for recording in self.recordings]
        if len(set(identities)) != len(identities):
            raise ValueError("a repeated (namespace, value) is an error, never a merge")
        return self

    @model_validator(mode="after")
    def _outputs_are_one_per_kind_and_satisfiable(self) -> "InferenceWork":
        if not self.outputs:
            raise ValueError("a work must request at least one output")
        kinds = [output.kind for output in self.outputs]
        if len(set(kinds)) != len(kinds):
            raise ValueError("a work requests each output kind at most once")
        if "detections" in kinds and "scores" not in kinds:
            raise ValueError("a detections request needs a scores request beside it")
        return self


def work_digest(work: InferenceWork) -> str:
    """The scientific identity of a work: everything but its resource preferences."""
    return sha256_v1(work.model_dump(mode="json", exclude={"resources"}))


def partition(count: int, batch_size: int) -> tuple[tuple[int, ...], ...]:
    """Dense, ordered batches of recording positions covering every position exactly once."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    return tuple(
        tuple(range(start, min(start + batch_size, count)))
        for start in range(0, count, batch_size)
    )
