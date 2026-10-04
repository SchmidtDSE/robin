"""What a caller hands the engine: what to run, over what, with what asked for.

`frozen=True` does not reach inside `settings` and `resources`. Mutating `work.settings`
changes the `work_digest`; mutating `work.resources` does not, because the digest
excludes it.
"""

import math
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from robin_contracts.canonical import is_sha256_bytes, is_sha256_v1, sha256_v1
from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.output_contracts import (
    EmbeddingsContractId,
    OutputRequest,
    WorkContractId,
)
from robin_contracts.protocols import JsonScalar


def _bytes_digest(value: str) -> str:
    if not is_sha256_bytes(value):
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


class InputArtifact(BaseModel, frozen=True, extra="forbid"):
    """One file a work reads as input: where it is, and what its bytes hash to."""

    uri: NonEmptyText
    checksum: BytesDigest


class RecordingRef(BaseModel, frozen=True, extra="forbid"):
    """One recording: its identity, `namespace` and `value`, and where its audio is.

    In a work over saved embeddings, each recording names its own embeddings file in
    `embeddings`; in a work over audio, none does.
    """

    namespace: NonEmptyText
    value: NonEmptyText
    audio_uri: NonEmptyText
    duration_seconds: Annotated[float, AfterValidator(_positive_duration)] | None = None
    embeddings: InputArtifact | None = None


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


class AudioInput(BaseModel, frozen=True, extra="forbid"):
    """Inference runs over the recordings' own audio."""

    kind: Literal["audio"] = "audio"


class EmbeddingArtifactInput(BaseModel, frozen=True, extra="forbid"):
    """Inference runs over embeddings files the named backbone wrote.

    There is one file per recording, named on each recording.
    """

    kind: Literal["embedding_artifact"] = "embedding_artifact"
    contract_id: EmbeddingsContractId
    backbone: ModelCard


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
        identities = [(recording.namespace, recording.value) for recording in self.recordings]
        if len(set(identities)) != len(identities):
            raise ValueError("a repeated (namespace, value) is an error, never a merge")
        return self

    @model_validator(mode="after")
    def _recordings_name_an_embeddings_file_exactly_when_the_input_is_embeddings(
        self,
    ) -> "InferenceWork":
        over_embeddings = self.input.kind == "embedding_artifact"
        for recording in self.recordings:
            if over_embeddings and recording.embeddings is None:
                raise ValueError(
                    f"recording ({recording.namespace!r}, {recording.value!r}) names no "
                    f"embeddings file, but the work runs over embeddings"
                )
            if not over_embeddings and recording.embeddings is not None:
                raise ValueError(
                    f"recording ({recording.namespace!r}, {recording.value!r}) names an "
                    f"embeddings file, but the work runs over audio"
                )
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
    """The identity of a work: everything but its resource preferences.

    Audio is identified by its uri, not its bytes, so replacing a file at the same uri
    leaves the digest unchanged.
    """
    return sha256_v1(work.model_dump(mode="json", exclude={"resources"}))


def recording_work_digest(work: InferenceWork, recording: RecordingRef) -> str:
    """The digest of `work` with `recordings` holding only a single `recording`.

    The digest is independent of the other recordings in the work.
    Raises ValueError if `recording` is not in `work.recordings`.
    """
    if recording not in work.recordings:
        raise ValueError(
            f"recording ({recording.namespace!r}, {recording.value!r}) with audio "
            f"{recording.audio_uri!r} is not one of this work's recordings"
        )
    return work_digest(work.model_copy(update={"recordings": (recording,)}))


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
