"""Serializable model and head card meanings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from robin_contracts.canonical import sha256_v1
from robin_contracts.embedding_transforms import EmbeddingTransform


class ModelRef(BaseModel):
    """An immutable reference to a model card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    digest: str

    @field_validator("name", "version", "digest")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("model reference fields must be non-empty")
        return value

    @property
    def id(self) -> str:
        """The stable human-readable model identity."""
        return f"{self.name}/{self.version}"


class InferenceParam(BaseModel):
    """A model inference parameter and its declared type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type: Literal["float", "int"]


class AudioGeometry(BaseModel):
    """Audio handling facts a backbone declares."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    downmix: str | None = None
    resampler: str | None = None
    pad: str | None = None


class ModelCard(BaseModel):
    """Serializable backbone facts; parsing a card file belongs to the runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_version: str
    runtime: str
    segment_duration: float
    sample_rate: int
    min_detection_threshold: float
    weights_uri: str | None = None
    spectrogram_shape: tuple[int, int] | None = None
    audio: AudioGeometry | None = None
    backend: str | None = None
    inference_params: tuple[InferenceParam, ...] = ()
    can_emit_embeddings: bool = False
    embedding_dim: int | None = None
    taxa_registry_uri: str | None = None

    @field_validator("model_name", "model_version", "runtime")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("model card identity fields must be non-empty")
        return value

    @field_validator("segment_duration", "min_detection_threshold")
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("model card numeric fields must be finite")
        return value

    @field_validator("sample_rate")
    @classmethod
    def _positive_sample_rate(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return value

    @field_validator("embedding_dim")
    @classmethod
    def _positive_embedding_dim(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("embedding_dim must be positive when declared")
        return value


class HeadCard(BaseModel):
    """Serializable head facts and the backbone embedding it requires."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_version: str
    backbone: ModelRef
    weights_uri: str
    classes: tuple[str, ...]
    required_embedding_transform: EmbeddingTransform
    taxa_registry_uri: str

    @field_validator("model_name", "model_version", "weights_uri", "taxa_registry_uri")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("head card text fields must be non-empty")
        return value

    @field_validator("classes")
    @classmethod
    def _classes_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not label for label in value):
            raise ValueError("head card classes must be non-empty strings")
        return value


def card_digest(card: ModelCard | HeadCard) -> str:
    """Return the public canonical identity digest for a model or head card."""
    return sha256_v1(card)


def model_ref(card: ModelCard | HeadCard) -> ModelRef:
    """Build the immutable model reference for a public card."""
    return ModelRef(
        name=card.model_name,
        version=card.model_version,
        digest=card_digest(card),
    )
