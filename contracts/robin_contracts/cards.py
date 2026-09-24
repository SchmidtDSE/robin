"""Model and head cards, and the YAML files they are read from and written to.

A card says what a model is, not where its files are: the work pins those.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator

from robin_contracts.canonical import is_sha256_v1, sha256_v1
from robin_contracts.embedding_transforms import EmbeddingTransform


class ModelRef(BaseModel):
    """An immutable reference to a model card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    digest: str

    @field_validator("name", "version")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("model reference fields must be non-empty")
        return value

    @field_validator("digest")
    @classmethod
    def _a_card_digest(cls, value: str) -> str:
        if not is_sha256_v1(value):
            raise ValueError(f"expected 'sha256:v1:' and 64 hex characters, got {value!r}")
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
    """Serializable backbone facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_version: str
    runtime: str
    segment_duration: float
    sample_rate: int
    min_detection_threshold: float
    spectrogram_shape: tuple[int, int] | None = None
    audio: AudioGeometry | None = None
    backend: str | None = None
    inference_params: tuple[InferenceParam, ...] = ()
    can_emit_embeddings: bool = False
    embedding_dim: int | None = None

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
    classes: tuple[str, ...]
    required_embedding_transform: EmbeddingTransform

    @field_validator("model_name", "model_version")
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


# ModelCard and HeadCard each forbid extra fields and require one the other lacks, so
# exactly one of them accepts any valid card and no discriminator field is needed.
_ANY_CARD = TypeAdapter(ModelCard | HeadCard)


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a repeated key instead of keeping the last one."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while reading a mapping",
                    node.start_mark,
                    f"found the key {key!r} more than once",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def read_card(path: Path) -> ModelCard | HeadCard:
    """Read a card from a YAML file whose fields are the card's fields.

    Nothing is renamed, defaulted or resolved against the file's folder. A value YAML
    reads as a number or a bool where the card wants text is refused, not converted.
    Raises ValueError, naming the path, for a malformed or invalid card; file errors
    are raised as they are.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8 text: {error}") from error
    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{path} is not a readable card: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{path} must hold a mapping of card fields, got {type(loaded).__name__}"
        )
    try:
        return _ANY_CARD.validate_python(loaded)
    except ValidationError as error:
        raise ValueError(f"{path} is not a valid card: {error}") from error


def write_card(card: ModelCard | HeadCard, path: Path) -> None:
    """Write a card a program produced as YAML, in field order.

    The file holds the same plain values the card's digest is computed from. Comments
    are not kept, so never use this to rewrite a card someone edited by hand.
    """
    text = yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    Path(path).write_text(text, encoding="utf-8")
