"""Model and head cards, and the YAML files they are read from and written to.

A card says what a model is, not where its files are: the work pins those.
"""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from robin_contracts.canonical import is_sha256_bytes, is_sha256_v1, sha256_v1
from robin_contracts.embedding_transforms import EmbeddingTransform

PadPolicy = Literal["centre_crop_end_pad", "drop", "time_scaled"]

EmbeddingDtype = Literal["float16", "float32"]


def refuse_windows_that_do_not_advance(duration: float, overlap: float) -> None:
    """Refuse an overlap so long that the next window would not start after this one."""
    if overlap >= duration:
        raise ValueError(
            f"window_overlap {overlap} must be less than window_duration {duration}"
        )


class RunnerResampled(BaseModel, frozen=True):
    """This repository resampled the audio, and chose how."""
    by: Literal["runner"] = "runner"
    algorithm: Literal["soxr_hq", "librosa"]


class BackendResampled(BaseModel, frozen=True):
    """The model library resampled inside its own call."""
    by: Literal["backend"] = "backend"
    library: str
    version: str


Resampling = Annotated[RunnerResampled | BackendResampled, Field(discriminator="by")]


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
    """How a backbone's audio is reduced to one channel, resampled and padded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    downmix: Literal["mean", "first"]
    resampler: Resampling
    pad: PadPolicy


class ModelCard(BaseModel):
    """Serializable backbone facts, including every fact its recipe is built from.

    `min_detection_threshold` is the lowest score the model emits. At or below the
    score domain's minimum, the model emits every label for every window.
    `taxa_registry_digest` is the digest of the registry file the model's scores are
    labelled by, and a card that emits scores must declare it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_version: str
    runtime: str
    window_duration: float = Field(gt=0, allow_inf_nan=False)
    window_overlap: float = Field(ge=0, allow_inf_nan=False)  # seconds shared with the next
    sample_rate: int
    min_detection_threshold: float
    score_domain: Literal["probability"] | None  # None: the model emits no scores
    taxa_registry_digest: str | None = None  # sha256 of the registry file that labels the scores
    spectrogram_shape: tuple[int, int] | None = None
    audio: AudioGeometry
    backend: str
    embedding_transform: EmbeddingTransform
    dtype: EmbeddingDtype  # the precision embeddings are stored at
    inference_params: tuple[InferenceParam, ...] = ()
    can_emit_embeddings: bool = False
    embedding_dim: int | None = None
    embedding_dtype: EmbeddingDtype | None = None  # the precision the model emits

    @field_validator("model_name", "model_version", "runtime")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("model card identity fields must be non-empty")
        return value

    @field_validator("min_detection_threshold")
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

    @field_validator("taxa_registry_digest")
    @classmethod
    def _a_file_digest(cls, value: str | None) -> str | None:
        if value is not None and not is_sha256_bytes(value):
            raise ValueError(f"expected 'sha256:' and 64 hex characters, got {value!r}")
        return value

    @field_validator("embedding_dim")
    @classmethod
    def _positive_embedding_dim(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("embedding_dim must be positive when declared")
        return value

    @model_validator(mode="after")
    def _windows_advance(self) -> "ModelCard":
        refuse_windows_that_do_not_advance(self.window_duration, self.window_overlap)
        return self

    @model_validator(mode="after")
    def _registry_is_pinned_exactly_when_scores_are_emitted(self) -> "ModelCard":
        if self.score_domain is not None and self.taxa_registry_digest is None:
            raise ValueError("a card that emits scores must declare taxa_registry_digest")
        if self.score_domain is None and self.taxa_registry_digest is not None:
            raise ValueError("a card that emits no scores must not declare taxa_registry_digest")
        return self

    @model_validator(mode="after")
    def _embeddings_are_described_exactly_when_emitted(self) -> "ModelCard":
        if self.can_emit_embeddings:
            if self.embedding_dim is None or self.embedding_dtype is None:
                raise ValueError(
                    "a card with can_emit_embeddings true must declare embedding_dim "
                    "and embedding_dtype"
                )
        elif self.embedding_dim is not None or self.embedding_dtype is not None:
            raise ValueError(
                "a card with can_emit_embeddings false must declare neither embedding_dim "
                "nor embedding_dtype"
            )
        return self


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
