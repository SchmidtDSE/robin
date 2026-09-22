"""The interface between an installed model distribution and the engine."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable

from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.inputs import Input
from robin_contracts.records import WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.specs import Recipe

JsonScalar = str | int | float | bool | None

Log = Callable[[str], None]

ScoreRetention = Literal["full", "thresholded", "top_k"]

EmbeddingDtype = Literal["float16", "float32"]


def noop(text: str) -> None:
    """Discard a log line. The default an adapter gets when the caller wants silence."""
    return None


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What this configured instance can emit — not what its class or card can."""

    emits_scores: bool
    emits_embeddings: bool
    score_domain: Literal["probability"] | None = None
    supported_retention: frozenset[ScoreRetention] = frozenset()
    native_score_floor: float | None = None
    embedding_dim: int | None = None
    embedding_dtype: EmbeddingDtype | None = None
    vocabulary: tuple[str, ...] | None = None


@runtime_checkable
class Model(Protocol):
    """What an adapter satisfies: one pass over one input, plus two lifecycle hooks."""

    recipe: Recipe
    capabilities: ModelCapabilities

    def run(self, input: Input) -> Iterator[WindowOutput]: ...
    def after_recording(self) -> None: ...
    def clean_up(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ModelContext:
    """Everything an adapter is given. It acquires nothing for itself."""

    card: ModelCard | HeadCard
    registry: TaxonRegistry | None
    weights: Mapping[str, Path]
    settings: Mapping[str, JsonScalar]
    scratch_dir: Path
    emit_embeddings: bool
    log: Log = noop
