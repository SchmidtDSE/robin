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
    native_top_k: int | None = None
    embedding_dim: int | None = None
    embedding_dtype: EmbeddingDtype | None = None

    def __post_init__(self) -> None:
        cap = self.native_top_k
        # bool is an int subclass, and a cap of True is a type error, not a k of one.
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError(f"native_top_k must be a positive int or None, got {cap!r}")
        if (cap is not None) != ("top_k" in self.supported_retention):
            raise ValueError(
                f"native_top_k={cap!r} must be set exactly when supported_retention "
                f"includes 'top_k', got {sorted(self.supported_retention)}"
            )
        if cap is not None and self.supported_retention != frozenset({"top_k"}):
            raise ValueError(
                f"native_top_k={cap!r} requires supported_retention to be only "
                f"'top_k', got {sorted(self.supported_retention)}"
            )


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
    files: Mapping[str, Path]  # role -> verified local file, the registry's included
    settings: Mapping[str, JsonScalar]
    scratch_dir: Path
    emit_embeddings: bool
    log: Log = noop
