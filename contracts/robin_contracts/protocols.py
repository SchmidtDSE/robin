"""The interface between an installed model distribution and the engine."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable

from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.inputs import Input
from robin_contracts.records import WindowOutput
from robin_contracts.registry import TaxonRegistry

JsonScalar = str | int | float | bool | None

Log = Callable[[str], None]

ScoreRetention = Literal["full", "thresholded", "top_k"]


def noop(text: str) -> None:
    """Discard a log line. The default an adapter gets when the caller wants silence."""
    return None


@runtime_checkable
class Model(Protocol):
    """What an adapter satisfies: one pass over one input, plus two lifecycle hooks."""

    def run(self, input: Input) -> Iterator[WindowOutput]: ...
    def after_recording(self) -> None: ...
    def clean_up(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ModelContext:
    """Everything an adapter is given. It acquires nothing for itself."""

    card: ModelCard | HeadCard
    registry: TaxonRegistry | None
    files: Mapping[str, Path]  # role -> verified local file
    settings: Mapping[str, JsonScalar]
    resources: Mapping[str, JsonScalar]
    scratch_dir: Path
    emit_embeddings: bool
    log: Log = noop
