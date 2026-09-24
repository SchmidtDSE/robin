"""What one pass over one window produced, as the adapter hands it over."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class ClassScore:
    """One label's score for one window."""

    label: str
    score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowOutput:
    """Scores, an embedding, or both, for one window of the recording being run.

    The window does not name its recording: one run call handles one recording.
    Shallow-frozen: the dataclass does not freeze the embedding array. The engine
    copies the values as it accepts them and never holds the adapter's array.
    """

    start: float
    end: float
    scores: tuple[ClassScore, ...] = ()
    embedding: "np.ndarray | None" = None
