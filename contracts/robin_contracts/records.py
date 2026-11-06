"""What one pass over one window produced, as the adapter hands it over."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowOutput:
    """Scores, an embedding, or both, for one window of the recording being run.

    `scores` is one row: a 1-D array with one score for each registry entry, in
    `class_index` order. It is None when the model gives no scores.
    The window does not name its recording: one run call handles one recording.
    Shallow-frozen: the dataclass does not freeze the arrays. The engine copies the
    score row and the embedding as it accepts them, and never holds the adapter's arrays.
    """

    start: float
    end: float
    scores: "np.ndarray | None" = None
    embedding: "np.ndarray | None" = None
