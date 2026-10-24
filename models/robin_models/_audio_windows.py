"""What the BirdNET and Perch adapters share: each cuts a recording into windows, scores
them in batches, and makes one output for each window.

Each adapter keeps its own window reader and its own network call, and passes them in.
Each function takes `owner`, the name the error gives, such as "model perch/v8". Importing
this module loads NumPy and soundfile, but no ML runtime.
"""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from robin_contracts.protocols import Log
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.specs import WindowGeometry, window_bounds

Bounds = Sequence[tuple[float, float]]


def run_windows(
    path: Path,
    *,
    geometry: WindowGeometry,
    width: int,
    batch_size: int,
    read_window: Callable[[sf.SoundFile, float, float], np.ndarray],
    run_batch: Callable[[Bounds, np.ndarray], Iterator[WindowOutput]],
    log: Log,
    owner: str,
) -> Iterator[WindowOutput]:
    """Read and score the recording one batch at a time, yielding each window in order.

    `read_window(audio, start, end)` gives one window's samples at the model's rate, not
    yet padded. `run_batch(bounds, batch)` gives the outputs of one batch of padded rows.
    """
    with sf.SoundFile(str(path)) as audio:
        if audio.frames == 0:
            raise ValueError(f"{owner} cannot run {path}: it has no audio frames")
        bounds = window_bounds(audio.frames / audio.samplerate, geometry)
        log(f"running {len(bounds)} windows of {path.name}")
        for first in range(0, len(bounds), batch_size):
            batch_bounds = bounds[first : first + batch_size]
            batch = np.zeros((len(batch_bounds), width), dtype=np.float32)
            for row, (start, end) in zip(batch, batch_bounds):
                window = read_window(audio, start, end)
                # The rest of the row stays zero, which pads the window at the end.
                row[: len(window)] = window
            yield from run_batch(batch_bounds, batch)


def refuse_unusable_logits(
    logits: np.ndarray, bounds: Bounds, labels: Sequence[str], owner: str
) -> None:
    """Refuse logits not shaped (windows, labels), or not finite."""
    if logits.shape != (len(bounds), len(labels)):
        raise ValueError(
            f"{owner} gave logits of shape {logits.shape} for {len(bounds)} "
            f"windows of {len(labels)} labels"
        )
    found = np.argwhere(~np.isfinite(logits))
    if len(found):
        window, label = found[0]
        raise ValueError(
            f"{owner} gave the logit {float(logits[window, label])!r} for "
            f"{labels[label]!r} in the window at {bounds[window][0]} s"
        )


def get_window_outputs(
    bounds: Bounds,
    scores: np.ndarray | None,
    embeddings: np.ndarray | None,
    labels: Sequence[str] | None,
) -> Iterator[WindowOutput]:
    """One output for each window of a batch, in order.

    `scores` is None when there are no labels, and `embeddings` is None when none were
    asked for.
    """
    if scores is None:
        score_rows = [()] * len(bounds)
    else:
        score_rows = [_get_class_scores(labels, row) for row in scores.tolist()]
    embedding_rows = [None] * len(bounds) if embeddings is None else list(embeddings)
    # strict refuses a network that gives another number of embedding rows than windows.
    for (start, end), class_scores, embedding in zip(
        bounds, score_rows, embedding_rows, strict=True
    ):
        # Each window gets its own copy, so no two share the batch's memory.
        embedding = None if embedding is None else embedding.copy()
        yield WindowOutput(start=start, end=end, scores=class_scores, embedding=embedding)


def _get_class_scores(labels: Sequence[str], scores: list[float]) -> tuple[ClassScore, ...]:
    return tuple(ClassScore(label=label, score=score) for label, score in zip(labels, scores))
