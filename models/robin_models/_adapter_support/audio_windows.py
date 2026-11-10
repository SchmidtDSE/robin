"""What the BirdNET and Perch adapters share: each cuts a recording into windows, scores
them in batches, and makes one output for each window.

Each adapter keeps its own block reader, window cutter and network call, and passes them
in. Each check takes `owner`, the name the error gives, such as "model perch/v8".
Importing this module loads NumPy and soundfile, but no ML runtime.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from robin_contracts.protocols import Log
from robin_contracts.records import WindowOutput
from robin_contracts.specs import WindowGeometry, window_bounds

from robin_models._adapter_support.batch_pipeline import (
    BatchPipeline,
    BatchRunner,
    BlockReader,
    Bounds,
    WindowCutter,
    WindowResources,
)


def run_windows(
    path: Path,
    *,
    geometry: WindowGeometry,
    width: int,
    resources: WindowResources,
    read_block: BlockReader,
    cut_window: WindowCutter,
    run_batch: BatchRunner,
    log: Log,
    owner: str,
) -> Iterator[WindowOutput]:
    """Read and score the recording one batch at a time, yielding each window in order.

    `read_block(audio, bounds)` decodes the audio of a block of windows, given their
    bounds, and returns it in the adapter's own form. `cut_window(block, start, end)`
    returns one window's samples from that block at the model's rate, not yet padded.
    `run_batch(bounds, batch)` returns the outputs of one batch. Every batch has
    `resources.batch_size` rows: after the last window, the rows are zero. `bounds` holds
    only the real windows, so the adapter keeps only that many rows of the network's
    outputs.

    The batches are read on other threads, ahead of the model: see
    `batch_pipeline.BatchPipeline`. The readers stay at most
    `resources.read_ahead_batches` batches ahead of the model, plus one batch for each
    reader. No thread starts before the first window is asked for. An error in any thread
    is raised here, after the windows of the batches before it. Closing the iterator early
    stops every thread and closes every file.

    When the last window is yielded, the loop logs the seconds spent reading (summed over
    the readers), in the model, and waiting for audio. The model seconds count only the
    `run_batch` call, not the iteration over its outputs.
    """
    # Opened only to count the frames: each reader opens the file again.
    with sf.SoundFile(str(path)) as audio:
        if audio.frames == 0:
            raise ValueError(f"{owner} cannot run {path}: it has no audio frames")
        bounds = window_bounds(audio.frames / audio.samplerate, geometry)
    log(f"running {len(bounds)} windows of {path.name}")
    pipeline = BatchPipeline(
        path,
        bounds,
        width=width,
        resources=resources,
        read_block=read_block,
        cut_window=cut_window,
        run_batch=run_batch,
    )
    yield from pipeline.get_outputs()
    seconds = pipeline.seconds
    log(
        f"ran {len(bounds)} windows of {path.name}: read {sum(seconds.reads):.1f} s, "
        f"model {seconds.model:.1f} s, waited for audio {seconds.waited:.1f} s"
    )


def mix_down(frames: np.ndarray) -> np.ndarray:
    """Return one channel as it is, or the mean of several channels in float32."""
    if frames.shape[1] == 1:
        return frames[:, 0]
    return np.mean(frames, axis=1, dtype=np.float32)


def drop_padded_rows(
    rows: np.ndarray | None, batch: np.ndarray, bounds: Bounds, name: str, owner: str
) -> np.ndarray | None:
    """Refuse outputs without one row for each row of the batch, then keep the windows' rows.

    A batch ends in zero rows after its windows, and their outputs are dropped. The count
    is checked first, so a wrong count is refused even when no row is dropped.
    """
    if rows is None:
        return None
    if len(rows) != len(batch):
        raise ValueError(
            f"{owner} gave {len(rows)} rows of {name} for a batch of {len(batch)} rows"
        )
    return rows[: len(bounds)]


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
) -> Iterator[WindowOutput]:
    """Yield one output for each window of a batch, in order.

    `scores` is the batch's (windows, labels) array, or None when there are no labels.
    Each window's scores are one row of it, in registry order; the engine copies the
    row. `embeddings` is None when none were asked for.
    """
    score_rows = [None] * len(bounds) if scores is None else list(scores)
    embedding_rows = [None] * len(bounds) if embeddings is None else list(embeddings)
    # The adapters refuse a wrong row count first. strict guards any other caller.
    for (start, end), row, embedding in zip(bounds, score_rows, embedding_rows, strict=True):
        # Each window gets its own copy, so no two share the batch's memory.
        embedding = None if embedding is None else embedding.copy()
        yield WindowOutput(start=start, end=end, scores=row, embedding=embedding)
