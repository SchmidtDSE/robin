"""The threads of the window loop: see `BatchPipeline`.

Importing this module loads NumPy and soundfile, but no ML runtime.
"""

import math
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from pydantic import Field, PositiveInt
from robin_contracts.records import WindowOutput

from robin_models._adapter_support.adapter_setup import Resources

Bounds = Sequence[tuple[float, float]]
# The callbacks that an adapter passes to `audio_windows.run_windows`, which describes them.
BlockReader = Callable[[sf.SoundFile, Bounds], object]
WindowCutter = Callable[[object, float, float], np.ndarray]
BatchRunner = Callable[[Bounds, np.ndarray], Iterator[WindowOutput]]
# A batch that a reader has read: its windows' bounds and its padded rows.
_ReadBatch = tuple[Bounds, np.ndarray]

# Each reader reads one batch at a time, with its own file handle. Two CPUs stay free for
# the model thread and the engine.
READERS = max(1, min(3, (os.cpu_count() or 1) - 2))

# Every thread of the window loop has a name with this prefix.
THREAD_NAME_PREFIX = "robin-windows-"


class WindowResources(Resources):
    """The resources of an adapter that runs the window loop.

    `read_ahead_batches` bounds the batches read but not yet given to the model, and so
    the memory that the readers hold. The readers can also hold one batch each. A reader
    decodes the windows of its batch in blocks of at most `read_block_seconds`. A whole
    number of seconds, such as 60, is accepted.
    """

    read_ahead_batches: PositiveInt = 4
    read_block_seconds: float = Field(60.0, gt=0, allow_inf_nan=False)


def _get_blocks(bounds: Bounds, block_seconds: float) -> list[Bounds]:
    """Group a batch's windows, in order, into blocks to decode at once.

    A block spans from its first window's start to its last window's end, at most
    `block_seconds`, and it always holds at least one window. So no window crosses the
    edge of a block.
    """
    blocks: list[Bounds] = []
    current: list[tuple[float, float]] = []
    for start, end in bounds:
        # The window starts a new block if the current block would span too long with it.
        if current and end - current[0][0] > block_seconds:
            blocks.append(current)
            current = []
        current.append((start, end))
    if current:
        blocks.append(current)
    return blocks


@dataclass
class _Seconds:
    """The seconds spent in each part of the window loop. Each thread adds to its own."""

    reads: list[float]
    model: float = 0.0
    waited: float = 0.0


@dataclass(frozen=True)
class _Failure:
    """An exception that a thread passes through its queue in place of a batch."""

    error: BaseException


# Put in the output queue after the last batch.
_END = object()

# How often a thread that waits checks whether the loop has stopped.
_POLL_SECONDS = 0.1


class BatchPipeline:
    """Read one recording's batches on reader threads, and run the model on them on one
    model thread.

    Each of up to `READERS` readers takes the next batch number, decodes that batch, and
    puts it in `_ready`. The model thread runs the batches in number order: a batch that
    arrives before its turn waits until the batches before it have run. `get_outputs`
    yields the outputs in window order. An error in any thread goes through the queues in
    place of its batch, so it is raised after the windows of the batches before it.
    `seconds` holds the time that each thread spent.
    """

    def __init__(
        self,
        path: Path,
        bounds: Bounds,
        *,
        width: int,
        resources: WindowResources,
        read_block: BlockReader,
        cut_window: WindowCutter,
        run_batch: BatchRunner,
    ) -> None:
        self._path = path
        self._bounds = bounds
        self._width = width
        self._batch_size = resources.batch_size
        self._block_seconds = resources.read_block_seconds
        self._read_block = read_block
        self._cut_window = cut_window
        self._run_batch = run_batch
        self.seconds = _Seconds(reads=[0.0] * READERS)
        self._batch_count = math.ceil(len(bounds) / self._batch_size)
        # Each batch number is taken by one reader, in order.
        self._numbers: queue.SimpleQueue[int] = queue.SimpleQueue()
        for number in range(self._batch_count):
            self._numbers.put(number)
        # A reader takes a slot before it takes a batch, and the model gives the slot back
        # when it takes that batch. So the batches read but not yet taken stay bounded.
        self._slots = threading.Semaphore(resources.read_ahead_batches + READERS)
        # Each item is a batch number and its batch, or the error that its reader raised.
        self._ready: queue.Queue[tuple[int, _ReadBatch | _Failure]] = queue.Queue()
        # Each item is the outputs of one batch, the error that stopped the loop, or _END.
        self._outputs: queue.Queue[object] = queue.Queue(maxsize=resources.read_ahead_batches)
        self._stop = threading.Event()

    def get_outputs(self) -> Iterator[WindowOutput]:
        """Start the threads, and yield the outputs."""
        started: list[threading.Thread] = []
        try:
            # Inside the try, so that if one thread fails to start, those already started
            # are stopped and joined.
            for thread in self._create_threads():
                thread.start()
                started.append(thread)
            while True:
                got = self._outputs.get()
                if got is _END:
                    return
                if isinstance(got, _Failure):
                    raise got.error
                yield from got
        finally:
            # Also reached when the caller closes the iterator early. The threads stop at
            # their next step, and each reader closes its own file. The queues are emptied
            # after the joins, because a reader can still put a batch while it stops.
            self._stop.set()
            for thread in started:
                thread.join()
            self._empty_queues()

    def _create_threads(self) -> list[threading.Thread]:
        """Return the reader threads and the model thread, not yet started."""
        readers = [
            threading.Thread(
                target=self._read_batches,
                args=(position,),
                name=f"{THREAD_NAME_PREFIX}reader-{position}",
                daemon=True,
            )
            for position in range(READERS)
        ]
        model = threading.Thread(
            target=self._score_batches, name=f"{THREAD_NAME_PREFIX}model", daemon=True
        )
        return [*readers, model]

    def _empty_queues(self) -> None:
        """Remove every item from the queues, so that no batch is held after the loop stops."""
        for waiting in (self._ready, self._outputs):
            while True:
                try:
                    waiting.get_nowait()
                except queue.Empty:
                    break

    def _read_batches(self, position: int) -> None:
        """Read batches until none is left or the loop stops.

        The reader opens its own handle on the file when it takes its first batch, so an
        error in opening the file takes that batch's place. `position` is this reader's
        place in `_Seconds.reads`.
        """
        audio: sf.SoundFile | None = None
        try:
            while (number := self._take_batch_number()) is not None:
                try:
                    if audio is None:
                        audio = sf.SoundFile(str(self._path))
                    batch = self._read_batch(audio, number, position)
                except BaseException as error:
                    self._ready.put((number, _Failure(error)))
                    return
                if batch is None:
                    return
                self._ready.put((number, batch))
                # The model owns the batch now. Holding it while the next one is read
                # would keep its array alive outside the bound of the slots.
                del batch
        finally:
            if audio is not None:
                audio.close()

    def _read_batch(self, audio: sf.SoundFile, number: int, position: int) -> _ReadBatch | None:
        """Read batch `number` into a padded array of full size, and add the seconds it took
        to the reader at `position`.

        Returns None if the loop stopped first.
        """
        started = perf_counter()
        first = number * self._batch_size
        batch_bounds = self._bounds[first : first + self._batch_size]
        rows = np.zeros((self._batch_size, self._width), dtype=np.float32)
        row = 0
        for block_bounds in _get_blocks(batch_bounds, self._block_seconds):
            if self._stop.is_set():
                return None
            block = self._read_block(audio, block_bounds)
            for start, end in block_bounds:
                window = self._cut_window(block, start, end)
                # The rest of the row stays zero, which pads the window at the end.
                rows[row, : len(window)] = window
                row += 1
        self.seconds.reads[position] += perf_counter() - started
        return batch_bounds, rows

    def _score_batches(self) -> None:
        """Run the model on each batch in turn, and put its outputs in `_outputs`."""
        # The batches that readers have finished and the model has not yet taken, by
        # number. Local, not an attribute, so that no batch in it is kept after the loop
        # stops.
        arrived: dict[int, _ReadBatch | _Failure] = {}
        try:
            for number in range(self._batch_count):
                batch = self._wait_for_batch(number, arrived)
                if batch is None:
                    return
                self._slots.release()
                if isinstance(batch, _Failure):
                    self._put_output(batch)
                    return
                outputs = self._score_batch(batch)
                # Dropped before the wait for the next batch, so that its array is freed.
                del batch
                if not self._put_output(outputs):
                    return
            self._put_output(_END)
        except BaseException as error:
            self._put_output(_Failure(error))

    def _wait_for_batch(
        self, number: int, arrived: dict[int, _ReadBatch | _Failure]
    ) -> _ReadBatch | _Failure | None:
        """Wait until batch `number` has arrived, then remove it from `arrived` and return it.

        Adds the seconds spent waiting. Returns None if the loop stopped first.
        """
        started = perf_counter()
        while number not in arrived:
            got = self._get_next_read_batch()
            if got is None:
                return None
            got_number, got_batch = got
            arrived[got_number] = got_batch
        self.seconds.waited += perf_counter() - started
        return arrived.pop(number)

    def _score_batch(self, batch: _ReadBatch) -> list[WindowOutput]:
        """Run the model on one batch, and return its outputs as a list.

        A list, so an error in making the outputs is raised here, before any window of this
        batch reaches the caller.
        """
        bounds, rows = batch
        started = perf_counter()
        produced = self._run_batch(bounds, rows)
        self.seconds.model += perf_counter() - started
        return list(produced)

    def _take_batch_number(self) -> int | None:
        """Take a slot when one is free, then the next batch number.

        Returns None, with no slot held, if no batch is left or the loop stopped first.
        """
        while not self._stop.is_set():
            if self._slots.acquire(timeout=_POLL_SECONDS):
                try:
                    return self._numbers.get_nowait()
                except queue.Empty:
                    self._slots.release()
                    return None
        return None

    def _put_output(self, item: object) -> bool:
        """Put `item` in `outputs` when there is room. Returns False if the loop stopped first."""
        while not self._stop.is_set():
            try:
                self._outputs.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _get_next_read_batch(self) -> tuple[int, _ReadBatch | _Failure] | None:
        """Take the next batch that a reader has read. Returns None if the loop stopped first."""
        while not self._stop.is_set():
            try:
                return self._ready.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue
        return None
