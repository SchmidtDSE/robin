"""The window loop that the BirdNET and Perch adapters share."""

import re
import threading
import time
import weakref

import numpy as np
import pytest
import soundfile

from robin_contracts.records import WindowOutput
from robin_contracts.specs import WindowGeometry
from robin_models._adapter_support import audio_windows, batch_pipeline
from robin_models._adapter_support.adapter_setup import get_resources

RATE = 100


class FakeClock:
    """A clock that moves only when a test's stand-in function moves it.

    Each thread has its own time, because the window loop reads and runs the model on
    several threads at once. One thread's steps then never fall inside another's timing.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def now(self) -> float:
        return getattr(self._local, "now", 0.0)

    @now.setter
    def now(self, value: float) -> None:
        self._local.now = value

    def __call__(self) -> float:
        return self.now


def get_seconds(line: str, stage: str) -> float:
    """Return the seconds that a log line gives for one stage."""
    found = re.search(rf"\b{stage} (\d+\.\d) s\b", line)
    assert found, line
    return float(found.group(1))


GEOMETRY = WindowGeometry(window_duration=1.0, window_overlap=0.0, pad="drop")
# Long enough for any test thread to reach the point it waits for, short enough that a
# test of a missing behavior fails instead of hanging.
TIMEOUT = 5.0


def write_seconds(tmp_path, seconds: int):
    path = tmp_path / f"{seconds}_seconds.wav"
    soundfile.write(path, np.zeros(seconds * RATE, dtype=np.float32), RATE)
    return path


def read_nothing(audio, bounds):
    """A block reader that decodes nothing: `cut_silence` makes each window."""
    return None


def cut_silence(block, start, end):
    return np.zeros(RATE, dtype=np.float32)


def get_outputs_without_scores(bounds, batch):
    return iter([WindowOutput(start=start, end=end) for start, end in bounds])


def get_windows(
    path,
    *,
    batch_size=3,
    read_block=read_nothing,
    cut_window=cut_silence,
    run_batch=get_outputs_without_scores,
    log=lambda line: None,
    **resources,
):
    """Run the window loop. `resources` gives the other fields of `WindowResources`."""
    return audio_windows.run_windows(
        path,
        geometry=GEOMETRY,
        width=RATE,
        resources=batch_pipeline.WindowResources(batch_size=batch_size, **resources),
        read_block=read_block,
        cut_window=cut_window,
        run_batch=run_batch,
        log=log,
        owner="model test/1",
    )


def get_loop_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(batch_pipeline.THREAD_NAME_PREFIX) and thread.is_alive()
    ]


def take_until_raised(windows) -> tuple[list[float], BaseException]:
    starts = []
    with pytest.raises(Exception) as raised:
        for window in windows:
            starts.append(window.start)
    return starts, raised.value


def test_read_and_model_seconds_are_counted_apart(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(batch_pipeline, "perf_counter", clock)

    def cut_window(block, start, end):
        clock.now += 1
        return cut_silence(block, start, end)

    def run_batch(bounds, batch):
        clock.now += 10
        return get_outputs_without_scores(bounds, batch)

    lines: list[str] = []
    windows = list(
        get_windows(
            write_seconds(tmp_path, 3),
            batch_size=2,
            cut_window=cut_window,
            run_batch=run_batch,
            log=lines.append,
        )
    )

    assert len(windows) == 3
    timed = [line for line in lines if " s" in line]
    assert len(timed) == 1, lines
    assert get_seconds(timed[0], "read") == 3.0
    assert get_seconds(timed[0], "model") == 20.0


def test_windows_arrive_in_order_across_many_batches(tmp_path):
    windows = list(get_windows(write_seconds(tmp_path, 29), batch_size=3))

    assert [window.start for window in windows] == [float(n) for n in range(29)]
    assert not get_loop_threads()


def test_a_read_error_is_raised_after_the_windows_of_earlier_batches(tmp_path):
    failure = OSError("the disk went away")

    def cut_window(block, start, end):
        if start == 7.0:
            raise failure
        return cut_silence(block, start, end)

    windows = get_windows(write_seconds(tmp_path, 12), cut_window=cut_window)
    starts, raised = take_until_raised(windows)

    # Window 7 is in the third batch of 3, so the first two batches arrive.
    assert starts == [float(n) for n in range(6)]
    assert raised is failure
    assert not get_loop_threads()


def test_a_model_error_is_raised_after_the_windows_of_earlier_batches(tmp_path):
    failure = ValueError("the network gave no rows")

    def run_batch(bounds, batch):
        if bounds[0][0] == 6.0:
            raise failure
        return get_outputs_without_scores(bounds, batch)

    windows = get_windows(write_seconds(tmp_path, 12), run_batch=run_batch)
    starts, raised = take_until_raised(windows)

    assert starts == [float(n) for n in range(6)]
    assert raised is failure
    assert not get_loop_threads()


def test_closing_the_windows_early_stops_every_thread_and_closes_the_file(tmp_path):
    opened = []

    def read_block(audio, bounds):
        opened.append(audio)
        return None

    windows = get_windows(write_seconds(tmp_path, 30), read_block=read_block)
    next(windows)
    windows.close()

    assert not get_loop_threads()
    # Each reader has its own handle, and every one is closed.
    assert opened and all(audio.closed for audio in opened)


@pytest.mark.parametrize("read_ahead_batches", [1, 4])
def test_the_reader_stays_a_bounded_number_of_batches_ahead(tmp_path, read_ahead_batches):
    batch_size = 2
    # The batches read ahead, one being read by each reader, and one the model holds.
    read_threads = 2
    bound = (read_ahead_batches + read_threads + 1) * batch_size
    reads = []
    release = threading.Event()

    def cut_window(block, start, end):
        reads.append(start)
        return cut_silence(block, start, end)

    def run_batch(bounds, batch):
        release.wait(TIMEOUT)
        return get_outputs_without_scores(bounds, batch)

    windows = get_windows(
        write_seconds(tmp_path, 60),
        batch_size=batch_size,
        read_ahead_batches=read_ahead_batches,
        read_threads=read_threads,
        cut_window=cut_window,
        run_batch=run_batch,
    )
    consumer = threading.Thread(target=lambda: next(windows, None))
    consumer.start()
    # Wait until the readers stop: they are at their bound, or they read only as needed.
    deadline = time.monotonic() + TIMEOUT
    seen = -1
    while time.monotonic() < deadline and len(reads) != seen:
        seen = len(reads)
        time.sleep(0.2)
    stalled = len(reads)
    release.set()
    consumer.join(TIMEOUT)
    windows.close()

    # More than one batch shows the readers run ahead of the model; the bound keeps them close.
    assert batch_size < stalled <= bound


def test_the_next_batch_is_read_while_the_model_runs(tmp_path):
    batch_size = 3
    next_batch_started = threading.Event()
    waited = []

    def read_block(audio, bounds):
        if bounds[0][0] == float(batch_size):
            next_batch_started.set()
        return None

    def run_batch(bounds, batch):
        if bounds[0][0] == 0.0:
            waited.append(next_batch_started.wait(TIMEOUT))
        return get_outputs_without_scores(bounds, batch)

    windows = list(
        get_windows(
            write_seconds(tmp_path, 9),
            batch_size=batch_size,
            read_block=read_block,
            run_batch=run_batch,
        )
    )

    assert len(windows) == 9
    assert waited == [True]


def test_each_block_is_decoded_once(tmp_path):
    # 1 s windows in 2 s blocks: each batch of 4 is two blocks of 2.
    blocks = []
    cut = []

    def read_block(audio, bounds):
        # The block's bounds are its identity, so readers on two threads cannot share one.
        block = tuple(bounds)
        blocks.append(block)
        return block

    def cut_window(block, start, end):
        cut.append((block, start, end))
        return cut_silence(block, start, end)

    windows = list(
        get_windows(
            write_seconds(tmp_path, 8),
            batch_size=4,
            read_block_seconds=2.0,
            read_block=read_block,
            cut_window=cut_window,
        )
    )

    assert len(windows) == 8
    expected = [((2.0 * n, 2.0 * n + 1.0), (2.0 * n + 1.0, 2.0 * n + 2.0)) for n in range(4)]
    assert sorted(blocks) == expected
    # Each window is cut once, from the block that holds it.
    assert sorted(cut) == sorted((block, *window) for block in expected for window in block)


def test_a_block_holds_one_window_even_if_it_is_longer_than_a_block(tmp_path):
    blocks = []

    def read_block(audio, bounds):
        blocks.append(tuple(bounds))
        return None

    list(get_windows(write_seconds(tmp_path, 3), read_block_seconds=0.5, read_block=read_block))

    assert blocks == [((0.0, 1.0),), ((1.0, 2.0),), ((2.0, 3.0),)]


def test_every_batch_has_the_full_size(tmp_path):
    given = []

    def cut_window(block, start, end):
        return np.ones(RATE, dtype=np.float32)

    def run_batch(bounds, batch):
        given.append((list(bounds), batch.copy()))
        return get_outputs_without_scores(bounds, batch)

    windows = list(
        get_windows(
            write_seconds(tmp_path, 5), batch_size=4, cut_window=cut_window, run_batch=run_batch
        )
    )

    assert len(windows) == 5
    assert [batch.shape for _, batch in given] == [(4, RATE), (4, RATE)]
    assert [len(bounds) for bounds, _ in given] == [4, 1]
    last = given[1][1]
    assert np.array_equal(last[0], np.ones(RATE, dtype=np.float32))
    assert not last[1:].any()


class TrackedNumpy:
    """NumPy for the window loop, except that it keeps a weak reference to each array
    made with `zeros`. The loop makes each batch's input array with `zeros`."""

    def __init__(self) -> None:
        self.arrays: list[weakref.ref] = []

    def __getattr__(self, name: str):
        return getattr(np, name)

    def zeros(self, *args, **kwargs) -> np.ndarray:
        array = np.zeros(*args, **kwargs)
        self.arrays.append(weakref.ref(array))
        return array


def test_a_batch_the_model_has_finished_is_not_kept_alive(tmp_path, monkeypatch):
    tracked = TrackedNumpy()
    monkeypatch.setattr(batch_pipeline, "np", tracked)
    first_batch_given = threading.Event()
    first_batch_alive = []

    def cut_window(block, start, end):
        if start == 2.0:
            # Batch 1 is being read, and the caller has every window of batch 0. So the
            # model has finished batch 0, and nothing needs its input array.
            first_batch_given.wait(TIMEOUT)
            first_batch_alive.append(tracked.arrays[0]() is not None)
        return cut_silence(block, start, end)

    starts = []
    # One reader, so the input arrays are made in batch order.
    windows = get_windows(
        write_seconds(tmp_path, 6), batch_size=2, cut_window=cut_window, read_threads=1
    )
    for window in windows:
        starts.append(window.start)
        if window.start == 1.0:
            first_batch_given.set()

    assert starts == [float(n) for n in range(6)]
    assert first_batch_alive == [False]


def test_a_batch_read_while_the_loop_stops_is_not_kept(tmp_path, monkeypatch):
    tracked = TrackedNumpy()
    monkeypatch.setattr(batch_pipeline, "np", tracked)
    pipelines = []

    class KeptPipeline(batch_pipeline.BatchPipeline):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            pipelines.append(self)

    monkeypatch.setattr(audio_windows, "BatchPipeline", KeptPipeline)

    def read_block(audio, bounds):
        if bounds[0][0] >= 2.0:
            # The reader finishes this batch only after the loop has started to stop.
            pipelines[0]._stop.wait(TIMEOUT)
            time.sleep(0.2)
        return None

    windows = get_windows(
        write_seconds(tmp_path, 6), batch_size=2, read_block=read_block, read_threads=1
    )
    next(windows)
    windows.close()

    # The pipeline is still referenced here, as a traceback can keep it.
    assert pipelines
    assert [array() is None for array in tracked.arrays] == [True] * len(tracked.arrays)


def test_a_thread_that_fails_to_start_stops_the_threads_already_started(tmp_path, monkeypatch):
    failure = RuntimeError("can't start new thread")
    start = threading.Thread.start

    def start_or_fail(thread):
        if thread.name == f"{batch_pipeline.THREAD_NAME_PREFIX}model":
            raise failure
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_or_fail)
    opened = []

    def read_block(audio, bounds):
        opened.append(audio)
        return None

    windows = get_windows(write_seconds(tmp_path, 30), read_block=read_block)
    with pytest.raises(RuntimeError) as raised:
        next(windows)

    assert raised.value is failure
    assert not get_loop_threads()
    assert all(audio.closed for audio in opened)


def test_a_whole_number_of_block_seconds_is_accepted():
    resources = {"batch_size": 8, "read_block_seconds": 30}
    got = get_resources(batch_pipeline.WindowResources, resources, "model test/1")
    assert got.read_block_seconds == 30.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("read_ahead_batches", 0),
        ("read_ahead_batches", 4.0),
        ("read_block_seconds", 0),
        ("read_block_seconds", float("nan")),
        ("read_block_seconds", float("inf")),
        ("read_threads", 0),
        ("read_threads", 2.0),
    ],
    ids=repr,
)
def test_a_read_setting_that_is_not_a_positive_number_is_refused(name, value):
    resources = {"batch_size": 8, name: value}
    with pytest.raises(ValueError, match=name):
        get_resources(batch_pipeline.WindowResources, resources, "model test/1")


@pytest.mark.parametrize("read_threads", [1, 4])
def test_the_loop_reads_on_the_given_number_of_threads_with_the_same_windows(
    tmp_path, read_threads
):
    readers = set()

    def run_batch(bounds, batch):
        prefix = f"{batch_pipeline.THREAD_NAME_PREFIX}reader-"
        readers.update(t.name for t in get_loop_threads() if t.name.startswith(prefix))
        return get_outputs_without_scores(bounds, batch)

    windows = get_windows(
        write_seconds(tmp_path, 30), batch_size=2, run_batch=run_batch, read_threads=read_threads
    )

    assert [window.start for window in windows] == [float(n) for n in range(30)]
    assert len(readers) == read_threads


@pytest.mark.parametrize(
    ("cpu_max", "cpus", "expected"),
    [
        pytest.param(None, 4, 2, id="no_cgroup_file"),
        pytest.param("max 100000\n", 8, 6, id="no_limit"),
        pytest.param("max 100000\n", 32, 30, id="no_cap"),
        pytest.param("200000 100000\n", 64, 1, id="two_cpu_limit_on_a_large_host"),
        pytest.param("650000 100000\n", 64, 4, id="part_of_a_cpu_is_not_counted"),
        pytest.param("800000 100000\n", 4, 2, id="limit_above_the_hosts_cpus"),
        pytest.param(None, None, 1, id="cpu_count_unknown"),
    ],
)
def test_the_default_read_threads_leave_two_cpus_free(
    tmp_path, monkeypatch, cpu_max, cpus, expected
):
    cpu_max_file = tmp_path / "cpu.max"
    if cpu_max is not None:
        cpu_max_file.write_text(cpu_max)
    monkeypatch.setattr(batch_pipeline, "CPU_MAX_FILE", cpu_max_file)
    monkeypatch.setattr(batch_pipeline.os, "cpu_count", lambda: cpus)

    assert batch_pipeline.WindowResources(batch_size=8).read_threads == expected
