"""The window loop that the BirdNET and Perch adapters share."""

import re

import numpy as np
import soundfile

from robin_contracts.records import WindowOutput
from robin_contracts.specs import WindowGeometry
from robin_models import _audio_windows as audio_windows

RATE = 100


class FakeClock:
    """A clock that moves only when a test's stand-in function moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def get_seconds(line: str, stage: str) -> float:
    """Return the seconds that a log line gives for one stage."""
    found = re.search(rf"\b{stage} (\d+\.\d) s\b", line)
    assert found, line
    return float(found.group(1))


def test_read_and_model_seconds_are_counted_apart(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(audio_windows, "perf_counter", clock)
    path = tmp_path / "three_seconds.wav"
    soundfile.write(path, np.zeros(3 * RATE, dtype=np.float32), RATE)
    geometry = WindowGeometry(window_duration=1.0, window_overlap=0.0, pad="drop")

    def read_window(audio, start, end):
        clock.now += 1
        return np.zeros(RATE, dtype=np.float32)

    def run_batch(bounds, batch):
        clock.now += 10
        return iter([WindowOutput(start=start, end=end) for start, end in bounds])

    lines: list[str] = []
    windows = list(
        audio_windows.run_windows(
            path,
            geometry=geometry,
            width=RATE,
            batch_size=2,
            read_window=read_window,
            run_batch=run_batch,
            log=lines.append,
            owner="model test/1",
        )
    )

    assert len(windows) == 3
    timed = [line for line in lines if " s" in line]
    assert len(timed) == 1, lines
    assert get_seconds(timed[0], "read") == 3.0
    assert get_seconds(timed[0], "model") == 20.0
