"""The OWL adapter: its bundled card and registry, and the adapter against a fake runtime
on real audio files."""

import contextlib
import csv
import hashlib
import importlib
import os
import sys
import threading
import types
from collections import Counter
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pytest
import soundfile

import robin_models.owl
from robin_contracts.cards import ModelCard, read_card
from robin_contracts.inputs import AudioClip, Embeddings
from robin_contracts.protocols import Model, ModelContext
from robin_contracts.specs import recipe, window_bounds
from robin_inference_engine.load_registry import load_registry

RESOURCES = files("robin_models.owl") / "resources"

with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)
    with _path.open(newline="", encoding="utf-8") as _file:
        REGISTRY_ROWS = list(csv.DictReader(_file))

GEOMETRY = recipe(CARD, {}).audio.geometry
AUDIO = CARD.audio.model_dump()
RATES = (8000, 16000, 22050, 32000, 44100, 48000, 96000)


# The bundled registry and card.


def test_the_registry_loads_with_51_entries_of_two_kinds():
    kinds = Counter(entry.label_kind for entry in REGISTRY.entries)
    assert len(REGISTRY.entries) == 51
    assert kinds == {"taxon": 44, "non_taxonomic": 7}
    assert kinds["unresolved"] == 0


def test_every_rows_label_kind_follows_from_its_kind():
    def expected(kind: str) -> str:
        return {"non_taxon": "non_taxonomic", "unresolved": "unresolved"}.get(kind, "taxon")

    for row in REGISTRY_ROWS:
        assert row["label_kind"] == expected(row["kind"]), row


def test_the_genus_rows_are_taxa_with_a_key():
    genus_labels = {row["label"] for row in REGISTRY_ROWS if row["kind"] == "genus"}
    assert genus_labels == {"POEC", "SITT", "SPRU", "TAMI", "WHIS"}
    for label in genus_labels:
        entry = REGISTRY.by_label(label)
        assert entry.label_kind == "taxon"
        assert entry.gbif_taxon_key is not None


def test_the_non_taxonomic_rows_have_no_name_and_no_key():
    rows = [entry for entry in REGISTRY.entries if entry.label_kind == "non_taxonomic"]
    assert len(rows) == 7
    for entry in rows:
        assert entry.scientific_name is None
        assert entry.gbif_taxon_key is None


def test_the_card_names_the_bundled_registry_by_its_digest():
    data = (RESOURCES / "taxa_registry.csv").read_bytes()

    assert CARD.taxa_registry_digest == "sha256:" + hashlib.sha256(data).hexdigest()


# What the adapter built OWL's recipe from before its card stated every fact itself. It
# gave a hop of 12 s, which is 12 s windows that do not overlap.
RECIPE_BEFORE_THE_CARD_STATED_IT = {
    "audio": {
        "downmix": "first",
        "pad": "drop",
        "resampler": {"algorithm": "soxr_hq", "by": "runner"},
        "sample_rate": 8000,
        "window_duration": 12.0,
        "window_overlap": 0.0,
    },
    "backend": "h5-fp32",
    "dtype": "float32",
    "settings": {},
    "version": 1,
}


def test_the_recipe_the_card_states_is_the_one_the_adapter_built():
    stated = recipe(CARD, {}).model_dump(mode="json")

    assert stated.pop("model") == {
        "name": "owl",
        "version": "v4",
        "digest": recipe(CARD, {}).model.digest,
    }
    assert stated == RECIPE_BEFORE_THE_CARD_STATED_IT


# OWL's window bounds, at the durations soundfile reports.


def test_a_day_of_windows_starts_on_exact_multiples_of_the_window():
    bounds = window_bounds(24 * 3600.0, GEOMETRY)
    assert len(bounds) == 7200
    for i, (start, end) in enumerate(bounds):
        assert start == i * 12.0
        assert end - start == 12.0
        assert start % 12.0 == 0.0


def soundfile_duration(frames: int, rate: int) -> float:
    """A file's duration as soundfile reports it."""
    return frames / rate


@pytest.mark.parametrize("rate", RATES)
def test_a_file_one_frame_short_of_three_windows_yields_two(rate):
    bounds = window_bounds(soundfile_duration(36 * rate - 1, rate), GEOMETRY)
    assert len(bounds) == 2


@pytest.mark.parametrize("rate", RATES)
def test_a_file_of_exactly_three_windows_yields_three(rate):
    bounds = window_bounds(soundfile_duration(36 * rate, rate), GEOMETRY)
    assert len(bounds) == 3


@pytest.mark.parametrize("rate", RATES)
def test_one_extra_sample_adds_no_window(rate):
    bounds = window_bounds(soundfile_duration(36 * rate + 1, rate), GEOMETRY)
    assert len(bounds) == 3


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("extra_frames", [-1, 0, 1])
def test_each_window_starts_on_a_whole_frame_at_the_files_rate(rate, extra_frames):
    duration = soundfile_duration(36 * rate + extra_frames, rate)
    starts = [start for start, _ in window_bounds(duration, GEOMETRY)]
    assert [round(start * rate) for start in starts] == [
        i * 12 * rate for i in range(len(starts))
    ]
    assert len(starts) == {-1: 2, 0: 3, 1: 3}[extra_frames]


def test_a_recording_shorter_than_one_window_yields_no_windows():
    assert window_bounds(5.0, GEOMETRY) == []


# The adapter, run against stand-ins for its runtime, on real audio files.

ADAPTER = "robin_models.owl.adapter"
SHAPE = (257, 1000)
# The test files hold a ramp: frame n has the value n / RAMP_SCALE, exact in float64.
RAMP_SCALE = 2**24
# The fake spectrogram writes a window's first frame into its first pixels, in this base.
DIGIT_BASE = 250
DIGITS = 3


def get_real_palette() -> list[int]:
    """sox_numpy's palette: 4 fixed colours, then 251 grays, as flat RGB triples."""
    palette = [0, 0, 0, 255, 255, 255, 191, 191, 191, 127, 127, 127]
    for i in range(251):
        gray = int(0.5 + 255 * i / 250)
        palette.extend([gray, gray, gray])
    return palette


PALETTE = get_real_palette()


def get_gray(index: int) -> np.float32:
    """The value the network gets for a palette index: its gray, divided by 255."""
    return np.float32(PALETTE[3 * index]) / np.float32(255)


def get_first_frame_pixels(first_frame: int) -> list[int]:
    """The pixels that the fake spectrogram writes first: the frame number's digits, plus 4."""
    digits = []
    for _ in range(DIGITS):
        first_frame, digit = divmod(first_frame, DIGIT_BASE)
        digits.append(4 + digit)
    return digits


def get_first_frame(scores: np.ndarray) -> int:
    """The first frame of the window whose scores the fake network made from its pixels."""
    grays = {get_gray(index): index for index in range(4, 255)}
    digits = [grays[np.float32(value)] - 4 for value in scores[:DIGITS]]
    return sum(digit * DIGIT_BASE**place for place, digit in enumerate(digits))


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class FakeNetwork:
    """Scores each row with its first pixel values, so that each score shows its window."""

    def __init__(self, runtime: "FakeRuntime") -> None:
        self.runtime = runtime
        self.input_shape = runtime.input_shape

    def __call__(self, images: FakeTensor) -> FakeTensor:
        values = images.values
        rows = values.reshape(len(values), -1)
        self.runtime.batches.append(
            {"shape": values.shape, "dtype": values.dtype, "device": self.runtime.device,
             "first_pixels": rows[:, :DIGITS + 4].copy()}
        )
        return FakeTensor(rows[:, : self.runtime.score_count].copy())


class FakeRuntime:
    """Records what the adapter asked of its runtime, and fails where told to.

    The device scope is kept for each thread, as TensorFlow keeps it.
    """

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.load_error: Exception | None = None
        self.input_shape = (None, 257, 1000, 1)
        self.load_devices: list[str | None] = []
        self.gpus: list[str] = []
        self.score_count = 51
        # One entry for each spectrogram call: its samples, rate, shape, and device.
        self.rendered: list[dict] = []
        self.fail_render_at_frame: int | None = None
        # Returns the pixels for one window's samples. None gives the frame-number pixels.
        self.pixels = None
        self.batches: list[dict] = []
        self.palette = list(PALETTE)
        # OMP_NUM_THREADS each time the adapter took a name from sox_numpy.
        self.omp_threads_at_import: list[str | None] = []
        self._local = threading.local()

    @property
    def device(self) -> str | None:
        return getattr(self._local, "device", None)

    def load_model(self, path):
        self.load_devices.append(self.device)
        self.loaded.append(str(path))
        if self.load_error is not None:
            raise self.load_error
        return FakeNetwork(self)

    @contextlib.contextmanager
    def on_device(self, name: str):
        previous, self._local.device = self.device, name
        try:
            yield
        finally:
            self._local.device = previous

    def list_physical_devices(self, kind: str) -> list[str]:
        return list(self.gpus) if kind == "GPU" else ["CPU:0"]

    def convert_to_tensor(self, value, dtype=None):
        assert dtype is None or dtype == "float32"
        return FakeTensor(np.asarray(value))

    def spectrogram(self, audio_array, shape, sample_rate=None):
        first_frame = round(float(audio_array[0]) * RAMP_SCALE)
        self.rendered.append(
            {"samples": np.array(audio_array, copy=True), "rate": sample_rate,
             "shape": tuple(shape), "device": self.device, "first_frame": first_frame}
        )
        if first_frame == self.fail_render_at_frame:
            raise RuntimeError(f"the clip from frame {first_frame} could not be rendered")
        if self.pixels is not None:
            return self.pixels(audio_array)
        pixels = np.full(shape, 4, dtype=np.uint8)
        pixels.reshape(-1)[:DIGITS] = get_first_frame_pixels(first_frame)
        return pixels

    def get_palette(self, spectrum_points: int = 251) -> list[int]:
        return list(self.palette)

    def get_sox_numpy_attribute(self, name: str):
        """Give the adapter a name from sox_numpy, and record OMP_NUM_THREADS as it was then."""
        self.omp_threads_at_import.append(os.environ.get("OMP_NUM_THREADS"))
        names = {"spectrogram": self.spectrogram, "get_palette": self.get_palette}
        if name not in names:
            raise AttributeError(name)
        return names[name]


@pytest.fixture
def runtime(monkeypatch):
    """Import the adapter against a fake runtime, and remove it afterwards."""
    fake = FakeRuntime()
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.keras = types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=fake.load_model)
    )
    tensorflow.device = fake.on_device
    tensorflow.config = types.SimpleNamespace(list_physical_devices=fake.list_physical_devices)
    tensorflow.convert_to_tensor = fake.convert_to_tensor
    tensorflow.float32 = "float32"
    # No attributes of its own, so each name the adapter takes goes through __getattr__.
    sox_numpy = types.ModuleType("sox_numpy")
    sox_numpy.__getattr__ = fake.get_sox_numpy_attribute
    stubs = {
        "tensorflow": tensorflow,
        "tf_keras": types.ModuleType("tf_keras"),
        "sox_numpy": sox_numpy,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    # The adapter sets OMP_NUM_THREADS. monkeypatch puts back the value from before the test.
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    fake.adapter = import_adapter()
    yield fake
    sys.modules.pop(ADAPTER, None)
    if hasattr(robin_models.owl, "adapter"):
        delattr(robin_models.owl, "adapter")


def import_adapter():
    """Import the adapter again, against the runtime that sys.modules holds now."""
    sys.modules.pop(ADAPTER, None)
    return importlib.import_module(ADAPTER)


def owl_context(tmp_path, **changes) -> ModelContext:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    fields = dict(
        card=CARD,
        registry=REGISTRY,
        files={"weights": tmp_path / "weights.h5", "taxa_registry": tmp_path / "registry.csv"},
        settings={},
        resources={},
        scratch_dir=scratch,
        emit_embeddings=False,
    )
    fields.update(changes)
    return ModelContext(**fields)


def write_ramp(tmp_path, seconds: float = 36.0, rate: int = 1000, channels: int = 1) -> Path:
    """Write a WAV file whose frame n holds n / RAMP_SCALE on channel 0.

    Channel 1, if any, holds the negative, so a mix of the channels is all zeros.
    """
    frames = round(seconds * rate)
    ramp = np.arange(frames, dtype=np.float64) / RAMP_SCALE
    samples = np.stack([ramp, -ramp][:channels], axis=1)
    path = tmp_path / f"ramp-{seconds}s-{rate}hz-{channels}ch.wav"
    soundfile.write(path, samples, rate, subtype="DOUBLE")
    return path


def run(model, path) -> list:
    return list(model.run(AudioClip(path=path)))


def test_a_built_model_satisfies_the_protocol(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    assert isinstance(model, Model)


# Resources. The checks of each value are tested once, in test_adapter_setup.


@pytest.mark.parametrize(("name", "value"), [("batch_size", 0), ("device", "tpu")])
def test_a_bad_resource_is_refused_before_the_weights_load(runtime, tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        runtime.adapter.build(owl_context(tmp_path, resources={name: value}))
    assert runtime.loaded == []


@pytest.mark.parametrize("resources", [{}, {"device": "cpu"}], ids=["absent", "cpu"])
def test_the_weights_load_on_the_cpu(runtime, tmp_path, resources):
    runtime.adapter.build(owl_context(tmp_path, resources=resources))
    assert runtime.load_devices == ["/CPU:0"]


def test_with_device_gpu_the_weights_load_on_the_gpu(runtime, tmp_path):
    runtime.gpus = ["GPU:0"]
    runtime.adapter.build(owl_context(tmp_path, resources={"device": "gpu"}))
    assert runtime.load_devices == ["/GPU:0"]


def test_device_gpu_is_refused_before_the_weights_load_when_tensorflow_sees_no_gpu(
    runtime, tmp_path
):
    with pytest.raises(ValueError, match="no GPU"):
        runtime.adapter.build(owl_context(tmp_path, resources={"device": "gpu"}))
    assert runtime.loaded == []


@pytest.mark.parametrize(
    ("resources", "gpus", "device"),
    [({}, [], "/CPU:0"), ({"device": "gpu"}, ["GPU:0"], "/GPU:0")],
    ids=["cpu", "gpu"],
)
def test_each_spectrogram_is_made_outside_any_device_scope_and_each_batch_on_the_built_device(
    runtime, tmp_path, resources, gpus, device
):
    runtime.gpus = gpus
    model = runtime.adapter.build(owl_context(tmp_path, resources={**resources, "batch_size": 2}))
    run(model, write_ramp(tmp_path))
    assert [call["device"] for call in runtime.rendered] == [None] * 3
    assert [batch["device"] for batch in runtime.batches] == [device] * 2


# soxr's OpenMP threads, set by the adapter before sox_numpy loads soxr.


def test_importing_the_adapter_sets_omp_num_threads_to_1(runtime):
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_importing_the_adapter_keeps_an_omp_num_threads_the_caller_set(runtime, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    import_adapter()
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_omp_num_threads_is_set_before_sox_numpy_is_imported(runtime):
    assert runtime.omp_threads_at_import
    assert set(runtime.omp_threads_at_import) == {"1"}


# What OWL does, whatever its card says.


def changed_card(**changes) -> ModelCard:
    return ModelCard.model_validate(CARD.model_dump() | changes)


@pytest.mark.parametrize(
    ("field", "changes"),
    [
        pytest.param("window_duration", {"window_duration": 5.0}, id="window_duration"),
        pytest.param("sample_rate", {"sample_rate": 16000}, id="sample_rate"),
        pytest.param("audio", {"audio": {**AUDIO, "downmix": "mean"}}, id="downmix"),
        pytest.param(
            "audio",
            {"audio": {**AUDIO, "resampler": {"by": "runner", "algorithm": "librosa"}}},
            id="resampler",
        ),
        pytest.param("audio", {"audio": {**AUDIO, "pad": "time_scaled"}}, id="pad"),
        pytest.param(
            "score_domain",
            {"score_domain": None, "taxa_registry_digest": None},
            id="score_domain",
        ),
        pytest.param(
            "can_emit_embeddings",
            {"can_emit_embeddings": True, "embedding_dim": 4, "embedding_dtype": "float32"},
            id="can_emit_embeddings",
        ),
    ],
)
def test_a_card_stating_what_owl_does_not_do_is_refused_before_the_weights_load(
    runtime, tmp_path, field, changes
):
    with pytest.raises(ValueError, match=f"card field {field} is"):
        runtime.adapter.build(owl_context(tmp_path, card=changed_card(**changes)))
    assert runtime.loaded == []


@pytest.mark.parametrize("shape", [[257, 500], None], ids=["another_shape", "no_shape"])
def test_a_spectrogram_shape_the_loaded_network_does_not_take_is_refused(
    runtime, tmp_path, shape
):
    card = changed_card(spectrogram_shape=shape)
    with pytest.raises(ValueError, match="spectrogram_shape .* network takes \\(257, 1000\\)"):
        runtime.adapter.build(owl_context(tmp_path, card=card))


@pytest.mark.parametrize(
    "palette",
    [PALETTE[:-3], PALETTE[:12] + [1, 2, 3] + PALETTE[15:]],
    ids=["one_entry_short", "not_gray"],
)
def test_a_palette_that_is_not_255_grays_is_refused(runtime, tmp_path, palette):
    runtime.palette = palette
    with pytest.raises(ValueError, match="palette"):
        runtime.adapter.build(owl_context(tmp_path))


# Building and running.


def test_a_missing_registry_is_refused_before_the_weights_load(runtime, tmp_path):
    with pytest.raises(ValueError, match="registry"):
        runtime.adapter.build(owl_context(tmp_path, registry=None))
    assert runtime.loaded == []


def test_missing_weights_are_refused_naming_the_role(runtime, tmp_path):
    files = {"taxa_registry": tmp_path / "registry.csv"}
    with pytest.raises(ValueError, match="weights"):
        runtime.adapter.build(owl_context(tmp_path, files=files))
    assert runtime.loaded == []


def test_weights_that_fail_to_load_fail_construction(runtime, tmp_path):
    runtime.load_error = OSError("not an HDF5 file")
    with pytest.raises(OSError, match="not an HDF5 file"):
        runtime.adapter.build(owl_context(tmp_path))


def test_the_weights_load_once_at_construction_and_not_per_recording(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    assert runtime.loaded == [str(tmp_path / "weights.h5")]
    path = write_ramp(tmp_path)
    for _ in range(2):
        run(model, path)
        model.after_recording()
    assert len(runtime.loaded) == 1


def test_run_yields_every_window_with_every_label_scored_in_output_order(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    windows = run(model, write_ramp(tmp_path))
    assert [(window.start, window.end) for window in windows] == [
        (0.0, 12.0),
        (12.0, 24.0),
        (24.0, 36.0),
    ]
    for window in windows:
        assert window.scores.shape == (len(REGISTRY.entries),)
        assert get_first_frame(window.scores) == round(window.start * 1000)
        assert window.embedding is None


def test_each_windows_samples_are_cut_from_its_block_at_the_files_rate(runtime, tmp_path):
    rate = 1000
    path = write_ramp(tmp_path, seconds=60.0, rate=rate)
    context = owl_context(
        tmp_path, settings={"window_overlap": 6.0}, resources={"read_block_seconds": 30}
    )
    model = runtime.adapter.build(context)
    windows = run(model, path)
    starts = [window.start for window in windows]
    assert starts == [6.0 * n for n in range(9)]
    calls = sorted(runtime.rendered, key=lambda call: call["first_frame"])
    assert len(calls) == len(starts)
    for start, call in zip(starts, calls, strict=True):
        first = round(start * rate)
        expected = np.arange(first, first + 12 * rate, dtype=np.float64) / RAMP_SCALE
        assert call["samples"].dtype == np.float64
        assert np.array_equal(call["samples"], expected)
        assert call["rate"] == rate
        assert call["shape"] == SHAPE


def test_only_the_first_channel_reaches_the_spectrogram(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    run(model, write_ramp(tmp_path, seconds=12.0, channels=2))
    [call] = runtime.rendered
    assert np.array_equal(call["samples"], np.arange(12000, dtype=np.float64) / RAMP_SCALE)


def test_the_network_gets_each_windows_gray_values_divided_by_255(runtime, tmp_path):
    indices = np.full(SHAPE, 100, dtype=np.uint8)
    indices.reshape(-1)[:4] = [4, 254, 5, 129]
    runtime.pixels = lambda samples: indices
    model = runtime.adapter.build(owl_context(tmp_path))
    run(model, write_ramp(tmp_path, seconds=24.0))
    [batch] = runtime.batches
    assert batch["shape"] == (2, 257, 1000, 1)
    assert batch["dtype"] == np.float32
    expected = np.array([get_gray(int(i)) for i in [4, 254, 5, 129, 100, 100, 100]])
    for row in batch["first_pixels"]:
        assert np.array_equal(row, expected)
    assert expected[0] == 0.0 and expected[1] == 1.0


def test_the_last_batch_reaches_the_network_with_only_its_windows(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    windows = run(model, write_ramp(tmp_path, seconds=131 * 12 + 7.3))
    assert [batch["shape"][0] for batch in runtime.batches] == [64, 64, 3]
    assert [window.start for window in windows] == [12.0 * n for n in range(131)]
    for window in windows:
        assert get_first_frame(window.scores) == round(window.start * 1000)


@pytest.mark.parametrize("count", [50, 52], ids=["one_too_few", "one_too_many"])
def test_a_network_giving_a_score_count_other_than_the_registrys_is_refused(
    runtime, tmp_path, count
):
    runtime.score_count = count
    model = runtime.adapter.build(owl_context(tmp_path))
    with pytest.raises(ValueError, match="scores"):
        run(model, write_ramp(tmp_path))


def test_run_renders_nothing_until_it_is_iterated(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    windows = model.run(AudioClip(path=write_ramp(tmp_path)))
    assert runtime.rendered == []
    next(windows)
    assert runtime.rendered != []
    windows.close()


def test_a_window_that_fails_to_render_fails_the_recording_after_the_batches_before_it(
    runtime, tmp_path
):
    runtime.fail_render_at_frame = 4 * 12000
    model = runtime.adapter.build(owl_context(tmp_path, resources={"batch_size": 2}))
    windows = model.run(AudioClip(path=write_ramp(tmp_path, seconds=72.0)))
    yielded = []
    with pytest.raises(RuntimeError, match="from frame 48000 could not be rendered"):
        for window in windows:
            yielded.append(window.start)
    assert yielded == [0.0, 12.0, 24.0, 36.0]


def test_a_recording_shorter_than_one_window_is_not_rendered_or_scored(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    assert run(model, write_ramp(tmp_path, seconds=5.0)) == []
    assert runtime.rendered == []
    assert runtime.batches == []


def test_a_file_with_no_audio_frames_is_refused(runtime, tmp_path):
    path = tmp_path / "empty.wav"
    soundfile.write(path, np.zeros((0, 1)), 1000, subtype="DOUBLE")
    model = runtime.adapter.build(owl_context(tmp_path))
    with pytest.raises(ValueError, match="no audio frames"):
        run(model, path)


def test_after_recording_is_safe_before_any_run(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    model.after_recording()
    model.after_recording()


def test_run_refuses_an_input_that_is_not_audio_before_reading_anything(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    embeddings = Embeddings(
        starts=np.array([0.0]),
        ends=np.array([12.0]),
        values=np.zeros((1, 1), dtype=np.float32),
    )
    with pytest.raises(TypeError, match="Embeddings"):
        list(model.run(embeddings))
    assert runtime.rendered == []


def test_owl_refuses_a_flac_with_no_length_in_its_header(runtime, tmp_path):
    # A FLAC whose STREAMINFO gives 0 total samples, which means "unknown". The 36-bit
    # total is the low 4 bits of byte 21 and bytes 22 to 25 of the file.
    path = tmp_path / "no_length.flac"
    soundfile.write(path, np.zeros(36 * 8000, dtype=np.float32), 8000, format="FLAC")
    data = bytearray(path.read_bytes())
    data[21] &= 0xF0
    data[22:26] = bytes(4)
    path.write_bytes(bytes(data))
    assert soundfile.info(str(path)).frames == 2**63 - 1
    model = runtime.adapter.build(owl_context(tmp_path))

    with pytest.raises(ValueError, match="no length") as raised:
        run(model, path)

    assert str(path) in str(raised.value)
    assert runtime.rendered == []
