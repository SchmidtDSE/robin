"""The OWL adapter: its bundled card and registry, and the adapter against a fake runtime."""

import csv
import hashlib
import importlib
import sys
import types
from collections import Counter
from importlib.resources import as_file, files
from pathlib import Path

import pytest

import robin_models.owl
from robin_contracts.cards import ModelCard, read_card
from robin_contracts.inputs import AudioClip, Embedding
from robin_contracts.protocols import Model, ModelContext
from robin_contracts.records import ClassScore
from robin_contracts.specs import recipe, window_bounds
from robin_inference_engine.load_registry import load_registry

RESOURCES = files("robin_models.owl") / "resources"

with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)
    with _path.open(newline="", encoding="utf-8") as _file:
        REGISTRY_ROWS = list(csv.DictReader(_file))

GEOMETRY = recipe(CARD).audio.geometry
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


def test_the_card_declares_probability_scores_no_embeddings_and_no_settings():
    assert CARD.score_domain == "probability"
    assert CARD.min_detection_threshold == 0.0
    assert CARD.can_emit_embeddings is False
    assert CARD.embedding_dim is None
    assert CARD.embedding_dtype is None
    assert CARD.inference_params == ()


def test_the_card_names_the_bundled_registry_by_its_digest():
    data = (RESOURCES / "taxa_registry.csv").read_bytes()

    assert CARD.taxa_registry_digest == "sha256:" + hashlib.sha256(data).hexdigest()


# What the adapter built OWL's recipe from before its card stated every fact itself. It
# gave a hop of 12 s, which is 12 s windows that do not overlap.
RECIPE_BEFORE_THE_CARD_STATED_IT = {
    "audio": {
        "downmix": "first",
        "pad": "time_scaled",
        "resampler": {"algorithm": "soxr_hq", "by": "runner"},
        "sample_rate": 8000,
        "window_duration": 12.0,
        "window_overlap": 0.0,
    },
    "backend": "h5-fp32",
    "dtype": "float32",
    "version": 1,
}


def test_the_recipe_the_card_states_is_the_one_the_adapter_built():
    stated = recipe(CARD).model_dump(mode="json")

    assert stated.pop("model") == {
        "name": "owl",
        "version": "v4",
        "digest": recipe(CARD).model.digest,
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
def test_a_file_one_frame_short_of_three_windows_yields_three(rate):
    bounds = window_bounds(soundfile_duration(36 * rate - 1, rate), GEOMETRY)
    assert len(bounds) == 3


@pytest.mark.parametrize("rate", RATES)
def test_a_file_of_exactly_three_windows_yields_three(rate):
    bounds = window_bounds(soundfile_duration(36 * rate, rate), GEOMETRY)
    assert len(bounds) == 3


@pytest.mark.parametrize("rate", RATES)
def test_one_extra_sample_adds_a_window(rate):
    bounds = window_bounds(soundfile_duration(36 * rate + 1, rate), GEOMETRY)
    assert len(bounds) == 4
    assert bounds[3][0] == 36.0


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("extra_frames", [-1, 0, 1])
def test_each_window_starts_on_a_whole_frame_at_the_files_rate(rate, extra_frames):
    duration = soundfile_duration(36 * rate + extra_frames, rate)
    starts = [start for start, _ in window_bounds(duration, GEOMETRY)]
    assert [round(start * rate) for start in starts] == [
        i * 12 * rate for i in range(len(starts))
    ]
    assert len(starts) >= 3


def test_a_recording_shorter_than_one_window_yields_one_window():
    assert window_bounds(5.0, GEOMETRY) == [(0.0, 12.0)]


# The adapter, run against stand-ins for its runtime.

ADAPTER = "robin_models.owl.adapter"


class FakeRuntime:
    """Records what the adapter asked of its runtime, and fails where told to."""

    def __init__(self) -> None:
        self.duration = 30.0
        self.loaded: list[str] = []
        self.load_error: Exception | None = None
        self.inspected: list[str] = []
        self.input_shape = (None, 257, 1000, 1)
        self.rendered: list[float] = []
        self.fail_render_at: float | None = None
        self.predicted: list[list[str]] = []
        self.batch_sizes: list[int] = []
        self.score_count = 51

    def load_model(self, path):
        self.loaded.append(str(path))
        if self.load_error is not None:
            raise self.load_error
        return types.SimpleNamespace(input_shape=self.input_shape)

    def info(self, path):
        self.inspected.append(str(path))
        return types.SimpleNamespace(duration=self.duration)

    def render(self, path, *, start_time, duration, shape, dest):
        self.rendered.append(start_time)
        if start_time == self.fail_render_at:
            raise RuntimeError(f"the clip at {start_time} s could not be decoded")
        Path(dest).write_bytes(b"png")
        return dest

    def predict(self, network, paths, batch_size):
        self.predicted.append(list(paths))
        self.batch_sizes.append(batch_size)
        for index, path in enumerate(paths):
            assert Path(path).exists()
            yield index, fake_scores(index)[: self.score_count]


def fake_scores(index: int) -> list[float]:
    """52 distinct scores in [0, 1] that differ from window to window, unsorted.

    The network gives 51; the extra one lets a test give one too many.
    """
    return [((label * 7 + index) % 52) / 51 for label in range(52)]


@pytest.fixture
def runtime(monkeypatch):
    """Import the adapter against a fake runtime, and remove it afterwards."""
    fake = FakeRuntime()
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.keras = types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=fake.load_model)
    )
    soundfile = types.ModuleType("soundfile")
    soundfile.info = fake.info
    processor = types.ModuleType("sox_tensorflow.processor")
    processor.spectrogram_from_flac = fake.render
    stubs = {
        "tensorflow": tensorflow,
        "tf_keras": types.ModuleType("tf_keras"),
        "soundfile": soundfile,
        "sox_tensorflow": types.ModuleType("sox_tensorflow"),
        "sox_tensorflow.processor": processor,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop(ADAPTER, None)
    fake.adapter = importlib.import_module(ADAPTER)
    monkeypatch.setattr(fake.adapter, "_predict", fake.predict)
    yield fake
    sys.modules.pop(ADAPTER, None)
    if hasattr(robin_models.owl, "adapter"):
        delattr(robin_models.owl, "adapter")


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


def test_a_built_model_satisfies_the_protocol_and_restates_nothing_from_its_card(
    runtime, tmp_path
):
    model = runtime.adapter.build(owl_context(tmp_path))
    assert isinstance(model, Model)
    assert not hasattr(model, "recipe")
    assert not hasattr(model, "capabilities")


# The batch size, a resource.


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        pytest.param({}, 64, id="none_given"),
        pytest.param({"batch_size": 8}, 8, id="given"),
        pytest.param({"device": "cpu", "batch_size": 1}, 1, id="beside_another_key"),
        pytest.param({"device": "cpu"}, 64, id="only_an_unused_key"),
    ],
)
def test_the_batch_size_is_read_from_the_resources(runtime, resources, expected):
    assert runtime.adapter._read_batch_size(resources) == expected


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8", None], ids=repr)
def test_a_batch_size_that_is_not_a_positive_int_is_refused(runtime, value):
    with pytest.raises(ValueError, match="batch_size") as caught:
        runtime.adapter._read_batch_size({"batch_size": value})
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize(("resources", "expected"), [({}, 64), ({"batch_size": 8}, 8)])
def test_the_batch_size_reaches_the_network(runtime, tmp_path, resources, expected):
    model = runtime.adapter.build(owl_context(tmp_path, resources=resources))
    list(model.run(AudioClip(path=tmp_path / "a.flac")))
    assert runtime.batch_sizes == [expected]


def test_a_bad_batch_size_is_refused_before_the_weights_load(runtime, tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        runtime.adapter.build(owl_context(tmp_path, resources={"batch_size": 0}))
    assert runtime.loaded == []


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
        pytest.param("audio", {"audio": {**AUDIO, "pad": "drop"}}, id="pad"),
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
    for name in ("a.flac", "b.flac"):
        list(model.run(AudioClip(path=tmp_path / name)))
        model.after_recording()
    assert len(runtime.loaded) == 1


def test_run_yields_every_window_with_every_label_scored_in_output_order(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    windows = list(model.run(AudioClip(path=tmp_path / "a.flac")))
    labels = [entry.label for entry in REGISTRY.entries]
    assert [(window.start, window.end) for window in windows] == [
        (0.0, 12.0),
        (12.0, 24.0),
        (24.0, 36.0),
    ]
    for index, window in enumerate(windows):
        expected = tuple(
            ClassScore(label=label, score=score)
            for label, score in zip(labels, fake_scores(index)[:51], strict=True)
        )
        assert window.scores == expected
        assert window.embedding is None
    scores = [score.score for score in windows[0].scores]
    assert scores != sorted(scores, reverse=True)


def test_overlapping_windows_are_rendered_from_where_each_one_starts(runtime, tmp_path):
    card = changed_card(window_overlap=6.0)
    model = runtime.adapter.build(owl_context(tmp_path, card=card))
    windows = list(model.run(AudioClip(path=tmp_path / "a.flac")))
    assert runtime.rendered == [0.0, 6.0, 12.0, 18.0]
    assert [window.start for window in windows] == runtime.rendered


@pytest.mark.parametrize("count", [50, 52], ids=["one_too_few", "one_too_many"])
def test_a_network_giving_a_score_count_other_than_the_registrys_is_refused(
    runtime, tmp_path, count
):
    runtime.score_count = count
    model = runtime.adapter.build(owl_context(tmp_path))
    with pytest.raises(ValueError):
        list(model.run(AudioClip(path=tmp_path / "a.flac")))


def test_run_renders_nothing_until_it_is_iterated(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    windows = model.run(AudioClip(path=tmp_path / "a.flac"))
    assert runtime.rendered == []
    next(windows)
    assert runtime.rendered == [0.0, 12.0, 24.0]


def test_a_window_that_fails_to_render_fails_the_recording(runtime, tmp_path):
    runtime.fail_render_at = 12.0
    model = runtime.adapter.build(owl_context(tmp_path))
    with pytest.raises(RuntimeError, match="the clip at 12.0 s could not be decoded"):
        list(model.run(AudioClip(path=tmp_path / "a.flac")))
    assert runtime.rendered == [0.0, 12.0]
    assert runtime.predicted == []


def test_after_a_rendering_failure_the_recordings_spectrograms_are_removed(runtime, tmp_path):
    runtime.fail_render_at = 12.0
    context = owl_context(tmp_path)
    unrelated = context.scratch_dir / "unrelated.txt"
    unrelated.write_text("kept")
    model = runtime.adapter.build(context)
    with pytest.raises(RuntimeError):
        list(model.run(AudioClip(path=tmp_path / "a.flac")))
    assert list(context.scratch_dir.rglob("*.png")) != []
    model.after_recording()
    assert list(context.scratch_dir.iterdir()) == [unrelated]
    model.after_recording()


def test_after_recording_is_safe_before_any_run(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    model.after_recording()
    model.after_recording()


def test_run_refuses_an_input_that_is_not_audio_before_reading_anything(runtime, tmp_path):
    model = runtime.adapter.build(owl_context(tmp_path))
    with pytest.raises(TypeError, match="Embedding"):
        list(model.run(Embedding(start=0.0, end=12.0, values=(0.0,))))
    assert runtime.inspected == []
    assert runtime.rendered == []
