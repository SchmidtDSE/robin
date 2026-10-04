"""Perch v8: its bundled card and registry, and the adapter against a fake TensorFlow."""

import contextlib
import csv
import gc
import hashlib
import importlib
import math
import os
import subprocess
import sys
import types
import warnings
import weakref
from collections import Counter
from importlib.resources import as_file, files

import numpy as np
import pyarrow as pa
import pytest
import scipy.signal
import soundfile

import robin_models.perch
from robin_adapters.artifact_writer.local import LocalArtifactWriter
from robin_adapters.file_provider.local import LocalFileProvider
from robin_contracts.cards import ModelCard, model_ref, read_card
from robin_contracts.inputs import AudioClip, Embeddings
from robin_contracts.layout import artifact_path
from robin_contracts.output_contracts import EmbeddingsRequest, ScoresRequest
from robin_contracts.protocols import Model, ModelContext
from robin_contracts.results import InferenceSuccess
from robin_contracts.work import AudioInput, InferenceWork, PinnedFile, PinnedModel, RecordingRef
from robin_inference_engine.artifacts.embeddings import read_embeddings
from robin_inference_engine.artifacts.scores import read_scores
from robin_inference_engine.engine import run_work
from robin_inference_engine.load_registry import load_registry

RESOURCES = files("robin_models.perch") / "resources"

with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)
    with _path.open(newline="", encoding="utf-8") as _file:
        REGISTRY_ROWS = list(csv.DictReader(_file))

REGISTRY_SHA256 = "ca28f370cb5924af9966fee9fcd0a12d57632bba85a01a3f5908f9011ef99fd9"
RUNTIME_MODULES = ("tensorflow", "scipy", "soundfile")


def runtime_modules_loaded_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter and return which runtime modules it imported."""
    report = f"import sys; print(','.join(m for m in {RUNTIME_MODULES!r} if m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", f"{code}\n{report}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {name for name in result.stdout.strip().split(",") if name}


def test_the_card_states_every_field():
    assert CARD.model_dump() == {
        "model_name": "perch",
        "model_version": "v8",
        "runtime": "tensorflow",
        "window_duration": 5.0,
        "window_overlap": 0.0,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": "sha256:" + REGISTRY_SHA256,
        "spectrogram_shape": None,
        "audio": {
            "downmix": "mean",
            "resampler": {"by": "runner", "algorithm": "scipy_polyphase_with_context"},
            "pad": "centre_crop_end_pad",
        },
        "backend": "pb-fp32",
        "dtype": "float32",
        "inference_params": (),
        "can_emit_embeddings": True,
        "embedding_dim": 1280,
        "embedding_dtype": "float32",
    }


def test_the_card_reads_as_perch_v8():
    assert model_ref(CARD).id == "perch/v8"


def test_the_card_names_the_bundled_registry_by_its_digest():
    digest = hashlib.sha256((RESOURCES / "taxa_registry.csv").read_bytes()).hexdigest()

    assert digest == REGISTRY_SHA256
    assert CARD.taxa_registry_digest == "sha256:" + digest


def test_the_registry_loads_with_10932_entries_taxa_and_unresolved():
    kinds = Counter(entry.label_kind for entry in REGISTRY.entries)

    assert len(REGISTRY.entries) == 10932
    assert kinds == {"taxon": 10917, "unresolved": 15}


def test_every_rows_label_kind_follows_from_its_kind():
    for row in REGISTRY_ROWS:
        expected = "unresolved" if row["kind"] == "unresolved" else "taxon"
        assert row["label_kind"] == expected, row


def test_the_unresolved_classes_are_the_fifteen_with_no_key():
    unresolved = [entry for entry in REGISTRY.entries if entry.label_kind == "unresolved"]

    assert [(e.class_index, e.label, e.scientific_name) for e in unresolved] == [
        (1590, "braeme2", "Riccordia bracei bracei"),
        (1591, "braeme3", "Riccordia bracei elegans"),
        (3468, "fotdro4", "Dicrurus divaricatus"),
        (3641, "gnfhum2", "Leucolia viridifrons wagneri"),
        (3700, "goftyr4", "Zimmerius viridiflavus flavidifrons"),
        (3721, "gollor1", "Glossoptila goldiei"),
        (5545, "masyel3", "Geothlypis semiflava chiriquensis"),
        (7108, "rebwoo4", "Hylexetastes uniformis brigidai"),
        (7178, "reepar3", "Calamornis heudei"),
        (8941, "stisha2", "Leucocarbo chalconotus stewarti"),
        (9377, "thbvir2", "Vireo pallens approximans"),
        (9683, "vemdro1", "Dicrurus atactus"),
        (9684, "vemdro5", "Dicrurus atactus"),
        (9702, "verfly8", "Pyrocephalus nanus dubius"),
        (9934, "wetsab3", "Pampa curvipennis pampa"),
    ]
    for entry in unresolved:
        assert entry.gbif_taxon_key is None


def test_every_taxon_has_a_scientific_name_and_a_gbif_key():
    for entry in REGISTRY.entries:
        if entry.label_kind == "taxon":
            assert entry.scientific_name, entry
            assert entry.gbif_taxon_key is not None, entry


def test_importing_the_perch_package_imports_no_runtime():
    assert runtime_modules_loaded_after("import robin_models.perch") == set()


# The adapter, against a fake TensorFlow.

ADAPTER = "robin_models.perch.adapter"
LABELS = tuple(entry.label for entry in REGISTRY.entries)
AUDIO = CARD.audio.model_dump()
WINDOW = 160_000
CPU = "/CPU:0"

# The real signature's outputs and their shapes. The real model lists them in an order that
# differs between TensorFlow versions.
OUTPUT_SHAPES = {
    "embedding": [None, 1280],
    "label": [None, 10932],
    "genus": [None, 2333],
    "family": [None, 249],
    "order": [None, 41],
    "frontend": [None, 500, 160],
}
# The order the fake returns them in, which is not the order the adapter reads them.
RETURNED_ORDER = ("order", "frontend", "label", "genus", "embedding", "family")


class FakeSpec:
    """A tensor spec: only its shape, as a list."""

    def __init__(self, shape: list) -> None:
        self.shape = types.SimpleNamespace(as_list=lambda: list(shape))


class FakeTensor:
    """A returned output. Reading its values is recorded by name."""

    def __init__(self, runtime: "FakeTensorFlow", name: str, values: np.ndarray) -> None:
        self._runtime = runtime
        self._name = name
        self._values = values

    def numpy(self) -> np.ndarray:
        self._runtime.read.append(self._name)
        return np.array(self._values, dtype=np.float32, copy=True)


class FakeSignature:
    """One SavedModel signature. It refers to the runtime, never to the loaded object."""

    def __init__(self, runtime: "FakeTensorFlow", name: str) -> None:
        self._runtime = runtime
        self._name = name
        self.structured_input_signature = (
            (),
            {name: FakeSpec(shape) for name, shape in runtime.inputs.items()},
        )
        self.structured_outputs = {
            name: FakeSpec(shape) for name, shape in runtime.output_shapes.items()
        }

    def __call__(self, *, inputs):
        runtime = self._runtime
        call = types.SimpleNamespace(
            signature=self._name,
            given=inputs,
            values=np.array(inputs, copy=True),
            device=runtime.device,
            returned=None,
        )
        runtime.calls.append(call)
        if runtime.call_error is not None:
            raise runtime.call_error
        batch = np.asarray(inputs)
        values = {
            "embedding": runtime.embedding(batch),
            "label": runtime.logits(batch),
        }
        returned = {}
        for name in RETURNED_ORDER:
            if name not in runtime.output_shapes:
                continue
            if name not in values:
                # Never read as scores or embeddings: NaN would fail a test that did.
                shape = [len(batch), *runtime.output_shapes[name][1:]]
                values[name] = np.full(shape, np.nan, dtype=np.float32)
            returned[name] = FakeTensor(runtime, name, np.asarray(values[name], np.float32))
        call.returned = returned
        return returned


class FakeLoaded:
    """What `tf.saved_model.load` returns: its signatures."""

    def __init__(self, signatures: dict) -> None:
        self.signatures = signatures


class FakeTensorFlow:
    """Records what the adapter asked of TensorFlow, and fails where told to."""

    def __init__(self) -> None:
        self.device = None
        self.converted: list = []
        self.loads: list = []
        self.load_error: Exception | None = None
        self.loaded: list[weakref.ref] = []
        self.gpu_visible = False
        self.call_error: Exception | None = None
        self.calls: list = []
        self.read: list[str] = []
        self.signature_names = ["serving_default", "infer_tf"]
        self.inputs = {"inputs": [None, WINDOW]}
        self.output_shapes = {name: list(shape) for name, shape in OUTPUT_SHAPES.items()}
        self.embedding = fake_embeddings
        self.logits = fake_logits

    def module(self) -> types.ModuleType:
        tf = types.ModuleType("tensorflow")
        tf.float32 = "float32"
        tf.device = self.on_device
        tf.convert_to_tensor = self.convert_to_tensor
        tf.saved_model = types.SimpleNamespace(load=self.load)
        tf.config = types.SimpleNamespace(list_physical_devices=self.list_physical_devices)
        return tf

    @contextlib.contextmanager
    def on_device(self, name: str):
        previous, self.device = self.device, name
        try:
            yield
        finally:
            self.device = previous

    def convert_to_tensor(self, value, dtype):
        self.converted.append(types.SimpleNamespace(value=value, dtype=dtype, device=self.device))
        return value

    def load(self, path):
        folder = os.fspath(path)
        self.loads.append(
            types.SimpleNamespace(path=folder, device=self.device, contents=folder_contents(folder))
        )
        if self.load_error is not None:
            raise self.load_error
        loaded = FakeLoaded({name: FakeSignature(self, name) for name in self.signature_names})
        self.loaded.append(weakref.ref(loaded))
        return loaded

    def list_physical_devices(self, kind=None):
        if self.gpu_visible and kind in (None, "GPU"):
            return [types.SimpleNamespace(name="/physical_device:GPU:0", device_type="GPU")]
        return []

    def serving_calls(self) -> list:
        return [call for call in self.calls if call.signature == "serving_default"]


def folder_contents(folder: str) -> dict[str, tuple[bool, str | None]]:
    """Each file under `folder`: whether it is a symbolic link, and where it points."""
    contents = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            link = os.path.islink(path)
            contents[os.path.relpath(path, folder)] = (link, os.readlink(path) if link else None)
    return contents


# Each label's default logit is a fixed bias, from -40 to 10 in an order unrelated to the
# registry's, plus the window's mean: distinct per label and per window.
LOGIT_BIAS = np.linspace(-40.0, 10.0, 10932)[np.random.default_rng(0).permutation(10932)]


def fake_logits(batch: np.ndarray) -> np.ndarray:
    return (np.float64(batch).mean(axis=1, keepdims=True) + LOGIT_BIAS).astype(np.float32)


def fake_embeddings(batch: np.ndarray) -> np.ndarray:
    """Finite embeddings, distinct per window. Element 0 is the window's mean."""
    offset = np.float64(batch).mean(axis=1, keepdims=True)
    return (np.arange(1280) / 1280 + offset).astype(np.float32)


@pytest.fixture
def runtime(monkeypatch):
    """Import the adapter against a fake TensorFlow, and remove it afterwards."""
    fake = FakeTensorFlow()
    monkeypatch.setitem(sys.modules, "tensorflow", fake.module())
    sys.modules.pop(ADAPTER, None)
    fake.adapter = importlib.import_module(ADAPTER)
    yield fake
    sys.modules.pop(ADAPTER, None)
    if hasattr(robin_models.perch, "adapter"):
        delattr(robin_models.perch, "adapter")


def labels_text(codes=LABELS, header: str | None = "ebird2021", newline: str = "\r\n") -> str:
    """A labels file as shipped: a header, then one code per line, each line ending in CRLF."""
    lines = list(codes) if header is None else [header, *codes]
    return newline.join(lines) + newline


def perch_files(tmp_path, labels: str | bytes | None = None) -> dict:
    """Small stand-ins for the pinned files. The fake TensorFlow reads none of them."""
    paths = {
        "saved_model": tmp_path / "saved_model.pb",
        "variables_data": tmp_path / "variables.data-00000-of-00001",
        "variables_index": tmp_path / "variables.index",
    }
    for role, path in paths.items():
        path.write_bytes(role.encode())
    labels_file = tmp_path / "label.csv"
    text = labels_text() if labels is None else labels
    labels_file.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    registry = tmp_path / "taxa_registry.csv"
    registry.write_bytes((RESOURCES / "taxa_registry.csv").read_bytes())
    return paths | {"labels": labels_file, "taxa_registry": registry}


def perch_context(tmp_path, **changes) -> ModelContext:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    fields = dict(
        card=CARD,
        registry=REGISTRY,
        settings={},
        resources={},
        scratch_dir=scratch,
        emit_embeddings=False,
    )
    fields.update(changes)
    if "files" not in fields:
        fields["files"] = perch_files(tmp_path)
    return ModelContext(**fields)


def without(files: dict, *roles: str) -> dict:
    return {role: path for role, path in files.items() if role not in roles}


def embeddings_only(tmp_path, **changes) -> ModelContext:
    files = without(perch_files(tmp_path), "labels", "taxa_registry")
    return perch_context(tmp_path, files=files, registry=None, **changes)


def changed_card(**changes) -> ModelCard:
    return ModelCard.model_validate(CARD.model_dump() | changes)


def test_a_built_model_satisfies_the_protocol_and_restates_nothing_from_its_card(
    runtime, tmp_path
):
    model = runtime.adapter.build(perch_context(tmp_path))
    assert isinstance(model, Model)
    assert not hasattr(model, "recipe")
    assert not hasattr(model, "capabilities")


# What Perch does, whatever its card says.


@pytest.mark.parametrize(
    ("field", "changes"),
    [
        pytest.param("window_duration", {"window_duration": 3.0}, id="window_duration"),
        pytest.param("sample_rate", {"sample_rate": 48000}, id="sample_rate"),
        pytest.param("audio", {"audio": {**AUDIO, "downmix": "first"}}, id="downmix"),
        pytest.param(
            "audio",
            {
                "audio": {
                    **AUDIO,
                    "resampler": {"by": "runner", "algorithm": "scipy_fft_per_window"},
                }
            },
            id="resampler",
        ),
        pytest.param("audio", {"audio": {**AUDIO, "pad": "drop"}}, id="pad"),
        pytest.param(
            "score_domain",
            {"score_domain": None, "taxa_registry_digest": None},
            id="score_domain",
        ),
        pytest.param(
            "min_detection_threshold",
            {"min_detection_threshold": 0.005},
            id="min_detection_threshold",
        ),
        pytest.param("spectrogram_shape", {"spectrogram_shape": [500, 160]}, id="spectrogram"),
        pytest.param(
            "can_emit_embeddings",
            {"can_emit_embeddings": False, "embedding_dim": None, "embedding_dtype": None},
            id="can_emit_embeddings",
        ),
        pytest.param("embedding_dim", {"embedding_dim": 1024}, id="embedding_dim"),
        pytest.param("embedding_dtype", {"embedding_dtype": "float16"}, id="embedding_dtype"),
        pytest.param("backend", {"backend": "onnx-fp32"}, id="backend"),
    ],
)
def test_a_card_stating_what_perch_does_not_do_is_refused_before_the_model_loads(
    runtime, tmp_path, field, changes
):
    card = changed_card(**changes)
    with pytest.raises(ValueError, match=f"card field {field} is") as caught:
        runtime.adapter.build(perch_context(tmp_path, card=card))
    message = str(caught.value)
    assert repr(getattr(card, field)) in message
    assert f"model perch/v8 does {runtime.adapter.BEHAVIOUR[field]!r}" in message
    assert runtime.loads == []


def test_overlapping_windows_build(runtime, tmp_path):
    runtime.adapter.build(perch_context(tmp_path, card=changed_card(window_overlap=2.5)))


# Resources.


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        pytest.param({}, 32, id="none_given"),
        pytest.param({"batch_size": 1}, 1, id="one"),
        pytest.param({"batch_size": 64}, 64, id="sixty_four"),
        pytest.param({"device": "cpu"}, 32, id="only_the_device"),
    ],
)
def test_the_batch_size_is_read_from_the_resources(runtime, tmp_path, resources, expected):
    model = runtime.adapter.build(perch_context(tmp_path, resources=resources))
    assert model._batch_size == expected


@pytest.mark.parametrize("value", [True, 0, -1, 32.0, "32"], ids=repr)
def test_a_batch_size_that_is_not_a_positive_int_is_refused_before_the_model_loads(
    runtime, tmp_path, value
):
    with pytest.raises(ValueError, match="batch_size") as caught:
        runtime.adapter.build(perch_context(tmp_path, resources={"batch_size": value}))
    assert repr(value) in str(caught.value)
    assert runtime.loads == []


@pytest.mark.parametrize("resources", [{}, {"device": "cpu"}], ids=["absent", "cpu"])
def test_the_cpu_is_the_device_that_builds(runtime, tmp_path, resources):
    runtime.adapter.build(perch_context(tmp_path, resources=resources))


@pytest.mark.parametrize("value", ["gpu", "/GPU:0", "CPU", ""], ids=repr)
def test_any_other_device_is_refused_before_the_model_loads(runtime, tmp_path, value):
    with pytest.raises(ValueError, match="device") as caught:
        runtime.adapter.build(perch_context(tmp_path, resources={"device": value}))
    assert repr(value) in str(caught.value)
    assert runtime.loads == []


# The pinned files.


@pytest.mark.parametrize("role", ["saved_model", "variables_data", "variables_index"])
def test_a_missing_savedmodel_file_is_refused_naming_the_role(runtime, tmp_path, role):
    files = without(perch_files(tmp_path), role)
    with pytest.raises(ValueError, match=f"'{role}'") as caught:
        runtime.adapter.build(perch_context(tmp_path, files=files))
    assert str(sorted(files)) in str(caught.value)
    assert runtime.loads == []


def test_with_a_registry_a_missing_labels_file_is_refused_naming_the_role(runtime, tmp_path):
    files = without(perch_files(tmp_path), "labels")
    with pytest.raises(ValueError, match="'labels'") as caught:
        runtime.adapter.build(perch_context(tmp_path, files=files))
    assert str(sorted(files)) in str(caught.value)
    assert runtime.loads == []


def test_without_a_registry_neither_labels_nor_registry_file_is_needed(runtime, tmp_path):
    runtime.adapter.build(embeddings_only(tmp_path))


def test_the_savedmodel_folder_is_rebuilt_from_links_to_the_pinned_files(runtime, tmp_path):
    context = perch_context(tmp_path)
    runtime.adapter.build(context)
    [load] = runtime.loads
    folder = os.path.realpath(load.path)
    scratch = os.path.realpath(context.scratch_dir)
    assert os.path.commonpath([folder, scratch]) == scratch
    assert folder != scratch
    assert load.contents == {
        "saved_model.pb": (True, str(context.files["saved_model"])),
        os.path.join("variables", "variables.data-00000-of-00001"): (
            True,
            str(context.files["variables_data"]),
        ),
        os.path.join("variables", "variables.index"): (
            True,
            str(context.files["variables_index"]),
        ),
    }
    assert not os.path.exists(os.path.join(load.path, "assets"))
    assert load.device == CPU


def test_two_models_sharing_a_scratch_dir_get_separate_folders(runtime, tmp_path):
    context = perch_context(tmp_path)
    runtime.adapter.build(context)
    runtime.adapter.build(context)
    first, second = runtime.loads
    assert first.path != second.path


# The serving signature.


@pytest.mark.parametrize("found", [[], ["infer_tf"]], ids=["none", "another"])
def test_a_missing_serving_signature_is_refused_naming_those_found(runtime, tmp_path, found):
    runtime.signature_names = found
    with pytest.raises(ValueError, match="serving_default") as caught:
        runtime.adapter.build(perch_context(tmp_path))
    assert str(found) in str(caught.value)


@pytest.mark.parametrize(
    ("inputs", "named"),
    [
        pytest.param({"audio": [None, WINDOW]}, "['audio']", id="input_name"),
        pytest.param(
            {"inputs": [None, WINDOW], "other": [None, WINDOW]},
            "['inputs', 'other']",
            id="two_inputs",
        ),
        pytest.param({"inputs": [None, 144000]}, "[None, 144000]", id="input_width"),
        pytest.param({"inputs": [32, WINDOW]}, "[32, 160000]", id="fixed_batch"),
    ],
)
def test_an_input_the_card_does_not_describe_is_refused(runtime, tmp_path, inputs, named):
    runtime.inputs = inputs
    with pytest.raises(ValueError, match="serving_default") as caught:
        runtime.adapter.build(perch_context(tmp_path))
    assert named in str(caught.value)


@pytest.mark.parametrize(
    ("shape", "found"), [(None, "None"), ([None, 1024], "[None, 1024]")], ids=["absent", "width"]
)
def test_an_embedding_output_other_than_the_cards_is_refused(runtime, tmp_path, shape, found):
    if shape is None:
        del runtime.output_shapes["embedding"]
    else:
        runtime.output_shapes["embedding"] = shape
    with pytest.raises(ValueError, match="'embedding'") as caught:
        runtime.adapter.build(perch_context(tmp_path))
    assert "[None, 1280]" in str(caught.value)
    assert f"found {found}" in str(caught.value)


@pytest.mark.parametrize(
    ("shape", "found"), [(None, "None"), ([None, 6522], "[None, 6522]")], ids=["absent", "width"]
)
def test_with_a_registry_a_label_output_that_does_not_fit_is_refused(
    runtime, tmp_path, shape, found
):
    if shape is None:
        del runtime.output_shapes["label"]
    else:
        runtime.output_shapes["label"] = shape
    with pytest.raises(ValueError, match="'label'") as caught:
        runtime.adapter.build(perch_context(tmp_path))
    assert "[None, 10932]" in str(caught.value)
    assert f"found {found}" in str(caught.value)


@pytest.mark.parametrize("shape", [None, [None, 6522]], ids=["absent", "width"])
def test_without_a_registry_the_label_output_is_not_checked(runtime, tmp_path, shape):
    if shape is None:
        del runtime.output_shapes["label"]
    else:
        runtime.output_shapes["label"] = shape
    runtime.adapter.build(embeddings_only(tmp_path))


# The labels file shipped with the weights.


def labels_file_context(tmp_path, labels: str | bytes) -> ModelContext:
    return perch_context(tmp_path, files=perch_files(tmp_path, labels=labels))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(labels_text(), id="crlf_as_shipped"),
        pytest.param(labels_text(newline="\n"), id="lf"),
        pytest.param(labels_text(newline="\n")[:-1], id="lf_without_a_final_newline"),
    ],
)
def test_a_labels_file_equal_to_the_registry_builds(runtime, tmp_path, text):
    runtime.adapter.build(labels_file_context(tmp_path, text))


@pytest.mark.parametrize(
    ("header", "first"),
    [(None, LABELS[0]), ("ebird2022", "ebird2022")],
    ids=["no_header", "another_header"],
)
def test_a_labels_file_without_its_header_is_refused_naming_its_first_line(
    runtime, tmp_path, header, first
):
    with pytest.raises(ValueError, match="ebird2021") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, labels_text(header=header)))
    assert repr(first) in str(caught.value)


@pytest.mark.parametrize(
    ("codes", "count"),
    [(LABELS[:-1], 10931), (LABELS + ("extra1",), 10933)],
    ids=["one_too_few", "one_too_many"],
)
def test_a_labels_file_of_another_length_is_refused(runtime, tmp_path, codes, count):
    with pytest.raises(ValueError, match="labels") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, labels_text(codes)))
    assert str(count) in str(caught.value)
    assert "10932" in str(caught.value)


def test_a_labels_file_in_another_order_is_refused_at_the_first_difference(runtime, tmp_path):
    codes = list(LABELS)
    codes[10], codes[11] = codes[11], codes[10]
    with pytest.raises(ValueError, match="position 10") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, labels_text(codes)))
    assert repr(LABELS[10]) in str(caught.value)
    assert repr(LABELS[11]) in str(caught.value)


def test_a_labels_file_with_another_code_is_refused(runtime, tmp_path):
    codes = list(LABELS)
    codes[500] = "zzzzzz1"
    with pytest.raises(ValueError, match="position 500") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, labels_text(codes)))
    assert repr(LABELS[500]) in str(caught.value)
    assert "'zzzzzz1'" in str(caught.value)


def test_a_labels_file_that_is_not_utf8_is_refused(runtime, tmp_path):
    text = labels_text().encode("utf-8") + b"\xff\r\n"
    with pytest.raises(ValueError, match="UTF-8") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, text))
    assert "label.csv" in str(caught.value)


def test_a_model_that_fails_to_load_fails_the_build(runtime, tmp_path):
    runtime.load_error = OSError("not a SavedModel")
    with pytest.raises(OSError, match="not a SavedModel"):
        runtime.adapter.build(perch_context(tmp_path))


# Running a recording.


def write_audio(path, samples: np.ndarray, rate: int):
    """Write 32-bit float WAV, so the samples read back are exactly those written."""
    soundfile.write(str(path), samples, rate, subtype="FLOAT")
    return path


def noise(frames: int, channels: int | None = None, seed: int = 0) -> np.ndarray:
    shape = (frames,) if channels is None else (frames, channels)
    return np.random.default_rng(seed).uniform(-0.5, 0.5, shape).astype(np.float32)


def run(model, path) -> list:
    return list(model.run(AudioClip(path=path)))


def rows_given_to(runtime) -> np.ndarray:
    return np.concatenate([call.values for call in runtime.serving_calls()])


def model_and_noise(runtime, tmp_path, frames: int, rate: int = 32000, **changes):
    model = runtime.adapter.build(perch_context(tmp_path, **changes))
    samples = noise(frames)
    return model, samples, write_audio(tmp_path / "a.wav", samples, rate)


def up_and_down(rate: int) -> tuple[int, int]:
    g = math.gcd(rate, 32000)
    return 32000 // g, rate // g


def whole_recording_resampled(samples: np.ndarray, rate: int) -> np.ndarray:
    """The whole recording, mixed down by its float32 channel mean and resampled at once."""
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1, dtype=np.float32)
    return scipy.signal.resample_poly(mono, *up_and_down(rate)).astype(np.float32)


def padded_slice(whole: np.ndarray, first: int) -> np.ndarray:
    row = np.zeros(WINDOW, dtype=np.float32)
    piece = whole[first : first + WINDOW]
    row[: len(piece)] = piece
    return row


# Windowing.


@pytest.mark.parametrize(
    ("frames", "count"),
    [(2 * WINDOW - 1, 2), (2 * WINDOW, 2), (2 * WINDOW + 1, 3)],
    ids=["one_frame_short", "exact", "one_frame_over"],
)
def test_a_32khz_file_yields_one_window_per_started_5_seconds(runtime, tmp_path, frames, count):
    model, _, path = model_and_noise(runtime, tmp_path, frames)
    windows = run(model, path)
    assert [(w.start, w.end) for w in windows] == [(5.0 * i, 5.0 * i + 5.0) for i in range(count)]


def test_the_last_window_is_padded_with_zeros_at_its_end(runtime, tmp_path):
    model, samples, path = model_and_noise(runtime, tmp_path, WINDOW + 1000)
    run(model, path)
    last = rows_given_to(runtime)[1]
    assert np.array_equal(last[:1000], samples[WINDOW:])
    assert np.array_equal(last[1000:], np.zeros(WINDOW - 1000, dtype=np.float32))


def test_a_mono_window_reaches_the_model_bit_for_bit(runtime, tmp_path):
    model, samples, path = model_and_noise(runtime, tmp_path, WINDOW)
    run(model, path)
    assert np.array_equal(rows_given_to(runtime)[0], samples)


def test_a_stereo_window_reaches_the_model_as_its_channel_mean(runtime, tmp_path):
    model = runtime.adapter.build(perch_context(tmp_path))
    samples = noise(WINDOW, channels=2)
    run(model, write_audio(tmp_path / "a.wav", samples, 32000))
    expected = np.mean(samples, axis=1, dtype=np.float32)
    assert np.array_equal(rows_given_to(runtime)[0], expected)


@pytest.mark.parametrize("channels", [None, 2], ids=["mono", "stereo"])
@pytest.mark.parametrize("rate", [48000, 44100])
def test_each_window_at_another_rate_equals_the_whole_recording_resampled(
    runtime, tmp_path, rate, channels
):
    model = runtime.adapter.build(perch_context(tmp_path, resources={"batch_size": 2}))
    samples = noise(17 * rate + 123, channels=channels)
    windows = run(model, write_audio(tmp_path / "a.wav", samples, rate))
    whole = whole_recording_resampled(samples, rate)
    rows = rows_given_to(runtime)
    assert len(windows) == len(rows) == 4
    assert len(runtime.serving_calls()) == 2
    for i, row in enumerate(rows):
        assert np.array_equal(row, padded_slice(whole, WINDOW * i)), i


@pytest.mark.parametrize("rate", [48000, 44100])
def test_a_window_resampled_without_context_differs_from_the_whole_recording(rate):
    samples = noise(17 * rate + 123)
    own_frames = samples[5 * rate : 10 * rate]
    alone = scipy.signal.resample_poly(own_frames, *up_and_down(rate)).astype(np.float32)
    reference = padded_slice(whole_recording_resampled(samples, rate), WINDOW)
    assert not np.array_equal(alone[:3200], reference[:3200])


def test_overlapping_windows_at_another_rate_each_equal_the_whole_recording_resampled(
    runtime, tmp_path
):
    card = changed_card(window_overlap=2.5)
    model = runtime.adapter.build(perch_context(tmp_path, card=card))
    samples = noise(12 * 48000)
    windows = run(model, write_audio(tmp_path / "a.wav", samples, 48000))
    assert [w.start for w in windows] == [0.0, 2.5, 5.0, 7.5]
    whole = whole_recording_resampled(samples, 48000)
    for i, row in enumerate(rows_given_to(runtime)):
        assert np.array_equal(row, padded_slice(whole, 80000 * i)), i


def test_a_start_between_32khz_samples_is_refused_when_its_window_is_read(runtime, tmp_path):
    card = changed_card(window_overlap=2.4999847412109375)
    model = runtime.adapter.build(perch_context(tmp_path, card=card))
    windows = model.run(AudioClip(path=write_audio(tmp_path / "a.wav", noise(8 * 32000), 32000)))
    with pytest.raises(ValueError, match=r"2\.5000152587890625") as caught:
        list(windows)
    assert "32000" in str(caught.value)


@pytest.mark.parametrize(("rate", "runs"), [(48000, True), (44100, False)])
def test_a_start_on_the_grid_the_rates_share_runs_and_one_off_it_is_refused(
    runtime, tmp_path, rate, runs
):
    card = changed_card(window_overlap=4.9375)
    model = runtime.adapter.build(perch_context(tmp_path, card=card))
    samples = noise(6 * rate)
    path = write_audio(tmp_path / "a.wav", samples, rate)
    if runs:
        windows = run(model, path)
        assert [w.start for w in windows] == [0.0625 * i for i in range(17)]
        whole = whole_recording_resampled(samples, rate)
        for i, row in enumerate(rows_given_to(runtime)):
            assert np.array_equal(row, padded_slice(whole, 2000 * i)), i
    else:
        with pytest.raises(ValueError, match=r"0\.0625") as caught:
            run(model, path)
        assert str(rate) in str(caught.value)


def test_a_file_with_no_frames_is_refused_before_the_model_runs(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 0)
    with pytest.raises(ValueError, match="a.wav"):
        run(model, path)
    assert runtime.calls == []


# Scores and the sigmoid.

SEVEN_LOGITS = np.array([-120, -35.8, -1, 0, 1, 35.8, 120], dtype=np.float32)


def float64_sigmoid(x: np.ndarray) -> np.ndarray:
    e = np.exp(-np.abs(np.float64(x)))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def test_the_sigmoid_is_within_one_float32_step_and_raises_no_warning(runtime):
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        got = runtime.adapter.sigmoid(SEVEN_LOGITS)
    want = float64_sigmoid(SEVEN_LOGITS)
    assert got.dtype == np.float32
    assert not np.isnan(got).any()
    assert got[3] == 0.5
    for g, w in zip(got, want, strict=True):
        assert abs(np.float64(g) - w) <= np.spacing(np.float32(w))


def test_every_score_is_between_0_and_1_on_a_fine_grid(runtime):
    grid = np.linspace(-200, 200, 400_001, dtype=np.float32)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        scores = runtime.adapter.sigmoid(grid)
    assert scores.dtype == np.float32
    assert ((scores >= 0) & (scores <= 1)).all()


def test_run_gives_the_sigmoid_of_the_logits_exactly(runtime, tmp_path):
    runtime.logits = lambda batch: np.resize(SEVEN_LOGITS, (len(batch), 10932))
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    [window] = run(model, path)
    expected = runtime.adapter.sigmoid(np.resize(SEVEN_LOGITS, 10932)).tolist()
    assert [score.score for score in window.scores] == expected


def test_outputs_are_read_by_name_whatever_their_order(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 2 * WINDOW, emit_embeddings=True)
    windows = run(model, path)
    [call] = runtime.serving_calls()
    assert list(call.returned) == list(RETURNED_ORDER)
    for name in ("genus", "family", "order", "frontend"):
        assert np.isnan(call.returned[name]._values).all()
    assert sorted(runtime.read) == ["embedding", "label"]
    logits = call.returned["label"]._values
    embeddings = call.returned["embedding"]._values
    for window, row, embedding in zip(windows, logits, embeddings, strict=True):
        assert [s.score for s in window.scores] == runtime.adapter.sigmoid(row).tolist()
        assert np.array_equal(window.embedding, embedding)


def test_every_window_scores_every_label_once_in_registry_order(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 2 * WINDOW)
    windows = run(model, path)
    expected = runtime.adapter.sigmoid(fake_logits(rows_given_to(runtime)))
    for window, scores in zip(windows, expected, strict=True):
        assert tuple(score.label for score in window.scores) == LABELS
        assert all(type(score.score) is float for score in window.scores)
        assert [score.score for score in window.scores] == scores.tolist()
    assert windows[0].scores != windows[1].scores


def test_a_model_returning_fewer_logits_than_it_declares_fails_the_recording(runtime, tmp_path):
    runtime.logits = lambda batch: fake_logits(batch)[:, :10931]
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    with pytest.raises(ValueError, match="10931"):
        run(model, path)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan], ids=["inf", "minus_inf", "nan"])
def test_a_logit_that_is_not_finite_fails_the_recording_at_its_window(runtime, tmp_path, value):
    def logits(batch):
        out = fake_logits(batch)
        if len(runtime.serving_calls()) == 2:
            out[1, 100] = value
        return out

    runtime.logits = logits
    model, _, path = model_and_noise(runtime, tmp_path, 4 * WINDOW, resources={"batch_size": 2})
    yielded = []
    with pytest.raises(ValueError) as caught:
        for window in model.run(AudioClip(path=path)):
            yielded.append(window)
    message = str(caught.value)
    assert "15.0" in message
    assert repr(LABELS[100]) in message
    assert repr(float(value)) in message
    assert [window.start for window in yielded] == [0.0, 5.0]


# Embeddings, and both outputs at once.


def test_each_window_gets_its_own_copy_of_its_raw_embedding(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 3 * WINDOW, emit_embeddings=True)
    windows = run(model, path)
    [call] = runtime.serving_calls()
    expected = call.returned["embedding"]._values
    for window, row in zip(windows, expected, strict=True):
        embedding = window.embedding
        assert embedding.shape == (1280,)
        assert embedding.dtype == np.float32
        assert embedding.flags["C_CONTIGUOUS"]
        assert np.array_equal(embedding, row)
        assert not np.shares_memory(embedding, expected)
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        assert not np.shares_memory(windows[first].embedding, windows[second].embedding)


def test_without_emit_embeddings_no_embedding_is_read(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    [window] = run(model, path)
    assert window.embedding is None
    assert runtime.read == ["label"]


@pytest.mark.parametrize("asked", ["scores", "embeddings", "both"])
def test_each_batch_takes_one_call(runtime, tmp_path, asked):
    resources = {"batch_size": 2}
    if asked == "embeddings":
        context = embeddings_only(tmp_path, emit_embeddings=True, resources=resources)
    else:
        context = perch_context(tmp_path, emit_embeddings=asked == "both", resources=resources)
    model = runtime.adapter.build(context)
    windows = run(model, write_audio(tmp_path / "a.wav", noise(4 * WINDOW), 32000))
    assert len(windows) == 4
    assert len(runtime.calls) == len(runtime.serving_calls()) == 2


def test_asking_for_embeddings_does_not_change_a_score(runtime, tmp_path):
    path = write_audio(tmp_path / "a.wav", noise(3 * WINDOW), 32000)
    alone = run(runtime.adapter.build(perch_context(tmp_path)), path)
    both = run(runtime.adapter.build(perch_context(tmp_path, emit_embeddings=True)), path)
    assert [w.scores for w in both] == [w.scores for w in alone]


def test_with_a_registry_scores_come_even_when_only_embeddings_are_asked_for(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW, emit_embeddings=True)
    [window] = run(model, path)
    assert len(window.scores) == 10932
    assert window.embedding is not None


def test_without_a_registry_no_scores_come_and_the_label_output_is_not_read(runtime, tmp_path):
    runtime.logits = lambda batch: np.full((len(batch), 10932), np.nan, dtype=np.float32)
    model = runtime.adapter.build(embeddings_only(tmp_path, emit_embeddings=True))
    windows = run(model, write_audio(tmp_path / "a.wav", noise(2 * WINDOW), 32000))
    assert [window.scores for window in windows] == [(), ()]
    assert all(window.embedding is not None for window in windows)
    assert "label" not in runtime.read


# Batching and the device.


def test_windows_are_batched_within_one_recording_only(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 7 * WINDOW, resources={"batch_size": 3})
    run(model, path)
    run(model, write_audio(tmp_path / "b.wav", noise(2 * WINDOW, seed=1), 32000))
    batches = [call.given for call in runtime.serving_calls()]
    assert [len(batch) for batch in batches] == [3, 3, 1, 2]
    for batch in batches:
        assert batch.dtype == np.float32
        assert batch.shape[1] == WINDOW
        assert batch.flags["C_CONTIGUOUS"]


def test_with_a_gpu_visible_everything_runs_on_the_cpu(runtime, tmp_path):
    runtime.gpu_visible = True
    model, _, path = model_and_noise(runtime, tmp_path, 3 * WINDOW, emit_embeddings=True)
    run(model, path)
    assert [load.device for load in runtime.loads] == [CPU]
    assert len(runtime.calls) == len(runtime.converted) == 1
    assert {call.device for call in runtime.calls} == {CPU}
    assert {converted.device for converted in runtime.converted} == {CPU}
    assert {converted.dtype for converted in runtime.converted} == {"float32"}


# The model's lifecycle.


class RecordingSoundfile:
    """The real soundfile module, recording which of its functions were called."""

    def __init__(self) -> None:
        self.called: list[str] = []

    def __getattr__(self, name):
        real = getattr(soundfile, name)

        def recorded(*args, **kwargs):
            self.called.append(name)
            return real(*args, **kwargs)

        return recorded


def test_run_reads_nothing_until_it_is_iterated(runtime, tmp_path, monkeypatch):
    recorder = RecordingSoundfile()
    monkeypatch.setattr(runtime.adapter, "sf", recorder)
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    windows = model.run(AudioClip(path=path))
    assert recorder.called == []
    next(windows)
    assert recorder.called != []


def test_run_refuses_an_input_that_is_not_audio_before_reading_anything(
    runtime, tmp_path, monkeypatch
):
    recorder = RecordingSoundfile()
    monkeypatch.setattr(runtime.adapter, "sf", recorder)
    model = runtime.adapter.build(perch_context(tmp_path))
    embeddings = Embeddings(
        starts=np.array([0.0]),
        ends=np.array([5.0]),
        values=np.zeros((1, 1), dtype=np.float32),
    )
    with pytest.raises(TypeError, match="Embeddings"):
        model.run(embeddings)
    assert recorder.called == []


def test_a_file_that_cannot_be_read_fails_the_recording(runtime, tmp_path):
    model = runtime.adapter.build(perch_context(tmp_path))
    with pytest.raises(soundfile.LibsndfileError):
        run(model, tmp_path / "missing.wav")
    assert runtime.calls == []


def test_a_model_failure_fails_the_recording(runtime, tmp_path):
    runtime.call_error = RuntimeError("the model failed")
    model, _, path = model_and_noise(runtime, tmp_path, 2 * WINDOW)
    yielded = []
    with pytest.raises(RuntimeError, match="the model failed"):
        for window in model.run(AudioClip(path=path)):
            yielded.append(window)
    assert yielded == []


def test_the_model_loads_once_and_stays_loaded_after_each_recording(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    run(model, path)
    model.after_recording()
    assert len(run(model, path)) == 1
    model.after_recording()
    assert len(runtime.loads) == 1
    assert len(runtime.serving_calls()) == 2


def test_clean_up_releases_the_loaded_model_and_its_signature(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    run(model, path)
    [loaded] = runtime.loaded
    watched = [loaded, weakref.ref(loaded().signatures["serving_default"])]
    gc.collect()
    assert all(ref() is not None for ref in watched)
    model.clean_up()
    gc.collect()
    assert all(ref() is None for ref in watched)


# Through the engine.


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(path) -> PinnedFile:
    return PinnedFile(
        uri=str(path), digest=f"sha256:{sha256(path)}", size_bytes=path.stat().st_size
    )


def test_a_work_runs_through_the_engine_to_scores_and_embeddings(runtime, tmp_path):
    audio = write_audio(tmp_path / "a.wav", noise(7 * 48000), 48000)
    model_files = perch_files(tmp_path)
    work = InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=(
            RecordingRef(
                namespace="test", value="seven-seconds", audio_uri=str(audio), duration_seconds=7.0
            ),
        ),
        model=PinnedModel(
            card=CARD, files={role: pinned(path) for role, path in model_files.items()}
        ),
        input=AudioInput(),
        settings={},
        resources={"batch_size": 2},
        outputs=(
            ScoresRequest(contract_id="robin.scores.arrow/1", retention="full"),
            EmbeddingsRequest(contract_id="robin.embeddings.arrow/1"),
        ),
    )
    root = tmp_path / "published"
    result = run_work(
        work,
        model_files=LocalFileProvider(),
        inputs=LocalFileProvider(),
        artifacts=LocalArtifactWriter(root),
    )
    assert isinstance(result, InferenceSuccess), result
    records = {record.kind: record for record in result.artifacts}
    path = root / artifact_path("scores", "test", "seven-seconds")
    with read_scores(path, expected_checksum=records["scores"].checksum) as stream:
        scores = pa.Table.from_batches(list(stream.batches))
    assert scores.num_rows == 2 * 10932
    assert Counter(scores.column("window_start_s").to_pylist()) == {0.0: 10932, 5.0: 10932}
    path = root / artifact_path("embeddings", "test", "seven-seconds")
    with read_embeddings(path, expected_checksum=records["embeddings"].checksum) as stream:
        embeddings = pa.Table.from_batches(list(stream.batches))
    assert embeddings.num_rows == 2
    assert {len(row) for row in embeddings.column("embedding").to_pylist()} == {1280}
