"""BirdNET v2.4: its bundled card and registry, and the adapter against a fake TensorFlow."""

import contextlib
import csv
import gc
import hashlib
import importlib
import os
import sys
import types
import weakref
from collections import Counter
from importlib.resources import as_file, files

import numpy as np
import pyarrow as pa
import pytest
import scipy.signal
import soundfile

import robin_models.birdnet
from robin_adapters.artifact_writer.local import LocalArtifactWriter
from robin_adapters.file_provider.local import LocalFileProvider
from robin_contracts.cards import ModelCard, model_ref, read_card
from robin_contracts.inputs import AudioClip, Embeddings
from robin_contracts.layout import artifact_path
from robin_contracts.output_contracts import EmbeddingsRequest, ScoresRequest
from robin_contracts.protocols import Model, ModelContext
from robin_contracts.records import ClassScore
from robin_contracts.results import InferenceCompleted
from robin_contracts.work import AudioInput, InferenceWork, PinnedFile, PinnedModel, RecordingRef
from robin_inference_engine.artifacts.embeddings import read_embeddings
from robin_inference_engine.artifacts.scores import read_scores
from robin_inference_engine.engine import run_work
from robin_inference_engine.load_registry import load_registry

RESOURCES = files("robin_models.birdnet") / "resources"

with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)
    with _path.open(newline="", encoding="utf-8") as _file:
        REGISTRY_ROWS = list(csv.DictReader(_file))

REGISTRY_SHA256 = "b758a58bee475d45b1fb04a8f42cbaa13c794d2dc782f0e86968e6c1515d4c2d"


def test_the_card_states_every_field():
    assert CARD.model_dump() == {
        "model_name": "birdnet",
        "model_version": "v2p4",
        "runtime": "tensorflow",
        "window_duration": 3.0,
        "sample_rate": 48000,
        "min_detection_threshold": 0.0,
        "score_domain": "sigmoid",
        "taxa_registry_digest": "sha256:" + REGISTRY_SHA256,
        "spectrogram_shape": None,
        "audio": {
            "downmix": "mean",
            "resampler": {"by": "runner", "algorithm": "scipy_fft_per_window"},
            "pad": "centre_crop_end_pad",
        },
        "backend": "pb-fp32",
        "dtype": "float32",
        "inference_params": ({"name": "window_overlap", "type": "float"},),
        "can_emit_embeddings": True,
        "embedding_dim": 1024,
        "embedding_dtype": "float32",
    }


def test_the_card_reads_as_birdnet_v2p4():
    assert model_ref(CARD).id == "birdnet/v2p4"


def test_the_card_names_the_bundled_registry_by_its_digest():
    digest = hashlib.sha256((RESOURCES / "taxa_registry.csv").read_bytes()).hexdigest()

    assert digest == REGISTRY_SHA256
    assert CARD.taxa_registry_digest == "sha256:" + digest


def test_the_registry_loads_with_6522_entries_of_three_kinds():
    kinds = Counter(entry.label_kind for entry in REGISTRY.entries)

    assert len(REGISTRY.entries) == 6522
    assert kinds == {"taxon": 6510, "non_taxonomic": 11, "unresolved": 1}


def test_every_rows_label_kind_follows_from_its_kind():
    def expected(kind: str) -> str:
        return {"non_taxon": "non_taxonomic", "unresolved": "unresolved"}.get(kind, "taxon")

    for row in REGISTRY_ROWS:
        assert row["label_kind"] == expected(row["kind"]), row


def test_the_one_unresolved_class_is_the_glossy_backed_drongo_with_no_key():
    [entry] = [entry for entry in REGISTRY.entries if entry.label_kind == "unresolved"]

    assert entry.class_index == 1918
    assert entry.label == "Dicrurus divaricatus_Glossy-backed Drongo"
    assert entry.scientific_name == "Dicrurus divaricatus"
    assert entry.gbif_taxon_key is None


def test_the_non_taxonomic_classes_have_no_name_and_no_key():
    rows = [entry for entry in REGISTRY.entries if entry.label_kind == "non_taxonomic"]

    assert len(rows) == 11
    for entry in rows:
        assert entry.scientific_name is None
        assert entry.gbif_taxon_key is None


# The adapter, run against a fake TensorFlow. NumPy, SciPy and soundfile are real.

ADAPTER = "robin_models.birdnet.adapter"
LABELS = tuple(entry.label for entry in REGISTRY.entries)
AUDIO = CARD.audio.model_dump()
WINDOW = 144_000
CPU = "/CPU:0"
KERNEL = "CLASS_DENSE_LAYER/kernel:0"
BIAS = "CLASS_DENSE_LAYER/bias:0"


class FakeSpec:
    """A tensor spec: only its shape, as a list."""

    def __init__(self, shape: list) -> None:
        self.shape = types.SimpleNamespace(as_list=lambda: list(shape))


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def numpy(self) -> np.ndarray:
        return np.array(self._values, dtype=np.float32, copy=True)


class FakeProduct(FakeTensor):
    """What `tf.matmul` returns: adding the bias to it is recorded."""

    def __init__(self, runtime: "FakeTensorFlow", embeddings: np.ndarray, values) -> None:
        super().__init__(values)
        self._runtime = runtime
        self._embeddings = embeddings

    def __add__(self, other: "FakeVariable") -> FakeTensor:
        runtime = self._runtime
        runtime.additions.append(
            types.SimpleNamespace(added=weakref.ref(other), device=runtime.device)
        )
        if runtime.logits is not None:
            return FakeTensor(np.asarray(runtime.logits(self._embeddings), dtype=np.float32))
        return FakeTensor((self._values + other.values).astype(np.float32))


class FakeVariable:
    """A variable of the loaded model: its name, its shape, and its values."""

    def __init__(self, name: str, values: np.ndarray) -> None:
        self.name = name
        self.values = np.asarray(values, dtype=np.float32)
        shape = list(self.values.shape)
        self.shape = types.SimpleNamespace(as_list=lambda: list(shape))


class FakeSignature:
    """One SavedModel signature. It refers to the runtime, never to the loaded object."""

    def __init__(self, runtime: "FakeTensorFlow", output: str, spec: dict) -> None:
        self._runtime = runtime
        self._output = output
        self.structured_input_signature = ((), {spec["input_name"]: FakeSpec(spec["input"])})
        self.structured_outputs = {output: FakeSpec(spec["output"])}

    def __call__(self, *, inputs):
        runtime = self._runtime
        call = types.SimpleNamespace(
            output=self._output,
            given=inputs,
            values=np.array(inputs, copy=True),
            device=runtime.device,
            returned=None,
        )
        runtime.calls.append(call)
        if runtime.call_error is not None:
            raise runtime.call_error
        call.returned = FakeTensor(runtime.outputs[self._output](np.asarray(inputs)))
        return {self._output: call.returned}


class FakeLoaded:
    """What `tf.saved_model.load` returns: its signatures, and the model holding its variables."""

    def __init__(self, signatures: dict, variables: list | None) -> None:
        self.signatures = signatures
        if variables is not None:
            self.model = types.SimpleNamespace(variables=variables)


class FakeTensorFlow:
    """Records what the adapter asked of TensorFlow, and fails where told to."""

    def __init__(self) -> None:
        self.device = None
        self.converted: list = []
        self.loads: list = []
        self.load_error: Exception | None = None
        self.loaded: list[weakref.ref] = []
        self.call_error: Exception | None = None
        self.calls: list = []
        self.signatures = {
            "basic": {"input_name": "inputs", "input": [None, WINDOW], "output": [None, 6522]},
            "embeddings": {
                "input_name": "inputs",
                "input": [None, WINDOW],
                "output": [None, 1024],
            },
        }
        self.missing: set[str] = set()
        self.outputs = {"scores": self.default_scores, "embeddings": self.default_embeddings}
        # The classifier layer's values by variable name, made into new variables on each load.
        self.head = {KERNEL: HEAD_KERNEL, BIAS: HEAD_BIAS}
        self.has_model = True
        self.matmuls: list = []
        self.additions: list = []
        self.logits = None

    def module(self) -> types.ModuleType:
        tf = types.ModuleType("tensorflow")
        tf.float32 = "float32"
        tf.device = self.on_device
        tf.convert_to_tensor = self.convert_to_tensor
        tf.matmul = self.matmul
        tf.saved_model = types.SimpleNamespace(load=self.load)
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

    def matmul(self, a: FakeTensor, b: FakeVariable) -> FakeProduct:
        # The variable is held weakly, so recording it does not keep it alive.
        self.matmuls.append(types.SimpleNamespace(a=a, b=weakref.ref(b), device=self.device))
        # OpenBLAS on arm64 macOS raises floating-point warnings for some finite products.
        with np.errstate(all="ignore"):
            product = np.matmul(a._values, b.values).astype(np.float32)
        return FakeProduct(self, a._values, product)

    def load(self, path):
        folder = os.fspath(path)
        self.loads.append(
            types.SimpleNamespace(path=folder, device=self.device, contents=folder_contents(folder))
        )
        if self.load_error is not None:
            raise self.load_error
        signatures = {
            name: FakeSignature(self, output, self.signatures[name])
            for name, output in (("basic", "scores"), ("embeddings", "embeddings"))
            if name not in self.missing
        }
        variables = [FakeVariable("POST_BN_1/gamma:0", np.ones(1024))]
        variables += [FakeVariable(name, values) for name, values in self.head.items()]
        loaded = FakeLoaded(signatures, variables if self.has_model else None)
        self.loaded.append(weakref.ref(loaded))
        return loaded

    def default_scores(self, batch: np.ndarray) -> np.ndarray:
        return head_logits(self.default_embeddings(batch))

    def default_embeddings(self, batch: np.ndarray) -> np.ndarray:
        return fake_embeddings(batch, self.signatures["embeddings"]["output"][1])

    def calls_to(self, output: str) -> list:
        return [call for call in self.calls if call.output == output]


def folder_contents(folder: str) -> dict[str, tuple[bool, str | None]]:
    """Each file under `folder`: whether it is a symbolic link, and where it points."""
    contents = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            link = os.path.islink(path)
            contents[os.path.relpath(path, folder)] = (link, os.readlink(path) if link else None)
    return contents


LABEL_ORDER = np.random.default_rng(0).permutation(6523)

# The default classifier layer gives each label its bias plus the embedding's element 0,
# which is the window's mean: logits distinct per label and per window, from below -15 to
# above 15.
HEAD_KERNEL = np.zeros((1024, 6522), dtype=np.float32)
HEAD_KERNEL[0] = 1.0
HEAD_BIAS = np.linspace(-20.0, 20.0, 6523)[LABEL_ORDER][:6522].astype(np.float32)


def head_logits(embeddings: np.ndarray) -> np.ndarray:
    """The default classifier layer's logits for these embeddings."""
    return (embeddings[:, :1] + HEAD_BIAS).astype(np.float32)


def fake_embeddings(batch: np.ndarray, width: int) -> np.ndarray:
    """Finite embeddings, distinct per window. Element 0 is the window's mean."""
    offset = np.float64(batch).mean(axis=1, keepdims=True)
    return (np.arange(width) / width + offset).astype(np.float32)


@pytest.fixture
def runtime(monkeypatch):
    """Import the adapter against a fake TensorFlow, and remove it afterwards."""
    fake = FakeTensorFlow()
    monkeypatch.setitem(sys.modules, "tensorflow", fake.module())
    sys.modules.pop(ADAPTER, None)
    fake.adapter = importlib.import_module(ADAPTER)
    yield fake
    sys.modules.pop(ADAPTER, None)
    if hasattr(robin_models.birdnet, "adapter"):
        delattr(robin_models.birdnet, "adapter")


def birdnet_files(tmp_path, labels: str | bytes | None = None) -> dict:
    """Small stand-ins for the pinned files. The fake TensorFlow reads none of them."""
    paths = {
        "saved_model": tmp_path / "saved_model.pb",
        "variables_data": tmp_path / "variables.data-00000-of-00001",
        "variables_index": tmp_path / "variables.index",
    }
    for role, path in paths.items():
        path.write_bytes(role.encode())
    labels_file = tmp_path / "en_us.txt"
    text = "\n".join(LABELS) if labels is None else labels
    labels_file.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    registry = tmp_path / "taxa_registry.csv"
    registry.write_bytes((RESOURCES / "taxa_registry.csv").read_bytes())
    return paths | {"labels": labels_file, "taxa_registry": registry}


def birdnet_context(tmp_path, **changes) -> ModelContext:
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
        fields["files"] = birdnet_files(tmp_path)
    return ModelContext(**fields)


def without(files: dict, *roles: str) -> dict:
    return {role: path for role, path in files.items() if role not in roles}


def changed_card(**changes) -> ModelCard:
    return ModelCard.model_validate(CARD.model_dump() | changes)


# What BirdNET does, whatever its card says.


@pytest.mark.parametrize(
    ("field", "changes"),
    [
        pytest.param("window_duration", {"window_duration": 5.0}, id="window_duration"),
        pytest.param("sample_rate", {"sample_rate": 32000}, id="sample_rate"),
        pytest.param("audio", {"audio": {**AUDIO, "downmix": "first"}}, id="downmix"),
        pytest.param(
            "audio",
            {"audio": {**AUDIO, "resampler": {"by": "runner", "algorithm": "soxr_hq"}}},
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
        pytest.param("spectrogram_shape", {"spectrogram_shape": [96, 511]}, id="spectrogram"),
        pytest.param(
            "can_emit_embeddings",
            {"can_emit_embeddings": False, "embedding_dim": None, "embedding_dtype": None},
            id="can_emit_embeddings",
        ),
        pytest.param("embedding_dim", {"embedding_dim": 512}, id="embedding_dim"),
        pytest.param("embedding_dtype", {"embedding_dtype": "float16"}, id="embedding_dtype"),
        pytest.param("backend", {"backend": "onnx-fp32"}, id="backend"),
    ],
)
def test_a_card_stating_what_birdnet_does_not_do_is_refused_before_the_model_loads(
    runtime, tmp_path, field, changes
):
    card = changed_card(**changes)
    with pytest.raises(ValueError, match=f"card field {field} is") as caught:
        runtime.adapter.build(birdnet_context(tmp_path, card=card))
    message = str(caught.value)
    assert repr(getattr(card, field)) in message
    assert f"model birdnet/v2p4 does {runtime.adapter.BEHAVIOUR[field]!r}" in message
    assert runtime.loads == []


def test_overlapping_windows_build(runtime, tmp_path):
    runtime.adapter.build(birdnet_context(tmp_path, settings={"window_overlap": 1.0}))


# Resources.


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        pytest.param({}, 32, id="none_given"),
        pytest.param({"batch_size": 8}, 8, id="given"),
        pytest.param({"device": "cpu", "batch_size": 1}, 1, id="beside_the_device"),
        pytest.param({"device": "cpu"}, 32, id="only_the_device"),
    ],
)
def test_the_batch_size_is_read_from_the_resources(runtime, resources, expected):
    assert runtime.adapter._read_batch_size(resources) == expected


@pytest.mark.parametrize("value", [0, -1, True, 32.0, "8", None], ids=repr)
def test_a_batch_size_that_is_not_a_positive_int_is_refused(runtime, value):
    with pytest.raises(ValueError, match="batch_size") as caught:
        runtime.adapter._read_batch_size({"batch_size": value})
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize("resources", [{}, {"device": "cpu"}], ids=["absent", "cpu"])
def test_the_cpu_is_the_device_that_builds(runtime, tmp_path, resources):
    runtime.adapter.build(birdnet_context(tmp_path, resources=resources))


@pytest.mark.parametrize("value", ["gpu", "gpu:0", "CPU", "/CPU:0", 0, None], ids=repr)
def test_any_other_device_is_refused(runtime, tmp_path, value):
    with pytest.raises(ValueError, match="device") as caught:
        runtime.adapter.build(birdnet_context(tmp_path, resources={"device": value}))
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize(
    "resources", [{"batch_size": 0}, {"device": "gpu"}], ids=["batch_size", "device"]
)
def test_a_bad_resource_is_refused_before_the_model_loads(runtime, tmp_path, resources):
    with pytest.raises(ValueError):
        runtime.adapter.build(birdnet_context(tmp_path, resources=resources))
    assert runtime.loads == []


# The pinned files.


@pytest.mark.parametrize("role", ["saved_model", "variables_data", "variables_index"])
def test_a_missing_savedmodel_file_is_refused_naming_the_role(runtime, tmp_path, role):
    files = without(birdnet_files(tmp_path), role)
    with pytest.raises(ValueError, match=f"'{role}'") as caught:
        runtime.adapter.build(birdnet_context(tmp_path, files=files))
    assert str(sorted(files)) in str(caught.value)
    assert runtime.loads == []


def test_with_a_registry_a_missing_labels_file_is_refused_naming_the_role(runtime, tmp_path):
    files = without(birdnet_files(tmp_path), "labels")
    with pytest.raises(ValueError, match="'labels'") as caught:
        runtime.adapter.build(birdnet_context(tmp_path, files=files))
    assert str(sorted(files)) in str(caught.value)
    assert runtime.loads == []


def test_without_a_registry_neither_labels_nor_registry_file_is_needed(runtime, tmp_path):
    files = without(birdnet_files(tmp_path), "labels", "taxa_registry")
    runtime.adapter.build(birdnet_context(tmp_path, files=files, registry=None))


def test_the_savedmodel_folder_is_rebuilt_from_links_to_the_pinned_files(runtime, tmp_path):
    context = birdnet_context(tmp_path)
    runtime.adapter.build(context)
    [load] = runtime.loads
    folder = os.path.realpath(load.path)
    assert os.path.commonpath([folder, os.path.realpath(context.scratch_dir)]) == (
        os.path.realpath(context.scratch_dir)
    )
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
    assert load.device == CPU


def test_two_models_sharing_a_scratch_dir_get_separate_folders(runtime, tmp_path):
    context = birdnet_context(tmp_path)
    runtime.adapter.build(context)
    runtime.adapter.build(context)
    first, second = runtime.loads
    assert first.path != second.path


# The SavedModel's signatures.


def test_without_the_basic_signature_the_model_builds(runtime, tmp_path):
    runtime.missing.add("basic")
    runtime.adapter.build(birdnet_context(tmp_path))


def test_a_missing_signature_is_refused(runtime, tmp_path):
    runtime.missing.add("embeddings")
    with pytest.raises(ValueError, match="embeddings") as caught:
        runtime.adapter.build(birdnet_context(tmp_path))
    assert "['basic']" in str(caught.value)


@pytest.mark.parametrize(
    ("spec", "expected", "found"),
    [
        pytest.param({"input_name": "audio"}, "'inputs'", "['audio']", id="input_name"),
        pytest.param(
            {"input": [None, 96000]}, "[None, 144000]", "[None, 96000]", id="input_width"
        ),
        pytest.param({"input": [1, WINDOW]}, "[None, 144000]", "[1, 144000]", id="fixed_batch"),
    ],
)
def test_an_input_the_card_does_not_describe_is_refused(runtime, tmp_path, spec, expected, found):
    runtime.signatures["embeddings"].update(spec)
    with pytest.raises(ValueError, match="embeddings") as caught:
        runtime.adapter.build(birdnet_context(tmp_path))
    assert expected in str(caught.value)
    assert found in str(caught.value)


def test_an_embedding_width_other_than_the_cards_is_refused(runtime, tmp_path):
    runtime.signatures["embeddings"]["output"] = [None, 512]
    with pytest.raises(ValueError, match="embeddings") as caught:
        runtime.adapter.build(birdnet_context(tmp_path))
    assert "[None, 1024]" in str(caught.value)
    assert "[None, 512]" in str(caught.value)


# The SavedModel's classifier layer.


def zeros(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


@pytest.mark.parametrize(
    ("head", "named", "expected", "found"),
    [
        pytest.param({KERNEL: None}, KERNEL, None, None, id="no_kernel"),
        pytest.param({BIAS: None}, BIAS, None, None, id="no_bias"),
        pytest.param(
            {KERNEL: zeros(1024, 6521)}, KERNEL, "[1024, 6522]", "[1024, 6521]", id="kernel_labels"
        ),
        pytest.param(
            {KERNEL: zeros(512, 6522)}, KERNEL, "[1024, 6522]", "[512, 6522]", id="kernel_width"
        ),
        pytest.param({BIAS: zeros(6521)}, BIAS, "[6522]", "[6521]", id="bias_labels"),
    ],
)
def test_with_a_registry_a_classifier_layer_that_does_not_fit_is_refused(
    runtime, tmp_path, head, named, expected, found
):
    for name, values in head.items():
        if values is None:
            del runtime.head[name]
        else:
            runtime.head[name] = values
    with pytest.raises(ValueError) as caught:
        runtime.adapter.build(birdnet_context(tmp_path))
    message = str(caught.value)
    assert repr(named) in message
    if expected is not None:
        assert expected in message
        assert found in message


def test_with_a_registry_a_loaded_model_without_its_variables_is_refused(runtime, tmp_path):
    runtime.has_model = False
    with pytest.raises(ValueError) as caught:
        runtime.adapter.build(birdnet_context(tmp_path))
    assert "'model'" in str(caught.value)


@pytest.mark.parametrize("missing", ["head", "model"])
def test_without_a_registry_the_classifier_layer_is_not_looked_for(runtime, tmp_path, missing):
    if missing == "head":
        runtime.head.clear()
    else:
        runtime.has_model = False
    runtime.adapter.build(embeddings_only(tmp_path, emit_embeddings=True))


# The labels file shipped with the weights.


def labels_file_context(tmp_path, labels: str | bytes) -> ModelContext:
    return birdnet_context(tmp_path, files=birdnet_files(tmp_path, labels=labels))


@pytest.mark.parametrize("ending", ["", "\n"], ids=["as_shipped", "trailing_newline"])
def test_a_labels_file_equal_to_the_registry_builds(runtime, tmp_path, ending):
    runtime.adapter.build(labels_file_context(tmp_path, "\n".join(LABELS) + ending))


@pytest.mark.parametrize(
    ("lines", "count"),
    [(LABELS[:-1], 6521), (LABELS + ("Extra species_Extra",), 6523)],
    ids=["one_too_few", "one_too_many"],
)
def test_a_labels_file_of_another_length_is_refused(runtime, tmp_path, lines, count):
    with pytest.raises(ValueError, match="labels") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, "\n".join(lines)))
    assert str(count) in str(caught.value)
    assert "6522" in str(caught.value)


def test_a_labels_file_in_another_order_is_refused_at_the_first_difference(runtime, tmp_path):
    lines = list(LABELS)
    lines[10], lines[11] = lines[11], lines[10]
    with pytest.raises(ValueError, match="position 10") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, "\n".join(lines)))
    assert repr(LABELS[10]) in str(caught.value)
    assert repr(LABELS[11]) in str(caught.value)


def test_a_labels_file_with_another_common_name_is_refused(runtime, tmp_path):
    lines = list(LABELS)
    lines[0] = lines[0].split("_")[0] + "_Another Name"
    with pytest.raises(ValueError, match="position 0") as caught:
        runtime.adapter.build(labels_file_context(tmp_path, "\n".join(lines)))
    assert repr(LABELS[0]) in str(caught.value)
    assert repr(lines[0]) in str(caught.value)


def test_a_labels_file_that_is_not_utf8_is_refused(runtime, tmp_path):
    text = "\n".join(LABELS).encode("utf-8") + b"\n\xff"
    with pytest.raises(ValueError, match="UTF-8"):
        runtime.adapter.build(labels_file_context(tmp_path, text))


def test_a_model_that_fails_to_load_fails_the_build(runtime, tmp_path):
    runtime.load_error = OSError("not a SavedModel")
    with pytest.raises(OSError, match="not a SavedModel"):
        runtime.adapter.build(birdnet_context(tmp_path))


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


def rows_given_to(runtime, output: str = "embeddings") -> np.ndarray:
    return np.concatenate([call.values for call in runtime.calls_to(output)])


def model_and_noise(runtime, tmp_path, frames: int, rate: int = 48000, **changes):
    model = runtime.adapter.build(birdnet_context(tmp_path, **changes))
    samples = noise(frames)
    return model, samples, write_audio(tmp_path / "a.wav", samples, rate)


def embeddings_only(tmp_path, **changes) -> ModelContext:
    files = without(birdnet_files(tmp_path), "labels", "taxa_registry")
    return birdnet_context(tmp_path, files=files, registry=None, **changes)


def reference_sigmoid(x: np.ndarray) -> np.ndarray:
    """flat_sigmoid_logaddexp_fast from birdnet 0.2.12, at sensitivity -1.0 and bias 1.0."""
    y = -np.clip(x, -15, 15)
    e = np.exp(-np.abs(y), dtype=np.float32)
    return np.where(y >= 0, e / (1 + e), 1 / (1 + e))


def test_a_built_model_satisfies_the_protocol_and_restates_nothing_from_its_card(
    runtime, tmp_path
):
    model = runtime.adapter.build(birdnet_context(tmp_path))
    assert isinstance(model, Model)
    assert not hasattr(model, "recipe")
    assert not hasattr(model, "capabilities")


# Windowing.


@pytest.mark.parametrize(
    ("frames", "count"),
    [(2 * WINDOW - 1, 2), (2 * WINDOW, 2), (2 * WINDOW + 1, 3)],
    ids=["one_frame_short", "exact", "one_frame_over"],
)
def test_a_48khz_file_yields_one_window_per_started_3_seconds(runtime, tmp_path, frames, count):
    model, _, path = model_and_noise(runtime, tmp_path, frames)
    windows = run(model, path)
    assert [(w.start, w.end) for w in windows] == [(3.0 * i, 3.0 * i + 3.0) for i in range(count)]


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
    model = runtime.adapter.build(birdnet_context(tmp_path))
    samples = noise(WINDOW, channels=2)
    run(model, write_audio(tmp_path / "a.wav", samples, 48000))
    expected = np.mean(samples, axis=1, dtype=np.float32)
    assert np.array_equal(rows_given_to(runtime)[0], expected)


@pytest.mark.parametrize("rate", [44100, 32000])
def test_a_window_at_another_rate_is_resampled_on_its_own_with_scipy(runtime, tmp_path, rate):
    model, samples, path = model_and_noise(runtime, tmp_path, 3 * rate, rate=rate)
    run(model, path)
    expected = scipy.signal.resample(samples, WINDOW)
    row = rows_given_to(runtime)[0]
    assert len(expected) == WINDOW
    assert np.array_equal(row, expected)


def test_a_short_last_window_is_resampled_then_padded(runtime, tmp_path):
    rate = 44100
    model, samples, path = model_and_noise(runtime, tmp_path, 3 * rate + rate, rate=rate)
    run(model, path)
    last = samples[3 * rate :]
    expected = np.zeros(WINDOW, dtype=np.float32)
    expected[:48000] = scipy.signal.resample(last, round(len(last) / rate * 48000))
    padded_first = scipy.signal.resample(np.pad(last, (0, 3 * rate - len(last))), WINDOW)
    row = rows_given_to(runtime)[1]
    assert np.array_equal(row, expected)
    assert not np.array_equal(row, padded_first.astype(np.float32))


def test_a_file_with_no_frames_is_refused_before_the_model_runs(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 0)
    with pytest.raises(ValueError, match="a.wav"):
        run(model, path)
    assert runtime.calls == []


@pytest.mark.parametrize("rate", [96000, 192000])
@pytest.mark.parametrize(
    ("frames", "start"), [(1, 0.0), ("window_and_one", 3.0)], ids=["one_frame", "window_and_one"]
)
def test_a_window_that_would_resample_to_nothing_is_refused_before_scipy_runs(
    runtime, tmp_path, monkeypatch, rate, frames, start
):
    frames = 3 * rate + 1 if frames == "window_and_one" else frames
    targets = []
    real = runtime.adapter.resample

    def recording_resample(window, target):
        targets.append(target)
        return real(window, target)

    monkeypatch.setattr(runtime.adapter, "resample", recording_resample)
    model, _, path = model_and_noise(runtime, tmp_path, frames, rate=rate)
    with pytest.raises(ValueError) as caught:
        run(model, path)
    message = str(caught.value)
    for named in (f"{start}", "1 frame", f"{rate}", "0 samples"):
        assert named in message
    assert 0 not in targets


def test_overlapping_windows_each_reach_the_model_as_their_own_slice(runtime, tmp_path):
    model = runtime.adapter.build(birdnet_context(tmp_path, settings={"window_overlap": 1.0}))
    n = 9 * 48000
    samples = np.arange(n, dtype=np.float32) / n
    windows = run(model, write_audio(tmp_path / "a.wav", samples, 48000))
    assert [w.start for w in windows] == [0.0, 2.0, 4.0, 6.0]
    for window, row in zip(windows, rows_given_to(runtime), strict=True):
        first = round(window.start * 48000)
        assert np.array_equal(row, samples[first : first + WINDOW])


def test_overlapping_windows_at_an_odd_rate_each_read_one_full_window(runtime, tmp_path):
    model = runtime.adapter.build(birdnet_context(tmp_path, settings={"window_overlap": 0.5}))
    samples = noise(12 * 11025)
    windows = run(model, write_audio(tmp_path / "a.wav", samples, 11025))
    full = [(w, row) for w, row in zip(windows, rows_given_to(runtime)) if w.end <= 12.0]
    assert [w.start for w, _ in full] == [0.0, 2.5, 5.0, 7.5]
    for window, row in full:
        first = round(window.start * 11025)
        expected = scipy.signal.resample(samples[first : first + 33_075], WINDOW)
        assert np.array_equal(row, expected.astype(np.float32))


# Scores and the sigmoid.

FIVE_LOGITS = np.array([-20, -15, 0, 15, 20], dtype=np.float32)


def test_the_sigmoid_is_the_birdnet_librarys_on_the_clip_edges(runtime):
    result = runtime.adapter.sigmoid(FIVE_LOGITS)
    assert result.dtype == np.float32
    assert np.array_equal(result, reference_sigmoid(FIVE_LOGITS))
    assert result[0] == result[1]
    assert result[2] == 0.5
    assert result[3] == result[4]


def test_the_sigmoid_is_the_birdnet_librarys_at_every_point_of_a_fine_grid(runtime):
    grid = np.linspace(-16, 16, 200_001, dtype=np.float32)
    expected = reference_sigmoid(grid)
    assert np.array_equal(runtime.adapter.sigmoid(grid), expected)
    # Another float32 sigmoid differs from the library's by a few float32 steps.
    assert not np.array_equal(1 / (1 + np.exp(-np.clip(grid, -15, 15))), expected)


def test_the_sigmoid_is_within_five_float32_epsilons_of_the_exact_one(runtime):
    grid = np.linspace(-16, 16, 200_001, dtype=np.float32)
    exact = 1 / (1 + np.exp(-np.clip(grid.astype(np.float64), -15, 15)))
    relative_error = np.abs(runtime.adapter.sigmoid(grid) - exact) / exact
    # On AVX-512 CPUs numpy's float32 exp may be off by 4 ULP. The addition and
    # the division each add at most half an epsilon.
    assert relative_error.max() <= 5 * np.finfo(np.float32).eps


def test_run_gives_the_sigmoid_of_the_logits_exactly(runtime, tmp_path):
    runtime.logits = lambda embeddings: np.resize(FIVE_LOGITS, (len(embeddings), 6522))
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    [window] = run(model, path)
    expected = np.resize(reference_sigmoid(FIVE_LOGITS), 6522).tolist()
    assert [score.score for score in window.scores] == expected


def loaded_classifier_layer(runtime) -> tuple[FakeVariable, FakeVariable]:
    [loaded] = runtime.loaded
    variables = {variable.name: variable for variable in loaded().model.variables}
    return variables[KERNEL], variables[BIAS]


def test_every_window_scores_every_label_once_in_registry_order(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 2 * WINDOW)
    windows = run(model, path)
    kernel, bias = loaded_classifier_layer(runtime)
    [call] = runtime.calls_to("embeddings")
    embeddings = call.returned.numpy()
    expected = reference_sigmoid(np.matmul(embeddings, kernel.values) + bias.values)
    [matmul] = runtime.matmuls
    assert matmul.a is call.returned
    assert matmul.b() is kernel
    [addition] = runtime.additions
    assert addition.added() is bias
    for window, scores in zip(windows, expected, strict=True):
        assert tuple(score.label for score in window.scores) == LABELS
        assert all(type(score.score) is float for score in window.scores)
        assert [score.score for score in window.scores] == scores.tolist()
    assert windows[0].scores != windows[1].scores


def test_a_model_returning_fewer_scores_than_it_declares_fails_the_recording(runtime, tmp_path):
    runtime.logits = lambda embeddings: head_logits(embeddings)[:, :6521]
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    with pytest.raises(ValueError, match="6521"):
        run(model, path)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan], ids=["inf", "minus_inf", "nan"])
def test_a_logit_that_is_not_finite_fails_the_recording_at_its_window(runtime, tmp_path, value):
    def logits(embeddings):
        out = head_logits(embeddings)
        if len(runtime.calls_to("embeddings")) == 2:
            out[1, 100] = value
        return out

    runtime.logits = logits
    model, _, path = model_and_noise(
        runtime, tmp_path, 4 * WINDOW, resources={"batch_size": 2}
    )
    yielded = []
    with pytest.raises(ValueError) as caught:
        for window in model.run(AudioClip(path=path)):
            yielded.append(window)
    message = str(caught.value)
    assert "9.0" in message
    assert repr(LABELS[100]) in message
    assert repr(float(value)) in message
    assert [window.start for window in yielded] == [0.0, 3.0]


# Embeddings, and both outputs at once.


def test_scores_and_embeddings_take_one_network_pass_per_batch(runtime, tmp_path):
    model, _, path = model_and_noise(
        runtime, tmp_path, 4 * WINDOW, emit_embeddings=True, resources={"batch_size": 2}
    )
    windows = run(model, path)
    assert len(runtime.calls_to("embeddings")) == 2
    assert len(runtime.matmuls) == 2
    assert len(runtime.additions) == 2
    assert runtime.calls_to("scores") == []
    assert all(len(window.scores) == 6522 for window in windows)
    assert all(window.embedding is not None for window in windows)


def test_scores_alone_come_from_the_embeddings_signature_and_emit_no_embedding(
    runtime, tmp_path
):
    model, _, path = model_and_noise(runtime, tmp_path, 4 * WINDOW, resources={"batch_size": 2})
    windows = run(model, path)
    assert len(runtime.calls_to("embeddings")) == 2
    assert runtime.calls_to("scores") == []
    assert all(len(window.scores) == 6522 for window in windows)
    assert all(window.embedding is None for window in windows)


def test_a_model_without_the_basic_signature_scores(runtime, tmp_path):
    runtime.missing.add("basic")
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    [window] = run(model, path)
    assert len(window.scores) == 6522
    assert runtime.calls_to("scores") == []


def test_each_window_gets_its_own_copy_of_its_embedding(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 3 * WINDOW, emit_embeddings=True)
    windows = run(model, path)
    expected = fake_embeddings(rows_given_to(runtime, "embeddings"), 1024)
    for window, row in zip(windows, expected, strict=True):
        embedding = window.embedding
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        assert embedding.flags["C_CONTIGUOUS"]
        assert np.array_equal(embedding, row)
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        assert not np.shares_memory(windows[first].embedding, windows[second].embedding)


def test_asking_for_embeddings_does_not_change_a_score(runtime, tmp_path):
    samples = noise(3 * WINDOW)
    path = write_audio(tmp_path / "a.wav", samples, 48000)
    alone = run(runtime.adapter.build(birdnet_context(tmp_path)), path)
    assert len(runtime.calls_to("embeddings")) == 1
    runtime.calls.clear()
    both = run(runtime.adapter.build(birdnet_context(tmp_path, emit_embeddings=True)), path)
    assert len(runtime.calls_to("embeddings")) == 1
    assert runtime.calls_to("scores") == []
    assert [w.scores for w in both] == [w.scores for w in alone]


def test_with_a_registry_scores_are_computed_even_when_only_embeddings_are_asked_for(
    runtime, tmp_path
):
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW, emit_embeddings=True)
    [window] = run(model, path)
    assert len(runtime.calls_to("embeddings")) == 1
    assert len(runtime.matmuls) == 1
    assert runtime.calls_to("scores") == []
    assert len(window.scores) == 6522


def test_without_a_registry_no_scores_are_computed(runtime, tmp_path):
    model = runtime.adapter.build(embeddings_only(tmp_path, emit_embeddings=True))
    windows = run(model, write_audio(tmp_path / "a.wav", noise(2 * WINDOW), 48000))
    assert runtime.calls_to("scores") == []
    assert runtime.matmuls == []
    assert [window.scores for window in windows] == [(), ()]
    assert all(window.embedding is not None for window in windows)


# Batching and the device.


def test_windows_are_batched_within_one_recording_only(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 5 * WINDOW, resources={"batch_size": 2})
    run(model, path)
    run(model, write_audio(tmp_path / "b.wav", noise(3 * WINDOW, seed=1), 48000))
    batches = [call.given for call in runtime.calls_to("embeddings")]
    assert [len(batch) for batch in batches] == [2, 2, 1, 2, 1]
    for batch in batches:
        assert batch.dtype == np.float32
        assert batch.shape[1] == WINDOW
        assert batch.flags["C_CONTIGUOUS"]


def test_every_call_runs_on_the_cpu(runtime, tmp_path):
    model, _, path = model_and_noise(runtime, tmp_path, 3 * WINDOW, emit_embeddings=True)
    run(model, path)
    assert [load.device for load in runtime.loads] == [CPU]
    assert {call.device for call in runtime.calls} == {CPU}
    assert {converted.device for converted in runtime.converted} == {CPU}
    assert {matmul.device for matmul in runtime.matmuls} == {CPU}
    assert {addition.device for addition in runtime.additions} == {CPU}
    assert len(runtime.matmuls) == len(runtime.additions) == 1
    assert len(runtime.calls) == 1


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
    model = runtime.adapter.build(birdnet_context(tmp_path))
    embeddings = Embeddings(
        starts=np.array([0.0]),
        ends=np.array([3.0]),
        values=np.zeros((1, 1), dtype=np.float32),
    )
    with pytest.raises(TypeError, match="Embeddings"):
        model.run(embeddings)
    assert recorder.called == []


def test_a_file_that_cannot_be_read_fails_the_recording(runtime, tmp_path):
    model = runtime.adapter.build(birdnet_context(tmp_path))
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
    assert len(runtime.calls_to("embeddings")) == 2


def test_clean_up_releases_the_loaded_model_its_signature_and_classifier_layer(
    runtime, tmp_path
):
    model, _, path = model_and_noise(runtime, tmp_path, WINDOW)
    run(model, path)
    [loaded] = runtime.loaded
    kernel, bias = loaded_classifier_layer(runtime)
    watched = [
        loaded,
        weakref.ref(loaded().signatures["embeddings"]),
        weakref.ref(kernel),
        weakref.ref(bias),
    ]
    del kernel, bias
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
    model_files = birdnet_files(tmp_path)
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
            ScoresRequest(contract_id="robin.scores.parquet/1", retention="full"),
            EmbeddingsRequest(contract_id="robin.embeddings.parquet/1"),
        ),
    )
    root = tmp_path / "published"
    result = run_work(
        work,
        model_files=LocalFileProvider(),
        inputs=LocalFileProvider(),
        artifacts=LocalArtifactWriter(root),
    )
    assert isinstance(result, InferenceCompleted), result
    assert result.failed == (), result.failed
    records = {record.kind: record for record in result.artifacts}
    path = root / artifact_path("scores", "test", "seven-seconds")
    with read_scores(path, expected_checksum=records["scores"].checksum) as stream:
        scores = pa.Table.from_batches(list(stream.batches))
    assert scores.num_rows == 3 * 6522
    assert Counter(scores.column("window_start_s").to_pylist()) == {0.0: 6522, 3.0: 6522, 6.0: 6522}
    path = root / artifact_path("embeddings", "test", "seven-seconds")
    with read_embeddings(path, expected_checksum=records["embeddings"].checksum) as stream:
        embeddings = pa.Table.from_batches(list(stream.batches))
    assert embeddings.num_rows == 3
    assert {len(row) for row in embeddings.column("embedding").to_pylist()} == {1024}
