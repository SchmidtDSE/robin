"""The ONNX head runtime, against a fake onnxruntime. NumPy is real."""

import gc
import hashlib
import importlib
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import robin_models.onnx_head
from robin_adapters.artifact_writer.local import LocalArtifactWriter
from robin_adapters.file_provider.local import LocalFileProvider
from robin_contracts.cards import AudioGeometry, HeadCard, ModelCard, RunnerResampled, model_ref
from robin_contracts.inputs import AudioClip, Embeddings
from robin_contracts.layout import artifact_path
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.protocols import Model, ModelContext
from robin_contracts.results import InferenceSuccess
from robin_contracts.specs import recipe
from robin_contracts.work import (
    AudioInput,
    EmbeddingArtifactInput,
    InferenceWork,
    InputArtifact,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts.embeddings import CONTRACT_ID, EmbeddingsWriter
from robin_inference_engine.artifacts.metadata import embedding_metadata, required_metadata
from robin_inference_engine.artifacts.scores import read_scores
from robin_inference_engine.artifacts.staging import checksum_file
from robin_inference_engine.engine import run_work
from robin_inference_engine.load_registry import load_registry

ADAPTER = "robin_models.onnx_head.adapter"
DIM = 4
REGISTRY_CSV = b"class_index,label,label_kind\n0,chorus,non_taxonomic\n1,rain,non_taxonomic\n"
LABELS = ("chorus", "rain")
CPU_PROVIDERS = ["CPUExecutionProvider"]


def build_backbone() -> ModelCard:
    return ModelCard(
        model_name="test-backbone",
        model_version="1",
        runtime="none",
        window_duration=3.0,
        window_overlap=0.0,
        sample_rate=16000,
        min_detection_threshold=0.0,
        score_domain=None,
        taxa_registry_digest=None,
        audio=AudioGeometry(
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        backend="none",
        dtype="float32",
        can_emit_embeddings=True,
        embedding_dim=DIM,
        embedding_dtype="float32",
    )


BACKBONE = build_backbone()


def build_head(**overrides) -> HeadCard:
    fields = {
        "model_name": "test-head",
        "model_version": "1",
        "runtime": "onnx",
        "backbone": model_ref(BACKBONE),
        "embedding_dim": DIM,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": "sha256:" + hashlib.sha256(REGISTRY_CSV).hexdigest(),
    }
    return HeadCard(**(fields | overrides))


HEAD = build_head()


# A fake onnxruntime.


class FakeNode:
    """One graph input or output, as onnxruntime describes it."""

    def __init__(self, name: str, shape: list, type: str) -> None:
        self.name = name
        self.shape = list(shape)
        self.type = type


class FakeSession:
    """An inference session. It refers to the runtime, which records what it was asked."""

    def __init__(self, runtime: "FakeOnnxRuntime") -> None:
        self._runtime = runtime

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode(**node) for node in self._runtime.inputs]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode(**node) for node in self._runtime.outputs]

    def run(self, output_names, input_feed):
        runtime = self._runtime
        [(name, array)] = input_feed.items()
        runtime.calls.append(
            types.SimpleNamespace(
                output_names=list(output_names),
                input_name=name,
                values=np.array(array, copy=True),
                dtype=array.dtype,
                c_contiguous=array.flags["C_CONTIGUOUS"],
            )
        )
        if runtime.run_error is not None:
            raise runtime.run_error
        return [runtime.output(np.asarray(array))]


class FakeOnnxRuntime:
    """Records each session built and each call run, and fails where told to."""

    def __init__(self) -> None:
        self.constructions: list = []
        self.sessions: list[weakref.ref] = []
        self.construct_error: Exception | None = None
        self.run_error: Exception | None = None
        self.calls: list = []
        self.inputs = [{"name": "embedding", "shape": ["batch", DIM], "type": "tensor(float)"}]
        self.outputs = [
            {"name": "scores", "shape": ["batch", len(LABELS)], "type": "tensor(float)"}
        ]
        self.output = default_scores

    def module(self) -> types.ModuleType:
        ort = types.ModuleType("onnxruntime")
        ort.InferenceSession = self.inference_session
        return ort

    def inference_session(self, *args, **kwargs) -> FakeSession:
        self.constructions.append(types.SimpleNamespace(args=args, kwargs=kwargs))
        if self.construct_error is not None:
            raise self.construct_error
        session = FakeSession(self)
        self.sessions.append(weakref.ref(session))
        return session


def default_scores(batch: np.ndarray) -> np.ndarray:
    """Scores in [0, 1] from each row's sum: distinct per label, and per row whose sums
    differ. A sine, so large sums do not all give 1."""
    sums = batch.sum(axis=1, keepdims=True, dtype=np.float64)
    return (0.5 + 0.5 * np.sin(sums + np.arange(len(LABELS)) / 2)).astype(np.float32)


@pytest.fixture
def runtime(monkeypatch):
    """Import the adapter against a fake onnxruntime, and remove it afterwards."""
    fake = FakeOnnxRuntime()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake.module())
    sys.modules.pop(ADAPTER, None)
    fake.adapter = importlib.import_module(ADAPTER)
    yield fake
    sys.modules.pop(ADAPTER, None)
    if hasattr(robin_models.onnx_head, "adapter"):
        delattr(robin_models.onnx_head, "adapter")


def head_files(tmp_path) -> dict:
    """A stand-in graph, which the fake reads nothing of, and the registry."""
    graph = tmp_path / "head.onnx"
    graph.write_bytes(b"graph")
    registry = tmp_path / "taxa_registry.csv"
    registry.write_bytes(REGISTRY_CSV)
    return {"graph": graph, "taxa_registry": registry}


def head_context(tmp_path, **changes) -> ModelContext:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    files = changes.pop("files", None) or head_files(tmp_path)
    fields = dict(
        card=HEAD,
        registry=load_registry(head_files(tmp_path)["taxa_registry"]),
        files=files,
        settings={},
        resources={},
        scratch_dir=scratch,
        emit_embeddings=False,
    )
    return ModelContext(**(fields | changes))


def without(files: dict, *roles: str) -> dict:
    return {role: path for role, path in files.items() if role not in roles}


# The card.


def test_a_head_card_with_the_onnx_runtime_builds_a_model(runtime, tmp_path):
    model = runtime.adapter.build(head_context(tmp_path))
    assert isinstance(model, Model)
    assert len(runtime.constructions) == 1


def test_a_card_naming_another_runtime_is_refused_naming_both(runtime, tmp_path):
    with pytest.raises(ValueError) as caught:
        runtime.adapter.build(head_context(tmp_path, card=build_head(runtime="tflite")))
    assert "'tflite'" in str(caught.value)
    assert "'onnx'" in str(caught.value)
    assert runtime.constructions == []


def test_a_model_card_is_refused_naming_its_type(runtime, tmp_path):
    with pytest.raises(ValueError, match="ModelCard"):
        runtime.adapter.build(head_context(tmp_path, card=BACKBONE))
    assert runtime.constructions == []


# The pinned files.


@pytest.mark.parametrize("role", ["graph", "taxa_registry"])
def test_a_missing_role_is_refused_naming_it_and_the_roles_given(runtime, tmp_path, role):
    files = without(head_files(tmp_path), role)
    with pytest.raises(ValueError, match=f"'{role}'") as caught:
        runtime.adapter.build(head_context(tmp_path, files=files))
    assert str(sorted(files)) in str(caught.value)
    assert runtime.constructions == []


def test_a_context_with_no_registry_is_refused_naming_the_registry_role(runtime, tmp_path):
    with pytest.raises(ValueError, match="'taxa_registry'"):
        runtime.adapter.build(head_context(tmp_path, registry=None))
    assert runtime.constructions == []


# Resources.


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        pytest.param({}, 256, id="none_given"),
        pytest.param({"batch_size": 1}, 1, id="one"),
        pytest.param({"batch_size": 7}, 7, id="seven"),
    ],
)
def test_the_batch_size_is_read_from_the_resources(runtime, tmp_path, resources, expected):
    assert runtime.adapter._read_batch_size(resources) == expected
    runtime.adapter.build(head_context(tmp_path, resources=resources))


@pytest.mark.parametrize("value", [0, -1, True, 256.0, "256"], ids=repr)
def test_a_batch_size_that_is_not_a_positive_int_is_refused(runtime, tmp_path, value):
    with pytest.raises(ValueError, match="batch_size") as caught:
        runtime.adapter.build(head_context(tmp_path, resources={"batch_size": value}))
    assert repr(value) in str(caught.value)
    assert runtime.constructions == []


@pytest.mark.parametrize("resources", [{}, {"device": "cpu"}], ids=["absent", "cpu"])
def test_the_cpu_is_the_device_that_builds(runtime, tmp_path, resources):
    runtime.adapter.build(head_context(tmp_path, resources=resources))


@pytest.mark.parametrize("value", ["gpu", "cuda", None], ids=repr)
def test_any_other_device_is_refused(runtime, tmp_path, value):
    with pytest.raises(ValueError, match="device") as caught:
        runtime.adapter.build(head_context(tmp_path, resources={"device": value}))
    assert repr(value) in str(caught.value)
    assert runtime.constructions == []


# Loading.


def test_the_session_reads_the_pinned_graph_on_the_cpu_provider_alone(runtime, tmp_path):
    context = head_context(tmp_path)
    runtime.adapter.build(context)
    [construction] = runtime.constructions
    assert construction.args == (str(context.files["graph"]),)
    assert type(construction.args[0]) is str
    assert construction.kwargs == {"providers": CPU_PROVIDERS}


def test_a_session_that_fails_to_load_fails_the_build(runtime, tmp_path):
    runtime.construct_error = RuntimeError("the graph is not ONNX")
    with pytest.raises(RuntimeError, match="the graph is not ONNX"):
        runtime.adapter.build(head_context(tmp_path))


# The graph against the card and the registry.


def node(name: str, shape: list, type: str = "tensor(float)") -> dict:
    return {"name": name, "shape": shape, "type": type}


INPUT = node("embedding", ["batch", DIM])
OUTPUT = node("scores", ["batch", len(LABELS)])


@pytest.mark.parametrize(
    ("inputs", "outputs", "found", "needed"),
    [
        pytest.param(
            [INPUT, node("extra", ["batch", DIM])], [OUTPUT], "2 inputs", "one input",
            id="two_inputs",
        ),
        pytest.param(
            [INPUT], [OUTPUT, node("extra", ["batch", 2])], "2 outputs", "one output",
            id="two_outputs",
        ),
        pytest.param(
            [node("embedding", ["batch", DIM], "tensor(double)")], [OUTPUT],
            "'tensor(double)'", "'tensor(float)'",
            id="double_input",
        ),
        pytest.param(
            [INPUT], [node("scores", ["batch", 2], "tensor(float16)")],
            "'tensor(float16)'", "'tensor(float)'",
            id="float16_output",
        ),
        pytest.param(
            [node("embedding", [None, 1280, 1])], [OUTPUT], "[None, 1280, 1]", "[batch, 4]",
            id="three_dimensional_input",
        ),
        pytest.param(
            [node("embedding", [1, DIM])], [OUTPUT], "[1, 4]", "not fixed",
            id="fixed_input_batch",
        ),
        pytest.param(
            [INPUT], [node("scores", [8, 2])], "[8, 2]", "not fixed",
            id="fixed_output_batch",
        ),
        pytest.param(
            [node("embedding", ["batch", DIM - 1])], [OUTPUT], "['batch', 3]",
            "embedding_dim 4",
            id="narrow_input",
        ),
        pytest.param(
            [INPUT], [node("scores", ["batch", 3])], "['batch', 3]", "2 labels",
            id="wide_output",
        ),
    ],
)
def test_a_graph_that_does_not_fit_the_card_is_refused(
    runtime, tmp_path, inputs, outputs, found, needed
):
    runtime.inputs, runtime.outputs = inputs, outputs
    with pytest.raises(ValueError) as caught:
        runtime.adapter.build(head_context(tmp_path))
    message = str(caught.value)
    assert found in message
    assert needed in message


@pytest.mark.parametrize("batch", ["batch", "unk__6", None], ids=repr)
def test_a_batch_dimension_that_is_symbolic_or_unknown_builds(runtime, tmp_path, batch):
    runtime.inputs = [node("embedding", [batch, DIM])]
    runtime.outputs = [node("scores", [batch, len(LABELS)])]
    runtime.adapter.build(head_context(tmp_path))


# Running.


def embeddings(rows: int, offset: float = 0.0) -> Embeddings:
    """`rows` windows 3 s apart, each with its own vector."""
    starts = 3.0 * np.arange(rows, dtype=np.float64)
    values = (np.arange(rows * DIM, dtype=np.float32).reshape(rows, DIM) + 1) / 3 + offset
    return Embeddings(starts=starts, ends=starts + 3.0, values=values.astype(np.float32))


def built(runtime, tmp_path, **resources):
    return runtime.adapter.build(head_context(tmp_path, resources=resources))


def run(model, given: Embeddings) -> list:
    return list(model.run(given))


def test_rows_reach_the_session_in_batches_of_the_batch_size(runtime, tmp_path):
    given = embeddings(7)
    run(built(runtime, tmp_path, batch_size=3), given)
    assert [len(call.values) for call in runtime.calls] == [3, 3, 1]
    for call, first in zip(runtime.calls, [0, 3, 6], strict=True):
        assert call.dtype == np.float32
        assert call.c_contiguous
        assert call.values.shape == (len(call.values), DIM)
        assert call.input_name == "embedding"
        assert call.output_names == ["scores"]
        assert np.array_equal(call.values, given.values[first : first + 3])


def test_the_default_batch_is_256_rows(runtime, tmp_path):
    run(built(runtime, tmp_path), embeddings(300))
    assert [len(call.values) for call in runtime.calls] == [256, 44]


def test_each_row_is_one_window_scoring_every_label_in_registry_order(runtime, tmp_path):
    given = embeddings(5)
    windows = run(built(runtime, tmp_path, batch_size=2), given)
    expected = default_scores(given.values)
    assert len(windows) == 5
    for row, window in enumerate(windows):
        assert type(window.start) is float and window.start == given.starts[row]
        assert type(window.end) is float and window.end == given.ends[row]
        assert tuple(score.label for score in window.scores) == LABELS
        assert all(type(score.score) is float for score in window.scores)
        assert [score.score for score in window.scores] == expected[row].tolist()
        assert window.embedding is None
    assert windows[0].scores != windows[1].scores


@pytest.mark.parametrize("value", [-0.5, 1.5, 37.0], ids=repr)
def test_the_graphs_output_is_the_score_unchanged(runtime, tmp_path, value):
    runtime.output = lambda batch: np.full((len(batch), len(LABELS)), value, dtype=np.float32)
    windows = run(built(runtime, tmp_path), embeddings(2))
    assert {score.score for window in windows for score in window.scores} == {value}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "minus_inf"])
def test_a_score_that_is_not_finite_fails_the_recording_at_its_window(runtime, tmp_path, value):
    def output(batch):
        scores = default_scores(batch)
        if len(runtime.calls) == 2:
            scores[1, 1] = value
        return scores

    runtime.output = output
    model = built(runtime, tmp_path, batch_size=3)
    yielded = []
    with pytest.raises(ValueError) as caught:
        for window in model.run(embeddings(7)):
            yielded.append(window)
    message = str(caught.value)
    assert "12.0" in message
    assert repr(LABELS[1]) in message
    assert repr(float(value)) in message
    assert [window.start for window in yielded] == [0.0, 3.0, 6.0]


@pytest.mark.parametrize(
    "change",
    [lambda scores: scores[:-1], lambda scores: np.hstack([scores, scores[:, :1]])],
    ids=["one_row_fewer", "one_column_more"],
)
def test_an_output_of_the_wrong_shape_fails_the_recording_naming_both(runtime, tmp_path, change):
    runtime.output = lambda batch: change(default_scores(batch))
    wrong = change(np.zeros((3, len(LABELS)))).shape
    with pytest.raises(ValueError) as caught:
        run(built(runtime, tmp_path), embeddings(3))
    assert str(wrong) in str(caught.value)
    assert str((3, len(LABELS))) in str(caught.value)


def test_run_refuses_audio_before_calling_the_session(runtime, tmp_path):
    model = built(runtime, tmp_path)
    with pytest.raises(TypeError, match="AudioClip"):
        model.run(AudioClip(path=tmp_path / "a.wav"))
    assert runtime.calls == []


def test_run_calls_the_session_only_when_iterated(runtime, tmp_path):
    windows = built(runtime, tmp_path).run(embeddings(3))
    assert runtime.calls == []
    next(windows)
    assert len(runtime.calls) == 1


def test_a_recording_with_no_rows_yields_nothing_and_calls_nothing(runtime, tmp_path):
    empty = Embeddings(
        starts=np.zeros(0), ends=np.zeros(0), values=np.zeros((0, DIM), dtype=np.float32)
    )
    assert run(built(runtime, tmp_path), empty) == []
    assert runtime.calls == []


def test_a_session_failure_fails_the_recording(runtime, tmp_path):
    runtime.run_error = RuntimeError("the graph failed")
    with pytest.raises(RuntimeError, match="the graph failed"):
        run(built(runtime, tmp_path), embeddings(3))


def test_one_session_runs_every_recording_and_stays_after_each(runtime, tmp_path):
    model = built(runtime, tmp_path)
    assert len(run(model, embeddings(3))) == 3
    model.after_recording()
    assert len(run(model, embeddings(2, offset=1.0))) == 2
    model.after_recording()
    gc.collect()
    assert len(runtime.constructions) == 1
    assert len(runtime.calls) == 2
    [session] = runtime.sessions
    assert session() is not None


def test_clean_up_releases_the_session(runtime, tmp_path):
    model = built(runtime, tmp_path)
    run(model, embeddings(3))
    [session] = runtime.sessions
    gc.collect()
    assert session() is not None
    model.clean_up()
    gc.collect()
    assert session() is None


# Through the engine.


def pinned(path: Path) -> PinnedFile:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PinnedFile(uri=str(path), digest=f"sha256:{digest}", size_bytes=path.stat().st_size)


def write_embeddings(path: Path, recording: RecordingRef, values: np.ndarray) -> None:
    """Write `values` as the backbone's embeddings file for `recording`, one row per 3 s."""
    weights = PinnedFile(uri="s3://b/weights.bin", digest="sha256:" + "0" * 64, size_bytes=8)
    work = InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=(recording,),
        model=PinnedModel(card=BACKBONE, files={"weights": weights}),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=(EmbeddingsRequest(contract_id=CONTRACT_ID),),
    )
    metadata = required_metadata(
        contract_id=CONTRACT_ID,
        work=work,
        recording=recording,
        recipe=recipe(BACKBONE),
        registry_uri=None,
        registry_fingerprint=None,
    ) | embedding_metadata(work, dim=DIM, source_dtype="float32", storage_dtype="float32")
    with EmbeddingsWriter(
        path, recording=recording, dim=DIM, storage_dtype="float32", metadata=metadata
    ) as writer:
        for row, vector in enumerate(values):
            start = 3.0 * row
            writer.write(
                AcceptedWindow(
                    recording=recording, start=start, end=start + 3.0, scores=(), embedding=vector
                )
            )
        writer.close()


def test_a_head_work_runs_through_the_engine_to_scores_and_detections(runtime, tmp_path):
    vectors = {name: embeddings(3, offset=offset).values for name, offset in [("a", 0), ("b", 9)]}
    recordings = []
    for name, values in vectors.items():
        recording = RecordingRef(namespace="test", value=name, audio_uri=f"s3://b/{name}.wav")
        path = tmp_path / f"{name}.arrow"
        write_embeddings(path, recording, values)
        named = InputArtifact(uri=str(path), checksum=checksum_file(path))
        recordings.append(recording.model_copy(update={"embeddings": named}))
    model_files = head_files(tmp_path)
    work = InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=tuple(recordings),
        model=PinnedModel(
            card=HEAD, files={role: pinned(path) for role, path in model_files.items()}
        ),
        input=EmbeddingArtifactInput(contract_id=CONTRACT_ID, backbone=BACKBONE),
        settings={},
        resources={"batch_size": 2},
        outputs=(
            ScoresRequest(contract_id="robin.scores.arrow/1", retention="full"),
            DetectionsRequest(
                contract_id="robin.detections.parquet/1", policy=ThresholdPolicy(min_score=0.0)
            ),
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
    assert [len(call.values) for call in runtime.calls] == [2, 1, 2, 1]
    checksums = {
        record.value: record.checksum for record in result.artifacts if record.kind == "scores"
    }
    for name, values in vectors.items():
        path = root / artifact_path("scores", "test", name)
        with read_scores(path, expected_checksum=checksums[name]) as stream:
            scores = pa.Table.from_batches(list(stream.batches)).to_pylist()
        assert len(scores) == 3 * len(LABELS)
        expected = default_scores(values)
        for score in scores:
            row = round(score["window_start_s"] / 3.0)
            column = LABELS.index(score["label"])
            assert score["score"] == expected[row, column].item()
    detections = [record for record in result.artifacts if record.kind == "detections"]
    assert detections
    for record in detections:
        assert pq.read_table(root / artifact_path("detections", "test", record.value)).num_rows
