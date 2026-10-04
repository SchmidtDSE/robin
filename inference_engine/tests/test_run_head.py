"""A head work, run through `run_work` over embeddings files a backbone work wrote."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from doubles import (
    HEAD_RUNTIME_GROUP,
    REGISTRY_CSV,
    CallLog,
    CopyingWriter,
    LocalFiles,
    ScriptedHead,
    ScriptedModel,
    WorkBuilder,
    head_work,
    installed_factory,
    registry_digest,
)
from robin_contracts.cards import (
    AudioGeometry,
    HeadCard,
    ModelCard,
    RunnerResampled,
    model_ref,
)
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.protocols import ModelContext
from robin_contracts.inputs import Embeddings
from robin_contracts.layout import artifact_path
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.results import FailureReport, InferenceFailure, InferenceSuccess
from robin_contracts.specs import recipe
from robin_contracts.work import InferenceWork, InputArtifact, RecordingRef
from robin_inference_engine import errors
from robin_inference_engine.artifacts.metadata import decode_metadata
from robin_inference_engine.artifacts.staging import checksum_file
from robin_inference_engine.engine import run_work

DIM = 4
HOP = 3.0


def build_backbone(**overrides) -> ModelCard:
    fields = {
        "model_name": "test-backbone",
        "model_version": "1",
        "runtime": "none",
        "window_duration": HOP,
        "window_overlap": 0.0,
        "sample_rate": 16000,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": registry_digest(REGISTRY_CSV),
        "audio": AudioGeometry(
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        "backend": "none",
        "dtype": "float32",
        "can_emit_embeddings": True,
        "embedding_dim": DIM,
        "embedding_dtype": "float32",
    }
    return ModelCard(**(fields | overrides))


BACKBONE = build_backbone()

# One label, unlike the backbone's two-label registry, so a header naming the wrong
# registry is caught.
HEAD_REGISTRY_CSV = b"class_index,label,label_kind\n0,bullfrog,non_taxonomic\n"
HEAD_LABELS = ("bullfrog",)


def build_head(**overrides) -> HeadCard:
    fields = {
        "model_name": "test-head",
        "model_version": "1",
        "runtime": "test-runtime",
        "backbone": model_ref(BACKBONE),
        "embedding_dim": DIM,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": registry_digest(HEAD_REGISTRY_CSV),
    }
    return HeadCard(**(fields | overrides))


HEAD = build_head()

# Rows each backbone recording writes.
ROWS = (3, 2)


def scores_request() -> ScoresRequest:
    return ScoresRequest(contract_id="robin.scores.arrow/1", retention="full")


def detections_request() -> DetectionsRequest:
    return DetectionsRequest(
        contract_id="robin.detections.parquet/1", policy=ThresholdPolicy(min_score=0.0)
    )


def backbone_window(recording: int, row: int, dim: int = DIM) -> WindowOutput:
    start = HOP * row
    embedding = np.arange(dim, dtype=np.float32) + 10.0 * recording + row / 8
    scores = (ClassScore(label="owl", score=0.25), ClassScore(label="rain", score=0.75))
    return WindowOutput(start=start, end=start + HOP, scores=scores, embedding=embedding)


def run_backbone(
    directory: Path, card: ModelCard = BACKBONE
) -> tuple[InferenceWork, InferenceSuccess]:
    """A backbone work that writes a scores and an embeddings file for each of ROWS'
    recordings."""
    calls: CallLog = []
    build = WorkBuilder(directory / "inputs", card)
    recordings = tuple(build.recording(str(n)) for n in range(len(ROWS)))
    work = build.work(
        recordings,
        settings={},
        outputs=(scores_request(), EmbeddingsRequest(contract_id="robin.embeddings.arrow/1")),
    )
    script = [
        [backbone_window(n, row, card.embedding_dim) for row in range(rows)]
        for n, rows in enumerate(ROWS)
    ]
    model = ScriptedModel(script=script, calls=calls)
    with installed_factory(directory, model_ref(card).id, lambda context: model):
        result = run_work(
            work,
            model_files=LocalFiles("model_files", build.model_paths, calls),
            inputs=LocalFiles("audio", build.audio_paths, calls),
            artifacts=CopyingWriter(directory / "published", calls),
        )
    assert isinstance(result, InferenceSuccess), result
    return work, result


def rewrite_header(path: Path, header: Mapping[str, str | None]) -> None:
    """Set (or, for None, drop) each `header` value of the Arrow stream at `path`."""
    with pa.ipc.open_stream(path) as reader:
        schema, batches = reader.schema, list(reader)
    metadata = {key.decode(): value.decode() for key, value in schema.metadata.items()}
    for key, value in header.items():
        if value is None:
            metadata.pop(key)
        else:
            metadata[key] = value
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(
        sink, schema.with_metadata(metadata)
    ) as writer:
        for batch in batches:
            writer.write_batch(batch.replace_schema_metadata(metadata))


def stored_vectors(path: Path) -> np.ndarray:
    """The file's vectors as stored, read with Arrow alone."""
    with pa.ipc.open_stream(path) as reader:
        column = reader.read_all().column("embedding").combine_chunks()
    return column.flatten().to_numpy(zero_copy_only=False).reshape(-1, column.type.list_size)


class HeadRig:
    """A backbone's published embeddings, a head's model files, and one call log."""

    def __init__(self, tmp_path: Path, *, head: HeadCard = HEAD) -> None:
        self.tmp_path = tmp_path
        self.calls: CallLog = []
        self.contexts: list[ModelContext] = []
        self.backbone_work, self.backbone_result = run_backbone(tmp_path / "backbone")
        self.build = WorkBuilder(tmp_path / "head", head, registry_csv=HEAD_REGISTRY_CSV)
        self.destination = tmp_path / "head-published"
        self.input_paths: dict[str, Path] = {}

    def work(self, *, outputs=None) -> InferenceWork:
        work, paths = head_work(
            self.backbone_work,
            self.backbone_result,
            self.build,
            outputs=outputs or (scores_request(),),
        )
        self.input_paths |= paths
        return work

    def input_path(self, work: InferenceWork, position: int) -> Path:
        return self.input_paths[work.recordings[position].embeddings.uri]

    def with_input(
        self, work: InferenceWork, position: int, path: Path, *, checksum: str | None = None
    ) -> InferenceWork:
        """`work`, with the recording at `position` naming the file at `path`."""
        uri = path.as_uri()
        self.input_paths[uri] = path
        embeddings = InputArtifact(uri=uri, checksum=checksum or checksum_file(path))
        recordings = list(work.recordings)
        named = recordings[position].model_dump() | {"embeddings": embeddings}
        recordings[position] = RecordingRef(**named)
        return InferenceWork(**(dict(work) | {"recordings": tuple(recordings)}))

    def head(self, **kwargs) -> ScriptedHead:
        return ScriptedHead(labels=HEAD_LABELS, calls=self.calls, **kwargs)

    def files(self, name: str, paths: dict[str, Path], **kwargs) -> LocalFiles:
        return LocalFiles(name, paths, self.calls, **kwargs)

    def run(self, work: InferenceWork, head=None, *, inputs=None, artifacts=None):
        head = head or self.head()

        def factory(context: ModelContext) -> ScriptedHead:
            self.contexts.append(context)
            return head

        with installed_factory(
            self.tmp_path, work.model.card.runtime, factory, group=HEAD_RUNTIME_GROUP
        ):
            return run_work(
                work,
                model_files=self.files("model_files", self.build.model_paths),
                inputs=inputs or self.files("embeddings", self.input_paths),
                artifacts=artifacts or CopyingWriter(self.destination, self.calls),
            )

    def runs(self) -> list[tuple[object, ...]]:
        return [call for call in self.calls if call[0] == "run"]


@pytest.fixture
def rig(tmp_path) -> HeadRig:
    return HeadRig(tmp_path)


def failure_of(result, *, code: str, stage: str) -> FailureReport:
    assert isinstance(result, InferenceFailure), result
    assert (result.failure.code, result.failure.stage) == (code, stage), result.failure
    return result.failure


# ---------------------------------------------------------------------------
# The head's runtime, found and built.
# ---------------------------------------------------------------------------


def test_a_head_work_is_built_by_its_runtimes_factory(rig):
    work = rig.work()

    rig.run(work)

    assert len(work.recordings) == 2
    assert len(rig.contexts) == 1
    context = rig.contexts[0]
    assert context.card == HEAD
    assert context.registry is not None
    assert context.registry.labels == frozenset(HEAD_LABELS)
    assert context.registry.fingerprint == registry_digest(HEAD_REGISTRY_CSV)
    assert context.emit_embeddings is False


# ---------------------------------------------------------------------------
# Each recording's input, fetched, read and checked.
# ---------------------------------------------------------------------------


def a_copy(rig: HeadRig, work: InferenceWork, position: int) -> Path:
    path = rig.tmp_path / f"copy-of-{position}.arrow"
    shutil.copyfile(rig.input_path(work, position), path)
    return path


def a_file_from_another_backbone(rig: HeadRig, work: InferenceWork) -> InferenceWork:
    path = a_copy(rig, work, 1)
    rewrite_header(path, {"robin.backbone_ref": "other-backbone/1"})
    return rig.with_input(work, 1, path)


def a_file_of_another_width(rig: HeadRig, work: InferenceWork) -> InferenceWork:
    wider = build_backbone(embedding_dim=DIM + 1)
    wider_work, wider_result = run_backbone(rig.tmp_path / "wider", wider)
    path = rig.tmp_path / "wider.arrow"
    named, paths = head_work(wider_work, wider_result, rig.build, outputs=(scores_request(),))
    shutil.copyfile(paths[named.recordings[1].embeddings.uri], path)
    # Named as the head's backbone, so only the width disagrees.
    rewrite_header(
        path,
        {
            "robin.backbone_ref": HEAD.backbone.id,
            "robin.backbone_card_digest": HEAD.backbone.digest,
        },
    )
    return rig.with_input(work, 1, path)


def another_recordings_file(rig: HeadRig, work: InferenceWork) -> InferenceWork:
    return rig.with_input(work, 1, rig.input_path(work, 0))


def a_file_named_by_another_checksum(rig: HeadRig, work: InferenceWork) -> InferenceWork:
    return rig.with_input(work, 1, rig.input_path(work, 1), checksum="sha256:" + "1" * 64)


def a_file_naming_no_recording_value(rig: HeadRig, work: InferenceWork) -> InferenceWork:
    path = a_copy(rig, work, 1)
    rewrite_header(path, {"robin.recording_value": None})
    return rig.with_input(work, 1, path)


@pytest.mark.parametrize(
    ("named", "code"),
    [
        (a_file_from_another_backbone, errors.HEAD_INPUT_BACKBONE_MISMATCH),
        (a_file_of_another_width, errors.HEAD_INPUT_WIDTH_MISMATCH),
        (another_recordings_file, errors.HEAD_INPUT_RECORDING_MISMATCH),
        (a_file_named_by_another_checksum, errors.ARTIFACT_CHECKSUM_MISMATCH),
        (a_file_naming_no_recording_value, errors.ARTIFACT_METADATA_INCOMPLETE),
    ],
)
def test_an_input_file_the_engine_does_not_trust_fails_the_work_at_its_recording(
    rig, named, code
):
    work = named(rig, rig.work())

    result = rig.run(work)

    failure = failure_of(result, code=code, stage=errors.READ_INPUT_ARTIFACT)
    assert (failure.namespace, failure.value) == ("test", "1")
    assert rig.runs() == [("run", "embeddings", ROWS[0])]
    assert not any(call[0] == "create" for call in rig.calls)


def test_the_head_is_given_each_recordings_stored_vectors_once(rig):
    work = rig.work()
    head = rig.head()

    result = rig.run(work, head)

    assert isinstance(result, InferenceSuccess), result
    assert rig.runs() == [("run", "embeddings", rows) for rows in ROWS]
    assert len(head.given) == len(ROWS)
    for position, given in enumerate(head.given):
        stored = stored_vectors(rig.input_path(work, position))
        assert given.values.dtype == np.float32
        assert np.array_equal(given.values, stored)
        assert np.array_equal(given.starts, HOP * np.arange(ROWS[position]))
        assert np.array_equal(given.ends, given.starts + HOP)


@pytest.mark.parametrize("fail", [None, RuntimeError("the graph would not run")])
def test_each_input_file_is_released_after_its_recording(rig, fail):
    work = rig.work()

    result = rig.run(work, rig.head(fail=fail))

    first = rig.input_path(work, 0)
    steps = [
        call
        for call in rig.calls
        if call[0] in ("run", "after_recording") or call[:2] == ("embeddings", "release")
    ]
    assert steps[:3] == [
        ("run", "embeddings", ROWS[0]),
        ("after_recording",),
        ("embeddings", "release", first),
    ]
    if fail is not None:
        failure_of(result, code=errors.MODEL_RUN_FAILED, stage=errors.INFER)
        assert len(steps) == 3


def test_an_input_the_port_cannot_fetch_is_refused_naming_its_uri(rig):
    work = rig.work()
    recording = work.recordings[0]
    inputs = rig.files(
        "embeddings",
        rig.input_paths,
        raise_on={recording.embeddings.uri: OSError("the bucket is closed")},
    )

    result = rig.run(work, inputs=inputs)

    failure = failure_of(result, code=errors.INPUT_UNAVAILABLE, stage=errors.ACQUIRE_INPUT)
    assert recording.embeddings.uri in failure.detail
    assert recording.audio_uri not in failure.detail
    assert rig.runs() == []


# ---------------------------------------------------------------------------
# The windows a head returns, against its input's rows.
# ---------------------------------------------------------------------------


def one_per_row(given: Embeddings) -> list[WindowOutput]:
    return [
        WindowOutput(
            start=float(start),
            end=float(end),
            scores=(ClassScore(label="bullfrog", score=0.5),),
        )
        for start, end in zip(given.starts, given.ends, strict=True)
    ]


def one_fewer(given: Embeddings) -> list[WindowOutput]:
    return one_per_row(given)[:-1]


def one_more(given: Embeddings) -> list[WindowOutput]:
    windows = one_per_row(given)
    last = windows[-1]
    return [*windows, WindowOutput(start=last.end, end=last.end + HOP, scores=last.scores)]


def a_moved_start(given: Embeddings) -> list[WindowOutput]:
    windows = one_per_row(given)
    moved = windows[1]
    windows[1] = WindowOutput(start=moved.start + 0.5, end=moved.end + 0.5, scores=moved.scores)
    return windows


def a_different_end(given: Embeddings) -> list[WindowOutput]:
    windows = one_per_row(given)
    windows[1] = WindowOutput(
        start=windows[1].start, end=windows[1].end + 0.5, scores=windows[1].scores
    )
    return windows


@pytest.mark.parametrize("windows", [one_fewer, one_more, a_moved_start, a_different_end])
def test_a_head_whose_windows_are_not_its_inputs_rows_fails_the_work(rig, windows):
    work = rig.work()

    result = rig.run(work, rig.head(windows=windows))

    failure = failure_of(
        result, code=errors.HEAD_WINDOWS_DISAGREE_WITH_INPUT, stage=errors.ACCEPT_WINDOW
    )
    assert (failure.namespace, failure.value) == ("test", "0")
    assert not any(call[0] == "create" for call in rig.calls)


# ---------------------------------------------------------------------------
# A head work from start to finish.
# ---------------------------------------------------------------------------


def published(record) -> Path:
    return Path(url2pathname(urlparse(record.uri).path))


def header_of(record) -> dict[str, str]:
    if record.kind == "detections":
        return decode_metadata(pq.read_schema(published(record)).metadata)
    with pa.ipc.open_stream(published(record)) as reader:
        return decode_metadata(reader.schema.metadata)


def full_head_work(rig: HeadRig) -> InferenceWork:
    return rig.work(outputs=(scores_request(), detections_request()))


def test_a_head_work_scores_every_window_of_its_input(rig):
    work = full_head_work(rig)

    result = rig.run(work)

    assert isinstance(result, InferenceSuccess), result
    scores = [record for record in result.artifacts if record.kind == "scores"]
    assert [(record.value, record.namespace) for record in scores] == [("0", "test"), ("1", "test")]
    for record, rows in zip(scores, ROWS, strict=True):
        with pa.ipc.open_stream(published(record)) as reader:
            table = reader.read_all()
        assert table.num_rows == rows * len(HEAD_LABELS)
        assert table.column("window_start_s").to_pylist() == [HOP * row for row in range(rows)]
        assert set(table.column("label").to_pylist()) == set(HEAD_LABELS)


def test_a_head_works_result_carries_its_backbones_recipe_and_names_the_head(rig):
    work = full_head_work(rig)

    result = rig.run(work)

    assert isinstance(result, InferenceSuccess), result
    stated = recipe(BACKBONE)
    assert result.recipe == stated
    assert result.window_geometry == stated.audio.geometry
    assert result.model == work.model
    assert result.model.card == HEAD
    assert result.registry_fingerprint == registry_digest(HEAD_REGISTRY_CSV)


def test_each_header_names_the_head_its_files_and_its_registry(rig):
    work = full_head_work(rig)

    result = rig.run(work)

    assert isinstance(result, InferenceSuccess), result
    head = model_ref(HEAD)
    files = {
        role: {"digest": pinned.digest, "size_bytes": pinned.size_bytes}
        for role, pinned in work.model.files.items()
    }
    kinds = sorted({record.kind for record in result.artifacts})
    assert kinds == ["detections", "scores"]
    for record in result.artifacts:
        header = header_of(record)
        assert header["robin.model_ref"] == head.id
        assert header["robin.model_card_digest"] == head.digest
        assert json.loads(header["robin.model_file_digests"]) == files
        assert header["robin.registry_fingerprint"] == registry_digest(HEAD_REGISTRY_CSV)
        assert header["robin.score_domain"] == "probability"
        assert header["robin.recipe_fingerprint"] == recipe(BACKBONE).id
        assert header["robin.recording_namespace"] == record.namespace
        assert header["robin.recording_value"] == record.value


def detection_ids(result: InferenceSuccess) -> set[str]:
    ids: set[str] = set()
    for record in result.artifacts:
        if record.kind == "detections":
            ids |= set(pq.read_table(published(record)).column("detection_id").to_pylist())
    return ids


def test_two_heads_over_one_input_give_different_detections(rig, tmp_path):
    first = rig.run(full_head_work(rig))
    other = build_head(model_name="other-head")
    rig.build = WorkBuilder(tmp_path / "other-head", other, registry_csv=HEAD_REGISTRY_CSV)
    rig.destination = tmp_path / "other-head-published"

    second = rig.run(full_head_work(rig))

    assert isinstance(first, InferenceSuccess), first
    assert isinstance(second, InferenceSuccess), second
    assert detection_ids(first)
    assert detection_ids(second)
    assert detection_ids(first).isdisjoint(detection_ids(second))


def test_a_head_published_at_its_backbones_destination_is_refused(rig, tmp_path):
    work = full_head_work(rig)
    writer = CopyingWriter(tmp_path / "backbone" / "published", rig.calls)

    result = rig.run(work, artifacts=writer)

    failure = failure_of(
        result, code=errors.ARTIFACT_PUBLICATION_FAILED, stage=errors.WRITE_ARTIFACT
    )
    assert (failure.namespace, failure.value) == ("test", "0")
    assert "scores" in failure.detail
    assert [call for call in rig.calls if call[0] == "create"] == [
        ("create", "scores", "test", "0")
    ]
    assert (tmp_path / "backbone" / "published" / artifact_path("scores", "test", "0")).exists()
