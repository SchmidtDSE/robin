"""One work in, one result out, through the ports and an installed model."""

import hashlib
import inspect
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from doubles import (
    REGISTRY_CSV,
    CallLog,
    CopyingWriter,
    LocalFiles,
    ScriptedModel,
    WorkBuilder,
    installed_factory,
    registry_digest,
)
from robin_contracts.canonical import canonical_json_bytes, checksum_file
from robin_contracts.cards import (
    AudioGeometry,
    HeadCard,
    InferenceParam,
    ModelCard,
    RunnerResampled,
    model_ref,
)
from robin_contracts.layout import artifact_path
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.ports import ArtifactWriter, FileProvider
from robin_contracts.protocols import ModelContext
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.results import (
    ArtifactRecord,
    FailureReport,
    InferenceFailure,
    InferenceSuccess,
)
from robin_contracts.specs import recipe
from robin_contracts.work import (
    REGISTRY_ROLE,
    EmbeddingArtifactInput,
    InferenceWork,
    InputArtifact,
    recording_work_digest,
    work_digest,
)
from robin_inference_engine import detections, engine, errors
from robin_inference_engine.artifacts.embeddings import read_embeddings
from robin_inference_engine.artifacts.metadata import decode_metadata
from robin_inference_engine.artifacts.scores import read_scores
from robin_inference_engine.coverage import check_completion_evidence
from robin_inference_engine.engine import run_work
from robin_inference_engine.load_registry import load_registry

# ---------------------------------------------------------------------------
# The ports: what a caller supplies, and that the doubles supply it.
# ---------------------------------------------------------------------------

PORTS = (
    (FileProvider, {"fetch", "release"}, LocalFiles),
    (ArtifactWriter, {"create"}, CopyingWriter),
)


@pytest.mark.parametrize(("port", "members", "double"), PORTS)
def test_each_double_has_the_signature_its_port_declares(port, members, double):
    for member in members:
        declared = inspect.signature(getattr(port, member))
        supplied = inspect.signature(getattr(double, member))
        assert list(supplied.parameters) == list(declared.parameters), member
        for name, parameter in declared.parameters.items():
            assert supplied.parameters[name].kind == parameter.kind, (member, name)


# ---------------------------------------------------------------------------
# A rig: real files on disk, the doubles over them, and one call log.
# ---------------------------------------------------------------------------

DIM = 4

def build_card(*, pad: str = "centre_crop_end_pad", **overrides) -> ModelCard:
    fields = {
        "model_name": "test-model",
        "model_version": "1",
        "runtime": "none",
        "window_duration": 3.0,
        "sample_rate": 16000,
        "min_detection_threshold": 0.0,
        "score_domain": "sigmoid",
        "taxa_registry_digest": registry_digest(REGISTRY_CSV),
        "audio": AudioGeometry(
            downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad=pad
        ),
        "backend": "none",
        "dtype": "float32",
        # The setting every work WorkBuilder builds carries.
        "inference_params": (InferenceParam(name="gain", type="float"),),
        "can_emit_embeddings": True,
        "embedding_dim": DIM,
        "embedding_dtype": "float32",
    }
    return ModelCard(**(fields | overrides))


# A model that emits every label for every window.
CARD = build_card()
REF = model_ref(CARD)
DROPPING_CARD = build_card(pad="drop")


def scores_request() -> ScoresRequest:
    return ScoresRequest(contract_id="robin.scores.parquet/1", retention="full")


def embeddings_request() -> EmbeddingsRequest:
    return EmbeddingsRequest(contract_id="robin.embeddings.parquet/1")


class Rig:
    """The inputs a caller holds, written under one test's `tmp_path`, and one call log."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        card: ModelCard | HeadCard = CARD,
        registry_csv: bytes | None = REGISTRY_CSV,
    ) -> None:
        self.tmp_path = tmp_path
        self.calls: CallLog = []
        self.contexts: list[ModelContext] = []
        self.build = WorkBuilder(tmp_path / "inputs", card, registry_csv=registry_csv)
        self.destination = tmp_path / "published"
        self.model_files = self.files("model_files", self.build.model_paths)

    def work(self, *, recordings: int | tuple = 1, **overrides) -> InferenceWork:
        """A scores work over `recordings`, or over that many recordings written for it."""
        if isinstance(recordings, int):
            recordings = tuple(self.build.recording(str(n)) for n in range(recordings))
        return self.build.work(recordings, **({"outputs": (scores_request(),)} | overrides))

    def model(self, script=(), **kwargs) -> ScriptedModel:
        return ScriptedModel(script=script, calls=self.calls, **kwargs)

    def files(self, name: str, paths: dict[str, Path], **kwargs) -> LocalFiles:
        return LocalFiles(name, paths, self.calls, **kwargs)

    def run(self, work, model=None, *, inputs=None, artifacts=None, install=True, **kwargs):
        model = model or self.model()

        def factory(context: ModelContext) -> ScriptedModel:
            self.contexts.append(context)
            return model

        def call():
            return run_work(
                work,
                model_files=self.model_files,
                inputs=inputs or self.files("audio", self.build.audio_paths),
                artifacts=artifacts or CopyingWriter(self.destination, self.calls),
                **kwargs,
            )

        if not install:
            return call()
        with installed_factory(self.tmp_path, model_ref(work.model.card).id, factory):
            return call()

    def model_file_uri(self, role: str) -> str:
        return self.build.files[role].uri

    def released_model_files(self) -> list[Path]:
        return [call[2] for call in self.calls if call[:2] == ("model_files", "release")]


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path)


def failure_of(result, work: InferenceWork, *, code: str, stage: str) -> FailureReport:
    assert isinstance(result, InferenceFailure), result
    assert result.schema_version == "robin.inference-result/1"
    assert result.work_digest == work_digest(work)
    assert (result.failure.code, result.failure.stage) == (code, stage), result.failure
    return result.failure


def assert_nothing_constructed(rig: Rig) -> None:
    """No instance, no audio, no artifact, and every fetched model file released."""
    engine_calls = {"run", "after_recording", "clean_up", "create"}
    assert not rig.contexts
    assert not any(call[0] in engine_calls or call[0] == "audio" for call in rig.calls)
    assert rig.released_model_files() == rig.model_files.returned


# ---------------------------------------------------------------------------
# Refusals before any model instance exists.
# ---------------------------------------------------------------------------


def test_a_backbone_given_saved_embeddings_is_refused_before_any_input_is_fetched(rig):
    source = EmbeddingArtifactInput(contract_id="robin.embeddings.parquet/1", backbone=CARD)
    embeddings = InputArtifact(uri="file:///embeddings.parquet", checksum="sha256:" + "1" * 64)
    recording = rig.build.recording("0", embeddings=embeddings)
    work = rig.work(recordings=(recording,), input=source)

    result = rig.run(work)

    failure_of(
        result, work, code=errors.INPUT_KIND_DISAGREES_WITH_CARD, stage=errors.VALIDATE_REQUEST
    )
    assert not any(call[0] == "audio" for call in rig.calls)
    assert_nothing_constructed(rig)


def test_model_files_are_fetched_in_role_order_and_a_failed_fetch_is_refused(rig):
    registry_uri = rig.model_file_uri(REGISTRY_ROLE)
    rig.model_files = rig.files(
        "model_files",
        rig.build.model_paths,
        raise_on={registry_uri: PermissionError("the bucket is closed")},
    )
    work = rig.work()

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.MODEL_FILE_UNAVAILABLE, stage=errors.ACQUIRE_MODEL
    )
    assert "PermissionError" in failure.detail
    assert "the bucket is closed" in failure.detail
    assert REGISTRY_ROLE in failure.detail
    assert rig.calls == [("model_files", "fetch", registry_uri)]
    assert_nothing_constructed(rig)


def test_a_failed_fetch_of_the_second_role_releases_the_first(rig):
    weights_uri = rig.model_file_uri("weights")
    rig.model_files = rig.files(
        "model_files", rig.build.model_paths, raise_on={weights_uri: OSError("timed out")}
    )
    work = rig.work()

    result = rig.run(work)

    failure_of(result, work, code=errors.MODEL_FILE_UNAVAILABLE, stage=errors.ACQUIRE_MODEL)
    assert rig.calls == [
        ("model_files", "fetch", rig.model_file_uri(REGISTRY_ROLE)),
        ("model_files", "fetch", weights_uri),
        ("model_files", "release", rig.build.model_path(REGISTRY_ROLE)),
    ]
    assert_nothing_constructed(rig)


def test_a_fetched_path_with_no_file_behind_it_is_refused(rig):
    rig.model_files = rig.files(
        "model_files", rig.build.model_paths, missing=(rig.model_file_uri("weights"),)
    )
    work = rig.work()

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.MODEL_FILE_UNAVAILABLE, stage=errors.ACQUIRE_MODEL
    )
    assert "weights.bin.absent" in failure.detail
    assert_nothing_constructed(rig)


def test_a_model_file_of_the_wrong_size_is_refused_naming_both(rig):
    work = rig.work()
    pinned = rig.build.files["weights"]
    rig.build.model_path("weights").write_bytes(b"x" * (pinned.size_bytes + 1))

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.MODEL_FILE_DIGEST_MISMATCH, stage=errors.ACQUIRE_MODEL
    )
    assert "'weights'" in failure.detail
    assert str(pinned.size_bytes) in failure.detail
    assert str(pinned.size_bytes + 1) in failure.detail
    assert_nothing_constructed(rig)


def test_a_model_file_with_the_wrong_digest_is_refused_naming_both(rig):
    work = rig.work()
    path = rig.build.model_path("weights")
    path.write_bytes(path.read_bytes().upper())  # same size, different bytes
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.MODEL_FILE_DIGEST_MISMATCH, stage=errors.ACQUIRE_MODEL
    )
    assert "'weights'" in failure.detail
    assert rig.build.files["weights"].digest in failure.detail
    assert actual in failure.detail
    assert_nothing_constructed(rig)


def test_scores_from_a_work_pinning_no_registry_file_are_refused(tmp_path):
    rig = Rig(tmp_path, registry_csv=None)
    work = rig.work()

    result = rig.run(work)

    failure_of(result, work, code=errors.REGISTRY_REQUIRED, stage=errors.VALIDATE_REQUEST)
    assert rig.released_model_files() == [rig.build.model_path("weights")]
    assert_nothing_constructed(rig)


def test_a_registry_that_cannot_be_read_is_refused(rig, monkeypatch):
    def load_a_deleted_file(path: Path):
        path.unlink()
        return load_registry(path)

    monkeypatch.setattr(engine, "load_registry", load_a_deleted_file)
    work = rig.work()

    result = rig.run(work)

    failure_of(result, work, code=errors.REGISTRY_UNREADABLE, stage=errors.LOAD_REGISTRY)
    assert_nothing_constructed(rig)


def test_a_malformed_registry_is_refused(tmp_path):
    rig = Rig(tmp_path, registry_csv=b"class_index,label\n0,owl\n")
    work = rig.work()

    result = rig.run(work)

    failure = failure_of(result, work, code=errors.REGISTRY_INVALID, stage=errors.LOAD_REGISTRY)
    assert "label_kind" in failure.detail
    assert_nothing_constructed(rig)


def test_a_registry_replaced_after_verification_is_refused(rig, monkeypatch):
    replacement = REGISTRY_CSV.replace(b"0,owl", b"0,hawk")

    def replace_then_load(path: Path):
        path.write_bytes(replacement)
        return load_registry(path)

    monkeypatch.setattr(engine, "load_registry", replace_then_load)
    work = rig.work()

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.REGISTRY_FINGERPRINT_MISMATCH, stage=errors.VALIDATE_REQUEST
    )
    assert rig.build.files[REGISTRY_ROLE].digest in failure.detail
    assert "sha256:" + hashlib.sha256(replacement).hexdigest() in failure.detail
    assert not any(call[0] == "run" for call in rig.calls)
    assert_nothing_constructed(rig)


def test_a_registry_the_card_does_not_name_is_refused_before_construction(tmp_path):
    rig = Rig(tmp_path, card=build_card(taxa_registry_digest=registry_digest(TAXA_CSV)))
    work = rig.work()

    result = rig.run(work)

    failure_of(
        result, work, code=errors.REGISTRY_DISAGREES_WITH_CARD, stage=errors.VALIDATE_REQUEST
    )
    assert_nothing_constructed(rig)


def test_a_model_nobody_installed_is_refused_at_construction(rig):
    work = rig.work()

    result = rig.run(work, install=False)

    failure = failure_of(
        result, work, code=errors.MODEL_NOT_INSTALLED, stage=errors.CONSTRUCT_MODEL
    )
    assert REF.id in failure.detail
    assert_nothing_constructed(rig)


def test_a_factory_that_raises_is_refused_and_nothing_is_cleaned_up(rig):
    work = rig.work()

    def factory(context: ModelContext):
        raise MemoryError("no room for the weights")

    with installed_factory(rig.tmp_path, REF.id, factory):
        result = rig.run(work, install=False)

    failure = failure_of(
        result, work, code=errors.MODEL_CONSTRUCTION_FAILED, stage=errors.CONSTRUCT_MODEL
    )
    assert "no room for the weights" in failure.detail
    assert_nothing_constructed(rig)


# ---------------------------------------------------------------------------
# Refusals the card decides, before any model instance exists.
# ---------------------------------------------------------------------------


def test_scores_from_a_card_that_emits_none_are_refused_before_construction(tmp_path):
    rig = Rig(tmp_path, card=build_card(score_domain=None, taxa_registry_digest=None))
    work = rig.work()

    result = rig.run(work)

    failure_of(result, work, code=errors.SCORES_NOT_EMITTED, stage=errors.VALIDATE_REQUEST)
    assert_nothing_constructed(rig)


def test_a_setting_the_card_does_not_declare_is_refused_before_construction(rig):
    work = rig.work(settings={"gain": 1.0, "top_k": 5})

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.SETTING_UNDECLARED, stage=errors.VALIDATE_REQUEST
    )
    assert "top_k" in failure.detail
    assert_nothing_constructed(rig)


def test_an_explicit_storage_width_the_card_does_not_declare_is_refused_before_construction(rig):
    request = EmbeddingsRequest(contract_id="robin.embeddings.parquet/1", storage_dtype="float16")
    work = rig.work(outputs=(request,))

    result = rig.run(work)

    failure_of(result, work, code=errors.EMBEDDING_DTYPE_DISAGREES, stage=errors.VALIDATE_REQUEST)
    assert_nothing_constructed(rig)


def test_a_head_given_audio_is_refused_before_construction(tmp_path):
    head = HeadCard(
        model_name="test-head",
        model_version="1",
        runtime="onnx",
        backbone=REF,
        embedding_dim=1280,
        min_detection_threshold=0.0,
        score_domain="sigmoid",
        taxa_registry_digest=registry_digest(REGISTRY_CSV),
    )
    rig = Rig(tmp_path, card=head)
    work = rig.work()

    result = rig.run(work)

    failure_of(
        result, work, code=errors.INPUT_KIND_DISAGREES_WITH_CARD, stage=errors.VALIDATE_REQUEST
    )
    assert_nothing_constructed(rig)


LABELS = ("owl", "rain")


def window(position: int, *, scores=True, embedding=False) -> WindowOutput:
    start = 3.0 * position
    return WindowOutput(
        start=start,
        end=start + 3.0,
        scores=tuple(ClassScore(label, 0.25 * (position + 1) % 1.0) for label in LABELS)
        if scores
        else (),
        embedding=np.full(DIM, position + 1.0, dtype=np.float32) if embedding else None,
    )


def script(*counts: int, **kwargs) -> list[list[WindowOutput]]:
    """For each recording in turn, that many windows."""
    return [[window(position, **kwargs) for position in range(count)] for count in counts]


def published(rig: Rig, record: ArtifactRecord) -> Path:
    return rig.destination / artifact_path(record.kind, record.namespace, record.value)


def published_files(rig: Rig) -> set[str]:
    """Every file under the destination, as a path relative to it."""
    if not rig.destination.exists():
        return set()
    return {
        path.relative_to(rig.destination).as_posix()
        for path in rig.destination.rglob("*")
        if path.is_file()
    }


def rows_of(rig: Rig, record: ArtifactRecord) -> list[dict]:
    if record.kind == "detections":
        path = published(rig, record)
        assert checksum_file(path) == record.checksum
        return pq.read_table(path).to_pylist()
    read = read_scores if record.kind == "scores" else read_embeddings
    with read(published(rig, record), expected_checksum=record.checksum) as stream:
        return [row for batch in stream.batches for row in batch.to_pylist()]


def artifact(result: InferenceSuccess, kind: str) -> ArtifactRecord:
    return next(one for one in result.artifacts if one.kind == kind)


def success_of(rig: Rig, result, work: InferenceWork) -> InferenceSuccess:
    """Check everything any success must say, and return it."""
    assert isinstance(result, InferenceSuccess), result
    assert result.work_digest == work_digest(work)
    assert [(row.namespace, row.value) for row in result.coverage] == [
        (one.namespace, one.value) for one in work.recordings
    ]
    counted = {
        (row.namespace, row.value): {
            "scores": row.score_rows,
            "embeddings": row.embedding_rows,
            "detections": row.detection_rows,
        }
        for row in result.coverage
    }
    # A record for every recording and kind with rows, holding exactly those rows.
    assert {
        (one.kind, one.namespace, one.value): one.rows for one in result.artifacts
    } == {
        (kind, *identity): count
        for identity, counts in counted.items()
        for kind, count in counts.items()
        if count
    }
    for record in result.artifacts:
        assert record.uri == published(rig, record).as_uri()
        assert len(rows_of(rig, record)) == record.rows
    requested_scores = any(isinstance(one, ScoresRequest) for one in work.outputs)
    assert (result.resolved_scores_request is not None) == requested_scores
    requested_detections = next(
        (one for one in work.outputs if isinstance(one, DetectionsRequest)), None
    )
    assert result.resolved_detection_policy == (
        requested_detections.policy if requested_detections is not None else None
    )
    check_completion_evidence(work, result)
    return result


# ---------------------------------------------------------------------------
# Completed works.
# ---------------------------------------------------------------------------


def test_a_scores_only_work_writes_every_score(rig):
    work = rig.work()

    result = success_of(rig, rig.run(work, rig.model(script(3))), work)

    assert [one.kind for one in result.artifacts] == ["scores"]
    assert (result.coverage[0].windows_completed, result.coverage[0].score_rows) == (3, 6)
    assert result.resolved_scores_request == scores_request()
    assert result.registry_uri == rig.build.files[REGISTRY_ROLE].uri
    assert result.registry_fingerprint == rig.build.files[REGISTRY_ROLE].digest
    assert result.model == work.model
    assert result.recipe == recipe(CARD, work.settings)
    assert result.window_geometry == recipe(CARD, work.settings).audio.geometry
    rows = rows_of(rig, artifact(result, "scores"))
    assert [(row["window_start_s"], row["label"]) for row in rows[:2]] == [
        (0.0, "owl"),
        (0.0, "rain"),
    ]


def test_an_embeddings_only_work_writes_one_vector_per_window(rig):
    work = rig.work(outputs=(embeddings_request(),))

    result = success_of(
        rig, rig.run(work, rig.model(script(2, scores=False, embedding=True))), work
    )

    assert [one.kind for one in result.artifacts] == ["embeddings"]
    assert result.resolved_scores_request is None
    rows = rows_of(rig, artifact(result, "embeddings"))
    assert [row["embedding"] for row in rows] == [[1.0] * DIM, [2.0] * DIM]


def test_a_work_asking_for_both_writes_both(rig):
    work = rig.work(outputs=(scores_request(), embeddings_request()))

    result = success_of(rig, rig.run(work, rig.model(script(2, embedding=True))), work)

    assert [one.kind for one in result.artifacts] == ["scores", "embeddings"]
    assert (result.coverage[0].score_rows, result.coverage[0].embedding_rows) == (4, 2)


def test_several_recordings_each_get_one_coverage_row_in_the_works_order(rig):
    work = rig.work(recordings=3)

    result = success_of(rig, rig.run(work, rig.model(script(2, 1, 4))), work)

    assert [row.windows_completed for row in result.coverage] == [2, 1, 4]
    assert [row.last_window_end_s for row in result.coverage] == [6.0, 3.0, 12.0]


def test_a_recording_with_one_window_completes(rig):
    work = rig.work()

    result = success_of(rig, rig.run(work, rig.model(script(1))), work)

    assert result.coverage[0].first_window_start_s == 0.0
    assert result.coverage[0].last_window_end_s == 3.0


def test_scores_nobody_asked_for_are_dropped_and_written_nowhere(rig):
    work = rig.work(outputs=(embeddings_request(),))

    result = success_of(rig, rig.run(work, rig.model(script(2, embedding=True))), work)

    assert [one.kind for one in result.artifacts] == ["embeddings"]
    assert [row.score_rows for row in result.coverage] == [0]
    assert result.coverage[0].windows_completed == 2


def test_recordings_sharing_a_value_in_two_namespaces_stay_apart(rig):
    recordings = (
        rig.build.recording("7", namespace="soundhub"),
        rig.build.recording("7", namespace="another-archive"),
    )
    work = rig.work(recordings=recordings)

    result = success_of(rig, rig.run(work, rig.model(script(2, 1))), work)

    assert [(row.namespace, row.value, row.windows_completed) for row in result.coverage] == [
        ("soundhub", "7", 2),
        ("another-archive", "7", 1),
    ]
    assert published_files(rig) == {
        artifact_path("scores", "soundhub", "7"),
        artifact_path("scores", "another-archive", "7"),
    }


def test_the_model_is_built_from_the_verified_files_and_the_work(rig):
    resources = {"batch_size": 8, "device": "cpu"}
    work = rig.work(outputs=(scores_request(), embeddings_request()), resources=resources)
    lines: list[str] = []

    rig.run(work, rig.model(script(1, embedding=True)), log=lines.append)

    (context,) = rig.contexts
    assert context.card == CARD
    assert context.registry is not None
    assert context.registry.fingerprint == rig.build.files[REGISTRY_ROLE].digest
    assert dict(context.files) == {role: rig.build.model_path(role) for role in rig.build.files}
    assert context.settings == work.settings == {"gain": 1.0}
    assert context.resources == resources
    assert context.emit_embeddings is True
    assert context.log == lines.append
    assert not context.scratch_dir.exists()


# ---------------------------------------------------------------------------
# What the request keeps, and what the card says the model emits.
# ---------------------------------------------------------------------------


def test_a_top_k_request_publishes_the_first_k_of_a_full_stream_per_window(taxa_rig):
    rig = taxa_rig
    request = ScoresRequest(
        contract_id="robin.scores.parquet/1", retention="top_k", min_score=0.0, top_k=2
    )
    work = rig.work(outputs=(request,))
    # The model emits every label in output order; owl and spotted tie in the first.
    model = rig.model([[taxa_window(0.0, 0.5, 0.9, 0.5), taxa_window(3.0, 0.1, 0.2, 0.3)]])

    result = success_of(rig, rig.run(work, model), work)

    record = artifact(result, "scores")
    rows = rows_of(rig, record)
    assert [(row["window_start_s"], row["label"], row["score"]) for row in rows] == [
        (0.0, "rain", pytest.approx(0.9)),
        (0.0, "owl", pytest.approx(0.5)),
        (3.0, "spotted", pytest.approx(0.3)),
        (3.0, "rain", pytest.approx(0.2)),
    ]
    assert result.coverage[0].score_rows == 4
    with read_scores(published(rig, record), expected_checksum=record.checksum) as stream:
        header = stream.metadata
    assert header["robin.score_retention"] == "top_k"
    assert header["robin.score_top_k"] == "2"
    assert header["robin.score_floor"] == "0.0"


def test_a_thresholded_request_above_the_models_floor_publishes_only_what_reaches_it(rig):
    request = ScoresRequest(
        contract_id="robin.scores.parquet/1", retention="thresholded", min_score=0.5
    )
    work = rig.work(outputs=(request,))
    scores = (ClassScore("owl", 0.4), ClassScore("rain", 0.5))

    result = success_of(
        rig, rig.run(work, rig.model([[WindowOutput(start=0.0, end=3.0, scores=scores)]])), work
    )

    rows = rows_of(rig, artifact(result, "scores"))
    assert [(row["label"], row["score"]) for row in rows] == [("rain", 0.5)]


def test_a_full_stream_missing_a_label_fails_the_work_under_a_reduced_request(rig):
    request = ScoresRequest(
        contract_id="robin.scores.parquet/1", retention="top_k", min_score=0.0, top_k=1
    )
    work = rig.work(outputs=(request,))
    short = WindowOutput(start=0.0, end=3.0, scores=(ClassScore("owl", 0.9),))

    result = rig.run(work, rig.model([[short]]))

    failure_of(result, work, code=errors.INCOMPLETE_FULL_SCORES, stage=errors.ACCEPT_WINDOW)
    assert_failed_cleanly(rig, result)


def test_an_embeddings_only_work_without_a_registry_succeeds_and_drops_every_score(tmp_path):
    rig = Rig(tmp_path, registry_csv=None)
    work = rig.work(outputs=(embeddings_request(),))
    # Scores no one asked for are dropped before they are checked, however malformed.
    junk = (ClassScore("hawk", math.nan), ClassScore("hawk", 7.0))
    embedding = np.ones(DIM, dtype=np.float32)
    windows = [
        WindowOutput(start=3.0 * n, end=3.0 * n + 3.0, scores=junk, embedding=embedding)
        for n in range(2)
    ]

    result = success_of(rig, rig.run(work, rig.model([windows])), work)

    assert result.registry_fingerprint is None
    assert [one.kind for one in result.artifacts] == ["embeddings"]
    assert (result.coverage[0].windows_completed, result.coverage[0].score_rows) == (2, 0)


@pytest.mark.parametrize(
    ("source", "storage"), [("float32", "float16"), ("float16", "float32")]
)
def test_embeddings_are_accepted_at_the_emitted_precision_and_stored_at_the_recipes(
    tmp_path, source, storage
):
    rig = Rig(tmp_path, card=build_card(embedding_dtype=source, dtype=storage))
    work = rig.work(outputs=(embeddings_request(),))
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=source)
    model = rig.model([[WindowOutput(start=0.0, end=3.0, embedding=values)]])

    result = success_of(rig, rig.run(work, model), work)

    record = artifact(result, "embeddings")
    with read_embeddings(published(rig, record), expected_checksum=record.checksum) as stream:
        header = stream.metadata
        (batch,) = list(stream.batches)
    assert header["robin.embedding_source_dtype"] == source
    assert header["robin.embedding_storage_dtype"] == storage
    stored = batch.column("embedding")
    assert stored.type.value_type == pa.from_numpy_dtype(np.dtype(storage))
    assert np.array_equal(
        np.asarray(stored[0].values.to_numpy(zero_copy_only=False)), values.astype(storage)
    )


def test_an_embedding_at_the_storage_precision_rather_than_the_emitted_one_fails(tmp_path):
    rig = Rig(tmp_path, card=build_card(embedding_dtype="float32", dtype="float16"))
    work = rig.work(outputs=(embeddings_request(),))
    stored_width = np.ones(DIM, dtype=np.float16)

    result = rig.run(work, rig.model([[WindowOutput(start=0.0, end=3.0, embedding=stored_width)]]))

    failure_of(result, work, code=errors.MALFORMED_EMBEDDING, stage=errors.ACCEPT_WINDOW)


# ---------------------------------------------------------------------------
# One file per recording and kind, at the place its recording gives.
# ---------------------------------------------------------------------------


def window_at(
    start: float, *, score: float | None = 0.75, embedding: float | None = 1.0
) -> WindowOutput:
    return WindowOutput(
        start=start,
        end=start + 3.0,
        scores=() if score is None else tuple(ClassScore(label, score) for label in LABELS),
        embedding=None if embedding is None else np.full(DIM, embedding, dtype=np.float32),
    )


def test_each_recording_gets_its_own_file_of_each_kind(rig):
    work = rig.work(recordings=2, outputs=(scores_request(), embeddings_request()))
    model = rig.model(
        [
            [window_at(0.0, score=0.25), window_at(3.0, score=0.25)],
            [window_at(30.0, score=0.5), window_at(33.0, score=0.5), window_at(36.0, score=0.5)],
        ]
    )

    result = success_of(rig, rig.run(work, model), work)

    assert [(one.kind, one.value) for one in result.artifacts] == [
        ("scores", "0"),
        ("embeddings", "0"),
        ("scores", "1"),
        ("embeddings", "1"),
    ]
    assert published_files(rig) == {
        artifact_path(kind, "test", value)
        for kind in ("scores", "embeddings")
        for value in ("0", "1")
    }
    starts = {
        (one.kind, one.value): sorted({row["window_start_s"] for row in rows_of(rig, one)})
        for one in result.artifacts
    }
    assert starts == {
        ("scores", "0"): [0.0, 3.0],
        ("embeddings", "0"): [0.0, 3.0],
        ("scores", "1"): [30.0, 33.0, 36.0],
        ("embeddings", "1"): [30.0, 33.0, 36.0],
    }
    for record in result.artifacts:
        recording = next(one for one in work.recordings if one.value == record.value)
        read = read_scores if record.kind == "scores" else read_embeddings
        with read(published(rig, record), expected_checksum=record.checksum) as stream:
            assert stream.metadata["robin.recording_work_digest"] == recording_work_digest(
                work, recording
            )
            assert stream.metadata["robin.recording_namespace"] == recording.namespace
            assert stream.metadata["robin.recording_value"] == recording.value


def test_a_recording_with_no_score_rows_still_gets_its_embeddings_file(tmp_path):
    rig, request = thresholded_rig(tmp_path)
    work = rig.work(recordings=2, outputs=(request, embeddings_request()))
    model = rig.model([[window_at(0.0), window_at(3.0)], [window_at(0.0, score=None)]])

    result = success_of(rig, rig.run(work, model), work)

    assert [(one.kind, one.value) for one in result.artifacts] == [
        ("scores", "0"),
        ("embeddings", "0"),
        ("embeddings", "1"),
    ]
    assert result.coverage[1].score_rows == 0
    assert artifact_path("scores", "test", "1") not in published_files(rig)
    assert artifact_path("embeddings", "test", "1") in published_files(rig)


def test_a_value_holding_a_slash_publishes_inside_its_own_folder(rig):
    work = rig.work(recordings=(rig.build.recording("a/b"),))

    result = success_of(rig, rig.run(work, rig.model(script(1))), work)

    (record,) = result.artifacts
    (relative,) = published_files(rig)
    assert relative == artifact_path("scores", "test", "a/b")
    assert relative.split("/") == [
        "scores",
        "recording_namespace=test",
        "recording_value=a%2Fb",
        "scores.parquet",
    ]
    assert record.uri == (rig.destination / relative).as_uri()


@pytest.mark.parametrize("kind", ["scores", "embeddings"])
def test_a_recordings_file_does_not_depend_on_its_batch(rig, kind):
    request = scores_request() if kind == "scores" else embeddings_request()

    def at(start: float, value: float) -> WindowOutput:
        # Scores a work did not request are dropped; an unrequested embedding is refused.
        return window_at(start, score=value, embedding=value if kind == "embeddings" else None)

    r, s_, t = (rig.build.recording(value) for value in ("r", "s", "t"))
    r_windows = [at(0.0, 0.25), at(3.0, 0.5)]
    first_work = rig.work(recordings=(r, s_), outputs=(request,))
    second_work = rig.work(recordings=(t, r), outputs=(request,))
    writer = CopyingWriter(rig.destination, rig.calls)
    location = rig.destination / artifact_path(kind, "test", "r")

    first = success_of(
        rig,
        rig.run(first_work, rig.model([r_windows, [at(6.0, 0.75)]]), artifacts=writer),
        first_work,
    )
    first_bytes = location.read_bytes()
    second = success_of(
        rig,
        rig.run(second_work, rig.model([[at(9.0, 0.5)], r_windows]), artifacts=writer),
        second_work,
    )

    assert work_digest(first_work) != work_digest(second_work)
    assert location.read_bytes() == first_bytes
    first_r, second_r = (
        next(one for one in result.artifacts if one.value == "r") for result in (first, second)
    )
    assert (first_r.uri, first_r.checksum) == (second_r.uri, second_r.checksum)
    assert writer.replayed == [location.as_uri()]


# ---------------------------------------------------------------------------
# Detections, selected from each recording's scores file.
# ---------------------------------------------------------------------------

# One class of each kind: a resolved taxon, an organism with no key, and a sound.
TAXA_CSV = (
    b"class_index,label,label_kind,scientific_name,common_name,gbif_taxon_key\n"
    b"0,owl,taxon,Strix varia,Barred Owl,2497921\n"
    b"1,rain,non_taxonomic,,,\n"
    b"2,spotted,unresolved,Strix occidentalis caurina,Northern Spotted Owl,\n"
)
POLICY = ThresholdPolicy(min_score=0.5)


@pytest.fixture
def taxa_rig(tmp_path) -> Rig:
    card = build_card(taxa_registry_digest=registry_digest(TAXA_CSV))
    return Rig(tmp_path, card=card, registry_csv=TAXA_CSV)


def detections_request(policy=POLICY) -> DetectionsRequest:
    return DetectionsRequest(contract_id="robin.detections.parquet/1", policy=policy)


def taxa_window(start: float, owl: float, rain: float, spotted: float, **kwargs) -> WindowOutput:
    """A window scoring every class, as full retention requires."""
    scores = (ClassScore("owl", owl), ClassScore("rain", rain), ClassScore("spotted", spotted))
    return WindowOutput(start=start, end=start + 3.0, scores=scores, **kwargs)


def test_each_recording_selecting_something_gets_one_detections_file(taxa_rig):
    rig = taxa_rig
    work = rig.work(recordings=2, outputs=(scores_request(), detections_request()))
    model = rig.model(
        [
            [taxa_window(0.0, 0.9, 0.6, 0.7), taxa_window(3.0, 0.1, 0.2, 0.5)],
            [taxa_window(0.0, 0.1, 0.2, 0.3)],
        ]
    )

    result = success_of(rig, rig.run(work, model), work)

    record = artifact(result, "detections")
    assert [(one.kind, one.value) for one in result.artifacts] == [
        ("scores", "0"),
        ("detections", "0"),
        ("scores", "1"),
    ]
    assert record.rows == result.coverage[0].detection_rows == 4
    assert result.coverage[1].detection_rows == 0
    assert result.resolved_detection_policy == POLICY
    assert artifact_path("detections", "test", "1") not in published_files(rig)
    rows = rows_of(rig, record)
    columns = ("window_start_s", "rank", "label", "label_kind", "gbif_taxon_key")
    assert [tuple(row[name] for name in columns) for row in rows] == [
        (0.0, 1, "owl", "taxon", 2497921),
        (0.0, 2, "spotted", "unresolved", None),
        (0.0, 3, "rain", "non_taxonomic", None),
        (3.0, 1, "spotted", "unresolved", None),
    ]
    assert rows[1]["scientific_name"] == "Strix occidentalis caurina"
    assert rows[2]["scientific_name"] is None


def test_a_detections_header_names_the_scores_file_it_was_selected_from(taxa_rig):
    rig = taxa_rig
    work = rig.work(outputs=(scores_request(), detections_request()))

    result = success_of(rig, rig.run(work, rig.model([[taxa_window(0.0, 0.9, 0.1, 0.1)]])), work)

    scores, record = artifact(result, "scores"), artifact(result, "detections")
    stored = decode_metadata(pq.read_schema(published(rig, record)).metadata)
    assert stored["robin.contract"] == "robin.detections.parquet/1"
    assert ThresholdPolicy.model_validate_json(stored["robin.detection_policy"]) == POLICY
    assert json.loads(stored["robin.source_artifacts"]) == [
        {"contract_id": scores.contract_id, "checksum": scores.checksum}
    ]
    assert stored["robin.recording_work_digest"] == recording_work_digest(
        work, work.recordings[0]
    )
    # The retention keys describe the scores the rows were ranked among.
    assert stored["robin.score_retention"] == "full"
    assert stored["robin.registry_fingerprint"] == result.registry_fingerprint


def test_a_recording_with_no_windows_has_no_detections(tmp_path):
    card = build_card(pad="drop", taxa_registry_digest=registry_digest(TAXA_CSV))
    rig = Rig(tmp_path, card=card, registry_csv=TAXA_CSV)
    recording = rig.build.recording("0", duration_seconds=2.0)
    work = rig.work(recordings=(recording,), outputs=(scores_request(), detections_request()))

    result = success_of(rig, rig.run(work), work)

    assert result.artifacts == ()
    assert result.coverage[0].detection_rows == 0
    assert result.resolved_detection_policy == POLICY


def test_a_recordings_detections_do_not_depend_on_its_batch(taxa_rig):
    rig = taxa_rig
    r, s_, t = (rig.build.recording(value) for value in ("r", "s", "t"))
    outputs = (scores_request(), detections_request())
    r_windows = [taxa_window(0.0, 0.9, 0.6, 0.7), taxa_window(3.0, 0.8, 0.1, 0.1)]
    other = [taxa_window(0.0, 0.6, 0.6, 0.6)]
    first_work = rig.work(recordings=(r, s_), outputs=outputs)
    second_work = rig.work(recordings=(t, r), outputs=outputs)
    writer = CopyingWriter(rig.destination, rig.calls)
    location = rig.destination / artifact_path("detections", "test", "r")

    first = rig.run(first_work, rig.model([r_windows, other]), artifacts=writer)
    success_of(rig, first, first_work)
    first_bytes = location.read_bytes()
    first_ids = pq.read_table(location).column("detection_id").to_pylist()
    second = rig.run(second_work, rig.model([other, r_windows]), artifacts=writer)
    success_of(rig, second, second_work)

    assert work_digest(first_work) != work_digest(second_work)
    assert location.read_bytes() == first_bytes
    assert pq.read_table(location).column("detection_id").to_pylist() == first_ids
    assert location.as_uri() in writer.replayed


def test_published_detections_read_as_one_dataset_naming_each_recording(taxa_rig):
    rig = taxa_rig
    recordings = (rig.build.recording("007"), rig.build.recording("x", namespace="other"))
    work = rig.work(recordings=recordings, outputs=(scores_request(), detections_request()))
    model = rig.model([[taxa_window(0.0, 0.9, 0.1, 0.1)], [taxa_window(0.0, 0.1, 0.8, 0.1)]])
    success_of(rig, rig.run(work, model), work)

    dataset = ds.dataset(
        rig.destination / "detections",
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("recording_namespace", pa.string()), ("recording_value", pa.string())]),
            flavor="hive",
        ),
    )
    rows = dataset.to_table().to_pylist()

    named = [(row["recording_namespace"], row["recording_value"], row["label"]) for row in rows]
    assert sorted(named) == [
        ("other", "x", "rain"),
        ("test", "007", "owl"),
    ]


def test_a_failed_aggregation_fails_the_work_and_publishes_nothing(taxa_rig, monkeypatch):
    rig = taxa_rig
    monkeypatch.setattr(
        detections,
        "build_detections_sql",
        lambda policy, *, recipe_fingerprint: ("SELECT label FROM nowhere", {}),
    )
    work = rig.work(recordings=2, outputs=(scores_request(), detections_request()))

    result = rig.run(work, rig.model([[taxa_window(0.0, 0.9, 0.1, 0.1)]] * 2))

    failure = failure_of(
        result, work, code=errors.AGGREGATION_FAILED, stage=errors.AGGREGATE
    )
    assert (failure.namespace, failure.value) == ("test", "0")
    assert not any(call[0] == "create" for call in rig.calls)
    assert_failed_cleanly(rig, result)


def test_scores_embeddings_and_detections_are_published_together(taxa_rig):
    rig = taxa_rig
    work = rig.work(
        outputs=(scores_request(), embeddings_request(), detections_request())
    )
    embedding = np.ones(DIM, dtype=np.float32)
    model = rig.model(
        [
            [
                taxa_window(0.0, 0.9, 0.1, 0.1, embedding=embedding),
                taxa_window(3.0, 0.6, 0.7, 0.1, embedding=embedding),
            ]
        ]
    )

    result = success_of(rig, rig.run(work, model), work)

    assert [one.kind for one in result.artifacts] == ["scores", "embeddings", "detections"]
    row = result.coverage[0]
    assert (row.score_rows, row.embedding_rows, row.detection_rows) == (6, 2, 3)


# ---------------------------------------------------------------------------
# Window overlap, which a work asks for in its settings.
# ---------------------------------------------------------------------------


def overlap_rig(tmp_path: Path) -> Rig:
    """A rig whose card declares a window overlap among its settings."""
    card = build_card(
        taxa_registry_digest=registry_digest(TAXA_CSV),
        inference_params=(
            InferenceParam(name="gain", type="float"),
            InferenceParam(name="window_overlap", type="float"),
        ),
    )
    return Rig(tmp_path, card=card, registry_csv=TAXA_CSV)


def test_a_work_with_an_overlap_accepts_windows_one_hop_apart(tmp_path):
    rig = overlap_rig(tmp_path)
    settings = {"gain": 1.0, "window_overlap": 1.0}
    work = rig.work(settings=settings)
    # A 3-second window that overlaps the next by 1 second starts every 2 seconds.
    windows = [taxa_window(start, 0.9, 0.1, 0.1) for start in (0.0, 2.0, 4.0)]

    result = success_of(rig, rig.run(work, rig.model([windows])), work)

    assert result.coverage[0].windows_completed == 3
    assert result.recipe.settings == settings
    assert result.window_geometry.hop == 2.0


def test_works_differing_only_in_overlap_publish_different_fingerprints(tmp_path):
    outputs = (scores_request(), detections_request())
    published_by_overlap = []
    for overlap in (1.0, 0.5):
        rig = overlap_rig(tmp_path / str(overlap))
        work = rig.work(settings={"gain": 1.0, "window_overlap": overlap}, outputs=outputs)
        result = success_of(
            rig, rig.run(work, rig.model([[taxa_window(0.0, 0.9, 0.1, 0.1)]])), work
        )
        path = published(rig, artifact(result, "detections"))
        header = decode_metadata(pq.read_schema(path).metadata)
        ids = pq.read_table(path).column("detection_id").to_pylist()
        published_by_overlap.append((header["robin.recipe_fingerprint"], ids))

    (first_fingerprint, first_ids), (second_fingerprint, second_ids) = published_by_overlap
    assert first_fingerprint != second_fingerprint
    assert set(first_ids).isdisjoint(second_ids)


# ---------------------------------------------------------------------------
# What completion evidence requires at its edges.
# ---------------------------------------------------------------------------


def thresholded_rig(tmp_path: Path, floor: float = 0.5) -> tuple[Rig, ScoresRequest]:
    """A rig whose model emits only the scores at or above `floor`, and a request for them."""
    request = ScoresRequest(
        contract_id="robin.scores.parquet/1", retention="thresholded", min_score=floor
    )
    return Rig(tmp_path, card=build_card(min_detection_threshold=floor)), request


def test_a_work_where_no_recording_has_score_rows_has_no_scores_records(tmp_path):
    # Also shows that an accepted window carrying no scores counts as completed.
    rig, request = thresholded_rig(tmp_path)
    work = rig.work(recordings=2, outputs=(request,))

    result = success_of(rig, rig.run(work, rig.model(script(3, 2, scores=False))), work)

    assert result.artifacts == ()
    assert result.resolved_scores_request == request
    assert [(row.windows_completed, row.score_rows) for row in result.coverage] == [
        (3, 0),
        (2, 0),
    ]
    assert published_files(rig) == set()


def test_a_recording_too_short_for_a_dropping_geometry_completes_with_its_reason(tmp_path):
    rig = Rig(tmp_path, card=DROPPING_CARD)
    recording = rig.build.recording("0", duration_seconds=2.0)
    work = rig.work(recordings=(recording,))

    result = success_of(rig, rig.run(work), work)

    (row,) = result.coverage
    assert (row.windows_completed, row.zero_window_reason) == (0, "shorter_than_window")
    assert result.artifacts == ()
    assert published_files(rig) == set()


def test_zero_windows_from_a_recording_of_unknown_duration_are_unexplained(tmp_path):
    rig = Rig(tmp_path, card=DROPPING_CARD)
    work = rig.work()

    result = rig.run(work)

    failure = failure_of(result, work, code=errors.UNEXPLAINED_ZERO_WINDOWS, stage=errors.INFER)
    assert (failure.namespace, failure.value) == ("test", "0")


def test_zero_windows_under_a_geometry_that_floors_at_one_are_unexplained(rig):
    recording = rig.build.recording("0", duration_seconds=2.0)
    work = rig.work(recordings=(recording,))

    result = rig.run(work)

    failure_of(result, work, code=errors.UNEXPLAINED_ZERO_WINDOWS, stage=errors.INFER)


# ---------------------------------------------------------------------------
# Failures once inference has started.
# ---------------------------------------------------------------------------


def assert_failed_cleanly(rig: Rig, result: InferenceFailure) -> None:
    """clean_up ran once, every model file was released, and no artifact is named."""
    assert rig.calls.count(("clean_up",)) == 1
    assert rig.released_model_files() == rig.model_files.returned
    text = canonical_json_bytes(result).decode()
    assert rig.destination.as_uri() not in text
    for relative in published_files(rig):
        assert checksum_file(rig.destination / relative) not in text


def test_an_adapter_raising_mid_recording_fails_the_work(rig):
    work = rig.work()
    model = rig.model([[window(0), ValueError("the tensor was the wrong shape")]])

    result = rig.run(work, model)

    failure = failure_of(result, work, code=errors.MODEL_RUN_FAILED, stage=errors.INFER)
    assert (failure.namespace, failure.value) == ("test", "0")
    assert "ValueError" in failure.detail
    assert "the tensor was the wrong shape" in failure.detail
    assert_failed_cleanly(rig, result)


def test_a_window_the_boundary_refuses_fails_the_work_where_it_happened(rig):
    work = rig.work()
    stray = WindowOutput(start=3.0, end=6.0, scores=(ClassScore("hawk", 0.5),))

    result = rig.run(work, rig.model([[window(0), stray]]))

    failure = failure_of(result, work, code=errors.UNKNOWN_LABEL, stage=errors.ACCEPT_WINDOW)
    assert (failure.namespace, failure.value, failure.window_start_s) == ("test", "0", 3.0)
    assert_failed_cleanly(rig, result)


def test_a_vector_too_large_for_its_storage_width_fails_the_work(tmp_path):
    rig = Rig(tmp_path, card=build_card(dtype="float16"))
    work = rig.work(outputs=(embeddings_request(),))
    huge = WindowOutput(start=0.0, end=3.0, embedding=np.full(DIM, 1e6, dtype=np.float32))

    result = rig.run(work, rig.model([[huge]]))

    failure = failure_of(
        result,
        work,
        code=errors.EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE,
        stage=errors.WRITE_ARTIFACT,
    )
    assert (failure.namespace, failure.value) == ("test", "0")
    assert_failed_cleanly(rig, result)


def test_a_writer_that_fails_on_the_second_create_fails_the_work(rig):
    work = rig.work(recordings=2)
    writer = CopyingWriter(rig.destination, rig.calls, fail_on=2)
    before = work_directories()

    result = rig.run(work, rig.model(script(2, 1)), artifacts=writer)

    failure = failure_of(
        result, work, code=errors.ARTIFACT_PUBLICATION_FAILED, stage=errors.WRITE_ARTIFACT
    )
    assert "OSError" in failure.detail
    assert (failure.namespace, failure.value) == ("test", "1")
    # The first recording's file was published, and the result still names none.
    assert published_files(rig) == {artifact_path("scores", "test", "0")}
    assert work_directories() == before
    assert_failed_cleanly(rig, result)


def test_an_input_the_port_cannot_fetch_fails_the_work_at_acquisition(rig):
    work = rig.work()
    inputs = rig.files("audio", {})

    result = rig.run(work, rig.model(script(1)), inputs=inputs)

    failure = failure_of(result, work, code=errors.INPUT_UNAVAILABLE, stage=errors.ACQUIRE_INPUT)
    assert (failure.namespace, failure.value) == ("test", "0")
    assert "KeyError" in failure.detail
    assert not any(call[:2] == ("audio", "release") for call in rig.calls)
    assert_failed_cleanly(rig, result)


# ---------------------------------------------------------------------------
# The lifecycle, read from one call log.
# ---------------------------------------------------------------------------


def model_file_fetches(rig: Rig) -> list[tuple]:
    return [("model_files", "fetch", rig.model_file_uri(role)) for role in sorted(rig.build.files)]


def model_file_releases(rig: Rig) -> list[tuple]:
    paths = [rig.build.model_path(role) for role in sorted(rig.build.files)]
    return [("model_files", "release", path) for path in paths]


def recordings_run(rig: Rig, work: InferenceWork, positions: range) -> list[tuple]:
    calls = []
    for position in positions:
        uri = work.recordings[position].audio_uri
        path = rig.build.audio_paths[uri]
        calls += [
            ("audio", "fetch", uri),
            ("run", path),
            ("after_recording",),
            ("audio", "release", path),
        ]
    return calls


def test_a_successful_work_calls_every_component_in_order(rig):
    work = rig.work(recordings=3)

    success_of(rig, rig.run(work, rig.model(script(1, 1, 1))), work)

    assert rig.calls == [
        *model_file_fetches(rig),
        *recordings_run(rig, work, range(3)),
        ("create", "scores", "test", "0"),
        ("create", "scores", "test", "1"),
        ("create", "scores", "test", "2"),
        ("clean_up",),
        *model_file_releases(rig),
    ]


def test_a_failure_in_the_second_recording_still_finishes_it_and_opens_no_third(rig):
    work = rig.work(recordings=3)
    model = rig.model([[window(0)], [RuntimeError("out of memory")], [window(0)]])

    result = rig.run(work, model)

    failure = failure_of(result, work, code=errors.MODEL_RUN_FAILED, stage=errors.INFER)
    assert (failure.namespace, failure.value) == ("test", "1")
    assert rig.calls == [
        *model_file_fetches(rig),
        *recordings_run(rig, work, range(2)),
        ("clean_up",),
        *model_file_releases(rig),
    ]


def test_a_failing_cleanup_step_cannot_hide_the_original_failure(rig):
    work = rig.work()
    model = rig.model(
        [[ValueError("the first failure")]],
        fail_after_recording=OSError("the second failure"),
        fail_clean_up=OSError("the third failure"),
    )
    rig.model_files = rig.files(
        "model_files", rig.build.model_paths, fail_release=OSError("the fourth failure")
    )
    lines: list[str] = []

    result = rig.run(work, model, log=lines.append)

    failure = failure_of(result, work, code=errors.MODEL_RUN_FAILED, stage=errors.INFER)
    assert "the first failure" in failure.detail
    for later in ("the second failure", "the third failure", "the fourth failure"):
        assert any(later in line for line in lines), later
    assert rig.calls[-3:] == [("clean_up",), *model_file_releases(rig)]


CLEANUP_FAILURES = {
    "after_recording": (
        {"fail_after_recording": OSError("could not close")},
        {},
        errors.MODEL_RUN_FAILED,
        errors.INFER,
        ("test", "0"),
    ),
    "input_release": (
        {},
        {"inputs": {"fail_release": OSError("disk busy")}},
        errors.INPUT_UNAVAILABLE,
        errors.ACQUIRE_INPUT,
        ("test", "0"),
    ),
    "clean_up": (
        {"fail_clean_up": OSError("could not free")},
        {},
        errors.MODEL_RUN_FAILED,
        errors.INFER,
        (None, None),
    ),
    "model_file_release": (
        {},
        {"model_files": {"fail_release": OSError("cache locked")}},
        errors.MODEL_FILE_UNAVAILABLE,
        errors.ACQUIRE_MODEL,
        (None, None),
    ),
}


@pytest.mark.parametrize("step", CLEANUP_FAILURES)
def test_a_cleanup_step_failing_on_an_otherwise_successful_path_is_the_failure(rig, step):
    model_kwargs, port_kwargs, code, stage, recording = CLEANUP_FAILURES[step]
    work = rig.work()
    if "model_files" in port_kwargs:
        rig.model_files = rig.files(
            "model_files", rig.build.model_paths, **port_kwargs["model_files"]
        )
    inputs = rig.files("audio", rig.build.audio_paths, **port_kwargs.get("inputs", {}))

    result = rig.run(work, rig.model(script(1), **model_kwargs), inputs=inputs)

    failure = failure_of(result, work, code=code, stage=stage)
    assert (failure.namespace, failure.value) == recording
    assert "OSError" in failure.detail
    assert rig.calls[-3:] == [("clean_up",), *model_file_releases(rig)]


# ---------------------------------------------------------------------------
# Nothing is left behind but what was published.
# ---------------------------------------------------------------------------


def listing(root: Path) -> set[Path]:
    return set(root.rglob("*"))


def work_directories() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("robin-work-*"))


LEFT_BEHIND_CASES = {
    "success": lambda rig: (rig.model(script(2, embedding=True)), {}),
    "adapter_raises": lambda rig: (rig.model([[window(0), ValueError("broken")]]), {}),
    "writer_raises": lambda rig: (
        rig.model(script(2, embedding=True)),
        {"artifacts": CopyingWriter(rig.destination, rig.calls, fail_on=2)},
    ),
    "input_unavailable": lambda rig: (rig.model(script(1)), {"inputs": rig.files("audio", {})}),
    "refused_window": lambda rig: (
        rig.model([[WindowOutput(start=0.0, end=3.0, scores=(ClassScore("hawk", 0.5),))]]),
        {},
    ),
    "refused_before_construction": lambda rig: (
        rig.model(),
        {"model_files": {"missing": (rig.model_file_uri("weights"),)}},
    ),
}


@pytest.mark.parametrize("case", LEFT_BEHIND_CASES)
def test_the_engine_leaves_nothing_behind(rig, case):
    work = rig.work(outputs=(scores_request(), embeddings_request()))
    model, ports = LEFT_BEHIND_CASES[case](rig)
    if "model_files" in ports:
        rig.model_files = rig.files(
            "model_files", rig.build.model_paths, **ports.pop("model_files")
        )
    before, directories = listing(rig.tmp_path), work_directories()

    rig.run(work, model, **ports)

    added = listing(rig.tmp_path) - before
    assert all(path == rig.destination or rig.destination in path.parents for path in added)
    assert before <= listing(rig.tmp_path)
    assert work_directories() == directories


# ---------------------------------------------------------------------------
# The same work gives the same result, and accepted rows are the engine's own.
# ---------------------------------------------------------------------------


def test_the_same_work_run_twice_gives_byte_identical_results(rig):
    work = rig.work(recordings=2, outputs=(scores_request(), embeddings_request()))

    first = rig.run(work, rig.model(script(2, 3, embedding=True)))
    second = rig.run(work, rig.model(script(2, 3, embedding=True)))

    assert isinstance(first, InferenceSuccess)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_an_adapter_reusing_one_buffer_cannot_change_rows_already_accepted(rig):
    work = rig.work(outputs=(embeddings_request(),))
    model = rig.model(script(3, scores=False, embedding=True), reuse_buffer=True)

    result = success_of(rig, rig.run(work, model), work)

    rows = rows_of(rig, artifact(result, "embeddings"))
    assert [row["embedding"] for row in rows] == [[1.0] * DIM, [2.0] * DIM, [3.0] * DIM]


# ---------------------------------------------------------------------------
# The caller's log.
# ---------------------------------------------------------------------------


def test_the_callers_log_receives_lines_and_changes_nothing_else(rig, capsys):
    work = rig.work(recordings=2)
    lines: list[str] = []

    logged = rig.run(work, rig.model(script(1, 2)), log=lines.append)
    silent = rig.run(work, rig.model(script(1, 2)))

    assert lines
    assert all(isinstance(line, str) and line for line in lines)
    assert canonical_json_bytes(logged) == canonical_json_bytes(silent)
    assert capsys.readouterr().out == ""
