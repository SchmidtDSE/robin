"""One work in, one result out, through the ports and an installed model."""

import hashlib
import inspect
import tempfile
from collections import Counter
from pathlib import Path
from typing import get_protocol_members

import numpy as np
import pytest

from doubles import (
    REGISTRY_CSV,
    CallLog,
    CopyingWriter,
    LocalFiles,
    ScriptedModel,
    WorkBuilder,
    installed_factory,
)
from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.cards import HeadCard, ModelCard, ModelRef, model_ref
from robin_contracts.embedding_transforms import Identity, L2Norm
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.protocols import ModelCapabilities, ModelContext
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.results import (
    ArtifactContractId,
    ArtifactKind,
    ArtifactRecord,
    FailureReport,
    InferenceFailure,
    InferenceSuccess,
)
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled
from robin_contracts.work import REGISTRY_ROLE, EmbeddingArtifactInput, InferenceWork, work_digest
from robin_inference_engine import engine, errors
from robin_inference_engine.artifacts.embeddings import read_embeddings
from robin_inference_engine.artifacts.scores import read_scores
from robin_inference_engine.coverage import check_completion_evidence
from robin_inference_engine.engine import run_work
from robin_inference_engine.load_registry import load_registry
from robin_inference_engine.ports import ArtifactWriter, FileAcquisition

# ---------------------------------------------------------------------------
# The ports: what a caller supplies, and that the doubles supply it.
# ---------------------------------------------------------------------------

PORTS = (
    (FileAcquisition, {"fetch", "release"}, LocalFiles),
    (ArtifactWriter, {"create"}, CopyingWriter),
)


@pytest.mark.parametrize(("port", "members", "_"), PORTS)
def test_each_port_declares_exactly_its_members(port, members, _):
    assert get_protocol_members(port) == members


@pytest.mark.parametrize(("port", "members", "double"), PORTS)
def test_each_double_has_the_signature_its_port_declares(port, members, double):
    for member in members:
        declared = inspect.signature(getattr(port, member))
        supplied = inspect.signature(getattr(double, member))
        assert list(supplied.parameters) == list(declared.parameters), member
        for name, parameter in declared.parameters.items():
            assert supplied.parameters[name].kind == parameter.kind, (member, name)


def test_the_writer_is_asked_for_a_declared_kind_and_contract():
    parameters = inspect.signature(ArtifactWriter.create).parameters

    assert parameters["kind"].annotation is ArtifactKind
    assert parameters["contract_id"].annotation is ArtifactContractId


# ---------------------------------------------------------------------------
# A rig: real files on disk, the doubles over them, and one call log.
# ---------------------------------------------------------------------------

DIM = 4

CARD = ModelCard(
    model_name="test-model",
    model_version="1",
    runtime="none",
    segment_duration=3.0,
    sample_rate=16000,
    min_detection_threshold=0.0,
    can_emit_embeddings=True,
    embedding_dim=DIM,
)
REF = model_ref(CARD)


def build_recipe(*, model: ModelRef = REF, **audio_overrides) -> Recipe:
    audio = {
        "sample_rate": 16000,
        "window": 3.0,
        "hop": 3.0,
        "downmix": "mean",
        "resampler": RunnerResampled(algorithm="soxr_hq"),
        "pad": "centre_crop_end_pad",
    }
    return Recipe(
        model=model,
        backend="none",
        audio=AudioSpec(**(audio | audio_overrides)),
        embedding_transform=Identity(),
        dtype="float32",
    )


def build_capabilities(**overrides) -> ModelCapabilities:
    fields = {
        "emits_scores": True,
        "emits_embeddings": True,
        "score_domain": "probability",
        "supported_retention": frozenset({"full"}),
        "embedding_dim": DIM,
        "embedding_dtype": "float32",
    }
    return ModelCapabilities(**(fields | overrides))


def scores_request() -> ScoresRequest:
    return ScoresRequest(contract_id="robin.scores.arrow/1", retention="full")


def embeddings_request() -> EmbeddingsRequest:
    return EmbeddingsRequest(contract_id="robin.embeddings.arrow/1")


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

    def model(self, script=(), *, recipe=None, capabilities=None, **kwargs) -> ScriptedModel:
        return ScriptedModel(
            recipe=recipe or build_recipe(model=model_ref(self.build.card)),
            capabilities=capabilities or build_capabilities(),
            script=script,
            calls=self.calls,
            **kwargs,
        )

    def files(self, name: str, paths: dict[str, Path], **kwargs) -> LocalFiles:
        return LocalFiles(name, paths, self.calls, **kwargs)

    def run(self, work, model=None, *, audio=None, artifacts=None, install=True, **kwargs):
        model = model or self.model()

        def factory(context: ModelContext) -> ScriptedModel:
            self.contexts.append(context)
            return model

        def call():
            return run_work(
                work,
                model_files=self.model_files,
                audio=audio or self.files("audio", self.build.audio_paths),
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


def test_a_detections_request_is_refused_before_anything_is_fetched(rig):
    detections = DetectionsRequest(
        contract_id="robin.detections.parquet/1", policy=ThresholdPolicy(min_score=0.5)
    )
    work = rig.work(outputs=(scores_request(), detections))

    result = rig.run(work)

    failure_of(
        result, work, code=errors.DETECTIONS_NOT_AVAILABLE, stage=errors.VALIDATE_REQUEST
    )
    assert rig.calls == []
    assert_nothing_constructed(rig)


def test_an_embedding_artifact_input_is_refused_before_anything_is_fetched(rig):
    source = EmbeddingArtifactInput(
        contract_id="robin.embeddings.arrow/1",
        uri="file:///embeddings.arrow",
        checksum="sha256:" + "1" * 64,
    )
    work = rig.work(input=source)

    result = rig.run(work)

    failure_of(
        result, work, code=errors.EMBEDDING_INPUT_NOT_AVAILABLE, stage=errors.VALIDATE_REQUEST
    )
    assert rig.calls == []
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


def test_a_head_declaring_a_class_the_registry_does_not_is_refused(tmp_path):
    head = HeadCard(
        model_name="test-head",
        model_version="1",
        backbone=REF,
        classes=("owl", "hawk"),
        required_embedding_transform=L2Norm(),
    )
    rig = Rig(tmp_path, card=head)
    work = rig.work()

    result = rig.run(work)

    failure = failure_of(
        result, work, code=errors.HEAD_CLASS_NOT_IN_REGISTRY, stage=errors.VALIDATE_REQUEST
    )
    assert "hawk" in failure.detail
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
# Refusals once an instance exists: it is cleaned up before its files are released.
# ---------------------------------------------------------------------------


def construction_then_clean_up(rig: Rig) -> list[tuple[object, ...]]:
    fetches = [
        ("model_files", "fetch", rig.model_file_uri(role)) for role in sorted(rig.build.files)
    ]
    releases = [("model_files", "release", path) for path in rig.model_files.returned]
    return fetches + [("clean_up",)] + releases


def test_an_instance_contradicting_the_request_is_refused_and_cleaned_up(rig):
    work = rig.work()
    silent = build_capabilities(
        emits_scores=False, score_domain=None, supported_retention=frozenset()
    )

    result = rig.run(work, rig.model(capabilities=silent))

    failure_of(result, work, code=errors.SCORES_NOT_EMITTED, stage=errors.VALIDATE_REQUEST)
    assert rig.calls == construction_then_clean_up(rig)
    assert len(rig.contexts) == 1


def test_an_instance_whose_recipe_names_another_model_is_refused_and_cleaned_up(rig):
    work = rig.work()
    stale = CARD.model_copy(update={"segment_duration": 5.0})

    result = rig.run(work, rig.model(recipe=build_recipe(model=model_ref(stale))))

    failure_of(result, work, code=errors.RECIPE_MODEL_DISAGREES, stage=errors.VALIDATE_REQUEST)
    assert rig.calls == construction_then_clean_up(rig)
    assert len(rig.contexts) == 1


# ---------------------------------------------------------------------------
# Windows, and what a completed result must say about them.
# ---------------------------------------------------------------------------

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
    return rig.destination / f"{record.kind}-{record.checksum.removeprefix('sha256:')}"


def rows_of(rig: Rig, record: ArtifactRecord) -> list[dict]:
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
    kinds = {one.kind: one for one in result.artifacts}
    for kind, field in (("scores", "score_rows"), ("embeddings", "embedding_rows")):
        per_recording = {(row.namespace, row.value): getattr(row, field) for row in result.coverage}
        if kind not in kinds:
            assert set(per_recording.values()) == {0}
            continue
        rows = rows_of(rig, kinds[kind])
        assert kinds[kind].rows == len(rows) == sum(per_recording.values())
        # Every row names a recording of this work, as many times as it produced rows.
        named = Counter((row["recording_namespace"], row["recording_value"]) for row in rows)
        assert {key: count for key, count in per_recording.items() if count} == named
    assert (result.resolved_scores_request is not None) == ("scores" in kinds)
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
    assert result.recipe == build_recipe()
    assert result.window_geometry == build_recipe().audio.geometry
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


def test_the_model_is_built_from_the_verified_files_and_the_work(rig):
    work = rig.work(outputs=(scores_request(), embeddings_request()))
    lines: list[str] = []

    rig.run(work, rig.model(script(1, embedding=True)), log=lines.append)

    (context,) = rig.contexts
    assert context.card == CARD
    assert context.registry is not None
    assert context.registry.fingerprint == rig.build.files[REGISTRY_ROLE].digest
    assert dict(context.files) == {role: rig.build.model_path(role) for role in rig.build.files}
    assert context.settings == work.settings
    assert context.emit_embeddings is True
    assert context.log == lines.append
    assert not context.scratch_dir.exists()


# ---------------------------------------------------------------------------
# What completion evidence requires at its edges.
# ---------------------------------------------------------------------------


def test_a_requested_kind_with_no_rows_still_gets_an_artifact(rig):
    # Also shows that an accepted window carrying no scores counts as completed.
    floor = 0.5
    request = ScoresRequest(
        contract_id="robin.scores.arrow/1", retention="thresholded", min_score=floor
    )
    capabilities = build_capabilities(
        supported_retention=frozenset({"thresholded"}), native_score_floor=floor
    )
    work = rig.work(outputs=(request,))

    result = success_of(
        rig, rig.run(work, rig.model(script(3, scores=False), capabilities=capabilities)), work
    )

    assert artifact(result, "scores").rows == 0
    assert rows_of(rig, artifact(result, "scores")) == []
    assert result.coverage[0].windows_completed == 3


def test_a_recording_too_short_for_a_dropping_geometry_completes_with_its_reason(rig):
    recording = rig.build.recording("0", duration_seconds=2.0)
    work = rig.work(recordings=(recording,))

    result = success_of(rig, rig.run(work, rig.model(recipe=build_recipe(pad="drop"))), work)

    (row,) = result.coverage
    assert (row.windows_completed, row.zero_window_reason) == (0, "shorter_than_window")
    assert artifact(result, "scores").rows == 0


def test_zero_windows_from_a_recording_of_unknown_duration_are_unexplained(rig):
    work = rig.work()

    result = rig.run(work, rig.model(recipe=build_recipe(pad="drop")))

    failure = failure_of(result, work, code=errors.UNEXPLAINED_ZERO_WINDOWS, stage=errors.INFER)
    assert (failure.namespace, failure.value) == ("test", "0")


def test_zero_windows_under_a_geometry_that_floors_at_one_are_unexplained(rig):
    recording = rig.build.recording("0", duration_seconds=2.0)
    work = rig.work(recordings=(recording,))

    result = rig.run(work, rig.model(recipe=build_recipe(pad="centre_crop_end_pad")))

    failure_of(result, work, code=errors.UNEXPLAINED_ZERO_WINDOWS, stage=errors.INFER)


# ---------------------------------------------------------------------------
# Failures once inference has started.
# ---------------------------------------------------------------------------


def assert_failed_cleanly(rig: Rig, result: InferenceFailure) -> None:
    """clean_up ran once, every model file was released, and no artifact is named."""
    assert rig.calls.count(("clean_up",)) == 1
    assert rig.released_model_files() == rig.model_files.returned
    text = canonical_json_bytes(result).decode()
    published_files = list(rig.destination.iterdir()) if rig.destination.exists() else []
    assert rig.destination.as_uri() not in text
    assert not any(path.name.split("-", 1)[1] in text for path in published_files)


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


def test_a_vector_too_large_for_its_storage_width_fails_the_work(rig):
    work = rig.work(outputs=(embeddings_request(),))
    recipe = build_recipe().model_copy(update={"dtype": "float16"})
    huge = WindowOutput(start=0.0, end=3.0, embedding=np.full(DIM, 1e6, dtype=np.float32))

    result = rig.run(work, rig.model([[huge]], recipe=recipe))

    failure = failure_of(
        result,
        work,
        code=errors.EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE,
        stage=errors.WRITE_ARTIFACT,
    )
    assert (failure.namespace, failure.value) == ("test", "0")
    assert_failed_cleanly(rig, result)


def test_a_writer_that_fails_on_the_second_create_fails_the_work(rig):
    work = rig.work(outputs=(scores_request(), embeddings_request()))
    writer = CopyingWriter(rig.destination, rig.calls, fail_on=2)

    result = rig.run(work, rig.model(script(2, embedding=True)), artifacts=writer)

    failure = failure_of(
        result, work, code=errors.ARTIFACT_PUBLICATION_FAILED, stage=errors.WRITE_ARTIFACT
    )
    assert "OSError" in failure.detail
    assert (failure.namespace, failure.value) == (None, None)
    # The first artifact's bytes were published, and the result still names none.
    assert [path.name.split("-")[0] for path in rig.destination.iterdir()] == ["scores"]
    assert_failed_cleanly(rig, result)


def test_audio_the_port_cannot_fetch_fails_the_work_at_acquisition(rig):
    work = rig.work()
    audio = rig.files("audio", {})

    result = rig.run(work, rig.model(script(1)), audio=audio)

    failure = failure_of(result, work, code=errors.AUDIO_UNAVAILABLE, stage=errors.ACQUIRE_AUDIO)
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
        ("create", "scores"),
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
    "audio_release": (
        {},
        {"audio": {"fail_release": OSError("disk busy")}},
        errors.AUDIO_UNAVAILABLE,
        errors.ACQUIRE_AUDIO,
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
    audio = rig.files("audio", rig.build.audio_paths, **port_kwargs.get("audio", {}))

    result = rig.run(work, rig.model(script(1), **model_kwargs), audio=audio)

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
    "audio_unavailable": lambda rig: (rig.model(script(1)), {"audio": rig.files("audio", {})}),
    "refused_instance": lambda rig: (
        rig.model(
            capabilities=build_capabilities(
                emits_scores=False, score_domain=None, supported_retention=frozenset()
            )
        ),
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
