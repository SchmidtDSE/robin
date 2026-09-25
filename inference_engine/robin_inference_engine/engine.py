"""Runs one work and returns its result.

Every expected failure is returned as an `InferenceFailure`. An exception from code the
engine does not own (a port, the registry loader, the model factory, the adapter) is
reported at the stage that called it. A `RuntimeError` from the engine's own components
is a defect in the engine and is not caught.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

from robin_contracts.cards import model_ref
from robin_contracts.inputs import AudioClip
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsContractId,
    EmbeddingsRequest,
    ResultContractId,
    ScoresContractId,
    ScoresRequest,
)
from robin_contracts.protocols import Log, Model, ModelContext, noop
from robin_contracts.records import WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.results import (
    ArtifactContractId,
    ArtifactRecord,
    FailureReport,
    InferenceFailure,
    InferenceResult,
    InferenceSuccess,
    ZeroWindowReason,
)
from robin_contracts.specs import WindowGeometry, window_count
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    RecordingRef,
    work_digest,
)
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptanceBoundary, AcceptedWindow
from robin_inference_engine.artifacts.embeddings import EmbeddingsWriter
from robin_inference_engine.artifacts.metadata import (
    embedding_metadata,
    required_metadata,
    score_metadata,
)
from robin_inference_engine.artifacts.scores import ScoresWriter
from robin_inference_engine.artifacts.staging import StagedArtifact, checksum_file
from robin_inference_engine.construct_model import construct_model
from robin_inference_engine.coverage import CoverageBuilder, check_completion_evidence
from robin_inference_engine.load_registry import load_registry
from robin_inference_engine.ports import ArtifactWriter, FileAcquisition
from robin_inference_engine.validate_request import refuse_instance, refuse_request

RESULT_CONTRACT_ID: ResultContractId = get_args(ResultContractId)[0]
SCORES_CONTRACT_ID: ScoresContractId = get_args(ScoresContractId)[0]
EMBEDDINGS_CONTRACT_ID: EmbeddingsContractId = get_args(EmbeddingsContractId)[0]

# A detail carries a message from code the engine does not own, which can be any size.
DETAIL_LIMIT = 2000


def run_work(
    work: InferenceWork,
    *,
    model_files: FileAcquisition,
    audio: FileAcquisition,
    artifacts: ArtifactWriter,
    log: Log = noop,
) -> InferenceResult:
    """Run every recording in `work` and return what was produced, or why it stopped."""
    digest = work_digest(work)
    try:
        return _run(
            work, digest, model_files=model_files, audio=audio, artifacts=artifacts, log=log
        )
    except errors.EngineError as error:
        return _failure(digest, error)


def _failure(digest: str, error: errors.EngineError) -> InferenceFailure:
    recording = error.recording
    return InferenceFailure(
        schema_version=RESULT_CONTRACT_ID,
        work_digest=digest,
        failure=FailureReport(
            code=error.code,
            stage=error.stage,
            namespace=recording.namespace if recording is not None else None,
            value=recording.value if recording is not None else None,
            window_start_s=error.window_start_s,
            detail=error.detail[:DETAIL_LIMIT],
        ),
    )


def _run(
    work: InferenceWork,
    digest: str,
    *,
    model_files: FileAcquisition,
    audio: FileAcquisition,
    artifacts: ArtifactWriter,
    log: Log,
) -> InferenceResult:
    _refuse_detections(work)
    _refuse_embedding_input(work)
    # One directory holds the model's scratch space and every staged artifact, so
    # removing it on the way out is the whole of the engine's own cleanup.
    with TemporaryDirectory(prefix="robin-work-") as temporary:
        fetched: list[Path] = []
        failed = True
        try:
            result = _run_model(
                work,
                digest,
                fetched=fetched,
                root=Path(temporary),
                model_files=model_files,
                audio=audio,
                artifacts=artifacts,
                log=log,
            )
            failed = False
            return result
        finally:
            # After clean_up, because the adapter may hold the files open until then.
            releases = [_releasing_model_file(model_files, path) for path in fetched]
            _finish(releases, failed=failed, log=log)


def _run_model(
    work: InferenceWork,
    digest: str,
    *,
    fetched: list[Path],
    root: Path,
    model_files: FileAcquisition,
    audio: FileAcquisition,
    artifacts: ArtifactWriter,
    log: Log,
) -> InferenceResult:
    """Fetch and check everything the model needs, build it, run it, and clean it up."""
    files = _fetch_model_files(work, model_files, fetched)
    registry = _load_registry(files)
    refuse_request(work, registry=registry)

    scratch_dir = root / "scratch"
    scratch_dir.mkdir()
    context = ModelContext(
        card=work.model.card,
        registry=registry,
        files=files,
        settings=work.settings,
        scratch_dir=scratch_dir,
        emit_embeddings=_embeddings(work) is not None,
        log=log,
    )
    model = construct_model(ref=model_ref(work.model.card), context=context)
    failed = True
    try:
        refuse_instance(work, capabilities=model.capabilities, recipe=model.recipe)
        staging = root / "staging"
        staging.mkdir()
        result = _infer(
            work,
            digest,
            model=model,
            registry=registry,
            audio=audio,
            artifacts=artifacts,
            staging=staging,
            log=log,
        )
        failed = False
        return result
    finally:
        clean_up = _calling(model.clean_up, "clean_up", errors.MODEL_RUN_FAILED, errors.INFER)
        _finish([clean_up], failed=failed, log=log)


def _infer(
    work: InferenceWork,
    digest: str,
    *,
    model: Model,
    registry: TaxonRegistry | None,
    audio: FileAcquisition,
    artifacts: ArtifactWriter,
    staging: Path,
    log: Log,
) -> InferenceSuccess:
    registry_uri = work.model.files[REGISTRY_ROLE].uri if registry is not None else None
    registry_fingerprint = registry.fingerprint if registry is not None else None
    scores = _scores(work)
    boundary = AcceptanceBoundary(
        geometry=model.recipe.audio.geometry,
        capabilities=model.capabilities,
        recordings=work.recordings,
        registry=registry,
        scores=scores,
        expect_embeddings=_embeddings(work) is not None,
    )
    coverage = CoverageBuilder(work.recordings)

    with _WindowWriters(
        work,
        model=model,
        registry_uri=registry_uri,
        registry_fingerprint=registry_fingerprint,
        staging=staging,
    ) as writers:
        for position, recording in enumerate(work.recordings):
            _run_recording(
                position,
                recording,
                model=model,
                audio=audio,
                boundary=boundary,
                coverage=coverage,
                writers=writers,
                log=log,
            )
        staged = writers.close()

    success = InferenceSuccess(
        schema_version=RESULT_CONTRACT_ID,
        work_digest=digest,
        recipe=model.recipe,
        model=work.model,
        registry_uri=registry_uri,
        registry_fingerprint=registry_fingerprint,
        window_geometry=model.recipe.audio.geometry,
        resolved_scores_request=scores,
        artifacts=tuple(_publish(artifacts, one) for one in staged),
        coverage=coverage.build(),
    )
    check_completion_evidence(work, success)
    return success


def _run_recording(
    position: int,
    recording: RecordingRef,
    *,
    model: Model,
    audio: FileAcquisition,
    boundary: AcceptanceBoundary,
    coverage: CoverageBuilder,
    writers: "_WindowWriters",
    log: Log,
) -> None:
    """Run one recording through the model, then always finish it."""
    boundary.begin_recording(position)
    coverage.begin_recording(position)
    path: Path | None = None
    windows: Iterator[WindowOutput] | None = None
    failed = True
    try:
        path = _fetch_audio(audio, recording)
        windows = _start(model, recording, AudioClip(path=path))
        accepted = 0
        for window in _each(windows, recording):
            kept = boundary.accept(window)
            coverage.record(kept)
            writers.write(kept)
            accepted += 1
        coverage.end_recording(
            zero_window_reason=None
            if accepted
            else _zero_window_reason(recording, model.recipe.audio.geometry)
        )
        log(f"recording {errors.named(recording)}: {accepted} windows accepted")
        failed = False
    finally:
        steps = _recording_cleanup(model, audio, recording, path=path, windows=windows)
        _finish(steps, failed=failed, log=log)


def _recording_cleanup(
    model: Model,
    audio: FileAcquisition,
    recording: RecordingRef,
    *,
    path: Path | None,
    windows: Iterator[WindowOutput] | None,
) -> list[Callable[[], None]]:
    """The model's hook runs before the audio is released: it may still hold the file."""
    steps = []
    # Close the run first: the adapter may keep files open until it is closed.
    close = getattr(windows, "close", None)
    if close is not None:
        steps.append(
            _calling(close, "closing the run", errors.MODEL_RUN_FAILED, errors.INFER, recording)
        )
    steps.append(
        _calling(
            model.after_recording,
            "after_recording",
            errors.MODEL_RUN_FAILED,
            errors.INFER,
            recording,
        )
    )
    if path is not None:
        steps.append(
            _calling(
                lambda: audio.release(path),
                "release",
                errors.AUDIO_UNAVAILABLE,
                errors.ACQUIRE_AUDIO,
                recording,
            )
        )
    return steps


def _zero_window_reason(
    recording: RecordingRef, geometry: WindowGeometry
) -> ZeroWindowReason | None:
    """`shorter_than_window` when the recording's known duration fits no window.

    Returns None when the duration is unknown, even if the model yielded no windows.
    """
    duration = recording.duration_seconds
    if duration is not None and window_count(duration, geometry) == 0:
        return "shorter_than_window"
    return None


def _scores(work: InferenceWork) -> ScoresRequest | None:
    return next((one for one in work.outputs if isinstance(one, ScoresRequest)), None)


def _embeddings(work: InferenceWork) -> EmbeddingsRequest | None:
    return next((one for one in work.outputs if isinstance(one, EmbeddingsRequest)), None)


# ---------------------------------------------------------------------------
# The two window artifacts, written as windows are accepted.
# ---------------------------------------------------------------------------


class _WindowWriters:
    """The writers this work requested, each holding an open file in staging."""

    def __init__(
        self,
        work: InferenceWork,
        *,
        model: Model,
        registry_uri: str | None,
        registry_fingerprint: str | None,
        staging: Path,
    ) -> None:
        def shared(contract_id: ArtifactContractId) -> dict[bytes, bytes]:
            return required_metadata(
                contract_id=contract_id,
                work=work,
                recipe=model.recipe,
                registry_uri=registry_uri,
                registry_fingerprint=registry_fingerprint,
            )

        capabilities = model.capabilities
        self._writers: list[ScoresWriter | EmbeddingsWriter] = []
        try:
            scores = _scores(work)
            if scores is not None:
                self._writers.append(
                    ScoresWriter(
                        staging / "scores.arrow",
                        metadata=shared(SCORES_CONTRACT_ID)
                        | score_metadata(scores, score_domain=capabilities.score_domain),
                    )
                )
            if _embeddings(work) is not None:
                # Stored at the recipe's width: it is inside the recipe fingerprint,
                # and a request naming a different one was refused before inference.
                storage_dtype = model.recipe.dtype
                self._writers.append(
                    EmbeddingsWriter(
                        staging / "embeddings.arrow",
                        dim=capabilities.embedding_dim,
                        storage_dtype=storage_dtype,
                        metadata=shared(EMBEDDINGS_CONTRACT_ID)
                        | embedding_metadata(
                            work,
                            dim=capabilities.embedding_dim,
                            source_dtype=capabilities.embedding_dtype,
                            storage_dtype=storage_dtype,
                        ),
                    )
                )
        except BaseException:
            self._release()
            raise

    def write(self, window: AcceptedWindow) -> None:
        for writer in self._writers:
            writer.write(window)

    def close(self) -> tuple[StagedArtifact, ...]:
        return tuple(writer.close() for writer in self._writers)

    def __enter__(self) -> "_WindowWriters":
        return self

    def __exit__(self, *exc: object) -> None:
        self._release()

    def _release(self) -> None:
        for writer in self._writers:
            writer.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Calls to the audio port, the writer and the adapter, reported at their stage.
# ---------------------------------------------------------------------------


def _publish(artifacts: ArtifactWriter, staged: StagedArtifact) -> ArtifactRecord:
    try:
        return artifacts.create(
            kind=staged.kind,
            contract_id=staged.contract_id,
            source=staged.path,
            checksum=staged.checksum,
            rows=staged.rows,
        )
    except Exception as exc:
        raise errors.EngineError(
            errors.ARTIFACT_PUBLICATION_FAILED,
            errors.WRITE_ARTIFACT,
            f"publishing the {staged.kind} artifact raised {type(exc).__name__}: {exc}",
        ) from exc


def _fetch_audio(audio: FileAcquisition, recording: RecordingRef) -> Path:
    try:
        return audio.fetch(recording.audio_uri)
    except Exception as exc:
        raise errors.EngineError(
            errors.AUDIO_UNAVAILABLE,
            errors.ACQUIRE_AUDIO,
            f"fetching {recording.audio_uri} raised {type(exc).__name__}: {exc}",
            recording=recording,
        ) from exc


def _start(model: Model, recording: RecordingRef, clip: AudioClip) -> Iterator[WindowOutput]:
    try:
        return iter(model.run(clip))
    except Exception as exc:
        raise _run_failure(recording, exc) from exc


def _each(windows: Iterator[WindowOutput], recording: RecordingRef) -> Iterator[WindowOutput]:
    """The adapter's windows, with only what the adapter raises reported as its failure."""
    while True:
        try:
            window = next(windows)
        except StopIteration:
            return
        except Exception as exc:
            raise _run_failure(recording, exc) from exc
        yield window


def _run_failure(recording: RecordingRef, exc: Exception) -> errors.EngineError:
    return errors.EngineError(
        errors.MODEL_RUN_FAILED,
        errors.INFER,
        f"the model raised {type(exc).__name__} on recording {errors.named(recording)}: {exc}",
        recording=recording,
    )


# ---------------------------------------------------------------------------
# Requests this engine refuses.
# ---------------------------------------------------------------------------


def _refuse_detections(work: InferenceWork) -> None:
    if any(isinstance(output, DetectionsRequest) for output in work.outputs):
        raise errors.EngineError(
            errors.DETECTIONS_NOT_AVAILABLE,
            errors.VALIDATE_REQUEST,
            "detections were requested, and this engine does not produce them",
        )


def _refuse_embedding_input(work: InferenceWork) -> None:
    if not isinstance(work.input, AudioInput):
        raise errors.EngineError(
            errors.EMBEDDING_INPUT_NOT_AVAILABLE,
            errors.VALIDATE_REQUEST,
            f"the work's input is an {work.input.kind}, and no model that reads one "
            f"is supported",
        )


# ---------------------------------------------------------------------------
# The model's files and registry, checked against what the work pins.
# ---------------------------------------------------------------------------


def _fetch_model_files(
    work: InferenceWork, model_files: FileAcquisition, fetched: list[Path]
) -> dict[str, Path]:
    """One verified local path per pinned role, fetched in sorted role order.

    Each path is added to `fetched` as soon as it is returned, so the caller releases
    it even when a later check fails.
    """
    paths: dict[str, Path] = {}
    for role in sorted(work.model.files):
        pinned = work.model.files[role]
        try:
            path = model_files.fetch(pinned.uri)
        except Exception as exc:
            raise _model_file_failure(
                errors.MODEL_FILE_UNAVAILABLE,
                f"fetching the {role!r} file {pinned.uri} raised {type(exc).__name__}: {exc}",
            ) from exc
        fetched.append(path)
        _verify_model_file(role, path, pinned)
        paths[role] = path
    return paths


def _verify_model_file(role: str, path: Path, pinned: PinnedFile) -> None:
    # Size first: it is cheap, and it refuses a truncated download without hashing it.
    try:
        if not path.is_file():
            raise _model_file_failure(
                errors.MODEL_FILE_UNAVAILABLE,
                f"the {role!r} file fetched from {pinned.uri} is not a file at {path}",
            )
        size = path.stat().st_size
        if size != pinned.size_bytes:
            raise _model_file_failure(
                errors.MODEL_FILE_DIGEST_MISMATCH,
                f"the work pins the {role!r} file at {pinned.size_bytes} bytes but {path} "
                f"holds {size}",
            )
        actual = checksum_file(path)
    except OSError as exc:
        raise _model_file_failure(
            errors.MODEL_FILE_UNAVAILABLE,
            f"the {role!r} file {path} could not be read: {type(exc).__name__}: {exc}",
        ) from exc
    if actual != pinned.digest:
        raise _model_file_failure(
            errors.MODEL_FILE_DIGEST_MISMATCH,
            f"the work pins the {role!r} file at {pinned.digest} but {path} hashes to "
            f"{actual}",
        )


def _model_file_failure(code: str, detail: str) -> errors.EngineError:
    return errors.EngineError(code, errors.ACQUIRE_MODEL, detail)


def _load_registry(files: dict[str, Path]) -> TaxonRegistry | None:
    path = files.get(REGISTRY_ROLE)
    if path is None:
        return None
    try:
        return load_registry(path)
    except OSError as exc:
        raise errors.EngineError(
            errors.REGISTRY_UNREADABLE,
            errors.LOAD_REGISTRY,
            f"the registry {path} could not be read: {type(exc).__name__}: {exc}",
        ) from exc
    except ValueError as exc:
        raise errors.EngineError(errors.REGISTRY_INVALID, errors.LOAD_REGISTRY, str(exc)) from exc


# ---------------------------------------------------------------------------
# Cleanup steps that all run, even when one of them raises.
# ---------------------------------------------------------------------------


def _releasing_model_file(model_files: FileAcquisition, path: Path) -> Callable[[], None]:
    return _calling(
        lambda: model_files.release(path),
        f"releasing the model file {path}",
        errors.MODEL_FILE_UNAVAILABLE,
        errors.ACQUIRE_MODEL,
    )


def _calling(
    call: Callable[[], object],
    name: str,
    code: str,
    stage: str,
    recording: RecordingRef | None = None,
) -> Callable[[], None]:
    """`call` as a cleanup step, with anything it raises reported under `code`."""

    def step() -> None:
        try:
            call()
        except Exception as exc:
            raise errors.EngineError(
                code, stage, f"{name} raised {type(exc).__name__}: {exc}", recording=recording
            ) from exc

    return step


def _finish(steps: list[Callable[[], None]], *, failed: bool, log: Log) -> None:
    """Run every step, even after one fails.

    After an earlier failure, a failing step is logged and cannot replace the failure
    that is already being reported. Otherwise the first step to fail is raised.
    """
    first: errors.EngineError | None = None
    for step in steps:
        try:
            step()
        except errors.EngineError as error:
            if failed:
                log(f"during cleanup after a failure: {error.detail}")
            elif first is None:
                first = error
    if first is not None:
        raise first
