"""Runs one work and returns its result.

Every expected failure is returned as an `InferenceFailure`. An exception from code the
engine does not own (a port, the registry loader, the model factory, the adapter) is
reported at the stage that called it. A `RuntimeError` from the engine's own components
is a defect in the engine and is not caught.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

from robin_contracts.cards import model_ref
from robin_contracts.inputs import AudioClip
from robin_contracts.output_contracts import ResultContractId
from robin_contracts.ports import ArtifactWriter, FileProvider
from robin_contracts.protocols import Log, Model, ModelContext, noop
from robin_contracts.records import WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.results import (
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
from robin_inference_engine.accept_window import AcceptanceBoundary
from robin_inference_engine.artifacts.staging import StagedArtifact, checksum_file
from robin_inference_engine.construct_model import construct_model
from robin_inference_engine.coverage import CoverageBuilder, check_completion_evidence
from robin_inference_engine.load_registry import load_registry
from robin_inference_engine.recording_outputs import RecordingOutputs
from robin_inference_engine.requested_outputs import embeddings_request, scores_request
from robin_inference_engine.validate_request import refuse_instance, refuse_request

RESULT_CONTRACT_ID: ResultContractId = get_args(ResultContractId)[0]

# A detail carries a message from code the engine does not own, which can be any size.
DETAIL_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class _Run:
    """The work, its digest, the ports and the log, which every step of the run reads.

    It holds only values fixed before the run starts, never anything the run creates.
    """

    work: InferenceWork
    digest: str
    model_files: FileProvider
    audio: FileProvider
    artifacts: ArtifactWriter
    log: Log


def run_work(
    work: InferenceWork,
    *,
    model_files: FileProvider,
    audio: FileProvider,
    artifacts: ArtifactWriter,
    log: Log = noop,
) -> InferenceResult:
    """Run every recording in `work` and return what was produced, or why it stopped."""
    run = _Run(
        work=work,
        digest=work_digest(work),
        model_files=model_files,
        audio=audio,
        artifacts=artifacts,
        log=log,
    )
    try:
        return _run(run)
    except errors.EngineError as error:
        return _failure(run.digest, error)


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


def _run(run: _Run) -> InferenceResult:
    _refuse_embedding_input(run.work)
    # One directory holds the model's scratch space and every staged artifact, so
    # removing it on the way out is the whole of the engine's own cleanup.
    with TemporaryDirectory(prefix="robin-work-") as temporary:
        fetched: list[Path] = []
        # After clean_up, because the adapter may hold the files open until then.
        with _cleanup_on_exit(
            lambda: [_releasing_model_file(run.model_files, path) for path in fetched],
            log=run.log,
        ):
            return _run_model(run, fetched=fetched, root=Path(temporary))


def _run_model(run: _Run, *, fetched: list[Path], root: Path) -> InferenceResult:
    """Fetch and check everything the model needs, build it, run it, and clean it up."""
    work = run.work
    files = _fetch_model_files(work, run.model_files, fetched)
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
        emit_embeddings=embeddings_request(work) is not None,
        log=run.log,
    )
    model = construct_model(ref=model_ref(work.model.card), context=context)
    clean_up = _cleanup_step(model.clean_up, "clean_up", errors.MODEL_RUN_FAILED, errors.INFER)
    with _cleanup_on_exit(lambda: [clean_up], log=run.log):
        refuse_instance(work, capabilities=model.capabilities, recipe=model.recipe)
        staging = root / "staging"
        staging.mkdir()
        return _infer(run, model=model, registry=registry, staging=staging)


def _infer(
    run: _Run, *, model: Model, registry: TaxonRegistry | None, staging: Path
) -> InferenceSuccess:
    work = run.work
    outputs = RecordingOutputs(work, model=model, registry=registry, staging=staging)
    boundary = AcceptanceBoundary(
        geometry=model.recipe.audio.geometry,
        capabilities=model.capabilities,
        recordings=work.recordings,
        registry=registry,
        scores=scores_request(work),
        expect_embeddings=embeddings_request(work) is not None,
    )
    coverage = CoverageBuilder(work.recordings)
    staged: list[StagedArtifact] = []
    for position, recording in enumerate(work.recordings):
        staged += _run_recording(
            run,
            position,
            recording,
            model=model,
            boundary=boundary,
            coverage=coverage,
            outputs=outputs,
        )
    # Publishing waits for the last recording, so a work that fails while running
    # publishes nothing.
    records = tuple(_publish(run.artifacts, one) for one in staged)
    return _success(run, model=model, outputs=outputs, records=records, coverage=coverage)


def _success(
    run: _Run,
    *,
    model: Model,
    outputs: RecordingOutputs,
    records: tuple[ArtifactRecord, ...],
    coverage: CoverageBuilder,
) -> InferenceSuccess:
    """The result of a completed work, checked against the work before it is returned."""
    detections = outputs.detections
    success = InferenceSuccess(
        schema_version=RESULT_CONTRACT_ID,
        work_digest=run.digest,
        recipe=model.recipe,
        model=run.work.model,
        registry_uri=outputs.registry_uri,
        registry_fingerprint=outputs.registry_fingerprint,
        window_geometry=model.recipe.audio.geometry,
        resolved_detection_policy=detections.policy if detections is not None else None,
        resolved_scores_request=scores_request(run.work),
        artifacts=records,
        coverage=coverage.build(),
    )
    check_completion_evidence(run.work, success)
    return success


def _run_recording(
    run: _Run,
    position: int,
    recording: RecordingRef,
    *,
    model: Model,
    boundary: AcceptanceBoundary,
    coverage: CoverageBuilder,
    outputs: RecordingOutputs,
) -> tuple[StagedArtifact, ...]:
    """Run one recording through the model, then always clean up after it.

    Returns the recording's staged files that hold rows.
    """
    boundary.begin_recording(position)
    coverage.begin_recording(position)
    path: Path | None = None
    windows: Iterator[WindowOutput] | None = None
    with _cleanup_on_exit(
        lambda: _recording_cleanup(model, run.audio, recording, path=path, windows=windows),
        log=run.log,
    ):
        path = _fetch_audio(run.audio, recording)
        with outputs.open_window_writers(position, recording) as writers:
            windows = _start(model, recording, AudioClip(path=path))
            accepted = 0
            for window in _each(windows, recording):
                kept = boundary.accept(window)
                coverage.record(kept)
                writers.write(kept)
                accepted += 1
            staged = writers.finish()
        detected = outputs.stage_detections(staged)
        if detected is not None:
            staged += (detected,)
        coverage.end_recording(
            zero_window_reason=None
            if accepted
            else _zero_window_reason(recording, model.recipe.audio.geometry),
            detection_rows=detected.rows if detected is not None else 0,
        )
        run.log(f"recording {errors.named(recording)}: {accepted} windows accepted")
        return staged


def _recording_cleanup(
    model: Model,
    audio: FileProvider,
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
            _cleanup_step(
                close, "closing the run", errors.MODEL_RUN_FAILED, errors.INFER, recording
            )
        )
    steps.append(
        _cleanup_step(
            model.after_recording,
            "after_recording",
            errors.MODEL_RUN_FAILED,
            errors.INFER,
            recording,
        )
    )
    if path is not None:
        steps.append(
            _cleanup_step(
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


# ---------------------------------------------------------------------------
# Calls to the audio port, the writer and the adapter, reported at their stage.
# ---------------------------------------------------------------------------


def _publish(artifacts: ArtifactWriter, staged: StagedArtifact) -> ArtifactRecord:
    try:
        return artifacts.create(
            kind=staged.kind,
            contract_id=staged.contract_id,
            namespace=staged.recording.namespace,
            value=staged.recording.value,
            source=staged.path,
            checksum=staged.checksum,
            rows=staged.rows,
        )
    except Exception as exc:
        raise errors.EngineError(
            errors.ARTIFACT_PUBLICATION_FAILED,
            errors.WRITE_ARTIFACT,
            f"publishing the {staged.kind} artifact of recording "
            f"{errors.named(staged.recording)} raised {type(exc).__name__}: {exc}",
            recording=staged.recording,
        ) from exc


def _fetch_audio(audio: FileProvider, recording: RecordingRef) -> Path:
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
    work: InferenceWork, model_files: FileProvider, fetched: list[Path]
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


def _releasing_model_file(model_files: FileProvider, path: Path) -> Callable[[], None]:
    return _cleanup_step(
        lambda: model_files.release(path),
        f"releasing the model file {path}",
        errors.MODEL_FILE_UNAVAILABLE,
        errors.ACQUIRE_MODEL,
    )


def _cleanup_step(
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


@contextmanager
def _cleanup_on_exit(steps: Callable[[], list[Callable[[], None]]], *, log: Log) -> Iterator[None]:
    """Run the cleanup `steps()` when the block ends, whether or not it raised.

    `steps` is called at the end, so it sees whatever the block set up.
    """
    failed = True
    try:
        yield
        failed = False
    finally:
        _run_cleanup_steps(steps(), failed=failed, log=log)


def _run_cleanup_steps(steps: list[Callable[[], None]], *, failed: bool, log: Log) -> None:
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
