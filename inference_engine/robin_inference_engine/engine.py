"""Runs one work and returns its result.

A failure before the first recording starts fails the whole work and is returned as an
`InferenceFailure`. After that, a recording that fails is reported in the result's
`failed`, and the work continues with the next one. An exception from code the engine
does not own (a port, the registry loader, the model factory, the adapter) is reported
at the stage that called it. A `RuntimeError` from the engine's own components is a
defect in the engine and is not caught. A cleanup step that fails is logged and does
not change the result.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

from robin_contracts.canonical import checksum_file
from robin_contracts.cards import HeadCard, model_ref
from robin_contracts.inputs import AudioClip, Input
from robin_contracts.output_contracts import ResultContractId
from robin_contracts.ports import ArtifactWriter, FileProvider
from robin_contracts.protocols import Log, Model, ModelContext, noop
from robin_contracts.records import WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.results import (
    ArtifactRecord,
    FailureReport,
    InferenceCompleted,
    InferenceFailure,
    InferenceResult,
    RecordingFailed,
    ZeroWindowReason,
)
from robin_contracts.specs import Recipe, WindowGeometry, recipe, window_count
from robin_contracts.work import (
    REGISTRY_ROLE,
    InferenceWork,
    PinnedFile,
    RecordingRef,
    work_digest,
)
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptanceBoundary
from robin_inference_engine.artifacts.staging import StagedArtifact
from robin_inference_engine.construct_model import construct_head, construct_model
from robin_inference_engine.coverage import CoverageBuilder, check_completion_evidence
from robin_inference_engine.load_registry import load_registry
from robin_inference_engine.recording_outputs import RecordingOutputs
from robin_inference_engine.requested_outputs import embeddings_request, scores_request
from robin_inference_engine.retention import retain_scores
from robin_inference_engine.validate_request import refuse_request
from robin_inference_engine.verify_head_input import read_head_input

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
    inputs: FileProvider
    artifacts: ArtifactWriter
    log: Log


def run_work(
    work: InferenceWork,
    *,
    model_files: FileProvider,
    inputs: FileProvider,
    artifacts: ArtifactWriter,
    log: Log = noop,
) -> InferenceResult:
    """Run every recording in `work` and return what was produced, or why it stopped."""
    run = _Run(
        work=work,
        digest=work_digest(work),
        model_files=model_files,
        inputs=inputs,
        artifacts=artifacts,
        log=log,
    )
    try:
        return _run(run)
    except errors.EngineError as error:
        return _failure(run.digest, error)


def _report(error: errors.EngineError) -> FailureReport:
    return FailureReport(
        code=error.code,
        stage=error.stage,
        window_start_s=error.window_start_s,
        detail=error.detail[:DETAIL_LIMIT],
    )


def _failure(digest: str, error: errors.EngineError) -> InferenceFailure:
    return InferenceFailure(
        schema_version=RESULT_CONTRACT_ID, work_digest=digest, failure=_report(error)
    )


def _recording_failed(recording: RecordingRef, error: errors.EngineError) -> RecordingFailed:
    # The identity is the loop's recording, not the error's: some readers raise without one.
    return RecordingFailed(
        namespace=recording.namespace, value=recording.value, failure=_report(error)
    )


def _run(run: _Run) -> InferenceResult:
    # One directory holds the model's scratch space and every staged artifact, so
    # removing it on the way out is the whole of the engine's own cleanup.
    temporary = TemporaryDirectory(prefix="robin-work-")
    with _cleanup_on_exit(
        lambda: [(f"removing the work folder {temporary.name}", temporary.cleanup)],
        log=run.log,
    ):
        fetched: list[Path] = []
        # After clean_up, because the adapter may hold the files open until then.
        with _cleanup_on_exit(
            lambda: [_releasing_model_file(run.model_files, path) for path in fetched],
            log=run.log,
        ):
            return _run_model(run, fetched=fetched, root=Path(temporary.name))


def _run_model(run: _Run, *, fetched: list[Path], root: Path) -> InferenceResult:
    """Fetch and check everything the model needs, build it, run it, and clean it up."""
    work = run.work
    files = _fetch_model_files(work, run.model_files, fetched)
    registry = _load_registry(files)
    refuse_request(work, registry=registry)
    # A head's windows are its backbone's, so a head work's recipe is its backbone's.
    # The refusals leave a head work only with an input naming that backbone's card.
    card = work.model.card
    stated = (
        recipe(work.input.backbone, work.input.backbone_settings)
        if isinstance(card, HeadCard)
        else recipe(card, work.settings)
    )

    scratch_dir = root / "scratch"
    scratch_dir.mkdir()
    context = ModelContext(
        card=work.model.card,
        registry=registry,
        files=files,
        settings=work.settings,
        resources=work.resources,
        scratch_dir=scratch_dir,
        emit_embeddings=embeddings_request(work) is not None,
        log=run.log,
    )
    model = (
        construct_head(runtime=card.runtime, context=context)
        if isinstance(card, HeadCard)
        else construct_model(ref=model_ref(card), context=context)
    )
    with _cleanup_on_exit(lambda: [("clean_up", model.clean_up)], log=run.log):
        staging = root / "staging"
        staging.mkdir()
        return _infer(run, model=model, recipe=stated, registry=registry, staging=staging)


def _infer(
    run: _Run,
    *,
    model: Model,
    recipe: Recipe,
    registry: TaxonRegistry | None,
    staging: Path,
) -> InferenceCompleted:
    work = run.work
    outputs = RecordingOutputs(work, recipe=recipe, registry=registry, staging=staging)
    boundary = AcceptanceBoundary(
        geometry=recipe.audio.geometry,
        card=work.model.card,
        recordings=work.recordings,
        registry=registry,
        scores=scores_request(work),
        expect_embeddings=embeddings_request(work) is not None,
    )
    coverage = CoverageBuilder(work.recordings)
    records: list[ArtifactRecord] = []
    failed: list[RecordingFailed] = []
    for position, recording in enumerate(work.recordings):
        try:
            records += _run_recording(
                run,
                position,
                recording,
                model=model,
                recipe=recipe,
                boundary=boundary,
                coverage=coverage,
                outputs=outputs,
            )
        except errors.EngineError as error:
            coverage.discard(position)
            failed.append(_recording_failed(recording, error))
            run.log(f"recording {errors.named(recording)} failed at {error.stage}: {error.code}")
    return _completed(
        run,
        recipe=recipe,
        outputs=outputs,
        records=tuple(records),
        coverage=coverage,
        failed=tuple(failed),
    )


def _completed(
    run: _Run,
    *,
    recipe: Recipe,
    outputs: RecordingOutputs,
    records: tuple[ArtifactRecord, ...],
    coverage: CoverageBuilder,
    failed: tuple[RecordingFailed, ...],
) -> InferenceCompleted:
    """The result of a work that ran, checked against the work before it is returned."""
    detections = outputs.detections
    completed = InferenceCompleted(
        schema_version=RESULT_CONTRACT_ID,
        work_digest=run.digest,
        recipe=recipe,
        model=run.work.model,
        registry_uri=outputs.registry_uri,
        registry_fingerprint=outputs.registry_fingerprint,
        window_geometry=recipe.audio.geometry,
        resolved_detection_policy=detections.policy if detections is not None else None,
        resolved_scores_request=scores_request(run.work),
        artifacts=records,
        coverage=coverage.build(),
        failed=failed,
    )
    check_completion_evidence(run.work, completed)
    return completed


def _run_recording(
    run: _Run,
    position: int,
    recording: RecordingRef,
    *,
    model: Model,
    recipe: Recipe,
    boundary: AcceptanceBoundary,
    coverage: CoverageBuilder,
    outputs: RecordingOutputs,
) -> tuple[ArtifactRecord, ...]:
    """Run one recording through the model, publish its files, then clean up after it.

    Each accepted window keeps only the scores the request asks for before it is
    written. Returns the records of the files that were published.
    """
    requested = scores_request(run.work)
    boundary.begin_recording(position)
    coverage.begin_recording(position)
    path: Path | None = None
    windows: Iterator[WindowOutput] | None = None
    with _cleanup_on_exit(
        lambda: _recording_cleanup(model, run.inputs, recording, path=path, windows=windows),
        log=run.log,
    ):
        path = _fetch_input(run.inputs, recording)
        given = _recording_input(run.work, recording, path, boundary, recipe.id)
        with _writing_locally(recording):
            with outputs.open_window_writers(position, recording) as writers:
                windows = _start(model, recording, given)
                accepted = 0
                for window in _each(windows, recording):
                    kept = boundary.accept(window)
                    if requested is not None:
                        kept = replace(kept, scores=retain_scores(kept.scores, requested))
                    coverage.record(kept)
                    writers.write(kept)
                    accepted += 1
                boundary.end_recording()
                staged = writers.finish()
            detected = outputs.stage_detections(staged)
        if detected is not None:
            staged += (detected,)
        coverage.end_recording(
            zero_window_reason=None
            if accepted
            else _zero_window_reason(recording, recipe.audio.geometry),
            detection_rows=detected.rows if detected is not None else 0,
        )
        # Publishing starts only after every file is staged, so a failure while staging
        # publishes none of this recording's files.
        records = tuple(_publish(run.artifacts, one) for one in staged)
        run.log(f"recording {errors.named(recording)}: {accepted} windows accepted")
        return records


def _recording_cleanup(
    model: Model,
    inputs: FileProvider,
    recording: RecordingRef,
    *,
    path: Path | None,
    windows: Iterator[WindowOutput] | None,
) -> list["_CleanupStep"]:
    """The model's hook runs before the input is released: it may still hold the file."""
    name = errors.named(recording)
    steps: list[_CleanupStep] = []
    # Close the run first: the adapter may keep files open until it is closed.
    close = getattr(windows, "close", None)
    if close is not None:
        steps.append((f"closing the run of recording {name}", close))
    steps.append((f"after_recording for recording {name}", model.after_recording))
    if path is not None:
        steps.append((f"releasing the input of recording {name}", lambda: inputs.release(path)))
    return steps


def _recording_input(
    work: InferenceWork,
    recording: RecordingRef,
    path: Path,
    boundary: AcceptanceBoundary,
    recipe_fingerprint: str,
) -> Input:
    """The audio as fetched, or for a head, the recording's embeddings, read and checked.

    A head's windows must then be exactly the rows it is given.
    """
    card = work.model.card
    if not isinstance(card, HeadCard):
        return AudioClip(path=path)
    embeddings = read_head_input(
        path, card=card, recording=recording, recipe_fingerprint=recipe_fingerprint
    )
    boundary.expect_windows(embeddings.starts, embeddings.ends)
    return embeddings


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
# Calls to the input port, the writer and the adapter, reported at their stage.
# ---------------------------------------------------------------------------


@contextmanager
def _writing_locally(recording: RecordingRef) -> Iterator[None]:
    """Report an `OSError` while staging this recording's files as that recording's failure.

    Only the writers can raise one here: the adapter's calls and the input reads already
    turn what they raise into engine failures.
    """
    try:
        yield
    except OSError as exc:
        raise errors.EngineError(
            errors.ARTIFACT_WRITE_FAILED,
            errors.WRITE_ARTIFACT,
            f"writing the files of recording {errors.named(recording)} on local disk raised "
            f"{type(exc).__name__}: {exc}",
            recording=recording,
        ) from exc


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


def _fetch_input(inputs: FileProvider, recording: RecordingRef) -> Path:
    """The recording's embeddings file when it names one, and its audio otherwise."""
    uri = recording.embeddings.uri if recording.embeddings is not None else recording.audio_uri
    try:
        return inputs.fetch(uri)
    except Exception as exc:
        raise errors.EngineError(
            errors.INPUT_UNAVAILABLE,
            errors.ACQUIRE_INPUT,
            f"fetching {uri} raised {type(exc).__name__}: {exc}",
            recording=recording,
        ) from exc


def _start(model: Model, recording: RecordingRef, given: Input) -> Iterator[WindowOutput]:
    try:
        return iter(model.run(given))
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


_CleanupStep = tuple[str, Callable[[], object]]


def _releasing_model_file(model_files: FileProvider, path: Path) -> _CleanupStep:
    return (f"releasing the model file {path}", lambda: model_files.release(path))


@contextmanager
def _cleanup_on_exit(steps: Callable[[], list[_CleanupStep]], *, log: Log) -> Iterator[None]:
    """Run the cleanup `steps()` when the block ends, whether or not it raised.

    `steps` is called at the end, so it sees whatever the block set up.
    """
    try:
        yield
    finally:
        _run_cleanup_steps(steps(), log=log)


def _run_cleanup_steps(steps: list[_CleanupStep], *, log: Log) -> None:
    """Run every step, and log each one that fails.

    A failed step never changes the result: the outputs were checked and checksummed
    before cleanup runs.
    """
    for name, call in steps:
        try:
            call()
        except Exception as exc:
            log(f"cleanup: {name} raised {type(exc).__name__}: {exc}")
