"""Runs one work and returns its result.

A failure before the first recording starts fails the whole work and is returned as an
`InferenceFailure`. After that, a recording that fails is reported in the result's
`failed`, and the work continues with the next one. An exception from code the engine
does not own (a port, the registry loader, the model factory, the adapter) is reported
at the stage that called it. A `RuntimeError` from the engine's own components is a
defect in the engine and is not caught. A cleanup step that fails is logged and does
not change the result.
"""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal, get_args


from robin_contracts.canonical import checksum_file
from robin_contracts.cards import HeadCard, model_ref
from robin_contracts.inputs import AudioClip, Input
from robin_contracts.output_contracts import ResultContractId
from robin_contracts.ports.storage import ArtifactWriter, FileProvider
from robin_contracts.protocols import Log, Model, ModelContext, discard_log_line
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
from robin_inference_engine.accept_window import AcceptanceBoundary, AcceptedWindow
from robin_inference_engine.artifacts.staging import StagedArtifact
from robin_inference_engine.construct_model import construct_head, construct_model
from robin_inference_engine.coverage import CoverageBuilder, check_completion_evidence
from robin_inference_engine.load_registry import load_registry
from robin_inference_engine.recording_outputs import RecordingOutputs
from robin_inference_engine.requested_outputs import embeddings_request, scores_request
from robin_inference_engine.retention import get_label_rank, retain_scores
from robin_inference_engine.validate_request import refuse_invalid_request
from robin_inference_engine.verify_head_input import read_head_input

RESULT_CONTRACT_ID: ResultContractId = get_args(ResultContractId)[0]

# A detail carries a message from code the engine does not own, which can be any size.
DETAIL_LIMIT = 2000

PREFETCH_THREAD_NAME = "robin-input-prefetch"


@dataclass(frozen=True, slots=True)
class _WorkContext:
    """The work, its digest, the ports and the log, which every step of the work reads.

    It holds only values fixed before the work starts, never anything the work creates.
    """

    work: InferenceWork
    digest: str
    model_files: FileProvider
    inputs: FileProvider
    artifacts: ArtifactWriter
    log: Log


class _Cleanup(ExitStack):
    """Steps that run when the block ends, whether or not it raised, last added first.

    A step that fails is logged and does not stop the others. It never changes the
    result: the outputs were checked and checksummed before cleanup runs.
    """

    def __init__(self, log: Log) -> None:
        super().__init__()
        self._log = log

    def add(self, name: str, call: Callable[[], object]) -> None:
        """Add a step that calls `call` when the block ends. `name` is used in the log."""
        self.callback(self._run_step, name, call)

    def _run_step(self, name: str, call: Callable[[], object]) -> None:
        try:
            call()
        except Exception as exc:
            self._log(f"cleanup: {name} raised {type(exc).__name__}: {exc}")


_Stage = Literal["fetch", "adapter", "engine", "detections", "publish"]


@dataclass(slots=True)
class _StageSeconds:
    """The seconds that one recording spends in each stage the engine times.

    `fetch` is the time that the recording waited for its input. The input is fetched
    while the recording before it runs, so this is often less than the fetch took.
    `adapter` is the time spent inside the adapter's calls. The engine time leaves it out.
    """

    fetch: float = 0.0
    adapter: float = 0.0
    engine: float = 0.0
    detections: float = 0.0
    publish: float = 0.0

    @contextmanager
    def measure(self, stage: _Stage) -> Iterator[None]:
        """Add the seconds that the block takes to `stage`, also when the block raises."""
        started = perf_counter()
        try:
            yield
        finally:
            setattr(self, stage, getattr(self, stage) + perf_counter() - started)

    def get_text(self) -> str:
        """Return the stage times for the log, in seconds with one decimal place."""
        return (
            f"fetch {self.fetch:.1f} s, engine {self.engine:.1f} s, "
            f"detections {self.detections:.1f} s, publish {self.publish:.1f} s"
        )


class _InputPrefetcher:
    """Fetches the recordings' inputs in order, in one background thread.

    Taking a recording's input starts the fetch of the next recording's input. So at most
    one input is fetched ahead of the recording that runs. Only the background thread
    calls `fetch`. The caller releases each input it takes.
    """

    def __init__(
        self, inputs: FileProvider, recordings: Sequence[RecordingRef], executor: ThreadPoolExecutor
    ) -> None:
        self._inputs = inputs
        self._recordings = recordings
        self._executor = executor
        # The input that is fetched next, or is being fetched now, and is not yet taken.
        # The first fetch starts when the first input is taken, so it is timed as a wait.
        self._next_position = 0
        self._next_fetch: Future[Path] | None = None

    def take_input(self, position: int) -> Path:
        """Wait for the input of the recording at `position`, then fetch the next one.

        Raises the recording's `input_unavailable` failure if its fetch failed.
        """
        if position != self._next_position:
            raise RuntimeError(
                f"the engine took the input of recording {position}, but the input of "
                f"recording {self._next_position} is the next one fetched"
            )
        if self._next_fetch is None:
            self._start_fetch(position)
        future = self._next_fetch
        failure = future.exception()  # waits for the fetch to end
        self._start_fetch(position + 1)
        if failure is not None:
            raise failure
        return future.result()

    def _start_fetch(self, position: int) -> None:
        """Start fetching the input at `position`. After the last recording, fetch nothing."""
        self._next_position = position
        self._next_fetch = (
            self._executor.submit(_fetch_input, self._inputs, self._recordings[position])
            if position < len(self._recordings)
            else None
        )

    def release_prefetched_input(self) -> None:
        """Release the input that was fetched for a recording that never ran, if any.

        A fetch that has not started is cancelled. A fetch that has started is waited
        for, and its file is released if it succeeded.
        """
        future = self._next_fetch
        if future is None or future.cancel():
            return
        try:
            path = future.result()
        except errors.EngineError:
            return
        self._inputs.release(path)


def run_work(
    work: InferenceWork,
    *,
    model_files: FileProvider,
    inputs: FileProvider,
    artifacts: ArtifactWriter,
    log: Log = discard_log_line,
) -> InferenceResult:
    """Run every recording in `work` and return what was produced, or why it stopped."""
    context = _WorkContext(
        work=work,
        digest=work_digest(work),
        model_files=model_files,
        inputs=inputs,
        artifacts=artifacts,
        log=log,
    )
    try:
        with _Cleanup(log) as cleanup:
            return _run_work(context, cleanup)
    except errors.EngineError as error:
        return _build_work_failure(context.digest, error)


def _run_work(context: _WorkContext, cleanup: _Cleanup) -> InferenceCompleted:
    """Set up the model and its inputs, then process every recording.

    Each step adds to `cleanup` what must be undone, as soon as there is something to
    undo. The steps run in reverse order: the model is cleaned up before its files are
    released, and the work folder is removed last.
    """
    work = context.work
    folder = _create_work_folder(cleanup)
    files = _fetch_model_files(work, context.model_files, cleanup)
    registry = _load_registry(files)
    refuse_invalid_request(work, registry=registry)
    work_recipe = _get_recipe(work)
    model = _build_model(context, files, registry, scratch_dir=_create_subfolder(folder, "scratch"))
    cleanup.add("clean_up", model.clean_up)
    processor = _RecordingProcessor(
        context,
        model=model,
        recipe=work_recipe,
        registry=registry,
        prefetcher=_start_input_prefetcher(context, cleanup),
        staging=_create_subfolder(folder, "staging"),
    )
    for position, recording in enumerate(work.recordings):
        try:
            processor.process_recording(position, recording)
        except errors.EngineError as error:
            # A recording that fails does not stop the work.
            processor.add_failed_recording(position, recording, error)
    return processor.build_completed_result()


def _create_work_folder(cleanup: _Cleanup) -> Path:
    # One folder holds the model's scratch space and every staged artifact, so
    # removing it on the way out is the whole of the engine's own cleanup.
    temporary = TemporaryDirectory(prefix="robin-work-")
    cleanup.add(f"removing the work folder {temporary.name}", temporary.cleanup)
    return Path(temporary.name)


def _create_subfolder(folder: Path, name: str) -> Path:
    subfolder = folder / name
    subfolder.mkdir()
    return subfolder


def _get_recipe(work: InferenceWork) -> Recipe:
    # A head's windows are its backbone's, so a head work's recipe is its backbone's.
    # The refusals leave a head work only with an input naming that backbone's card.
    card = work.model.card
    if isinstance(card, HeadCard):
        return recipe(work.input.backbone, work.input.backbone_settings)
    return recipe(card, work.settings)


def _build_model(
    context: _WorkContext,
    files: dict[str, Path],
    registry: TaxonRegistry | None,
    *,
    scratch_dir: Path,
) -> Model:
    """Build the work's model or head from its verified files."""
    work = context.work
    card = work.model.card
    model_context = ModelContext(
        card=card,
        registry=registry,
        files=files,
        settings=work.settings,
        resources=work.resources,
        scratch_dir=scratch_dir,
        emit_embeddings=embeddings_request(work) is not None,
        log=context.log,
    )
    if isinstance(card, HeadCard):
        return construct_head(runtime=card.runtime, context=model_context)
    return construct_model(ref=model_ref(card), context=model_context)


def _start_input_prefetcher(context: _WorkContext, cleanup: _Cleanup) -> _InputPrefetcher:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=PREFETCH_THREAD_NAME)
    cleanup.add("stopping the input prefetch thread", executor.shutdown)
    prefetcher = _InputPrefetcher(context.inputs, context.work.recordings, executor)
    cleanup.add("releasing the prefetched input", prefetcher.release_prefetched_input)
    return prefetcher


def _build_failure_report(error: errors.EngineError) -> FailureReport:
    """Return the code, stage, window and detail of `error`.

    The detail is cut to `DETAIL_LIMIT` characters.
    """
    return FailureReport(
        code=error.code,
        stage=error.stage,
        window_start_s=error.window_start_s,
        detail=error.detail[:DETAIL_LIMIT],
    )


def _build_work_failure(digest: str, error: errors.EngineError) -> InferenceFailure:
    """Return the result of a work that failed before its first recording started."""
    return InferenceFailure(
        schema_version=RESULT_CONTRACT_ID, work_digest=digest, failure=_build_failure_report(error)
    )


def _build_recording_failure(recording: RecordingRef, error: errors.EngineError) -> RecordingFailed:
    """Return the entry for one recording that failed, for the result's `failed`."""
    # The identity is the loop's recording, not the error's: some readers raise without one.
    return RecordingFailed(
        namespace=recording.namespace, value=recording.value, failure=_build_failure_report(error)
    )


# ---------------------------------------------------------------------------
# The recordings, one at a time.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StagedWindowFiles:
    """The window files staged for one recording, and the number of windows accepted."""

    files: tuple[StagedArtifact, ...]
    accepted_windows: int


class _RecordingProcessor:
    """Processes a work's recordings one at a time, and builds the work's result.

    It holds what every recording uses: the model, the checks, the coverage, the
    writers and the input prefetcher.
    """

    def __init__(
        self,
        context: _WorkContext,
        *,
        model: Model,
        recipe: Recipe,
        registry: TaxonRegistry | None,
        prefetcher: _InputPrefetcher,
        staging: Path,
    ) -> None:
        work = context.work
        self._context = context
        self._model = model
        self._recipe = recipe
        self._prefetcher = prefetcher
        self._outputs = RecordingOutputs(work, recipe=recipe, registry=registry, staging=staging)
        self._boundary = AcceptanceBoundary(
            geometry=recipe.audio.geometry,
            card=work.model.card,
            recordings=work.recordings,
            registry=registry,
            scores=scores_request(work),
            expect_embeddings=embeddings_request(work) is not None,
        )
        self._coverage = CoverageBuilder(work.recordings)
        self._requested = scores_request(work)
        # `top_k` needs each label's rank. It is computed once for the work.
        self._label_rank = get_label_rank(registry) if registry is not None else None
        self._records: list[ArtifactRecord] = []
        self._failed: list[RecordingFailed] = []

    def add_failed_recording(
        self, position: int, recording: RecordingRef, error: errors.EngineError
    ) -> None:
        """Discard the recording's coverage, and add it to the result's `failed`."""
        self._coverage.discard(position)
        self._failed.append(_build_recording_failure(recording, error))
        self._context.log(
            f"recording {errors.get_recording_name(recording)} failed at "
            f"{error.stage}: {error.code}"
        )

    def build_completed_result(self) -> InferenceCompleted:
        """Return the result of the work, after it is checked against the work."""
        work = self._context.work
        detections = self._outputs.detections
        completed = InferenceCompleted(
            schema_version=RESULT_CONTRACT_ID,
            work_digest=self._context.digest,
            recipe=self._recipe,
            model=work.model,
            registry_uri=self._outputs.registry_uri,
            registry_fingerprint=self._outputs.registry_fingerprint,
            window_geometry=self._recipe.audio.geometry,
            resolved_detection_policy=detections.policy if detections is not None else None,
            resolved_scores_request=self._requested,
            artifacts=tuple(self._records),
            coverage=self._coverage.build(),
            failed=tuple(self._failed),
        )
        check_completion_evidence(work, completed)
        return completed

    def process_recording(self, position: int, recording: RecordingRef) -> None:
        """Fetch the recording, run the model on it, and publish what it produces.

        Raises an `EngineError` if the recording fails. A recording that succeeds logs
        the seconds of each stage.
        """
        self._boundary.begin_recording(position)
        self._coverage.begin_recording(position)
        seconds = _StageSeconds()
        with _Cleanup(self._context.log) as cleanup:
            path = self._take_input(position, recording, cleanup, seconds)
            model_input = self._prepare_input(recording, path)
            with _report_local_write_errors(recording):
                window_files = self._run_model(position, recording, model_input, seconds)
                detections = self._stage_detections(window_files.files, seconds)
            self._end_coverage(recording, window_files.accepted_windows, detections)
            self._records += self._publish_files((*window_files.files, *detections), seconds)
        self._log_recording(recording, window_files.accepted_windows, seconds)

    def _take_input(
        self,
        position: int,
        recording: RecordingRef,
        cleanup: _Cleanup,
        seconds: _StageSeconds,
    ) -> Path:
        """Wait for the recording's input, and add the steps that run after the recording."""
        with seconds.measure("fetch"):
            path = self._prefetcher.take_input(position)
        name = errors.get_recording_name(recording)
        release = partial(self._context.inputs.release, path)
        cleanup.add(f"releasing the input of recording {name}", release)
        # Added last so that it runs first: the model may still hold the input file.
        cleanup.add(f"after_recording for recording {name}", self._model.after_recording)
        return path

    def _prepare_input(self, recording: RecordingRef, path: Path) -> Input:
        """Return the audio as fetched, or for a head, the recording's checked embeddings.

        A head's windows must then be exactly the rows it is given.
        """
        card = self._context.work.model.card
        if not isinstance(card, HeadCard):
            return AudioClip(path=path)
        embeddings = read_head_input(
            path, card=card, recording=recording, recipe_fingerprint=self._recipe.id
        )
        self._boundary.expect_windows(embeddings.starts, embeddings.ends)
        return embeddings

    def _end_coverage(
        self,
        recording: RecordingRef,
        accepted_windows: int,
        detections: tuple[StagedArtifact, ...],
    ) -> None:
        reason = (
            None
            if accepted_windows
            else _get_zero_window_reason(recording, self._recipe.audio.geometry)
        )
        self._coverage.end_recording(
            zero_window_reason=reason, detection_rows=sum(one.rows for one in detections)
        )

    def _publish_files(
        self, files: tuple[StagedArtifact, ...], seconds: _StageSeconds
    ) -> list[ArtifactRecord]:
        # This runs only after every file is staged, so a failure while staging
        # publishes none of the recording's files.
        with seconds.measure("publish"):
            return [_publish(self._context.artifacts, one) for one in files]

    def _log_recording(
        self, recording: RecordingRef, accepted_windows: int, seconds: _StageSeconds
    ) -> None:
        name = errors.get_recording_name(recording)
        self._context.log(
            f"recording {name}: {accepted_windows} windows accepted; {seconds.get_text()}"
        )

    def _run_model(
        self, position: int, recording: RecordingRef, model_input: Input, seconds: _StageSeconds
    ) -> _StagedWindowFiles:
        """Run the model on the recording, and stage the window files of what it accepts."""
        with (
            seconds.measure("engine"),
            self._outputs.open_window_writers(position, recording) as writers,
        ):
            accepted = self._write_windows(recording, model_input, writers.write, seconds)
            self._boundary.end_recording()
            files = writers.finish()
        # The adapter's calls ran inside the engine's block. The engine time leaves them out.
        seconds.engine -= seconds.adapter
        return _StagedWindowFiles(files=files, accepted_windows=accepted)

    def _stage_detections(
        self, window_files: tuple[StagedArtifact, ...], seconds: _StageSeconds
    ) -> tuple[StagedArtifact, ...]:
        """Return the detections file selected from the window files, or none."""
        with seconds.measure("detections"):
            detected = self._outputs.stage_detections(window_files)
        return () if detected is None else (detected,)

    def _write_windows(
        self,
        recording: RecordingRef,
        model_input: Input,
        write: Callable[[AcceptedWindow], object],
        seconds: _StageSeconds,
    ) -> int:
        """Check each of the model's windows, keep the scores asked for, and `write` it.

        Returns the number of windows accepted. The adapter's windows are closed as soon
        as they end or a window fails, because the adapter may hold the input file open
        and keep its threads running until then.
        """
        with _Cleanup(self._context.log) as cleanup:
            windows = _start_adapter_windows(self._model, recording, model_input, seconds)
            close = getattr(windows, "close", None)
            if close is not None:
                name = errors.get_recording_name(recording)
                cleanup.add(f"closing the windows of recording {name}", close)
            accepted = 0
            for window in _iterate_adapter_windows(windows, recording, seconds):
                kept = self._boundary.accept(window)
                if self._requested is not None:
                    scores, indices = retain_scores(
                        kept.scores, kept.score_indices, self._requested, self._label_rank
                    )
                    kept = replace(kept, scores=scores, score_indices=indices)
                self._coverage.count_window(kept)
                write(kept)
                accepted += 1
            return accepted


def _get_zero_window_reason(
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
def _report_local_write_errors(recording: RecordingRef) -> Iterator[None]:
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
            f"writing the files of recording {errors.get_recording_name(recording)} on "
            f"local disk raised {type(exc).__name__}: {exc}",
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
            f"{errors.get_recording_name(staged.recording)} raised {type(exc).__name__}: {exc}",
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


def _start_adapter_windows(
    model: Model, recording: RecordingRef, given: Input, seconds: _StageSeconds
) -> Iterator[WindowOutput]:
    with seconds.measure("adapter"):
        try:
            return iter(model.run(given))
        except Exception as exc:
            raise _build_model_failure(recording, exc) from exc


def _iterate_adapter_windows(
    windows: Iterator[WindowOutput], recording: RecordingRef, seconds: _StageSeconds
) -> Iterator[WindowOutput]:
    """The adapter's windows, with only what the adapter raises reported as its failure.

    The time spent inside the adapter is added to `seconds.adapter`.
    """
    while True:
        with seconds.measure("adapter"):
            try:
                window = next(windows)
            except StopIteration:
                return
            except Exception as exc:
                raise _build_model_failure(recording, exc) from exc
        yield window


def _build_model_failure(recording: RecordingRef, exc: Exception) -> errors.EngineError:
    return errors.EngineError(
        errors.MODEL_RUN_FAILED,
        errors.INFER,
        f"the model raised {type(exc).__name__} on recording "
        f"{errors.get_recording_name(recording)}: {exc}",
        recording=recording,
    )


# ---------------------------------------------------------------------------
# The model's files and registry, checked against what the work pins.
# ---------------------------------------------------------------------------


def _fetch_model_files(
    work: InferenceWork, model_files: FileProvider, cleanup: _Cleanup
) -> dict[str, Path]:
    """Return one verified local path for each pinned role, fetched in sorted role order.

    Each file's release is added to `cleanup` as soon as the file is fetched, so it is
    released even when a later check fails.
    """
    paths: dict[str, Path] = {}
    for role in sorted(work.model.files):
        pinned = work.model.files[role]
        try:
            path = model_files.fetch(pinned.uri)
        except Exception as exc:
            raise _build_model_file_error(
                errors.MODEL_FILE_UNAVAILABLE,
                f"fetching the {role!r} file {pinned.uri} raised {type(exc).__name__}: {exc}",
            ) from exc
        cleanup.add(f"releasing the model file {path}", partial(model_files.release, path))
        _verify_model_file(role, path, pinned)
        paths[role] = path
    return paths


def _verify_model_file(role: str, path: Path, pinned: PinnedFile) -> None:
    # Size first: it is cheap, and it refuses a truncated download without hashing it.
    try:
        if not path.is_file():
            raise _build_model_file_error(
                errors.MODEL_FILE_UNAVAILABLE,
                f"the {role!r} file fetched from {pinned.uri} is not a file at {path}",
            )
        size = path.stat().st_size
        if size != pinned.size_bytes:
            raise _build_model_file_error(
                errors.MODEL_FILE_DIGEST_MISMATCH,
                f"the work pins the {role!r} file at {pinned.size_bytes} bytes but {path} "
                f"holds {size}",
            )
        actual = checksum_file(path)
    except OSError as exc:
        raise _build_model_file_error(
            errors.MODEL_FILE_UNAVAILABLE,
            f"the {role!r} file {path} could not be read: {type(exc).__name__}: {exc}",
        ) from exc
    if actual != pinned.digest:
        raise _build_model_file_error(
            errors.MODEL_FILE_DIGEST_MISMATCH,
            f"the work pins the {role!r} file at {pinned.digest} but {path} hashes to "
            f"{actual}",
        )


def _build_model_file_error(code: str, detail: str) -> errors.EngineError:
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
