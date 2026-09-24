"""What the engine finished, counted as it happens and checked before it is returned."""

from collections.abc import Sequence

from robin_contracts.results import (
    ArtifactKind,
    InferenceSuccess,
    RecordingCoverage,
    ZeroWindowReason,
)
from robin_contracts.work import InferenceWork, RecordingId, RecordingRef, work_digest
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow


class CoverageBuilder:
    """Accumulates one coverage row per recording, from the windows that were accepted.

    It takes the accepted window rather than the adapter's, so what it counts is what
    the acceptance boundary let through. A recording's row is built when the recording
    ends and is immutable from then on.
    """

    def __init__(self, recordings: Sequence[RecordingRef]) -> None:
        self._recordings = tuple(recordings)
        self._begun: set[int] = set()
        self._rows: dict[int, RecordingCoverage] = {}
        self._open_recording: int | None = None
        self._windows_completed = 0
        self._score_rows = 0
        self._embedding_rows = 0
        self._first_start: float | None = None
        self._greatest_end: float | None = None

    def begin_recording(self, position: int) -> None:
        """Open the recording at this position in the work, whose windows arrive next."""
        if not 0 <= position < len(self._recordings):
            raise RuntimeError(f"position {position} is not one of this work's recordings")
        recording = self._recordings[position].id
        if position in self._begun:
            raise RuntimeError(f"recording {recording} has already been counted")
        if self._open_recording is not None:
            raise RuntimeError(
                f"cannot begin recording {recording} while recording "
                f"{self._open_id()} is still open"
            )
        self._begun.add(position)
        self._open_recording = position
        self._windows_completed = 0
        self._score_rows = 0
        self._embedding_rows = 0
        self._first_start = None
        self._greatest_end = None

    def record(self, window: AcceptedWindow) -> None:
        """Count one accepted window of the open recording, whether or not it carried anything."""
        if self._open_recording is None:
            raise RuntimeError("no recording is open, so no window can be counted")
        self._windows_completed += 1
        self._score_rows += len(window.scores)
        if window.embedding is not None:
            self._embedding_rows += 1
        if self._first_start is None:
            self._first_start = window.start
        # A cropped final window can end short, so the last end is not the greatest.
        self._greatest_end = (
            window.end if self._greatest_end is None else max(self._greatest_end, window.end)
        )

    def end_recording(self, *, zero_window_reason: ZeroWindowReason | None = None) -> None:
        """Close the open recording and build its row, refusing an unexplained zero."""
        if self._open_recording is None:
            raise RuntimeError("no recording is open to end")
        if self._windows_completed and zero_window_reason is not None:
            raise RuntimeError(
                f"recording {self._open_id()} completed {self._windows_completed} windows "
                f"but supplied zero-window reason {zero_window_reason!r}"
            )
        if not self._windows_completed and zero_window_reason is None:
            raise errors.EngineError(
                errors.UNEXPLAINED_ZERO_WINDOWS,
                errors.INFER,
                f"recording {self._open_id()} completed no window and nothing declares why",
                recording=self._open_id(),
            )
        recording = self._recordings[self._open_recording]
        self._rows[self._open_recording] = RecordingCoverage(
            recording=recording.id,
            audio_digest=recording.audio_digest,
            windows_completed=self._windows_completed,
            score_rows=self._score_rows,
            embedding_rows=self._embedding_rows,
            first_window_start_s=self._first_start,
            last_window_end_s=self._greatest_end,
            zero_window_reason=zero_window_reason,
        )
        self._open_recording = None

    def build(self) -> tuple[RecordingCoverage, ...]:
        """Every recording's row, in the work's order whatever order they ran in."""
        if self._open_recording is not None:
            raise RuntimeError(f"recording {self._open_id()} was begun and never ended")
        missing = [
            str(recording.id)
            for position, recording in enumerate(self._recordings)
            if position not in self._rows
        ]
        if missing:
            raise RuntimeError(f"recordings {missing} were never counted")
        return tuple(self._rows[position] for position in range(len(self._recordings)))

    def _open_id(self) -> RecordingId:
        return self._recordings[self._open_recording].id


def check_completion_evidence(work: InferenceWork, success: InferenceSuccess) -> None:
    """Refuse a result that does not answer the work it claims to answer.

    Every condition here is the engine contradicting itself after inference succeeded,
    so each raises rather than returning a code: there is no caller decision to make.
    """
    _check_work_digest(work, success)
    _check_recording_coverage(work, success)
    _check_requested_artifacts(work, success)


def _check_work_digest(work: InferenceWork, success: InferenceSuccess) -> None:
    expected = work_digest(work)
    if success.work_digest != expected:
        raise RuntimeError(
            f"result carries work digest {success.work_digest}, this work's is {expected}"
        )


def _check_recording_coverage(
    work: InferenceWork, success: InferenceSuccess
) -> None:
    expected = [recording.id for recording in work.recordings]
    covered = [row.recording for row in success.coverage]
    if covered != expected:
        raise RuntimeError(
            f"coverage names recordings {[str(one) for one in covered]}, in that order; "
            f"it must name this work's {[str(one) for one in expected]}, in the work's order"
        )
    for recording, row in zip(work.recordings, success.coverage):
        if row.audio_digest != recording.audio_digest:
            raise RuntimeError(
                f"coverage gives recording {recording.id} audio_digest {row.audio_digest}, "
                f"the work gives {recording.audio_digest}"
            )


def _check_requested_artifacts(
    work: InferenceWork, success: InferenceSuccess
) -> None:
    requested: set[ArtifactKind] = {output.kind for output in work.outputs}
    written = {artifact.kind for artifact in success.artifacts}
    if written != requested:
        raise RuntimeError(
            f"result names {sorted(written)}, this work requires {sorted(requested)}"
        )
