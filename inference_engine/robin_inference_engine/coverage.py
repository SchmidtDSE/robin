"""What the engine finished, counted as it happens and checked before it is returned."""

from collections.abc import Sequence

from robin_contracts.results import (
    ArtifactKind,
    InferenceSuccess,
    RecordingCoverage,
    ZeroWindowReason,
)
from robin_contracts.work import InferenceWork, work_digest
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow


class CoverageBuilder:
    """Accumulates one coverage row per recording, from the windows that were accepted.

    It takes the accepted window rather than the adapter's, so what it counts is what
    the acceptance boundary let through. A recording's row is built when the recording
    ends and is immutable from then on.
    """

    def __init__(self, recording_indices: Sequence[int]) -> None:
        self._expected = set(recording_indices)
        self._begun: set[int] = set()
        self._rows: dict[int, RecordingCoverage] = {}
        self._open_recording: int | None = None
        self._windows_completed = 0
        self._score_rows = 0
        self._embedding_rows = 0
        self._first_start: float | None = None
        self._greatest_end: float | None = None

    def begin_recording(self, recording_index: int) -> None:
        """Open the recording whose accepted windows arrive next."""
        if recording_index not in self._expected:
            raise RuntimeError(f"recording {recording_index} is not one of this work's")
        if recording_index in self._begun:
            raise RuntimeError(f"recording {recording_index} has already been counted")
        if self._open_recording is not None:
            raise RuntimeError(
                f"cannot begin recording {recording_index} while recording "
                f"{self._open_recording} is still open"
            )
        self._begun.add(recording_index)
        self._open_recording = recording_index
        self._windows_completed = 0
        self._score_rows = 0
        self._embedding_rows = 0
        self._first_start = None
        self._greatest_end = None

    def record(self, window: AcceptedWindow) -> None:
        """Count one accepted window, whether or not it carried anything."""
        if self._open_recording is None:
            raise RuntimeError("no recording is open, so no window can be counted")
        if window.recording_index != self._open_recording:
            raise RuntimeError(
                f"window names recording {window.recording_index} while recording "
                f"{self._open_recording} is open"
            )
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
                f"recording {self._open_recording} completed {self._windows_completed} windows "
                f"but supplied zero-window reason {zero_window_reason!r}"
            )
        if not self._windows_completed and zero_window_reason is None:
            raise errors.EngineError(
                errors.UNEXPLAINED_ZERO_WINDOWS,
                errors.INFER,
                f"recording {self._open_recording} completed no window and nothing declares why",
                recording_index=self._open_recording,
            )
        self._rows[self._open_recording] = RecordingCoverage(
            recording_index=self._open_recording,
            windows_completed=self._windows_completed,
            score_rows=self._score_rows,
            embedding_rows=self._embedding_rows,
            first_window_start_s=self._first_start,
            last_window_end_s=self._greatest_end,
            zero_window_reason=zero_window_reason,
        )
        self._open_recording = None

    def build(self) -> tuple[RecordingCoverage, ...]:
        """Every recording's row, in ascending index order whatever order they ran in."""
        if self._open_recording is not None:
            raise RuntimeError(f"recording {self._open_recording} was begun and never ended")
        missing = self._expected - set(self._rows)
        if missing:
            raise RuntimeError(f"recordings {sorted(missing)} were never counted")
        return tuple(self._rows[index] for index in sorted(self._rows))


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
    expected = sorted(recording.index for recording in work.recordings)
    covered = sorted(row.recording_index for row in success.coverage)
    if covered != expected:
        raise RuntimeError(f"coverage names recordings {covered}, this work's are {expected}")


def _check_requested_artifacts(
    work: InferenceWork, success: InferenceSuccess
) -> None:
    requested: set[ArtifactKind] = {output.kind for output in work.outputs}
    # The map resolves every recording_index, so a work requires it rather than asks.
    requested.add("recording_map")
    written = {artifact.kind for artifact in success.artifacts}
    written.add(success.recording_map.kind)
    if written != requested:
        raise RuntimeError(
            f"result names {sorted(written)}, this work requires {sorted(requested)}"
        )
