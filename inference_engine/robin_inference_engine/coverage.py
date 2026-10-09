"""What the engine finished, counted as it happens and checked before it is returned."""

from collections.abc import Sequence

from robin_contracts.results import (
    ArtifactKind,
    InferenceCompleted,
    RecordingCoverage,
    ZeroWindowReason,
)
from robin_contracts.work import InferenceWork, RecordingRef, work_digest
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow


class CoverageBuilder:
    """Accumulates one coverage row per recording, from the windows that were accepted.

    It takes the accepted window rather than the adapter's, so what it counts is what
    the acceptance boundary let through. A recording's row is built when the recording
    ends. It is fixed from then on, unless the recording fails later and its row is
    discarded.
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
        recording = errors.named(self._recordings[position])
        if position in self._begun:
            raise RuntimeError(f"recording {recording} has already been counted")
        if self._open_recording is not None:
            raise RuntimeError(
                f"cannot begin recording {recording} while recording "
                f"{errors.named(self._open())} is still open"
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

    def end_recording(
        self, *, zero_window_reason: ZeroWindowReason | None = None, detection_rows: int = 0
    ) -> None:
        """Close the open recording and build its row, refusing an unexplained zero.

        `detection_rows` is passed in: detections are counted after the windows, not per window.
        """
        if self._open_recording is None:
            raise RuntimeError("no recording is open to end")
        if self._windows_completed and zero_window_reason is not None:
            raise RuntimeError(
                f"recording {errors.named(self._open())} completed "
                f"{self._windows_completed} windows "
                f"but supplied zero-window reason {zero_window_reason!r}"
            )
        if not self._windows_completed and zero_window_reason is None:
            raise errors.EngineError(
                errors.UNEXPLAINED_ZERO_WINDOWS,
                errors.INFER,
                f"recording {errors.named(self._open())} completed no window and nothing "
                "declares why",
                recording=self._open(),
            )
        recording = self._recordings[self._open_recording]
        self._rows[self._open_recording] = RecordingCoverage(
            namespace=recording.namespace,
            value=recording.value,
            windows_completed=self._windows_completed,
            score_rows=self._score_rows,
            embedding_rows=self._embedding_rows,
            detection_rows=detection_rows,
            first_window_start_s=self._first_start,
            last_window_end_s=self._greatest_end,
            zero_window_reason=zero_window_reason,
        )
        self._open_recording = None

    def discard(self, position: int) -> None:
        """Forget the recording at this position, open or ended: it failed, so it has no row.

        It still counts as begun, so it cannot be begun again.
        """
        if position not in self._begun:
            raise RuntimeError(
                f"recording {errors.named(self._recordings[position])} was never begun"
            )
        if self._open_recording == position:
            self._open_recording = None
        self._rows.pop(position, None)

    def build(self) -> tuple[RecordingCoverage, ...]:
        """The rows of the recordings that ended and were not discarded, in the work's order."""
        if self._open_recording is not None:
            raise RuntimeError(
                f"recording {errors.named(self._open())} was begun and never ended"
            )
        return tuple(self._rows[position] for position in sorted(self._rows))

    def _open(self) -> RecordingRef:
        return self._recordings[self._open_recording]


def check_completion_evidence(work: InferenceWork, completed: InferenceCompleted) -> None:
    """Refuse a result that does not answer the work it claims to answer.

    Every condition here is the engine contradicting itself after inference succeeded,
    so each raises rather than returning a code: there is no caller decision to make.
    """
    _check_work_digest(work, completed)
    _check_partition(work, completed)
    _check_artifacts_were_requested(work, completed)


def _check_work_digest(work: InferenceWork, completed: InferenceCompleted) -> None:
    expected = work_digest(work)
    if completed.work_digest != expected:
        raise RuntimeError(
            f"result carries work digest {completed.work_digest}, this work's is {expected}"
        )


def _check_partition(work: InferenceWork, completed: InferenceCompleted) -> None:
    """Covered and failed recordings together are the work's, each list in the work's order."""
    expected = [(recording.namespace, recording.value) for recording in work.recordings]
    lists = {
        "coverage": [(row.namespace, row.value) for row in completed.coverage],
        "failed": [(one.namespace, one.value) for one in completed.failed],
    }
    named = sorted(lists["coverage"] + lists["failed"])
    if named != sorted(expected):
        raise RuntimeError(
            f"coverage names {lists['coverage']} and failed names {lists['failed']}; "
            f"together they must name this work's {expected}"
        )
    for name, listed in lists.items():
        members = set(listed)
        if listed != [one for one in expected if one in members]:
            raise RuntimeError(
                f"{name} names recordings {listed}, in that order; "
                "it must name them in the work's order"
            )


def _check_artifacts_were_requested(
    work: InferenceWork, completed: InferenceCompleted
) -> None:
    # A requested kind may have no artifacts: a recording with no rows has no file.
    requested: set[ArtifactKind] = {output.kind for output in work.outputs}
    unrequested = {artifact.kind for artifact in completed.artifacts} - requested
    if unrequested:
        raise RuntimeError(
            f"result names {sorted(unrequested)} artifacts, which this work did not "
            f"request; it requests {sorted(requested)}"
        )
