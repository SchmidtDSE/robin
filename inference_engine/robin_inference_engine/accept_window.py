"""The one gate every window an adapter yields passes through."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from robin_contracts.protocols import ModelCapabilities
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.specs import WindowGeometry
from robin_inference_engine import errors

# BirdNET's adapter rounds library-reported bounds to one decimal, so a bound carries
# up to 0.05 s of rounding error and a duration computed from two of them up to 0.1 s.
START_TOLERANCE_S = 0.05
DURATION_TOLERANCE_S = 0.1

PROBABILITY_RANGE = (0.0, 1.0)

# Resolved through a mapping rather than np.dtype(declared): np.dtype(None) is float64,
# so an instance that declared nothing would otherwise be compared against a dtype
# nobody chose.
EMBEDDING_DTYPES: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}


@dataclass(frozen=True, slots=True)
class AcceptedWindow:
    """A validated window the engine owns outright, including its embedding array."""

    recording_index: int
    start: float
    end: float
    scores: tuple[ClassScore, ...]
    embedding: np.ndarray | None


class AcceptanceBoundary:
    """Validates every window of one work, one recording at a time.

    Every check refuses and none repairs. A clamped score, a cast embedding or a sorted
    window stream would each turn a producer defect into a plausible published value.
    """

    def __init__(
        self,
        *,
        geometry: WindowGeometry,
        capabilities: ModelCapabilities,
        durations: Mapping[int, float | None],
        registry: TaxonRegistry | None = None,
        require_full_scores: bool = False,
        expect_embeddings: bool = False,
    ) -> None:
        if require_full_scores and registry is None:
            raise RuntimeError(
                "full score retention has no registry to compare a window against; "
                "such a request is refused before the engine runs"
            )
        self._geometry = geometry
        self._capabilities = capabilities
        self._durations = durations
        self._registry = registry
        self._require_full_scores = require_full_scores
        self._expect_embeddings = expect_embeddings
        self._open_recording: int | None = None
        self._processed: set[int] = set()
        self._last_start: float | None = None

    def begin_recording(self, recording_index: int) -> None:
        """Open the recording whose windows arrive next, and close the one before it."""
        if recording_index not in self._durations:
            raise RuntimeError(f"recording {recording_index} is not one of this work's")
        if recording_index in self._processed:
            raise RuntimeError(f"recording {recording_index} has already been processed")
        self._processed.add(recording_index)
        self._open_recording = recording_index
        self._last_start = None

    def accept(self, window: WindowOutput) -> AcceptedWindow:
        """Validate one window and return the engine's own copy of it."""
        if self._open_recording is None:
            raise RuntimeError("no recording is open, so no window can be accepted")

        self._check_recording(window)
        self._check_bounds(window)
        self._check_order(window)
        self._check_geometry(window)
        self._check_within_recording(window)
        self._check_scores(window)
        self._check_embedding(window)

        # Only now, so a refused window leaves nothing for the next one to be
        # compared against and no array the adapter still owns.
        self._last_start = window.start
        return AcceptedWindow(
            recording_index=window.recording_index,
            start=window.start,
            end=window.end,
            scores=window.scores,
            embedding=None if window.embedding is None else window.embedding.copy(),
        )

    def _check_recording(self, window: WindowOutput) -> None:
        if window.recording_index == self._open_recording:
            return
        if window.recording_index in self._durations:
            raise self._refuse(
                errors.WINDOW_OUT_OF_ORDER,
                window,
                f"window names recording {window.recording_index} while recording "
                f"{self._open_recording} is open",
            )
        raise self._refuse(
            errors.UNKNOWN_RECORDING_INDEX,
            window,
            f"recording {window.recording_index} is not one of this work's",
        )

    def _check_bounds(self, window: WindowOutput) -> None:
        if not math.isfinite(window.start) or not math.isfinite(window.end):
            raise self._refuse(
                errors.WINDOW_BOUNDS_INVALID,
                window,
                f"window bounds must be finite, got {window.start} to {window.end}",
            )
        if window.end <= window.start:
            raise self._refuse(
                errors.WINDOW_BOUNDS_INVALID,
                window,
                f"window end must be past its start, got {window.start} to {window.end}",
            )

    def _check_order(self, window: WindowOutput) -> None:
        if self._last_start is None or window.start > self._last_start:
            return
        if window.start == self._last_start:
            raise self._refuse(
                errors.DUPLICATE_WINDOW,
                window,
                f"recording {window.recording_index} already produced a window at "
                f"{window.start}",
            )
        raise self._refuse(
            errors.WINDOW_OUT_OF_ORDER,
            window,
            f"window at {window.start} arrived after {self._last_start}",
        )

    def _check_geometry(self, window: WindowOutput) -> None:
        # The remainder alone would accept a start just past a multiple and refuse one
        # just before the next, so the distance to the nearer multiple is what counts.
        # Python's % keeps it non-negative for a positive hop; math.fmod would not, and
        # would slip every negative start under the tolerance.
        remainder = window.start % self._geometry.hop
        off_grid = min(remainder, self._geometry.hop - remainder)
        if off_grid > START_TOLERANCE_S:
            raise self._refuse(
                errors.WINDOW_OFF_GEOMETRY,
                window,
                f"start {window.start} is {off_grid} from a multiple of hop "
                f"{self._geometry.hop}, past the {START_TOLERANCE_S} s tolerance",
            )

        span = window.end - window.start
        if abs(span - self._geometry.window) > DURATION_TOLERANCE_S:
            raise self._refuse(
                errors.WINDOW_OFF_GEOMETRY,
                window,
                f"window spans {span}, not the {self._geometry.window} the recipe "
                f"declares, past the {DURATION_TOLERANCE_S} s tolerance",
            )

    def _check_within_recording(self, window: WindowOutput) -> None:
        if window.start < 0:
            raise self._refuse(
                errors.WINDOW_OUTSIDE_RECORDING,
                window,
                f"start {window.start} is before the recording",
            )
        duration = self._durations[self._open_recording]
        if duration is None:
            return
        if window.start >= duration:
            raise self._refuse(
                errors.WINDOW_OUTSIDE_RECORDING,
                window,
                f"start {window.start} is at or past the recording's {duration} s",
            )
        if self._geometry.pad == "drop" and window.end > duration:
            raise self._refuse(
                errors.WINDOW_OUTSIDE_RECORDING,
                window,
                f"end {window.end} is past the recording's {duration} s "
                "under the drop policy",
            )
        if window.end > duration + self._geometry.window:
            raise self._refuse(
                errors.WINDOW_OUTSIDE_RECORDING,
                window,
                f"end {window.end} is more than one window past the recording's "
                f"{duration} s",
            )

    def _check_scores(self, window: WindowOutput) -> None:
        seen: set[str] = set()
        for score in window.scores:
            self._check_score_value(window, score)
            self._check_label(window, score)
            if score.label in seen:
                raise self._refuse(
                    errors.DUPLICATE_LABEL,
                    window,
                    f"label {score.label!r} appears more than once in this window",
                )
            seen.add(score.label)
        self._check_every_label_is_present(window, len(window.scores))

    def _check_score_value(self, window: WindowOutput, score: ClassScore) -> None:
        if self._capabilities.score_domain != "probability":
            raise self._refuse(
                errors.SCORE_OUT_OF_DOMAIN,
                window,
                f"score {score.score} for {score.label!r} has no domain to be inside; "
                f"this instance declares score_domain "
                f"{self._capabilities.score_domain!r}",
            )
        low, high = PROBABILITY_RANGE
        if not math.isfinite(score.score) or not low <= score.score <= high:
            raise self._refuse(
                errors.SCORE_OUT_OF_DOMAIN,
                window,
                f"score {score.score} for {score.label!r} is outside the probability "
                f"domain [{low}, {high}]",
            )

    def _check_label(self, window: WindowOutput, score: ClassScore) -> None:
        if not score.label:
            raise self._refuse(
                errors.UNKNOWN_LABEL, window, "a score carries an empty label"
            )
        if self._registry is None:
            raise self._refuse(
                errors.UNKNOWN_LABEL,
                window,
                f"label {score.label!r} is undeclared: this model declares no registry",
            )
        if score.label not in self._registry.labels:
            raise self._refuse(
                errors.UNKNOWN_LABEL,
                window,
                f"label {score.label!r} is not declared by registry "
                f"{self._registry.fingerprint}",
            )

    def _check_every_label_is_present(self, window: WindowOutput, count: int) -> None:
        if not self._require_full_scores:
            return
        # Membership and uniqueness are settled above, so an equal count is an equal
        # set and a several-thousand-label registry needs no second set per window.
        declared = len(self._registry.entries)
        if count != declared:
            raise self._refuse(
                errors.INCOMPLETE_FULL_SCORES,
                window,
                f"full retention needs each of registry "
                f"{self._registry.fingerprint}'s {declared} labels once, got {count}",
            )

    def _check_embedding(self, window: WindowOutput) -> None:
        if window.embedding is None:
            return
        self._check_embedding_was_asked_for(window)
        self._check_embedding_shape(window)
        if not np.isfinite(window.embedding).all():
            raise self._refuse(
                errors.EMBEDDING_NOT_FINITE,
                window,
                "embedding carries a value that is not finite",
            )

    def _check_embedding_was_asked_for(self, window: WindowOutput) -> None:
        if not self._expect_embeddings:
            raise self._refuse(
                errors.UNEXPECTED_EMBEDDING,
                window,
                "this work requested no embeddings",
            )
        if not self._capabilities.emits_embeddings:
            raise self._refuse(
                errors.UNEXPECTED_EMBEDDING,
                window,
                "this instance declares that it emits no embeddings",
            )

    def _check_embedding_shape(self, window: WindowOutput) -> None:
        """Refuse a mismatch rather than cast, reshape or copy it into shape."""
        embedding = window.embedding
        if embedding.ndim != 1:
            raise self._refuse(
                errors.MALFORMED_EMBEDDING,
                window,
                f"embedding must be 1-D, got shape {embedding.shape}",
            )
        declared = self._declared_embedding_dtype()
        # Compared as dtypes, not as dtype names: a non-native byte order reports the
        # same name, passes every other check here and narrows through astype, so a
        # name comparison would let it reach storage as bytes in the wrong order.
        if embedding.dtype != declared:
            raise self._refuse(
                errors.MALFORMED_EMBEDDING,
                window,
                f"embedding must be {declared}, got {embedding.dtype}",
            )
        if not embedding.flags["C_CONTIGUOUS"]:
            raise self._refuse(
                errors.MALFORMED_EMBEDDING, window, "embedding must be C-contiguous"
            )
        if embedding.size != self._capabilities.embedding_dim:
            raise self._refuse(
                errors.MALFORMED_EMBEDDING,
                window,
                f"embedding must be {self._capabilities.embedding_dim} values wide, "
                f"got {embedding.size}",
            )

    def _declared_embedding_dtype(self) -> np.dtype:
        # Every instance that emits embeddings without one of these is refused before
        # inference, so one reaching here is an engine defect, not an untrusted input.
        declared = self._capabilities.embedding_dtype
        if declared not in EMBEDDING_DTYPES:
            raise RuntimeError(
                f"an instance emitting embeddings declares embedding_dtype "
                f"{declared!r}; this engine accepts "
                f"{', '.join(EMBEDDING_DTYPES)}"
            )
        return EMBEDDING_DTYPES[declared]

    def _refuse(self, code: str, window: WindowOutput, detail: str) -> errors.EngineError:
        return errors.EngineError(
            code,
            errors.ACCEPT_WINDOW,
            detail,
            recording_index=window.recording_index,
            window_start_s=window.start,
        )
