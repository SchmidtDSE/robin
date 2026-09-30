"""The one gate every window an adapter yields passes through."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robin_contracts.cards import ModelCard
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.specs import WindowGeometry
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors

# Relative to the hop or window. Covers binary rounding in i * hop, nothing more.
GRID_ALLOWANCE = 1e-9

PROBABILITY_RANGE = (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class AcceptedWindow:
    """A validated window the engine owns outright, including its embedding array.

    `recording` is the recording that was open when the window arrived.
    """

    recording: RecordingRef
    start: float
    end: float
    scores: tuple[ClassScore, ...]
    embedding: np.ndarray | None


class AcceptanceBoundary:
    """Validates every window of one work, one recording at a time, against its card.

    Every check refuses and none repairs. A clamped score, a cast embedding or a sorted
    window stream would each turn a producer defect into a plausible published value.
    The scores are checked as the model's card says it emits them. The request's own
    floor and k are applied after this, so they are not checked here.
    """

    def __init__(
        self,
        *,
        geometry: WindowGeometry,
        card: ModelCard,
        recordings: Sequence[RecordingRef],
        registry: TaxonRegistry | None = None,
        scores: ScoresRequest | None = None,
        expect_embeddings: bool = False,
    ) -> None:
        self._card = card
        self._geometry = geometry
        # At or below the domain minimum the floor excludes nothing, so the model emits
        # every label for every window.
        self._full_stream = card.min_detection_threshold <= PROBABILITY_RANGE[0]
        self._recordings = tuple(recordings)
        self._registry = registry
        self._scores = scores
        self._expect_embeddings = expect_embeddings
        self._embedding_dtype = np.dtype(card.embedding_dtype) if expect_embeddings else None
        self._open_recording: int | None = None
        self._processed: set[int] = set()
        self._last_start: float | None = None

    def begin_recording(self, position: int) -> None:
        """Open the recording at this position in the work, and close the one before it."""
        if not 0 <= position < len(self._recordings):
            raise RuntimeError(f"position {position} is not one of this work's recordings")
        if position in self._processed:
            raise RuntimeError(
                f"recording {errors.named(self._recordings[position])} at position {position} "
                "was already processed"
            )
        self._processed.add(position)
        self._open_recording = position
        self._last_start = None

    def accept(self, window: WindowOutput) -> AcceptedWindow:
        """Validate one window and return the engine's own copy of it."""
        if self._open_recording is None:
            raise RuntimeError("no recording is open, so no window can be accepted")

        self._check_bounds(window)
        self._check_order(window)
        self._check_geometry(window)
        self._check_within_recording(window)
        # A model is never told whether scores are wanted, so producing unrequested
        # ones is not a defect: they are dropped before any score check applies.
        scores = window.scores if self._scores is not None else ()
        self._check_scores(window, scores)
        self._check_embedding(window)

        # Only now, so a refused window leaves nothing for the next one to be
        # compared against and no array the adapter still owns.
        self._last_start = window.start
        return AcceptedWindow(
            recording=self._open(),
            start=window.start,
            end=window.end,
            scores=scores,
            embedding=None if window.embedding is None else window.embedding.copy(),
        )

    def _open(self) -> RecordingRef:
        return self._recordings[self._open_recording]

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
                f"recording {errors.named(self._open())} already produced a window at "
                f"{window.start}",
            )
        raise self._refuse(
            errors.WINDOW_OUT_OF_ORDER,
            window,
            f"window at {window.start} arrived after {self._last_start}",
        )

    def _check_geometry(self, window: WindowOutput) -> None:
        # A start just below a multiple leaves a remainder of nearly a whole hop, so
        # measure to the nearer multiple. Python's %, unlike math.fmod, is never negative.
        hop = self._geometry.hop
        remainder = window.start % hop
        off_grid = min(remainder, hop - remainder)
        if off_grid > GRID_ALLOWANCE * hop:
            raise self._refuse(
                errors.WINDOW_OFF_GEOMETRY,
                window,
                f"start {window.start} is {off_grid} from a multiple of hop {hop}",
            )

        span = window.end - window.start
        declared = self._geometry.window_duration
        if abs(span - declared) > GRID_ALLOWANCE * declared:
            raise self._refuse(
                errors.WINDOW_OFF_GEOMETRY,
                window,
                f"window spans {span}, not the {declared} the recipe declares",
            )

    def _check_within_recording(self, window: WindowOutput) -> None:
        if window.start < 0:
            raise self._refuse(
                errors.WINDOW_OUTSIDE_RECORDING,
                window,
                f"start {window.start} is before the recording",
            )
        duration = self._recordings[self._open_recording].duration_seconds
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

    def _check_scores(self, window: WindowOutput, scores: tuple[ClassScore, ...]) -> None:
        seen: set[str] = set()
        for score in scores:
            self._check_score_value(window, score)
            self._check_label(window, score)
            if score.label in seen:
                raise self._refuse(
                    errors.DUPLICATE_LABEL,
                    window,
                    f"label {score.label!r} appears more than once in this window",
                )
            seen.add(score.label)
            # After the domain check, so a NaN never reaches a comparison that is false.
            self._check_score_floor(window, score)
        self._check_every_label_is_present(window, len(scores))

    def _check_score_value(self, window: WindowOutput, score: ClassScore) -> None:
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
        if score.label not in self._registry.labels:
            raise self._refuse(
                errors.UNKNOWN_LABEL,
                window,
                f"label {score.label!r} is not declared by registry "
                f"{self._registry.fingerprint}",
            )

    def _check_score_floor(self, window: WindowOutput, score: ClassScore) -> None:
        # Exact: the floor is a declared constant the model applied, so a tolerance
        # would only let a genuinely lower score through.
        floor = self._card.min_detection_threshold
        if self._full_stream or score.score >= floor:
            return
        raise self._refuse(
            errors.SCORE_BELOW_FLOOR,
            window,
            f"score {score.score} for {score.label!r} is below the model's own floor "
            f"{floor}, its card's min_detection_threshold",
        )

    def _check_every_label_is_present(self, window: WindowOutput, count: int) -> None:
        if self._scores is None or not self._full_stream:
            return
        # Membership and uniqueness are settled above, so an equal count is an equal
        # set and a several-thousand-label registry needs no second set per window.
        declared = len(self._registry.entries)
        if count != declared:
            raise self._refuse(
                errors.INCOMPLETE_FULL_SCORES,
                window,
                f"this model emits every label, so each of registry "
                f"{self._registry.fingerprint}'s {declared} labels must appear once, "
                f"got {count}",
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

    def _check_embedding_shape(self, window: WindowOutput) -> None:
        """Refuse a mismatch rather than cast, reshape or copy it into shape."""
        embedding = window.embedding
        if embedding.ndim != 1:
            raise self._refuse(
                errors.MALFORMED_EMBEDDING,
                window,
                f"embedding must be 1-D, got shape {embedding.shape}",
            )
        declared = self._embedding_dtype
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
        if embedding.size != self._card.embedding_dim:
            raise self._refuse(
                errors.MALFORMED_EMBEDDING,
                window,
                f"embedding must be {self._card.embedding_dim} values wide, "
                f"got {embedding.size}",
            )

    def _refuse(self, code: str, window: WindowOutput, detail: str) -> errors.EngineError:
        return errors.EngineError(
            code,
            errors.ACCEPT_WINDOW,
            detail,
            recording=self._open(),
            window_start_s=window.start,
        )
