"""Typed engine failures. A caller decides what to do from the code, not the message."""

ACCEPT_WINDOW = "accept_window"

UNKNOWN_RECORDING_INDEX = "unknown_recording_index"
WINDOW_BOUNDS_INVALID = "window_bounds_invalid"
DUPLICATE_WINDOW = "duplicate_window"
WINDOW_OFF_GEOMETRY = "window_off_geometry"
WINDOW_OUTSIDE_RECORDING = "window_outside_recording"
SCORE_OUT_OF_DOMAIN = "score_out_of_domain"
UNKNOWN_LABEL = "unknown_label"
DUPLICATE_LABEL = "duplicate_label"
INCOMPLETE_FULL_SCORES = "incomplete_full_scores"
UNEXPECTED_EMBEDDING = "unexpected_embedding"
MALFORMED_EMBEDDING = "malformed_embedding"
EMBEDDING_NOT_FINITE = "embedding_not_finite"
WINDOW_OUT_OF_ORDER = "window_out_of_order"

ACCEPTANCE_CODES = (
    UNKNOWN_RECORDING_INDEX,
    WINDOW_BOUNDS_INVALID,
    DUPLICATE_WINDOW,
    WINDOW_OFF_GEOMETRY,
    WINDOW_OUTSIDE_RECORDING,
    SCORE_OUT_OF_DOMAIN,
    UNKNOWN_LABEL,
    DUPLICATE_LABEL,
    INCOMPLETE_FULL_SCORES,
    UNEXPECTED_EMBEDDING,
    MALFORMED_EMBEDDING,
    EMBEDDING_NOT_FINITE,
    WINDOW_OUT_OF_ORDER,
)


class EngineError(Exception):
    """A failure the engine reports by code, with where it happened."""

    def __init__(
        self,
        code: str,
        stage: str,
        detail: str,
        *,
        recording_index: int | None = None,
        window_start_s: float | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.stage = stage
        self.detail = detail
        self.recording_index = recording_index
        self.window_start_s = window_start_s
