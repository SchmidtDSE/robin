"""Typed engine failures. A caller decides what to do from the code, not the message."""

# ---------------------------------------------------------------------------
# accept_window: what the engine refuses from an adapter's output.
# ---------------------------------------------------------------------------

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

ACCEPT_WINDOW_FAILURES = (
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


# ---------------------------------------------------------------------------
# validate_request: what the engine refuses to attempt at all.
# ---------------------------------------------------------------------------

VALIDATE_REQUEST = "validate_request"

SCORES_NOT_EMITTED = "scores_not_emitted"
EMBEDDINGS_NOT_EMITTED = "embeddings_not_emitted"
RETENTION_UNSUPPORTED = "retention_unsupported"
FULL_RETENTION_REDUCED = "full_retention_reduced"
MIN_SCORE_OUT_OF_DOMAIN = "min_score_out_of_domain"
EMBEDDING_EMISSION_DISAGREES = "embedding_emission_disagrees"
EMBEDDING_DIM_DISAGREES = "embedding_dim_disagrees"
REGISTRY_REQUIRED = "registry_required"
REGISTRY_FINGERPRINT_MISMATCH = "registry_fingerprint_mismatch"
HEAD_CLASS_NOT_IN_REGISTRY = "head_class_not_in_registry"

VALIDATE_REQUEST_FAILURES = (
    SCORES_NOT_EMITTED,
    EMBEDDINGS_NOT_EMITTED,
    RETENTION_UNSUPPORTED,
    FULL_RETENTION_REDUCED,
    MIN_SCORE_OUT_OF_DOMAIN,
    EMBEDDING_EMISSION_DISAGREES,
    EMBEDDING_DIM_DISAGREES,
    REGISTRY_REQUIRED,
    REGISTRY_FINGERPRINT_MISMATCH,
    HEAD_CLASS_NOT_IN_REGISTRY,
)


# ---------------------------------------------------------------------------
# infer: what the engine refuses to call a completed recording.
# ---------------------------------------------------------------------------

INFER = "infer"

UNEXPLAINED_ZERO_WINDOWS = "unexplained_zero_windows"

INFER_FAILURES = (UNEXPLAINED_ZERO_WINDOWS,)


# ---------------------------------------------------------------------------
# The failure itself, raised by every stage.
# ---------------------------------------------------------------------------


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
