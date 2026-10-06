"""Typed engine failures. A caller decides what to do from the code, not the message."""

from robin_contracts.work import RecordingRef

# ---------------------------------------------------------------------------
# accept_window: what the engine refuses from an adapter's output.
# ---------------------------------------------------------------------------

ACCEPT_WINDOW = "accept_window"

WINDOW_BOUNDS_INVALID = "window_bounds_invalid"
DUPLICATE_WINDOW = "duplicate_window"
WINDOW_OFF_GEOMETRY = "window_off_geometry"
WINDOW_OUTSIDE_RECORDING = "window_outside_recording"
SCORE_OUT_OF_DOMAIN = "score_out_of_domain"
UNKNOWN_LABEL = "unknown_label"
DUPLICATE_LABEL = "duplicate_label"
SCORE_BELOW_FLOOR = "score_below_floor"
INCOMPLETE_FULL_SCORES = "incomplete_full_scores"
UNEXPECTED_EMBEDDING = "unexpected_embedding"
MALFORMED_EMBEDDING = "malformed_embedding"
EMBEDDING_NOT_FINITE = "embedding_not_finite"
WINDOW_OUT_OF_ORDER = "window_out_of_order"
HEAD_WINDOWS_DISAGREE_WITH_INPUT = "head_windows_disagree_with_input"

ACCEPT_WINDOW_FAILURES = (
    WINDOW_BOUNDS_INVALID,
    DUPLICATE_WINDOW,
    WINDOW_OFF_GEOMETRY,
    WINDOW_OUTSIDE_RECORDING,
    SCORE_OUT_OF_DOMAIN,
    UNKNOWN_LABEL,
    DUPLICATE_LABEL,
    SCORE_BELOW_FLOOR,
    INCOMPLETE_FULL_SCORES,
    UNEXPECTED_EMBEDDING,
    MALFORMED_EMBEDDING,
    EMBEDDING_NOT_FINITE,
    WINDOW_OUT_OF_ORDER,
    HEAD_WINDOWS_DISAGREE_WITH_INPUT,
)


# ---------------------------------------------------------------------------
# validate_request: what the engine refuses to attempt at all.
# ---------------------------------------------------------------------------

VALIDATE_REQUEST = "validate_request"

SCORES_NOT_EMITTED = "scores_not_emitted"
EMBEDDINGS_NOT_EMITTED = "embeddings_not_emitted"
FULL_RETENTION_REDUCED = "full_retention_reduced"
SCORE_FLOOR_BELOW_MODEL_FLOOR = "score_floor_below_model_floor"
MIN_SCORE_OUT_OF_DOMAIN = "min_score_out_of_domain"
EMBEDDING_DTYPE_DISAGREES = "embedding_dtype_disagrees"
REGISTRY_REQUIRED = "registry_required"
REGISTRY_FINGERPRINT_MISMATCH = "registry_fingerprint_mismatch"
REGISTRY_DISAGREES_WITH_CARD = "registry_disagrees_with_card"
SETTING_UNDECLARED = "setting_undeclared"
SETTING_TYPE_MISMATCH = "setting_type_mismatch"
WINDOW_OVERLAP_INVALID = "window_overlap_invalid"
INPUT_KIND_DISAGREES_WITH_CARD = "input_kind_disagrees_with_card"
HEAD_BACKBONE_DISAGREES = "head_backbone_disagrees"
HEAD_EMITS_NO_EMBEDDINGS = "head_emits_no_embeddings"

VALIDATE_REQUEST_FAILURES = (
    SCORES_NOT_EMITTED,
    EMBEDDINGS_NOT_EMITTED,
    FULL_RETENTION_REDUCED,
    SCORE_FLOOR_BELOW_MODEL_FLOOR,
    MIN_SCORE_OUT_OF_DOMAIN,
    EMBEDDING_DTYPE_DISAGREES,
    REGISTRY_REQUIRED,
    REGISTRY_FINGERPRINT_MISMATCH,
    REGISTRY_DISAGREES_WITH_CARD,
    SETTING_UNDECLARED,
    SETTING_TYPE_MISMATCH,
    WINDOW_OVERLAP_INVALID,
    INPUT_KIND_DISAGREES_WITH_CARD,
    HEAD_BACKBONE_DISAGREES,
    HEAD_EMITS_NO_EMBEDDINGS,
)


# ---------------------------------------------------------------------------
# acquire_model: a pinned model file the engine could not fetch or does not trust.
# ---------------------------------------------------------------------------

ACQUIRE_MODEL = "acquire_model"

MODEL_FILE_UNAVAILABLE = "model_file_unavailable"
MODEL_FILE_DIGEST_MISMATCH = "model_file_digest_mismatch"

ACQUIRE_MODEL_FAILURES = (MODEL_FILE_UNAVAILABLE, MODEL_FILE_DIGEST_MISMATCH)


# ---------------------------------------------------------------------------
# load_registry: what the engine refuses to read as a label binding.
# ---------------------------------------------------------------------------

LOAD_REGISTRY = "load_registry"

REGISTRY_UNREADABLE = "registry_unreadable"
REGISTRY_INVALID = "registry_invalid"

LOAD_REGISTRY_FAILURES = (REGISTRY_UNREADABLE, REGISTRY_INVALID)


# ---------------------------------------------------------------------------
# construct_model: what the engine refuses to run as a model.
# ---------------------------------------------------------------------------

CONSTRUCT_MODEL = "construct_model"

MODEL_NOT_INSTALLED = "model_not_installed"
MODEL_REGISTERED_TWICE = "model_registered_twice"
MODEL_ENTRY_POINT_UNLOADABLE = "model_entry_point_unloadable"
MODEL_CONSTRUCTION_FAILED = "model_construction_failed"
MODEL_PROTOCOL_UNSATISFIED = "model_protocol_unsatisfied"

CONSTRUCT_MODEL_FAILURES = (
    MODEL_NOT_INSTALLED,
    MODEL_REGISTERED_TWICE,
    MODEL_ENTRY_POINT_UNLOADABLE,
    MODEL_CONSTRUCTION_FAILED,
    MODEL_PROTOCOL_UNSATISFIED,
)


# ---------------------------------------------------------------------------
# acquire_input: fetching a recording's input.
# ---------------------------------------------------------------------------

ACQUIRE_INPUT = "acquire_input"

INPUT_UNAVAILABLE = "input_unavailable"

ACQUIRE_INPUT_FAILURES = (INPUT_UNAVAILABLE,)


# ---------------------------------------------------------------------------
# infer: what the engine refuses to call a completed recording, and a model that
# failed while producing one.
# ---------------------------------------------------------------------------

INFER = "infer"

UNEXPLAINED_ZERO_WINDOWS = "unexplained_zero_windows"
MODEL_RUN_FAILED = "model_run_failed"

INFER_FAILURES = (UNEXPLAINED_ZERO_WINDOWS, MODEL_RUN_FAILED)


# ---------------------------------------------------------------------------
# read_input_artifact: what the engine refuses to believe about stored bytes.
#
# ---------------------------------------------------------------------------

READ_INPUT_ARTIFACT = "read_input_artifact"

ARTIFACT_CHECKSUM_MISMATCH = "artifact_checksum_mismatch"
ARTIFACT_CONTRACT_UNEXPECTED = "artifact_contract_unexpected"
ARTIFACT_SCHEMA_INVALID = "artifact_schema_invalid"
ARTIFACT_METADATA_INCOMPLETE = "artifact_metadata_incomplete"
ARTIFACT_MALFORMED = "artifact_malformed"
ARTIFACT_UNREADABLE = "artifact_unreadable"
HEAD_INPUT_BACKBONE_MISMATCH = "head_input_backbone_mismatch"
HEAD_INPUT_WIDTH_MISMATCH = "head_input_width_mismatch"
HEAD_INPUT_RECIPE_DIFFERS = "head_input_recipe_differs"
HEAD_INPUT_RECORDING_MISMATCH = "head_input_recording_mismatch"

READ_INPUT_ARTIFACT_FAILURES = (
    ARTIFACT_CHECKSUM_MISMATCH,
    ARTIFACT_CONTRACT_UNEXPECTED,
    ARTIFACT_SCHEMA_INVALID,
    ARTIFACT_METADATA_INCOMPLETE,
    ARTIFACT_MALFORMED,
    ARTIFACT_UNREADABLE,
    HEAD_INPUT_BACKBONE_MISMATCH,
    HEAD_INPUT_WIDTH_MISMATCH,
    HEAD_INPUT_RECIPE_DIFFERS,
    HEAD_INPUT_RECORDING_MISMATCH,
)


# ---------------------------------------------------------------------------
# write_artifact: what the engine refuses to put in a file, and a file it could not
# publish.
# ---------------------------------------------------------------------------

WRITE_ARTIFACT = "write_artifact"

EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE = "embedding_value_out_of_storage_dtype_range"
ARTIFACT_PUBLICATION_FAILED = "artifact_publication_failed"

WRITE_ARTIFACT_FAILURES = (
    EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE,
    ARTIFACT_PUBLICATION_FAILED,
)


# ---------------------------------------------------------------------------
# aggregate: a recording's scores the engine could not turn into detections.
# ---------------------------------------------------------------------------

AGGREGATE = "aggregate"

UNMATCHED_LABELS = "unmatched_labels"
AGGREGATION_FAILED = "aggregation_failed"

AGGREGATE_FAILURES = (UNMATCHED_LABELS, AGGREGATION_FAILED)


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
        recording: RecordingRef | None = None,
        window_start_s: float | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.stage = stage
        self.detail = detail
        self.recording = recording
        self.window_start_s = window_start_s


def named(recording: RecordingRef) -> str:
    """A recording's identity as messages spell it."""
    # Quoted as a pair: any separator could also appear inside a value.
    return f"({recording.namespace!r}, {recording.value!r})"
