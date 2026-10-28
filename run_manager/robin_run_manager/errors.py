"""Errors that an operation raises. The CLI and the API show them in the same shape."""

from collections.abc import Mapping
from typing import Any

# Reasons are published: Robin may add one, and never renames or removes one.
NOT_AUTHENTICATED = "not_authenticated"
NOT_FOUND = "not_found"
INVALID_REQUEST = "invalid_request"
INVALID_PROFILE = "invalid_profile"
DATABASE_NOT_FOUND = "database_not_found"
SERVER_NOT_VERIFIED = "server_not_verified"
SCHEMA_NOT_CURRENT = "schema_not_current"
MIGRATION_CHANGED = "migration_changed"
MIGRATION_UNKNOWN = "migration_unknown"
MODEL_CONFLICT = "model_conflict"
PRESET_CONFLICT = "preset_conflict"
UNKNOWN_MODEL = "unknown_model"
UNKNOWN_CONFIGURATION = "unknown_configuration"
INVALID_CURSOR = "invalid_cursor"
INVALID_CONFIGURATION = "invalid_configuration"
UNKNOWN_PRESET = "unknown_preset"
UNKNOWN_OUTPUT_VERSION = "unknown_output_version"
OUTPUT_VERSION_NOT_CURRENT = "output_version_not_current"
UNSUPPORTED_LOCATION = "unsupported_location"
INVALID_SELECTION = "invalid_selection"
SOURCE_REFUSED = "source_refused"

REFUSAL_REASONS = (
    NOT_AUTHENTICATED,
    NOT_FOUND,
    INVALID_REQUEST,
    INVALID_PROFILE,
    DATABASE_NOT_FOUND,
    SERVER_NOT_VERIFIED,
    SCHEMA_NOT_CURRENT,
    MIGRATION_CHANGED,
    MIGRATION_UNKNOWN,
    MODEL_CONFLICT,
    PRESET_CONFLICT,
    UNKNOWN_MODEL,
    UNKNOWN_CONFIGURATION,
    INVALID_CURSOR,
    INVALID_CONFIGURATION,
    UNKNOWN_PRESET,
    UNKNOWN_OUTPUT_VERSION,
    OUTPUT_VERSION_NOT_CURRENT,
    UNSUPPORTED_LOCATION,
    INVALID_SELECTION,
    SOURCE_REFUSED,
)


# Why a task failed. Published, like the refusal reasons. A task can also fail with the
# code of the worker's refusal, or the code of the work's failure.
SUBMISSION_REJECTED = "submission_rejected"
NO_RESULT = "no_result"
INVALID_RESULT = "invalid_result"
WORKER_ERROR = "worker_error"

TASK_FAILURE_REASONS = (
    SUBMISSION_REJECTED,
    OUTPUT_VERSION_NOT_CURRENT,
    NO_RESULT,
    INVALID_RESULT,
    WORKER_ERROR,
)


class OperationError(Exception):
    """An operation did not complete. `reason` is a stable code; `message` is for people."""

    reason: str = "internal_error"

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def body(self) -> dict[str, Any]:
        return {"reason": self.reason, "message": self.message, "details": self.details}


class Refused(OperationError):
    """The operation refused the request and changed nothing."""

    def __init__(
        self, reason: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message, details=details)
        self.reason = reason


class Unavailable(OperationError):
    """The database could not be used. Nothing changed, so the request is safe to repeat."""

    reason = "unavailable"


class NotConfigured(OperationError):
    """Something that this computer must supply is missing, such as AWS credentials.

    Nothing changed. The CLI gives it the exit code of a missing profile.
    """

    reason = "not_configured"


class SourceUnavailable(OperationError):
    """The catalog adapter cannot reach its source now. Nothing changed, so it is safe to repeat."""

    reason = "source_unavailable"


class CatalogFailed(OperationError):
    """The catalog adapter failed in a way that its port does not describe.

    The message names the adapter and the error's type only, because the error's text
    can hold a credential.
    """

    reason = "catalog_failed"


class OutcomeUncertain(OperationError):
    """The change may or may not have been made: the connection failed during commit,
    or the compute adapter failed after it may have started a task."""

    reason = "outcome_uncertain"


class ComputeUnavailable(OperationError):
    """The compute adapter failed. The task keeps its state, and a later pass tries again.

    The message names the error's type only, because its text can hold a credential.
    """

    reason = "compute_unavailable"


class StorageUnavailable(OperationError):
    """The task storage failed. The task keeps its state, and a later pass tries again.

    The message names the error's type only, because its text can hold a credential.
    """

    reason = "storage_unavailable"


class WorkFileConflict(OperationError):
    """The work file that Robin builds now is not the one it reserved, or another file
    is at its place. Robin wrote and submitted nothing. This is a defect, which Robin
    cannot repair."""

    reason = "work_file_conflict"
