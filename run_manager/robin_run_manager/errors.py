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


class OutcomeUncertain(OperationError):
    """The connection failed during commit. The change may or may not have been made."""

    reason = "outcome_uncertain"
