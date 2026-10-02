"""Canonical JSON and versioned digests for public SoundHub contracts."""

import hashlib
import json
import math
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from pydantic import BaseModel


_SHA256_V1 = re.compile(r"sha256:v1:[0-9a-f]{64}")
# `sha256:` hashes file bytes, `sha256:v1:` hashes canonical JSON, so a digest of one
# kind never matches one of the other. Strip the label where a path needs bare hex.
_SHA256_BYTES = re.compile(r"sha256:[0-9a-f]{64}")


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented in canonical JSON."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("canonical JSON object keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("canonical JSON does not allow non-finite floats")
        return 0.0 if value == 0 else value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise CanonicalizationError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a public contract value as deterministic UTF-8 JSON."""
    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_v1(value: Any) -> str:
    """Return the versioned SHA-256 digest of canonical JSON."""
    return "sha256:v1:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256_v1(value: str) -> bool:
    """Whether `value` is spelled the way `sha256_v1` spells what it returns."""
    return _SHA256_V1.fullmatch(value) is not None


def is_sha256_bytes(value: str) -> bool:
    """Whether `value` is a `sha256:` digest of a file's bytes."""
    return _SHA256_BYTES.fullmatch(value) is not None
