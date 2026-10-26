"""The catalog port: how a catalog adapter turns a selection into a list of recordings.

The profile's `catalog` section names a factory, such as `build_adapter`. Robin calls it
with the section's other fields, and it returns an object with a `select_recordings`
method. The factory raises `InvalidCatalogSettings` if it cannot use the settings.

`select_recordings` receives the operator's selection, a mapping of JSON values, and
returns `CatalogResult`. It raises:

- `InvalidSelection` if the operator can correct the selection or the source data.
- `SourceRefused` if the source refuses the operator's credentials.
- `SourceUnavailable` if the source cannot be reached now. The operator can try again.

Robin reports any other error by its type only, because its text can hold a credential.
"""

import sys
from collections.abc import Mapping
from typing import Annotated, Protocol

from pydantic import AfterValidator, BaseModel, JsonValue, field_validator

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.work import PositiveSeconds

_MAX_SOURCE_BYTES = 64 * 1024


def is_storable_text(text: str) -> bool:
    """Whether PostgreSQL can store `text`: no NUL character and no lone surrogate."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return "\x00" not in text


def _check_storable_text(value: str) -> str:
    if not value:
        raise ValueError("this field must be non-empty")
    if not is_storable_text(value):
        raise ValueError("this field holds a NUL character or a lone surrogate")
    return value


# Text that PostgreSQL can store, and not empty.
StorableText = Annotated[str, AfterValidator(_check_storable_text)]


class RequestedRecording(BaseModel, frozen=True, extra="forbid"):
    """One recording that a run asks for: its identity, its audio, and maybe its duration."""

    namespace: StorableText
    value: StorableText
    audio_uri: StorableText
    duration_seconds: PositiveSeconds | None = None


def serialize_source(source: Mapping[str, JsonValue]) -> bytes:
    """Returns the canonical JSON of a `CatalogResult.source`.

    Raises `ValueError` if `source` is not JSON, holds an integer outside the range of a
    64-bit float, or is more than 64 KiB.
    """
    _check_float_range(source)
    data = canonical_json_bytes(source)
    if len(data) > _MAX_SOURCE_BYTES:
        raise ValueError(f"the source is {len(data)} bytes; it must be at most 64 KiB")
    return data


def _check_float_range(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _check_float_range(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_float_range(item)
    elif isinstance(value, int) and abs(value) > sys.float_info.max:
        raise ValueError("the source holds an integer outside the range of a 64-bit float")


class CatalogResult(BaseModel, frozen=True, extra="forbid"):
    """What `select_recordings` returns.

    `source` holds facts that only the adapter has, because only the adapter reads the
    source: for example, the hash of the file that it read. Robin stores `source` in the
    run's provenance, beside the adapter name and the selection that Robin records itself.
    """

    recordings: tuple[RequestedRecording, ...]
    source: dict[str, JsonValue] = {}

    @field_validator("source")
    @classmethod
    def _check_source(cls, source: dict[str, JsonValue]) -> dict[str, JsonValue]:
        serialize_source(source)
        return source


class CatalogAdapter(Protocol):
    def select_recordings(self, selection: Mapping[str, JsonValue]) -> CatalogResult: ...


class InvalidCatalogSettings(Exception):
    """The factory cannot use the settings. The message says what to correct."""


class InvalidSelection(Exception):
    """The operator can correct the selection or the source data. The message says how."""


class SourceRefused(Exception):
    """The source refused the operator's credentials."""


class SourceUnavailable(Exception):
    """The source cannot be reached now. Nothing was stored, so the operator can try again."""
