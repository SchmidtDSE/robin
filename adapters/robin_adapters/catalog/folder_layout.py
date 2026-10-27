"""The `folder-layout` catalog adapter: recordings found by the paths of files in a folder.

The settings are:

- `namespace`: the namespace of each recording's identity.
- `root`: the folder, as an absolute path or a local `file://` URI.
- `pattern`: the path of a recording under `root`, such as
  `site{site}/{date}/{recorder}_{date}_{time}.wav`. Each `{field}` matches one or more
  characters that are not `/`. A field used twice must have the same text both times.
  A file whose path does not match is ignored.
- `identity`: the identity's value, made from the pattern's fields, such as
  `{recorder}_{date}_{time}`. Use fields that do not change when the files move.

The selection maps each field to a text value, a list of text values, or
`{"from": ..., "to": ...}`, a range that includes both ends and compares text. A field
that the selection does not name matches any text.

If a folder under `root` cannot be read, the selection stops with `SourceUnavailable`,
so that a run never gets a list of recordings with some missing.
"""

import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from string import Formatter
from typing import Any

from pydantic import BaseModel, JsonValue, field_validator, model_validator

from robin_contracts.local_files import get_file_uri, get_local_path
from robin_contracts.ports.catalog import (
    CatalogResult,
    InvalidSelection,
    RequestedRecording,
    SourceUnavailable,
    StorableText,
    check_settings,
    is_storable_text,
)

# Each part of a template is a literal text and the field after it, or None at the end.
_Template = list[tuple[str, str | None]]


class _FolderLayoutSettings(BaseModel, frozen=True, extra="forbid"):
    namespace: StorableText
    root: str
    pattern: str
    identity: str

    @field_validator("root")
    @classmethod
    def _check_root(cls, root: str) -> str:
        try:
            get_local_path(root)
        except ValueError:
            raise ValueError("give an absolute path or a local file:// URI") from None
        return root

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, pattern: str) -> str:
        if not pattern or pattern.startswith("/"):
            raise ValueError("give a path under root, with no leading /")
        _parse_template(pattern)
        return pattern

    @field_validator("identity")
    @classmethod
    def _check_identity(cls, identity: str) -> str:
        if not identity:
            raise ValueError("give a template made from the pattern's fields")
        _parse_template(identity)
        return identity

    @model_validator(mode="after")
    def _check_identity_fields(self) -> "_FolderLayoutSettings":
        unknown = _get_fields(_parse_template(self.identity)) - _get_fields(
            _parse_template(self.pattern)
        )
        if unknown:
            raise ValueError(
                f"identity uses {', '.join(sorted(unknown))}, which is not a field of the pattern"
            )
        return self


class _FolderLayoutAdapter:
    def __init__(self, settings: _FolderLayoutSettings) -> None:
        self._settings = settings
        self._folder = get_local_path(settings.root)
        pattern = _parse_template(settings.pattern)
        self._fields = _get_fields(pattern)
        self._identity = _parse_template(settings.identity)
        self._matcher = _get_matcher(pattern)

    def select_recordings(self, selection: Mapping[str, JsonValue]) -> CatalogResult:
        """Returns the recordings under `root` whose fields match `selection`, by path."""
        unknown = set(selection) - self._fields
        if unknown:
            raise InvalidSelection(
                f"The selection names {', '.join(sorted(unknown))}, which is not a field "
                f"of the pattern. Use {', '.join(sorted(self._fields))}."
            )
        tests = {field: _get_test(field, value) for field, value in selection.items()}
        if not self._folder.is_dir():
            raise SourceUnavailable(f"The folder {self._settings.root} cannot be found.")
        listed_at = datetime.now(UTC).isoformat()
        recordings = [
            self._get_recording(path, fields)
            for path, fields in self._get_matches()
            if all(test(fields[field]) for field, test in tests.items())
        ]
        return CatalogResult(
            recordings=tuple(recordings),
            source={
                "root": self._settings.root,
                "pattern": self._settings.pattern,
                "listed_at": listed_at,
            },
        )

    def _get_matches(self) -> list[tuple[str, dict[str, str]]]:
        prefix = str(self._folder).rstrip("/") + "/"
        matches = []
        for path in _list_files(self._folder):
            match = self._matcher.fullmatch(path.removeprefix(prefix))
            if path.startswith(prefix) and match is not None:
                matches.append((path, match.groupdict()))
        return matches

    def _get_recording(self, path: str, fields: dict[str, str]) -> RequestedRecording:
        if not is_storable_text(path):
            raise InvalidSelection(
                f"The file {path!r} has a name that Robin cannot store. Rename it."
            )
        value = "".join(text + (fields[field] if field else "") for text, field in self._identity)
        return RequestedRecording(
            namespace=self._settings.namespace,
            value=value,
            audio_uri=get_file_uri(path),
        )


def build_adapter(settings: Mapping[str, Any]) -> _FolderLayoutAdapter:
    """Returns a `folder-layout` adapter. Raises `InvalidCatalogSettings` for bad settings."""
    return _FolderLayoutAdapter(check_settings(_FolderLayoutSettings, settings))


def _list_files(folder: Path) -> list[str]:
    """Every file under `folder`, sorted. Raises `SourceUnavailable` if a folder cannot be read."""
    problems: list[OSError] = []
    paths = [
        os.path.join(directory, name)
        for directory, _, names in os.walk(folder, onerror=problems.append)
        for name in names
    ]
    if problems:
        raise SourceUnavailable(
            f"The folder {problems[0].filename} cannot be read, so some recordings "
            "would be missing. Make it readable, then try again."
        )
    return sorted(paths)


def _parse_template(template: str) -> _Template:
    try:
        parts = list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"{template!r} is not a valid template: {error}") from None
    for _, field, spec, conversion in parts:
        if field is not None and (not field.isidentifier() or spec or conversion):
            raise ValueError(f"{{{field}}} is not a field; write a name such as {{site}}")
    return [(text, field) for text, field, _, _ in parts]


def _get_fields(template: _Template) -> set[str]:
    return {field for _, field in template if field is not None}


def _get_matcher(pattern: _Template) -> re.Pattern[str]:
    regex = ""
    seen = set()
    for text, field in pattern:
        regex += re.escape(text)
        if field in seen:
            regex += f"(?P={field})"
        elif field is not None:
            regex += f"(?P<{field}>[^/]+)"
            seen.add(field)
    return re.compile(regex)


def _get_test(field: str, value: JsonValue) -> Callable[[str], bool]:
    if isinstance(value, str):
        return lambda text: text == value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return lambda text: text in value
    if (
        isinstance(value, dict)
        and set(value) == {"from", "to"}
        and all(isinstance(end, str) for end in value.values())
    ):
        return lambda text: value["from"] <= text <= value["to"]
    raise InvalidSelection(
        f"The selection for {field} is not valid. Give a text value, a list of text "
        'values, or {"from": ..., "to": ...} with text ends.'
    )
