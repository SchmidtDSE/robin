"""The `csv-list` catalog adapter: recordings from the rows of a CSV file.

The settings are:

- `namespace`: the namespace of each recording's identity.
- `id_column`: the column that holds each recording's identity value.
- `audio_column`: the column that holds each recording's audio.
- `duration_column` (optional): the column that holds each recording's duration in
  seconds. An empty cell means that the duration is not known.

The selection is `{"file": PATH}`, a local path. A relative path is read from the
current folder. The file is UTF-8, with a header row that holds the named columns.
Other columns and empty lines are ignored.

An audio cell is a URI, an absolute path, or a path relative to the CSV file's folder.
A path becomes a `file://` URI. A `file:` URI must be written as Robin writes it.

A problem in the file stops the selection with `InvalidSelection`, which gives the row.
Row 1 is the header, as in a spreadsheet.
"""

import csv
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, JsonValue, ValidationError

from robin_contracts.canonical import checksum_bytes
from robin_contracts.local_files import get_audio_uri
from robin_contracts.ports.catalog import (
    CatalogResult,
    InvalidSelection,
    RequestedRecording,
    StorableText,
    check_settings,
)

# The cell that gives each field of RequestedRecording.
_CELLS = {"value": "id", "audio_uri": "audio", "duration_seconds": "duration"}


class _CSVListSettings(BaseModel, frozen=True, extra="forbid"):
    namespace: StorableText
    id_column: str = Field(min_length=1)
    audio_column: str = Field(min_length=1)
    duration_column: str | None = Field(default=None, min_length=1)


class _RowProblem(Exception):
    pass


class _CSVListAdapter:
    def __init__(self, settings: _CSVListSettings) -> None:
        self._settings = settings

    def select_recordings(self, selection: Mapping[str, JsonValue]) -> CatalogResult:
        """Returns a recording for each row of the CSV file that `selection` names."""
        path = _get_csv_path(selection)
        data = _read_file(path)
        return CatalogResult(
            recordings=self._parse_table(path, _decode(path, data)),
            source={"file": str(path), "file_sha256": checksum_bytes(data)},
        )

    def _parse_table(self, path: Path, text: str) -> tuple[RequestedRecording, ...]:
        rows = csv.reader(io.StringIO(text, newline=""))
        try:
            columns = self._get_columns(next(rows, None))
            return tuple(
                self._parse_row(path.parent, row, columns, number)
                for number, row in enumerate(rows, start=2)
                if row
            )
        except csv.Error as error:
            raise InvalidSelection(f"{path} is not a valid CSV file: {error}.") from None

    def _get_columns(self, header: list[str] | None) -> dict[str, int]:
        if not header:
            raise InvalidSelection("The CSV file has no header row. Row 1 must name the columns.")
        named = {
            "id": self._settings.id_column,
            "audio": self._settings.audio_column,
            "duration": self._settings.duration_column,
        }
        columns = {}
        for role, name in named.items():
            if name is None:
                continue
            found = [index for index, cell in enumerate(header) if cell == name]
            if len(found) != 1:
                problem = "has no column" if not found else "has more than one column"
                raise InvalidSelection(f"The CSV file's header row {problem} {name!r}.")
            columns[role] = found[0]
        return columns

    def _parse_row(
        self, folder: Path, row: list[str], columns: dict[str, int], number: int
    ) -> RequestedRecording:
        cells = {
            role: row[index] if index < len(row) else "" for role, index in columns.items()
        }
        try:
            for role in ("id", "audio"):
                if not cells[role].strip():
                    raise _RowProblem(f"the {role} cell is empty")
            # RequestedRecording checks the values, so that its rules are not copied here.
            return RequestedRecording(
                namespace=self._settings.namespace,
                value=cells["id"],
                audio_uri=_get_audio_uri(folder, cells["audio"]),
                duration_seconds=_get_duration(cells.get("duration", "")),
            )
        except _RowProblem as error:
            problem = str(error)
        except ValidationError as error:
            problem = _get_cell_problem(error)
        raise InvalidSelection(f"Row {number} of the CSV file: {problem}.")


def build_adapter(settings: Mapping[str, Any]) -> _CSVListAdapter:
    """Returns a `csv-list` adapter. Raises `InvalidCatalogSettings` for bad settings."""
    return _CSVListAdapter(check_settings(_CSVListSettings, settings))


def _get_csv_path(selection: Mapping[str, JsonValue]) -> Path:
    file = selection.get("file")
    if set(selection) != {"file"} or not isinstance(file, str) or not file:
        raise InvalidSelection('The selection must be {"file": PATH}, with the path of a CSV file.')
    return Path(file).absolute()


def _read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        pass
    raise InvalidSelection(f"The CSV file {path} cannot be read.")


def _decode(path: Path, data: bytes) -> str:
    try:
        # A spreadsheet often starts its export with a byte order mark.
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    raise InvalidSelection(f"The CSV file {path} is not UTF-8 text.")


def _get_audio_uri(folder: Path, cell: str) -> str:
    try:
        # A relative path is read from the CSV file's folder.
        return get_audio_uri(cell, folder)
    except ValueError as error:
        problem = str(error)
    raise _RowProblem(f"the audio cell is {problem}")


def _get_duration(cell: str) -> float | None:
    if not cell.strip():
        return None
    try:
        return float(cell)
    except ValueError:
        pass
    raise _RowProblem("the duration cell is not a number of seconds")


def _get_cell_problem(error: ValidationError) -> str:
    problem = error.errors(include_url=False, include_input=False)[0]
    field = str(problem["loc"][0]) if problem["loc"] else ""
    reason = problem.get("ctx", {}).get("error", problem["msg"])
    return f"the {_CELLS.get(field, field)} cell is not valid: {reason}"
