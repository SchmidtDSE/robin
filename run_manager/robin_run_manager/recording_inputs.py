"""The two list files that `runs submit` reads: `--recordings` and `--uris`.

Both are UTF-8 with one entry on each line. Empty lines are skipped, but a line of only
spaces is refused.
"""

import codecs
import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple, NoReturn
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from robin_contracts.canonical import checksum_bytes
from robin_contracts.ports.catalog import RequestedRecording, is_storable_text
from robin_run_manager.errors import INVALID_REQUEST, Refused
from robin_run_manager.operator_files import field_problems

# The most line problems that one refusal reports.
MAX_PROBLEMS = 20

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_UTF16_MARKS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


class RecordingManifest(NamedTuple):
    """The recordings that a run asks for, and the provenance that the run keeps."""

    recordings: tuple[RequestedRecording, ...]
    provenance: dict[str, Any]


class _LineProblem(Exception):
    """One line cannot be used. The message says what is wrong and what to do."""


def read_recordings_file(path: Path) -> RecordingManifest:
    """Read a `--recordings` file. Each line is one JSON object for one recording."""
    data = _read_bytes(path)
    recordings, problems = [], []
    for number, line in _split_lines(path, data):
        try:
            recordings.append((number, _parse_recording(line)))
        except _LineProblem as problem:
            problems.append({"line": number, "problem": str(problem)})
    return _build_recording_manifest(path, data, recordings, problems, "recordings", {})


def read_uris_file(path: Path, namespace: str) -> RecordingManifest:
    """Read a `--uris` file. Each line is a URI or an absolute path in `namespace`.

    A `file://` line must have the spelling that Robin makes from its path, so that one
    file has one identity.
    """
    if not namespace:
        raise Refused(
            INVALID_REQUEST,
            "The namespace must not be empty. Give the name of the collection that the "
            "recordings belong to.",
            details={"file": str(path)},
        )
    data = _read_bytes(path)
    recordings, problems = [], []
    for number, line in _split_lines(path, data):
        try:
            uri = _parse_uri(line)
        except _LineProblem as problem:
            problems.append({"line": number, "problem": str(problem)})
            continue
        recording = RequestedRecording(namespace=namespace, value=uri, audio_uri=uri)
        recordings.append((number, recording))
    return _build_recording_manifest(
        path, data, recordings, problems, "uris", {"namespace": namespace}
    )


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        problem = error.strerror or "it cannot be read"
    raise Refused(
        INVALID_REQUEST,
        f"{path} cannot be read: {problem}.",
        details={"file": str(path), "problems": [], "truncated": False},
    )


def _split_lines(path: Path, data: bytes) -> list[tuple[int, str]]:
    """The lines that are not empty, with their numbers from 1."""
    if data.startswith(codecs.BOM_UTF8):
        _refuse_file(
            path,
            1,
            "The file starts with a byte order mark. Save it as UTF-8 without a byte order "
            "mark.",
        )
    if data.startswith(_UTF16_MARKS):
        _refuse_file(path, 1, "The file is UTF-16 text. Save it as UTF-8.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        number = data.count(b"\n", 0, error.start) + 1
        _refuse_file(path, number, f"Line {number} is not UTF-8 text. Save the file as UTF-8.")
    if "\x00" in text:
        number = text.count("\n", 0, text.index("\x00")) + 1
        _refuse_file(
            path,
            number,
            f"Line {number} holds a NUL character. The file may be UTF-16 text. Save it as "
            "UTF-8.",
        )
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    found = []
    for number, line in enumerate(lines, start=1):
        line = line.removesuffix("\r")
        if line:
            found.append((number, line))
    return found


def _parse_recording(line: str) -> RequestedRecording:
    if not line.strip():
        raise _LineProblem("The line holds only spaces. Remove it.")
    try:
        entry = json.loads(
            line, object_pairs_hook=_refuse_repeated_keys, parse_constant=_refuse_constant
        )
    except _LineProblem:
        raise
    except (ValueError, RecursionError):
        entry = None
    if not isinstance(entry, dict):
        raise _LineProblem(
            "The line is not a JSON object. Give one object with namespace, value, "
            "audio_uri, and optional duration_seconds."
        )
    if "embeddings" in entry:
        raise _LineProblem(
            "The entry has embeddings. Remove them: preparation pins the input of a head run."
        )
    # Checked before the model, so that the line message names the key.
    for key, value in entry.items():
        if isinstance(value, str) and not is_storable_text(value):
            raise _LineProblem(
                f"{key} holds a NUL character or a lone surrogate, which Robin cannot "
                "store. Correct the entry."
            )
    try:
        return RequestedRecording.model_validate(entry, strict=True)
    except ValidationError as error:
        problems = field_problems(error, "entry")
    raise _LineProblem(
        "; ".join(f"{problem['field']}: {problem['message']}" for problem in problems)
        + ". Correct the entry."
    )


def _refuse_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    entry = {}
    for key, value in pairs:
        if key in entry:
            raise _LineProblem(f"The key {key!r} appears more than once. Give each key once.")
        entry[key] = value
    return entry


def _refuse_constant(name: str) -> NoReturn:
    raise _LineProblem(f"{name} is not a number that Robin accepts. Give a finite number.")


def _parse_uri(line: str) -> str:
    if not line.strip():
        raise _LineProblem("The line holds only spaces. Remove it, or give a path or a URI.")
    if line.startswith("/"):
        return PurePosixPath(line).as_uri()
    if _SCHEME.match(line) is None:
        raise _LineProblem("The line is a relative path. Give an absolute path or a URI.")
    if not line.lower().startswith("file:"):
        return line
    try:
        path = unquote(urlsplit(line).path)
    except ValueError:
        path = ""
    if not path.startswith("/"):
        raise _LineProblem(
            "The line is not a valid file URI. Give an absolute path or a file:/// URI."
        )
    robins = PurePosixPath(path).as_uri()
    if line != robins:
        raise _LineProblem(
            f"The file URI is not in the spelling that Robin uses. Write it as {robins}."
        )
    return line


def get_repeated_identities(
    recordings: Sequence[RequestedRecording],
) -> dict[tuple[str, str], list[int]]:
    """Returns each `(namespace, value)` that is in `recordings` more than once.

    Each identity maps to the index, from 0, of every recording that has it.
    """
    indexes = {}
    for index, recording in enumerate(recordings):
        indexes.setdefault((recording.namespace, recording.value), []).append(index)
    return {identity: found for identity, found in indexes.items() if len(found) > 1}


def _build_recording_manifest(
    path: Path,
    data: bytes,
    numbered: list[tuple[int, RequestedRecording]],
    problems: list[dict],
    input: str,
    extra_provenance: dict[str, Any],
) -> RecordingManifest:
    lines = [number for number, _ in numbered]
    repeated = get_repeated_identities([recording for _, recording in numbered])
    for (namespace, value), indexes in repeated.items():
        first_line = lines[indexes[0]]
        for index in indexes[1:]:
            problems.append({
                "line": lines[index],
                "problem": f"The recording {value} in namespace {namespace} is also on "
                f"line {first_line}. Give each recording once.",
            })
    if problems:
        problems.sort(key=lambda problem: problem["line"])
        first = problems[0]
        more = f" The file has {len(problems)} problems." if len(problems) > 1 else ""
        raise Refused(
            INVALID_REQUEST,
            f"{path} cannot be submitted. Line {first['line']}: {first['problem']}{more}",
            details={
                "file": str(path),
                "problems": problems[:MAX_PROBLEMS],
                "truncated": len(problems) > MAX_PROBLEMS,
            },
        )
    if not numbered:
        raise Refused(
            INVALID_REQUEST,
            f"{path} holds no recordings. Give at least one.",
            details={"file": str(path), "problems": [], "truncated": False},
        )
    return RecordingManifest(
        recordings=tuple(recording for _, recording in numbered),
        provenance={
            "input": input,
            "file_sha256": checksum_bytes(data),
            "file_size_bytes": len(data),
            **extra_provenance,
        },
    )


def _refuse_file(path: Path, line: int, problem: str) -> NoReturn:
    raise Refused(
        INVALID_REQUEST,
        f"{path} cannot be submitted. {problem}",
        details={
            "file": str(path),
            "problems": [{"line": line, "problem": problem}],
            "truncated": False,
        },
    )
