"""The two list files that `runs submit` reads: `--recordings` and `--uris`.

Both are UTF-8 with one entry on each line. Empty lines are skipped, but a line of only
spaces is refused.
"""

import codecs
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from pydantic import ValidationError

from robin_contracts.canonical import checksum_bytes
from robin_contracts.local_files import get_audio_uri
from robin_contracts.ports.catalog import RequestedRecording, is_storable_text
from robin_contracts.text_formats import InvalidText, load_json
from robin_run_manager.errors import INVALID_REQUEST, Refused
from robin_run_manager.operator_files import field_problems

# The most line problems that one refusal reports.
MAX_PROBLEMS = 20

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
        entry = load_json(line)
    except InvalidText as error:
        # The problem only: a position inside one line would read as a line number.
        raise _LineProblem(f"The line {error.problem}.") from None
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


def _parse_uri(line: str) -> str:
    if not line.strip():
        raise _LineProblem("The line holds only spaces. Remove it, or give a path or a URI.")
    try:
        return get_audio_uri(line)
    except ValueError as error:
        problem = str(error)
    raise _LineProblem(f"The line is {problem}.")


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
