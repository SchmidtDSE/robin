"""One task: read the work file, check it, run it, and write the result beside it."""

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from robin_adapters.location import split_location
from robin_contracts.canonical import checksum_bytes
from robin_contracts.ports import ArtifactWriter
from robin_contracts.protocols import Log
from robin_contracts.results import InferenceResult
from robin_contracts.tasks import (
    InferenceTask,
    InferenceTaskResult,
    TaskRefusal,
    WorkerError,
    output_root_problem,
    output_version_id,
)
from robin_inference_engine.engine import run_work

from robin_worker import storage

TASK_SCHEMA = "robin.inference-task/1"
RESULT_SCHEMA = "robin.inference-task-result/1"
RESULT_FILE_NAME = "result.json"
DETAIL_LIMIT = 2000  # the engine's limit for failure detail

EXIT_REPORTED = 0
EXIT_REFUSED_OR_ERROR = 1
EXIT_RESULT_EXISTS = 3
EXIT_RESULT_NOT_WRITTEN = 4


class _Refused(Exception):
    """The worker refuses the task before inference starts."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail[:DETAIL_LIMIT]


@dataclass
class _Reading:
    """What the worker knows about the work file so far, for the result it writes."""

    started_at: datetime
    digest: str | None = None
    task_id: UUID | None = None


def run_task(work_uri: str, *, versions: Mapping[str, int], release: str, log: Log) -> int:
    """Run the task at `work_uri`, write its result beside the work file, and return the exit code.

    `versions` maps each output version identifier to the version this worker's code
    gives. A stop signal passes through, and no result is written.
    """
    reading = _Reading(started_at=_now())
    try:
        outcome = _run(work_uri, reading, versions=versions, log=log)
    except _Refused as refusal:
        outcome = TaskRefusal(code=refusal.code, detail=refusal.detail)
    except Exception as exc:
        outcome = WorkerError(detail=_described(exc)[:DETAIL_LIMIT])
    result = InferenceTaskResult(
        schema_version=RESULT_SCHEMA,
        task_id=reading.task_id,
        work_file_digest=reading.digest,
        robin_release=release,
        started_at=reading.started_at,
        # The wall clock can step back during a long task. The result must still be valid.
        finished_at=max(_now(), reading.started_at),
        result=outcome,
    )
    return _write_result(_result_uri(work_uri), result, log=log)


def _run(
    work_uri: str, reading: _Reading, *, versions: Mapping[str, int], log: Log
) -> InferenceResult:
    data = _read(work_uri)
    reading.digest = checksum_bytes(data)
    task = _parse(data)
    reading.task_id = task.task_id
    writer = _checked_writer(task.output_root)
    _check_output_version(task, versions)
    scratch = Path(tempfile.mkdtemp(prefix="robin-worker-"))
    try:
        files = storage.TaskFiles(scratch)  # one provider serves both ports
        return run_work(task.work, model_files=files, inputs=files, artifacts=writer, log=log)
    finally:
        _remove_scratch(scratch, log=log)


def _read(work_uri: str) -> bytes:
    try:
        return storage.read_bytes(work_uri)
    except Exception as exc:
        raise _Refused("work_file_unreadable", f"reading {work_uri}: {_described(exc)}") from exc


def _parse(data: bytes) -> InferenceTask:
    try:
        value = json.loads(data.decode("utf-8"))
    # ValueError covers bad UTF-8, bad JSON, and an integer too long to convert.
    # RecursionError is raised for very deep nesting.
    except (ValueError, RecursionError) as exc:
        raise _Refused("work_file_invalid", f"the work file is not UTF-8 JSON: {exc}") from exc
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if isinstance(schema, str) and schema != TASK_SCHEMA:
        raise _Refused(
            "unknown_schema_version",
            f"the work file has schema_version {schema!r}; this worker reads {TASK_SCHEMA!r}",
        )
    try:
        return InferenceTask.model_validate(value)
    except ValidationError as exc:
        raise _Refused("work_file_invalid", f"the work file is not a task: {exc}") from exc


def _checked_writer(output_root: str) -> ArtifactWriter:
    problem = output_root_problem(output_root, split_location=split_location)
    if problem is not None:
        raise _Refused("work_file_invalid", problem)
    return storage.artifact_writer(output_root)


def _check_output_version(task: InferenceTask, versions: Mapping[str, int]) -> None:
    identifier = output_version_id(task.work.model.card)
    expected = task.output_version
    if identifier != expected.model:
        raise _Refused(
            "model_identifier_mismatch",
            f"the task names {expected.model!r}, but its work runs {identifier!r}",
        )
    own = versions.get(identifier)
    if own != expected.version:
        raise _Refused(
            "output_version_mismatch",
            f"the task needs output version {expected.version} of {identifier!r}; "
            f"this worker has {'none' if own is None else own}",
        )


def _remove_scratch(scratch: Path, *, log: Log) -> None:
    # A failure here is only logged. Raising would replace the work's result, or a
    # stop signal on its way out, with an error.
    try:
        shutil.rmtree(scratch)
    except Exception as exc:
        log(f"removing the scratch folder {scratch} failed: {_described(exc)}")


def _write_result(uri: str, result: InferenceTaskResult, *, log: Log) -> int:
    try:
        storage.create(uri, result.model_dump_json().encode())
    except FileExistsError:
        log(f"a result already exists at {uri}; it was not changed")
        return EXIT_RESULT_EXISTS
    except Exception as exc:
        log(f"writing the result to {uri} failed: {_described(exc)}; a result may or may not exist")
        return EXIT_RESULT_NOT_WRITTEN
    if result.result.outcome in ("completed", "failure"):
        return EXIT_REPORTED
    return EXIT_REFUSED_OR_ERROR


def _result_uri(work_uri: str) -> str:
    return work_uri.rsplit("/", 1)[0] + "/" + RESULT_FILE_NAME


def _described(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _now() -> datetime:
    return datetime.now(timezone.utc)
