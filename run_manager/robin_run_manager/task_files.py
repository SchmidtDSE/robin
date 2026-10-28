"""The task folder, and the work file of a task.

Each task has its own folder under its run's frozen storage root. The work file is built
from the database only, so a rebuild gives the same bytes even if the profile changed.
"""

from dataclasses import dataclass
from uuid import UUID

import psycopg

from robin_contracts.canonical import canonical_json_bytes, checksum_bytes
from robin_contracts.local_files import get_file_uri
from robin_contracts.tasks import InferenceTask, ModelOutputVersion, output_version_id
from robin_contracts.work import InferenceWork, InputArtifact, RecordingRef
from robin_run_manager.configuration import Configuration
from robin_run_manager.models import get_stored_model
from robin_run_manager.stored_configurations import get_work_input

WORK_FILE_NAME = "work.json"
RESULT_FILE_NAME = "result.json"
OUTPUT_FOLDER_NAME = "outputs"


@dataclass(frozen=True)
class WorkFile:
    """The work file of a task: what it holds, its bytes, their checksum, and its URI."""

    task: InferenceTask
    data: bytes
    digest: str
    work_uri: str


def get_task_folder(root: str, run_id: UUID, task_id: UUID) -> str:
    """The folder of one task under the storage root `root`.

    A local root, a path or a `file://` URI, gives a `file://` URI. A trailing `/` on
    `root` is ignored. Links are not resolved.
    """
    base = get_file_uri(root) if root.startswith("/") else root
    return f"{base.removesuffix('/')}/runs/{run_id}/tasks/{task_id}"


def get_result_uri(work_uri: str) -> str:
    """The URI of the result file in the task folder of the work file `work_uri`."""
    return f"{work_uri.removesuffix(WORK_FILE_NAME)}{RESULT_FILE_NAME}"


def build_work_file(connection: psycopg.Connection, task_id: UUID) -> WorkFile:
    """Build the work file of `task_id` from the database, in the caller's transaction.

    For a head, each recording gets the embeddings file that preparation pinned.
    """
    row = connection.execute(
        "SELECT t.run_id, t.work_resources, r.execution_settings, i.output_versions, "
        "c.model_id, c.configuration "
        "FROM tasks t JOIN runs r USING (run_id) JOIN inference_runs i USING (run_id) "
        "JOIN configurations c ON c.digest = i.configuration_digest "
        "WHERE t.task_id = %s",
        (task_id,),
    ).fetchone()
    work = _build_work(connection, task_id, row)
    folder = get_task_folder(row["execution_settings"]["storage"]["root"], row["run_id"], task_id)
    model_id = output_version_id(work.model.card)
    task = InferenceTask(
        schema_version="robin.inference-task/1",
        task_id=task_id,
        output_root=f"{folder}/{OUTPUT_FOLDER_NAME}",
        output_version=ModelOutputVersion(
            model=model_id, version=row["output_versions"][model_id]
        ),
        work=work,
    )
    data = canonical_json_bytes(task.model_dump(mode="json"))
    return WorkFile(
        task=task,
        data=data,
        digest=checksum_bytes(data),
        work_uri=f"{folder}/{WORK_FILE_NAME}",
    )


def _build_work(connection: psycopg.Connection, task_id: UUID, row: dict) -> InferenceWork:
    """The work of the task: what its configuration computes, over its recordings."""
    configuration = Configuration.model_validate(row["configuration"])
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=_get_recordings(connection, task_id),
        model=get_stored_model(connection, row["model_id"]),
        input=get_work_input(connection, configuration.input),
        settings=configuration.settings,
        resources=row["work_resources"],
        outputs=configuration.outputs,
    )


def _get_recordings(connection: psycopg.Connection, task_id: UUID) -> tuple[RecordingRef, ...]:
    rows = connection.execute(
        "SELECT o.namespace, o.value, r.audio_uri, r.duration_seconds, "
        "f.uri AS input_uri, f.checksum AS input_checksum "
        "FROM recording_outcomes o "
        "JOIN run_recordings r USING (run_id, namespace, value) "
        "LEFT JOIN output_files f ON f.file_id = o.input_file_id "
        "WHERE o.task_id = %s ORDER BY o.namespace, o.value",
        (task_id,),
    ).fetchall()
    return tuple(
        RecordingRef(
            namespace=row["namespace"],
            value=row["value"],
            audio_uri=row["audio_uri"],
            duration_seconds=row["duration_seconds"],
            embeddings=None
            if row["input_uri"] is None
            else InputArtifact(uri=row["input_uri"], checksum=row["input_checksum"]),
        )
        for row in rows
    )
