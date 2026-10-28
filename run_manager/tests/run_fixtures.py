"""Helpers that write runs, outcomes, tasks, output files, and result files directly.

Robin has no operations yet that cancel or withdraw, so the tests write those rows here.
Tests also write outcome rows here to set up a state without running a whole task.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from psycopg.types.json import Json
from robin_adapters.storage.local import LocalTaskStorage
from robin_adapters.storage.location import split_location
from robin_contracts.local_files import get_local_path
from robin_contracts.ports.catalog import RequestedRecording
from robin_contracts.ports.compute import ComputeStatus
from robin_contracts.results import (
    ArtifactRecord,
    FailureReport,
    InferenceCompleted,
    RecordingCoverage,
    RecordingFailed,
)
from robin_contracts.specs import recipe
from robin_contracts.tasks import InferenceTaskResult
from robin_contracts.work import work_digest
from robin_run_manager.configuration import UnresolvedConfiguration, read_configuration
from robin_run_manager.database import Database, transaction
from robin_run_manager.launch import get_setup_parameters, reserve_tasks, submit_reserved_task
from robin_run_manager.models import register_model
from robin_run_manager.outcomes import update_task_from_compute
from robin_run_manager.preparation import prepare_run
from robin_run_manager.profile import Profile
from robin_run_manager.recording_inputs import RecordingManifest
from robin_run_manager.submit import submit_run
from robin_run_manager.task_files import WorkFile, build_work_file
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS

from compute_doubles import ScriptedCompute
from model_fixtures import read_pinned_model

FIXTURES = Path(__file__).parent / "fixtures"
OWL_CONFIGURATION = read_configuration(FIXTURES / "configuration-owl-v4.yaml")
PERCH_CONFIGURATION = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
HEAD_CONFIGURATION = read_configuration(FIXTURES / "configuration-songbird-head-v1.yaml")
NAMESPACE = "survey"
LOCAL_SETTINGS = {
    "compute": {"kind": "local"},
    "storage": {"kind": "local", "root": "/data/robin"},
}
KUBERNETES_SETTINGS = {
    "compute": {
        "kind": "kubernetes",
        "namespace": "robin",
        "s3_secret_name": "robin-s3",
        "s3_endpoint": "https://s3.example",
    },
    "storage": {"kind": "s3", "root": "s3://robin/"},
}
# A profile with local compute and local storage. Its database is never opened.
LOCAL_PROFILE = Profile.model_validate({
    "database": {"host": "localhost", "dbname": "robin", "user": "robin", "password": "x"},
    "compute": {"kind": "local"},
    "storage": {"kind": "local", "root": "/data/robin"},
})
# The setup of a pass with local compute and local task storage.
LOCAL_SETUP = get_setup_parameters(LOCAL_PROFILE, {"local"})


def register_fixture_models(database: Database) -> None:
    for name in ("owl-v4", "perch-v8", "songbird-head-v1"):
        register_model(database, read_pinned_model(name), actor="ana")


def make_recording_manifest(values) -> RecordingManifest:
    recordings = tuple(
        RequestedRecording(namespace=NAMESPACE, value=value, audio_uri=f"s3://audio/{value}.wav")
        for value in values
    )
    provenance = {"input": "recordings", "file_sha256": "sha256:" + "0" * 64, "file_size_bytes": 1}
    return RecordingManifest(recordings=recordings, provenance=provenance)


def submit(
    database: Database,
    values,
    configuration: UnresolvedConfiguration = OWL_CONFIGURATION,
    *,
    actor: str = "ana",
    release_versions=RELEASE_OUTPUT_VERSIONS,
    execution_settings=LOCAL_SETTINGS,
) -> UUID:
    return submit_run(
        database,
        preset=None,
        configuration=configuration,
        recordings=make_recording_manifest(values),
        execution_settings=execution_settings,
        actor=actor,
        release_versions=release_versions,
    ).run_id


RESERVED_STATES = ("submitting", "submitted", "running", "finalizing", "succeeded")
TASK_FAILURE = {"reason": "no_result", "message": "The task ended with no result.", "detail": None}
RECORDING_FAILURE = {
    "scope": "recording", "code": "audio_unreadable", "stage": "decode",
    "message": "The audio cannot be read.",
}
CONTRACT_IDS = {
    "embeddings": "robin.embeddings.parquet/1",
    "scores": "robin.scores.parquet/1",
    "detections": "robin.detections.parquet/1",
}


def add_task(database: Database, run_id: UUID, state: str = "succeeded", **columns) -> UUID:
    """A task of the run in `state`, with the columns that the state needs.

    A task past `queued` has a reservation, and a failed task has a failure. `columns`
    gives other columns, or replaces these values. A mapping is written as JSON.
    """
    values = {}
    if state in RESERVED_STATES:
        values["reserved_at"] = datetime.now(UTC)
        values["work_file_digest"] = "sha256:" + "2" * 64
    if state == "failed":
        values["failure"] = TASK_FAILURE
    values.update(columns)
    values = {
        name: Json(value) if isinstance(value, dict) else value for name, value in values.items()
    }
    names = ["run_id", "state", "created_by", *values]
    with transaction(database) as connection:
        return connection.execute(
            f"INSERT INTO tasks ({', '.join(names)}) VALUES ({', '.join(['%s'] * len(names))}) "
            "RETURNING task_id",
            (run_id, state, "ana", *values.values()),
        ).fetchone()["task_id"]


def set_outcome(
    database: Database, run_id: UUID, value: str, state: str, *, task_id: UUID | None = None
) -> None:
    """Write the outcome of one recording of the run, with the run's configuration and versions.

    A skipped recording is `in_progress`, and a failed one has a failure of its own.
    """
    skip_reason = "in_progress" if state == "skipped" else None
    failure = Json(RECORDING_FAILURE) if state == "failed" else None
    with transaction(database) as connection:
        connection.execute(
            "INSERT INTO recording_outcomes (run_id, namespace, value, configuration_digest, "
            "output_versions, state, skip_reason, task_id, failure) "
            "SELECT run_id, %s, %s, configuration_digest, output_versions, %s, %s, %s, %s "
            "FROM inference_runs WHERE run_id = %s "
            "ON CONFLICT (run_id, namespace, value) "
            "DO UPDATE SET state = EXCLUDED.state, failure = EXCLUDED.failure",
            (NAMESPACE, value, state, skip_reason, task_id, failure, run_id),
        )


def add_finished_run(
    database: Database,
    values,
    configuration: UnresolvedConfiguration = OWL_CONFIGURATION,
    state: str = "covered",
    **submit_options,
) -> UUID:
    """A ready run whose one task has the outcome `state` for each of `values`."""
    run_id = submit(database, values, configuration, **submit_options)
    task_id = add_task(database, run_id)
    with transaction(database) as connection:
        connection.execute("UPDATE runs SET state = 'ready' WHERE run_id = %s", (run_id,))
    for value in values:
        set_outcome(database, run_id, value, state, task_id=task_id)
    return run_id


def add_output_file(
    database: Database, run_id: UUID, value: str, kind: str = "embeddings", uri: str | None = None
) -> int:
    with transaction(database) as connection:
        task_id = connection.execute(
            "SELECT task_id FROM tasks WHERE run_id = %s LIMIT 1", (run_id,)
        ).fetchone()["task_id"]
        return connection.execute(
            "INSERT INTO output_files (run_id, task_id, namespace, value, kind, contract_id, "
            "uri, checksum) VALUES (%s, %s, %s, %s, %s, 'robin.embeddings.parquet/1', %s, %s) "
            "RETURNING file_id",
            (
                run_id, task_id, NAMESPACE, value, kind,
                uri or f"s3://out/{run_id}/{value}.parquet", "sha256:" + "1" * 64,
            ),
        ).fetchone()["file_id"]


def add_covered_result(
    database: Database, run_id: UUID, value: str, files=("scores",)
) -> UUID:
    """Make the run ready and give `value` a covered outcome, from a new succeeded task
    that wrote one output file of each kind in `files`. Returns the task's ID."""
    task_id = add_task(database, run_id)
    with transaction(database) as connection:
        connection.execute("UPDATE runs SET state = 'ready' WHERE run_id = %s", (run_id,))
        for kind in files:
            connection.execute(
                "INSERT INTO output_files (run_id, task_id, namespace, value, kind, "
                "contract_id, uri, checksum) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    run_id, task_id, NAMESPACE, value, kind, CONTRACT_IDS[kind],
                    f"s3://out/{run_id}/{value}.{kind}.parquet", "sha256:" + "1" * 64,
                ),
            )
    set_outcome(database, run_id, value, "covered", task_id=task_id)
    return task_id


def get_current_files(database: Database) -> list[dict]:
    """Every row of `current_files`, sorted by recording, configuration, and kind."""
    with transaction(database, read_only=True) as connection:
        return connection.execute(
            "SELECT * FROM current_files ORDER BY namespace, value, configuration_digest, kind"
        ).fetchall()


def cancel(database: Database, run_id: UUID) -> None:
    with transaction(database) as connection:
        connection.execute(
            "UPDATE runs SET cancelled_at = now(), cancelled_by = 'ana' WHERE run_id = %s",
            (run_id,),
        )


def withdraw(database: Database, run_id: UUID) -> None:
    with transaction(database) as connection:
        connection.execute(
            "UPDATE runs SET withdrawn_at = now(), withdrawn_by = 'ana' WHERE run_id = %s",
            (run_id,),
        )
        connection.execute(
            "UPDATE recording_outcomes SET state = 'withdrawn' "
            "WHERE run_id = %s AND state IN ('pending', 'covered')",
            (run_id,),
        )


def get_outcomes(database: Database, run_id: UUID) -> dict[str, dict]:
    """The run's outcome rows, by value."""
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT value, state, skip_reason, task_id, input_file_id FROM recording_outcomes "
            "WHERE run_id = %s",
            (run_id,),
        ).fetchall()
    return {row["value"]: row for row in rows}


def get_run_row(database: Database, run_id: UUID) -> dict:
    with transaction(database, read_only=True) as connection:
        return connection.execute(
            "SELECT * FROM runs WHERE run_id = %s", (run_id,)
        ).fetchone()


def get_tasks(database: Database, run_id: UUID) -> list[dict]:
    """The run's tasks in list order, each with the values it holds, sorted."""
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            "SELECT t.task_id, t.state, t.created_by, "
            "array_remove(array_agg(o.value ORDER BY o.namespace, o.value), NULL) AS values "
            "FROM tasks t LEFT JOIN recording_outcomes o ON o.task_id = t.task_id "
            "WHERE t.run_id = %s GROUP BY t.task_id ORDER BY t.created_at, t.task_id",
            (run_id,),
        ).fetchall()
    return rows


def add_queued_tasks(database, values, *, root, actor="ana", size=1, versions=None):
    """A prepared run of `values`, cut into tasks of `size`. Returns the run and its tasks."""
    release_versions = versions or RELEASE_OUTPUT_VERSIONS
    settings = {**LOCAL_SETTINGS, "storage": {"kind": "local", "root": root}}
    run_id = submit(
        database, values, actor=actor, execution_settings=settings,
        release_versions=release_versions,
    )
    prepare_run(database, run_id, recordings_per_task=size, release_versions=release_versions)
    return run_id, [task["task_id"] for task in get_tasks(database, run_id)]


def get_task(database, task_id) -> dict:
    with transaction(database, read_only=True) as connection:
        return connection.execute(
            "SELECT * FROM tasks WHERE task_id = %s", (task_id,)
        ).fetchone()


def count_in_flight(database) -> int:
    with transaction(database, read_only=True) as connection:
        return connection.execute(
            "SELECT count(*) AS n FROM tasks "
            "WHERE state IN ('submitting', 'submitted', 'running')"
        ).fetchone()["n"]


def set_execution_settings(database, run_id, settings) -> None:
    with transaction(database) as connection:
        connection.execute(
            "UPDATE runs SET execution_settings = %s WHERE run_id = %s", (Json(settings), run_id)
        )


ROBIN_RELEASE = "robin@sha256:" + "d" * 64
RESULT_STARTED = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
INPUT_FAILURE = FailureReport(code="input_unavailable", stage="acquire_input", detail="gone")


def get_work_file(database: Database, task_id: UUID) -> WorkFile:
    with transaction(database, read_only=True) as connection:
        return build_work_file(connection, task_id)


def build_completed(work_file: WorkFile, failed=()) -> InferenceCompleted:
    """A completed result of the work file's task. Each recording whose value is not in
    `failed` is covered, with one scores file under the task's output root."""
    task = work_file.task
    covered = [one for one in task.work.recordings if one.value not in failed]
    return InferenceCompleted(
        schema_version="robin.inference-result/1",
        work_digest=work_digest(task.work),
        recipe=recipe(task.work.model.card, task.work.settings),
        model=task.work.model,
        window_geometry={"window_duration": 3.0, "window_overlap": 0.0, "pad": "drop"},
        resolved_scores_request=task.work.outputs[0],
        artifacts=tuple(
            ArtifactRecord(
                kind="scores",
                contract_id="robin.scores.parquet/1",
                namespace=one.namespace,
                value=one.value,
                uri=f"{task.output_root}/scores/{one.value}.parquet",
                checksum="sha256:" + "e" * 64,
                size_bytes=100,
                rows=3,
            )
            for one in covered
        ),
        coverage=tuple(
            RecordingCoverage(
                namespace=one.namespace, value=one.value, windows_completed=3, score_rows=3,
                embedding_rows=0, detection_rows=0, first_window_start_s=0.0,
                last_window_end_s=9.0,
            )
            for one in covered
        ),
        failed=tuple(
            RecordingFailed(namespace=one.namespace, value=one.value, failure=INPUT_FAILURE)
            for one in task.work.recordings
            if one.value in failed
        ),
    )


def build_task_result(work_file: WorkFile, outcome, **fields) -> InferenceTaskResult:
    """A result of the work file's task with `outcome`. `fields` replace its other values."""
    values = {
        "schema_version": "robin.inference-task-result/1",
        "task_id": work_file.task.task_id,
        "work_file_digest": work_file.digest,
        "robin_release": ROBIN_RELEASE,
        "started_at": RESULT_STARTED,
        "finished_at": RESULT_STARTED,
        "result": outcome,
    }
    return InferenceTaskResult(**(values | fields))


def write_task_result(work_file: WorkFile, result: InferenceTaskResult | bytes) -> None:
    """Write `result.json` in the task's folder, as the worker does at its end."""
    path = get_local_path(work_file.work_uri).with_name("result.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result if isinstance(result, bytes) else result.model_dump_json().encode()
    path.write_bytes(data)


def add_submitted_task(database, root, values=("a", "b"), *, actor="ana"):
    """A submitted task of a new run that holds `values`. Gives the task and its compute."""
    _, [task_id] = add_queued_tasks(
        database, list(values), root=root, actor=actor, size=len(values)
    )
    reserve_tasks(database, actor=actor, profile=LOCAL_PROFILE, storage_kinds={"local"})
    compute = ScriptedCompute()
    assert submit_reserved_task(
        database, task_id, setup=LOCAL_SETUP, compute=compute, storage=LocalTaskStorage(),
        split_location=split_location,
    ) == "submitted"
    return task_id, compute


def add_finalizing_task(database, root, values=("a", "b")):
    """A finalizing task of a new run that holds `values`. Gives the task and its work file."""
    task_id, compute = add_submitted_task(database, root, values)
    compute.script("describe", ComputeStatus(state="ended"))
    assert update_task(database, task_id, compute) == "finalizing"
    return task_id, get_work_file(database, task_id)


def update_task(database, task_id, compute):
    return update_task_from_compute(database, task_id, setup=LOCAL_SETUP, compute=compute)
