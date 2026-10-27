"""Helpers that write runs, outcomes, tasks, and output files directly.

Robin has no operations yet that record outcomes, cancel, or withdraw, so the tests
write those rows here.
"""

from pathlib import Path
from uuid import UUID

from robin_contracts.ports.catalog import RequestedRecording
from robin_run_manager.configuration import UnresolvedConfiguration, read_configuration
from robin_run_manager.database import Database, transaction
from robin_run_manager.models import register_model
from robin_run_manager.recording_inputs import RecordingManifest
from robin_run_manager.submit import submit_run
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS

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


def add_task(database: Database, run_id: UUID, state: str = "succeeded") -> UUID:
    with transaction(database) as connection:
        return connection.execute(
            "INSERT INTO tasks (run_id, state, created_by) VALUES (%s, %s, 'ana') "
            "RETURNING task_id",
            (run_id, state),
        ).fetchone()["task_id"]


def set_outcome(
    database: Database, run_id: UUID, value: str, state: str, *, task_id: UUID | None = None
) -> None:
    """Write the outcome of one recording of the run, with the run's configuration and versions."""
    skip_reason = "in_progress" if state == "skipped" else None
    with transaction(database) as connection:
        connection.execute(
            "INSERT INTO recording_outcomes (run_id, namespace, value, configuration_digest, "
            "output_versions, state, skip_reason, task_id) "
            "SELECT run_id, %s, %s, configuration_digest, output_versions, %s, %s, %s "
            "FROM inference_runs WHERE run_id = %s "
            "ON CONFLICT (run_id, namespace, value) DO UPDATE SET state = EXCLUDED.state",
            (NAMESPACE, value, state, skip_reason, task_id, run_id),
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
