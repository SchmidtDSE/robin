"""The task folder, and the work file of a task built from the database."""

import hashlib
import uuid

import pytest
from psycopg.types.json import Json

from robin_adapters.storage.location import split_location
from robin_contracts.tasks import InferenceTask, ModelOutputVersion, output_root_problem
from robin_contracts.work import AudioInput, InputArtifact
from robin_run_manager.configuration import get_embedding_input
from robin_run_manager.database import transaction
from robin_run_manager.preparation import prepare_run
from robin_run_manager.stored_configurations import get_stored_configuration
from robin_run_manager.task_files import build_work_file, get_task_folder
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS
from run_fixtures import (
    HEAD_CONFIGURATION,
    LOCAL_SETTINGS,
    NAMESPACE,
    OWL_CONFIGURATION,
    PERCH_CONFIGURATION,
    add_covered_result,
    get_tasks,
    register_fixture_models,
    submit,
)

RUN_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
TASK_ID = uuid.UUID("00000000-0000-4000-8000-000000000002")
RESOURCES = {"threads": 3, "device": "cpu"}


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


def get_settings(root: str) -> dict:
    return {**LOCAL_SETTINGS, "storage": {"kind": "local", "root": root}}


def add_prepared_tasks(
    database,
    values,
    configuration=OWL_CONFIGURATION,
    *,
    size=100,
    release_versions=RELEASE_OUTPUT_VERSIONS,
    execution_settings=LOCAL_SETTINGS,
) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """A prepared run of `values`, and its task IDs in list order, each with RESOURCES."""
    run_id = submit(
        database, values, configuration,
        release_versions=release_versions, execution_settings=execution_settings,
    )
    prepare_run(database, run_id, recordings_per_task=size, release_versions=release_versions)
    task_ids = [task["task_id"] for task in get_tasks(database, run_id)]
    with transaction(database) as connection:
        connection.execute(
            "UPDATE tasks SET work_resources = %s WHERE run_id = %s", (Json(RESOURCES), run_id)
        )
    return run_id, task_ids


def build(database, task_id):
    with transaction(database, read_only=True) as connection:
        return build_work_file(connection, task_id)


def test_the_folder_is_under_the_runs_frozen_root(registered):
    run_id, [task_id] = add_prepared_tasks(
        registered, ["a"], execution_settings=get_settings("/data/robin")
    )
    # The profile's root changes, and a later run freezes the new root.
    add_prepared_tasks(registered, ["b"], execution_settings=get_settings("/moved"))

    work_file = build(registered, task_id)

    folder = f"file:///data/robin/runs/{run_id}/tasks/{task_id}"
    assert work_file.work_uri == f"{folder}/work.json"
    assert work_file.task.output_root == f"{folder}/outputs"


@pytest.mark.parametrize(
    ("root", "folder"),
    [
        ("/data/robin", f"file:///data/robin/runs/{RUN_ID}/tasks/{TASK_ID}"),
        ("file:///data/robin", f"file:///data/robin/runs/{RUN_ID}/tasks/{TASK_ID}"),
        ("s3://robin", f"s3://robin/runs/{RUN_ID}/tasks/{TASK_ID}"),
        ("s3://robin/team", f"s3://robin/team/runs/{RUN_ID}/tasks/{TASK_ID}"),
    ],
)
def test_a_root_with_a_trailing_slash_gives_the_same_folder(root, folder):
    assert get_task_folder(root, RUN_ID, TASK_ID) == folder
    assert get_task_folder(root + "/", RUN_ID, TASK_ID) == folder


@pytest.mark.parametrize("root", ["/data/robin", "file:///data/robin/", "s3://robin/"])
def test_output_root_is_accepted_by_the_result_check(root):
    output_root = get_task_folder(root, RUN_ID, TASK_ID) + "/outputs"

    assert output_root_problem(output_root, split_location=split_location) is None


def test_an_audio_work_holds_the_tasks_recordings_in_order(registered):
    run_id, [first, second] = add_prepared_tasks(registered, ["c", "b", "a"], size=2)
    with transaction(registered) as connection:
        connection.execute(
            "UPDATE run_recordings SET duration_seconds = 12.5 WHERE run_id = %s AND value = 'b'",
            (run_id,),
        )

    work = build(registered, first).task.work

    assert [
        (one.namespace, one.value, one.audio_uri, one.duration_seconds, one.embeddings)
        for one in work.recordings
    ] == [
        (NAMESPACE, "a", "s3://audio/a.wav", None, None),
        (NAMESPACE, "b", "s3://audio/b.wav", 12.5, None),
    ]
    assert work.input == AudioInput()
    assert [one.value for one in build(registered, second).task.work.recordings] == ["c"]


def test_a_head_work_copies_each_pinned_embeddings_file(registered):
    backbone = submit(registered, ["a", "b"], PERCH_CONFIGURATION)
    add_covered_result(registered, backbone, "a", files=("embeddings",))
    add_covered_result(registered, backbone, "b", files=("embeddings",))
    _, [task_id] = add_prepared_tasks(registered, ["a", "b"], HEAD_CONFIGURATION)
    with transaction(registered, read_only=True) as connection:
        stored = get_stored_configuration(
            connection, HEAD_CONFIGURATION.input.backbone_configuration
        )
        files = connection.execute(
            "SELECT value, uri, checksum FROM output_files WHERE run_id = %s ORDER BY value",
            (backbone,),
        ).fetchall()

    work = build(registered, task_id).task.work

    assert [one.embeddings for one in work.recordings] == [
        InputArtifact(uri=row["uri"], checksum=row["checksum"]) for row in files
    ]
    assert work.input == get_embedding_input(stored.configuration, stored.model)


def test_the_work_holds_the_tasks_recorded_resources(registered):
    _, [task_id] = add_prepared_tasks(registered, ["a"])

    assert build(registered, task_id).task.work.resources == RESOURCES


def test_the_output_version_is_the_runs_frozen_version_of_the_model(registered):
    newer = tuple(
        ModelOutputVersion(model=entry.model, version=7)
        if entry.model == "model:owl/v4" else entry
        for entry in RELEASE_OUTPUT_VERSIONS
    )
    _, [task_id] = add_prepared_tasks(registered, ["a"], release_versions=newer)

    task = build(registered, task_id).task

    assert task.output_version == ModelOutputVersion(model="model:owl/v4", version=7)
    assert task.task_id == task_id


def test_a_rebuild_gives_the_same_bytes(registered):
    _, [task_id] = add_prepared_tasks(registered, ["b", "a"])

    assert build(registered, task_id).data == build(registered, task_id).data


def test_the_digest_is_the_checksum_of_the_bytes(registered):
    _, [task_id] = add_prepared_tasks(registered, ["a"])

    work_file = build(registered, task_id)

    assert work_file.digest == "sha256:" + hashlib.sha256(work_file.data).hexdigest()
    assert InferenceTask.model_validate_json(work_file.data) == work_file.task
