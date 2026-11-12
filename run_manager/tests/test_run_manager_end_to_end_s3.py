"""A whole run with S3 storage and local compute: the run manager and the real worker.

The test calls the run manager's functions in its own process, not the CLI, so that it
can give the test model's output version to both the run manager and the worker.
"""

import dataclasses
import hashlib
import time
from functools import partial

import ondio
import pytest
from robin_adapters.compute.local.adapter import LocalCompute
from robin_adapters.storage.object_store.adapter import ObjectStoreTaskStorage
from robin_contracts.tasks import InferenceTaskResult, ModelOutputVersion
from robin_contracts.work import REGISTRY_ROLE, PinnedFile, PinnedModel
from robin_run_manager.compose import compose
from robin_run_manager.configuration import read_configuration
from robin_run_manager.models import register_model
from robin_run_manager.presets import create_preset
from robin_run_manager.profile import Profile, resolve_run_settings
from robin_run_manager.reconcile import reconcile_once
from robin_run_manager.recording_inputs import read_uris_file
from robin_run_manager.submit import submit_run
from robin_run_manager.task_files import RESULT_FILE_NAME, WORK_FILE_NAME, get_task_folder
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS
from run_fixtures import get_tasks

from worker_doubles import (
    REGISTRY_CSV,
    TEST_MODEL_ID,
    TEST_VERSIONS,
    build_card,
    installed_test_model,
    worker_command,
)

VERSIONS = (
    *RELEASE_OUTPUT_VERSIONS,
    ModelOutputVersion(model=TEST_MODEL_ID, version=TEST_VERSIONS[TEST_MODEL_ID]),
)
TEST_MODEL = "test-model/1"
CONFIGURATION = f"""\
model: {TEST_MODEL}
input:
  kind: audio
outputs:
  - kind: scores
    contract_id: robin.scores.parquet/1
    retention: full
"""
_RUN_TIMEOUT_S = 60
_TERMINAL_STATES = ("succeeded", "failed")


def put_object(endpoint: str, uri: str, data: bytes) -> None:
    ondio.create(uri, data, endpoint_url=endpoint)


def pin_file(endpoint: str, uri: str, data: bytes) -> PinnedFile:
    put_object(endpoint, uri, data)
    return PinnedFile(
        uri=uri, digest="sha256:" + hashlib.sha256(data).hexdigest(), size_bytes=len(data)
    )


def wait_for_task_to_end(services, database, run_id) -> dict:
    """Runs reconcile passes until the run's one task ends, and returns the task."""
    deadline = time.monotonic() + _RUN_TIMEOUT_S
    while True:
        report = reconcile_once(services, actor="ana", release_versions=VERSIONS)
        assert report.failures == []
        [task] = get_tasks(database, run_id)
        if task["state"] in _TERMINAL_STATES:
            return task
        if time.monotonic() > deadline:
            pytest.fail(f"The task is still {task['state']} after {_RUN_TIMEOUT_S} seconds.")
        time.sleep(0.2)


def test_a_run_on_s3_storage_ends_with_its_result_and_outputs(
    migrated_database, database_settings, s3_server, bucket, tmp_path
):
    endpoint = s3_server["endpoint"]
    root = f"s3://{bucket}/robin"
    profile = Profile.model_validate({
        "database": database_settings,
        "compute": {"setups": {"local": {"kind": "local"}}, "models": {TEST_MODEL: {}}},
        "storage": {"kind": "s3", "root": root, "endpoint": endpoint},
    })
    register_model(
        migrated_database,
        PinnedModel(
            card=build_card(),
            files={
                "weights": pin_file(endpoint, f"s3://{bucket}/model/weights.bin", b"weights"),
                REGISTRY_ROLE: pin_file(
                    endpoint, f"s3://{bucket}/model/registry.csv", REGISTRY_CSV
                ),
            },
        ),
        actor="ana",
    )
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(CONFIGURATION)
    create_preset(
        migrated_database, "e2e", read_configuration(configuration_path), actor="ana"
    )
    audio_uri = f"s3://{bucket}/audio/r1.wav"
    put_object(endpoint, audio_uri, b"audio r1")
    uris_path = tmp_path / "uris.txt"
    uris_path.write_text(f"{audio_uri}\n")

    composed = compose(profile)
    state_folder = tmp_path / "state"
    site = installed_test_model(tmp_path / "installed")
    compute = LocalCompute(
        command=worker_command(tmp_path, [], test_versions=True),
        env={"PYTHONPATH": str(site), "ROBIN_RELEASE": "test-release"},
        state_folder=state_folder,
    )
    services = dataclasses.replace(composed, compute={"local": compute})
    try:
        run_id = submit_run(
            migrated_database,
            preset="e2e",
            configuration=None,
            recordings=read_uris_file(uris_path, "survey"),
            resolve_run_settings=partial(resolve_run_settings, profile, compute=services.compute),
            actor="ana",
            release_versions=VERSIONS,
        ).run_id
        started = time.monotonic()
        task = wait_for_task_to_end(services, migrated_database, run_id)
        print(f"The run took {time.monotonic() - started:.1f} seconds.")
    finally:
        composed.database.close()

    log = state_folder / "tasks" / str(task["task_id"]) / "worker.log"
    assert task["state"] == "succeeded", log.read_text() if log.exists() else "no worker log"
    task_folder = get_task_folder(root, run_id, task["task_id"])
    work_uri = f"{task_folder}/{WORK_FILE_NAME}"
    result_uri = f"{task_folder}/{RESULT_FILE_NAME}"
    assert ondio.exists(work_uri, endpoint_url=endpoint)
    result = InferenceTaskResult.model_validate_json(
        ondio.read(result_uri, endpoint_url=endpoint)
    )
    artifacts = result.result.artifacts
    assert artifacts
    outputs = f"{task_folder}/outputs/"
    executions = set()
    for artifact in artifacts:
        assert artifact.uri.startswith(outputs), artifact.uri
        execution, _, path = artifact.uri.removeprefix(outputs).partition("/")
        assert path, artifact.uri
        executions.add(execution)
        assert ondio.exists(artifact.uri, endpoint_url=endpoint)
    assert len(executions) == 1
    with pytest.raises(FileExistsError):
        ObjectStoreTaskStorage(root, endpoint_url=endpoint).create(result_uri, b"another result")
