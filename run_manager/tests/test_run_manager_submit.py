"""Submit: store a run in `preparing`, with its configuration, versions, and recording list."""

from pathlib import Path
from unittest.mock import ANY

import pytest

from robin_contracts.ports.catalog import RequestedRecording
from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.configuration import read_configuration
from robin_run_manager.database import transaction
from robin_run_manager.errors import (
    INVALID_REQUEST,
    UNKNOWN_OUTPUT_VERSION,
    UNKNOWN_PRESET,
    UNSUPPORTED_LOCATION,
    Refused,
)
from robin_run_manager.models import register_model
from robin_run_manager.presets import create_preset
from robin_run_manager.recording_inputs import RecordingManifest
from robin_run_manager.submit import submit_run
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS

from model_fixtures import read_pinned_model
from run_fixtures import KUBERNETES_SETTINGS, LOCAL_SETTINGS

FIXTURES = Path(__file__).parent / "fixtures"
OWL_CONFIGURATION = read_configuration(FIXTURES / "configuration-owl-v4.yaml")
PERCH_CONFIGURATION = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
HEAD_CONFIGURATION = read_configuration(FIXTURES / "configuration-songbird-head-v1.yaml")
PROVENANCE = {
    "input": "uris",
    "file_sha256": "sha256:" + "0" * 64,
    "file_size_bytes": 812,
    "namespace": "survey",
}


def make_recording_manifest(count: int = 3) -> RecordingManifest:
    recordings = tuple(
        RequestedRecording(
            namespace="survey",
            value=f"file:///data/{index:05}.wav",
            audio_uri=f"file:///data/{index:05}.wav",
            duration_seconds=60.0 if index % 2 else None,
        )
        for index in range(count)
    )
    return RecordingManifest(recordings=recordings, provenance=dict(PROVENANCE))


@pytest.fixture
def registered(migrated_database):
    for name in ("owl-v4", "perch-v8", "songbird-head-v1"):
        register_model(migrated_database, read_pinned_model(name), actor="ana")
    return migrated_database


def get_rows(database, query: str, *parameters) -> list[dict]:
    with transaction(database, read_only=True) as connection:
        return connection.execute(query, parameters).fetchall()


def count_rows(database, table: str) -> int:
    return get_rows(database, f"SELECT count(*) AS n FROM {table}")[0]["n"]


def test_a_run_from_a_preset_is_stored_preparing(registered):
    preset = create_preset(registered, "owl", OWL_CONFIGURATION, actor="ana")
    submitted_list = make_recording_manifest()

    submitted = submit_run(
        registered,
        preset="owl",
        configuration=None,
        recordings=submitted_list,
        execution_settings=LOCAL_SETTINGS,
        actor="ben",
    )

    assert submitted.configuration_digest == preset.configuration_digest
    assert submitted.recording_count == 3
    [run] = get_rows(registered, "SELECT * FROM runs WHERE run_id = %s", submitted.run_id)
    assert (run["kind"], run["state"], run["submitted_by"]) == ("inference", "preparing", "ben")
    stored = get_rows(
        registered,
        "SELECT namespace, value, audio_uri, duration_seconds FROM run_recordings "
        "WHERE run_id = %s",
        submitted.run_id,
    )
    assert sorted(stored, key=lambda row: row["value"]) == [
        recording.model_dump() for recording in submitted_list.recordings
    ]


def test_a_preset_and_its_configuration_give_the_same_digest(registered):
    create_preset(registered, "owl", OWL_CONFIGURATION, actor="ana")

    from_preset = submit_run(
        registered,
        preset="owl",
        configuration=None,
        recordings=make_recording_manifest(),
        execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )
    from_file = submit_run(
        registered,
        preset=None,
        configuration=OWL_CONFIGURATION,
        recordings=make_recording_manifest(),
        execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )

    assert from_preset.configuration_digest == from_file.configuration_digest


def test_a_full_configuration_is_stored_for_later_heads(registered):
    submitted = submit_run(
        registered,
        preset=None,
        configuration=PERCH_CONFIGURATION,
        recordings=make_recording_manifest(),
        execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )

    assert submitted.configuration_digest == HEAD_CONFIGURATION.input.backbone_configuration
    assert get_rows(
        registered,
        "SELECT digest FROM configurations WHERE digest = %s",
        submitted.configuration_digest,
    )


@pytest.mark.parametrize("both", [True, False])
def test_both_or_neither_is_an_invalid_request(registered, both):
    create_preset(registered, "owl", OWL_CONFIGURATION, actor="ana")

    with pytest.raises(Refused) as refused:
        submit_run(
            registered,
            preset="owl" if both else None,
            configuration=OWL_CONFIGURATION if both else None,
            recordings=make_recording_manifest(),
            execution_settings=LOCAL_SETTINGS,
            actor="ana",
        )

    assert refused.value.reason == INVALID_REQUEST
    assert count_rows(registered, "runs") == 0


def test_an_unknown_preset_is_refused(registered):
    with pytest.raises(Refused) as refused:
        submit_run(
            registered, preset="absent", configuration=None, recordings=make_recording_manifest(),
            execution_settings=LOCAL_SETTINGS,
            actor="ana",
        )

    assert refused.value.reason == UNKNOWN_PRESET
    assert refused.value.details == {"name": "absent"}
    assert "robin presets list" in refused.value.message
    assert count_rows(registered, "runs") == 0


def test_a_model_not_in_the_release_is_refused_and_nothing_is_stored(registered):
    release_versions = tuple(
        version for version in RELEASE_OUTPUT_VERSIONS if version.model != "model:owl/v4"
    )

    with pytest.raises(Refused) as refused:
        submit_run(
            registered,
            preset=None,
            configuration=OWL_CONFIGURATION,
            recordings=make_recording_manifest(),
            execution_settings=LOCAL_SETTINGS,
            actor="ana",
            release_versions=release_versions,
        )

    assert refused.value.reason == UNKNOWN_OUTPUT_VERSION
    assert (count_rows(registered, "runs"), count_rows(registered, "configurations")) == (0, 0)


def test_a_head_run_freezes_both_versions(registered):
    create_preset(registered, "perch", PERCH_CONFIGURATION, actor="ana")
    release_versions = (
        *RELEASE_OUTPUT_VERSIONS[:-1],
        ModelOutputVersion(model="head_runtime:onnx", version=3),
    )

    submitted = submit_run(
        registered,
        preset=None,
        configuration=HEAD_CONFIGURATION,
        recordings=make_recording_manifest(),
        execution_settings=LOCAL_SETTINGS,
        actor="ana",
        release_versions=release_versions,
    )

    [run] = get_rows(
        registered,
        "SELECT output_versions FROM inference_runs WHERE run_id = %s",
        submitted.run_id,
    )
    assert run["output_versions"] == {"head_runtime:onnx": 3, "model:perch/v8": 1}


def test_the_provenance_is_stored_with_the_recording_count(registered):
    submitted = submit_run(
        registered,
        preset=None,
        configuration=OWL_CONFIGURATION,
        recordings=make_recording_manifest(5),
        execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )

    [run] = get_rows(
        registered,
        "SELECT provenance FROM inference_runs WHERE run_id = %s",
        submitted.run_id,
    )
    assert run["provenance"] == PROVENANCE | {"recording_count": 5}


def test_the_same_submission_twice_makes_two_runs(registered):
    first = submit_run(
        registered, preset=None, configuration=OWL_CONFIGURATION,
        recordings=make_recording_manifest(), execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )
    second = submit_run(
        registered, preset=None, configuration=OWL_CONFIGURATION,
        recordings=make_recording_manifest(), execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )

    assert first.run_id != second.run_id
    assert (count_rows(registered, "runs"), count_rows(registered, "run_recordings")) == (2, 6)


def test_a_list_of_20000_recordings_is_stored(registered):
    submitted = submit_run(
        registered,
        preset=None,
        configuration=OWL_CONFIGURATION,
        recordings=make_recording_manifest(20000),
        execution_settings=LOCAL_SETTINGS,
        actor="ana",
    )

    assert submitted.recording_count == 20000
    assert count_rows(registered, "run_recordings") == 20000


def make_s3_manifest(count: int = 3, provenance=PROVENANCE) -> RecordingManifest:
    recordings = tuple(
        RequestedRecording(
            namespace="survey", value=f"{index:05}", audio_uri=f"s3://audio/{index:05}.wav"
        )
        for index in range(count)
    )
    return RecordingManifest(recordings=recordings, provenance=dict(provenance))


def submit_configuration(
    database, recordings, execution_settings, configuration=OWL_CONFIGURATION
):
    return submit_run(
        database,
        preset=None,
        configuration=configuration,
        recordings=recordings,
        execution_settings=execution_settings,
        actor="ana",
    )


def register_with_files(database, name: str, **uris: str) -> None:
    """Register the fixture model `name`, with the file of each role in `uris` moved."""
    model = read_pinned_model(name)
    files = {
        role: pinned.model_copy(update={"uri": uris.get(role, pinned.uri)})
        for role, pinned in model.files.items()
    }
    register_model(database, model.model_copy(update={"files": files}), actor="ana")


def get_refusal(*arguments) -> Refused:
    with pytest.raises(Refused) as refused:
        submit_configuration(*arguments)
    return refused.value


def test_submit_freezes_the_execution_settings(registered):
    submitted = submit_configuration(registered, make_s3_manifest(), KUBERNETES_SETTINGS)

    [run] = get_rows(
        registered, "SELECT execution_settings FROM runs WHERE run_id = %s", submitted.run_id
    )
    assert run["execution_settings"] == KUBERNETES_SETTINGS


def test_submit_stores_the_source_inside_the_provenance(registered):
    provenance = {
        "input": "selection",
        "catalog": "csv-list",
        "selection": {"site": "a"},
        "source": {"file": "/data/lab/recordings.csv", "file_sha256": "sha256:" + "2" * 64},
    }

    submitted = submit_configuration(
        registered, make_s3_manifest(2, provenance), LOCAL_SETTINGS
    )

    [run] = get_rows(
        registered, "SELECT provenance FROM inference_runs WHERE run_id = %s", submitted.run_id
    )
    assert run["provenance"] == provenance | {"recording_count": 2}


def test_a_local_recording_under_kubernetes_is_refused_and_no_run_is_stored(registered):
    refusal = get_refusal(registered, make_recording_manifest(25), KUBERNETES_SETTINGS)

    assert refusal.reason == UNSUPPORTED_LOCATION
    assert len(refusal.details["problems"]) == 20
    assert refusal.details["truncated"] is True
    assert refusal.details["problems"][0] == {
        "role": "recording", "location": "file:///data/00000.wav", "accepted_forms": ANY,
    }
    assert (count_rows(registered, "runs"), count_rows(registered, "configurations")) == (0, 0)


def test_a_local_model_file_under_kubernetes_is_refused(migrated_database):
    register_model(migrated_database, read_pinned_model("perch-v8"), actor="ana")
    register_with_files(migrated_database, "songbird-head-v1", graph="file:///models/head.onnx")
    create_preset(migrated_database, "perch", PERCH_CONFIGURATION, actor="ana")

    refusal = get_refusal(
        migrated_database, make_s3_manifest(), KUBERNETES_SETTINGS, HEAD_CONFIGURATION
    )

    assert refusal.reason == UNSUPPORTED_LOCATION
    assert refusal.details["problems"] == [
        {"role": "model_file", "location": "file:///models/head.onnx", "accepted_forms": ANY},
    ]
    assert count_rows(migrated_database, "runs") == 0


def test_a_robin_models_registry_is_accepted_under_kubernetes(migrated_database):
    register_with_files(
        migrated_database, "owl-v4", taxa_registry="robin-models:owl/v4/taxa_registry.csv"
    )

    submitted = submit_configuration(migrated_database, make_s3_manifest(), KUBERNETES_SETTINGS)

    assert submitted.recording_count == 3
