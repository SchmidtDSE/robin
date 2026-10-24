"""Submit: store a run in `preparing`, with its configuration, versions, and recording list."""

from pathlib import Path

import pytest

from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.configuration import read_configuration
from robin_run_manager.database import transaction
from robin_run_manager.errors import (
    INVALID_REQUEST,
    UNKNOWN_OUTPUT_VERSION,
    UNKNOWN_PRESET,
    Refused,
)
from robin_run_manager.models import register_model
from robin_run_manager.presets import create_preset
from robin_run_manager.recording_inputs import RecordingFile, SubmittedRecording
from robin_run_manager.submit import submit_run
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS

from model_fixtures import read_pinned_model

FIXTURES = Path(__file__).parent / "fixtures"
OWL_CONFIGURATION = read_configuration(FIXTURES / "configuration-owl-v4.yaml")
PERCH_CONFIGURATION = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
HEAD_CONFIGURATION = read_configuration(FIXTURES / "configuration-songbird-head-v1.yaml")
OBSERVED = {
    "input": "uris",
    "file_sha256": "sha256:" + "0" * 64,
    "file_size_bytes": 812,
    "namespace": "survey",
}


def make_recording_file(count: int = 3) -> RecordingFile:
    recordings = tuple(
        SubmittedRecording(
            namespace="survey",
            value=f"file:///data/{index:05}.wav",
            audio_uri=f"file:///data/{index:05}.wav",
            duration_seconds=60.0 if index % 2 else None,
        )
        for index in range(count)
    )
    return RecordingFile(recordings=recordings, observed=dict(OBSERVED))


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
    submitted_list = make_recording_file()

    submitted = submit_run(
        registered, preset="owl", configuration=None, recordings=submitted_list, actor="ben"
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
        registered, preset="owl", configuration=None, recordings=make_recording_file(), actor="ana"
    )
    from_file = submit_run(
        registered,
        preset=None,
        configuration=OWL_CONFIGURATION,
        recordings=make_recording_file(),
        actor="ana",
    )

    assert from_preset.configuration_digest == from_file.configuration_digest


def test_a_full_configuration_is_stored_for_later_heads(registered):
    submitted = submit_run(
        registered,
        preset=None,
        configuration=PERCH_CONFIGURATION,
        recordings=make_recording_file(),
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
            recordings=make_recording_file(),
            actor="ana",
        )

    assert refused.value.reason == INVALID_REQUEST
    assert count_rows(registered, "runs") == 0


def test_an_unknown_preset_is_refused(registered):
    with pytest.raises(Refused) as refused:
        submit_run(
            registered, preset="absent", configuration=None, recordings=make_recording_file(),
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
            recordings=make_recording_file(),
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
        recordings=make_recording_file(),
        actor="ana",
        release_versions=release_versions,
    )

    [run] = get_rows(
        registered,
        "SELECT output_versions FROM inference_runs WHERE run_id = %s",
        submitted.run_id,
    )
    assert run["output_versions"] == {"head_runtime:onnx": 3, "model:perch/v8": 1}


def test_observed_provenance_is_stored_with_the_recording_count(registered):
    submitted = submit_run(
        registered,
        preset=None,
        configuration=OWL_CONFIGURATION,
        recordings=make_recording_file(5),
        actor="ana",
    )

    [run] = get_rows(
        registered,
        "SELECT observed_provenance FROM inference_runs WHERE run_id = %s",
        submitted.run_id,
    )
    assert run["observed_provenance"] == OBSERVED | {"recording_count": 5}


def test_the_same_submission_twice_makes_two_runs(registered):
    first = submit_run(
        registered, preset=None, configuration=OWL_CONFIGURATION,
        recordings=make_recording_file(), actor="ana",
    )
    second = submit_run(
        registered, preset=None, configuration=OWL_CONFIGURATION,
        recordings=make_recording_file(), actor="ana",
    )

    assert first.run_id != second.run_id
    assert (count_rows(registered, "runs"), count_rows(registered, "run_recordings")) == (2, 6)


def test_a_list_of_20000_recordings_is_stored(registered):
    submitted = submit_run(
        registered,
        preset=None,
        configuration=OWL_CONFIGURATION,
        recordings=make_recording_file(20000),
        actor="ana",
    )

    assert submitted.recording_count == 20000
    assert count_rows(registered, "run_recordings") == 20000
