"""The `current_files` view: the current output files of each recording and configuration."""

import pytest

from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS
from run_fixtures import (
    HEAD_CONFIGURATION,
    NAMESPACE,
    OWL_CONFIGURATION,
    PERCH_CONFIGURATION,
    add_covered_result,
    add_output_file,
    add_task,
    register_fixture_models,
    set_outcome,
    submit,
    withdraw,
)
from run_fixtures import get_current_files as get_rows


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


def get_release(versions: dict[str, int]) -> tuple[ModelOutputVersion, ...]:
    """The release's versions, with each one that `versions` names changed."""
    return tuple(
        ModelOutputVersion(model=entry.model, version=versions.get(entry.model, entry.version))
        for entry in RELEASE_OUTPUT_VERSIONS
    )


def get_current(database) -> list[tuple]:
    """Each row as (value, run_id, up_to_date, kind)."""
    return [
        (row["value"], row["run_id"], row["up_to_date"], row["kind"]) for row in get_rows(database)
    ]


def test_a_covered_recording_is_current_with_each_of_its_files(registered):
    run_id = submit(registered, ["a"])
    add_covered_result(registered, run_id, "a", files=("embeddings", "scores"))

    rows = get_rows(registered)

    assert [row["kind"] for row in rows] == ["embeddings", "scores"]
    for row, kind in zip(rows, ["embeddings", "scores"]):
        assert set(row) == {
            "namespace", "value", "configuration_digest", "output_versions", "up_to_date",
            "kind", "contract_id", "uri", "checksum", "run_id",
        }
        assert (row["namespace"], row["value"], row["run_id"]) == (NAMESPACE, "a", run_id)
        assert row["output_versions"] == {"model:owl/v4": 1}
        assert row["up_to_date"] is True
        assert row["contract_id"] == f"robin.{kind}.parquet/1"
        assert row["uri"] == f"s3://out/{run_id}/a.{kind}.parquet"
        assert row["checksum"] == "sha256:" + "1" * 64
    assert rows[0]["configuration_digest"] == rows[1]["configuration_digest"]


def test_a_withdrawn_run_is_not_current_and_an_older_version_returns(registered):
    older = submit(registered, ["a"])
    add_covered_result(registered, older, "a")
    newer = submit(registered, ["a"], release_versions=get_release({"model:owl/v4": 2}))
    add_covered_result(registered, newer, "a")
    assert get_current(registered) == [("a", newer, True, "scores")]

    withdraw(registered, newer)

    assert get_current(registered) == [("a", older, False, "scores")]


def test_the_newest_versions_are_current(registered):
    newer = submit(registered, ["a"], release_versions=get_release({"model:owl/v4": 2}))
    add_covered_result(registered, newer, "a")
    older = submit(registered, ["a"])
    add_covered_result(registered, older, "a")

    assert get_current(registered) == [("a", newer, True, "scores")]


def test_a_head_compares_both_versions(registered):
    submit(registered, ["b"], PERCH_CONFIGURATION)  # stores the backbone's configuration
    newer = submit(
        registered, ["a"], HEAD_CONFIGURATION,
        release_versions=get_release({"head_runtime:onnx": 1, "model:perch/v8": 2}),
    )
    add_covered_result(registered, newer, "a")
    older = submit(
        registered, ["a"], HEAD_CONFIGURATION,
        release_versions=get_release({"head_runtime:onnx": 1, "model:perch/v8": 1}),
    )
    add_covered_result(registered, older, "a")

    [row] = get_rows(registered)

    assert row["run_id"] == newer
    assert row["output_versions"] == {"head_runtime:onnx": 1, "model:perch/v8": 2}
    assert row["up_to_date"] is True


@pytest.mark.parametrize("higher_state", ["covered", "withdrawn"])
def test_up_to_date_is_false_once_a_higher_version_is_recorded_for_the_model(
    registered, higher_state
):
    other_configuration = OWL_CONFIGURATION.model_copy(
        update={"settings": {"window_overlap": 0.5}}
    )
    older = submit(registered, ["a"])
    add_covered_result(registered, older, "a")
    higher = submit(
        registered, ["a"], other_configuration, release_versions=get_release({"model:owl/v4": 2})
    )
    add_covered_result(registered, higher, "a")
    if higher_state == "withdrawn":
        withdraw(registered, higher)

    current = {row["run_id"]: row["up_to_date"] for row in get_rows(registered)}

    assert current[older] is False
    assert current.get(higher) == (True if higher_state == "covered" else None)


@pytest.mark.parametrize("state", ["failed", "pending", "skipped"])
def test_a_failed_pending_or_skipped_recording_has_no_row(registered, state):
    run_id = submit(registered, ["a"])
    task_id = add_task(registered, run_id, "running")
    add_output_file(registered, run_id, "a", kind="scores")
    set_outcome(registered, run_id, "a", state, task_id=task_id)

    assert get_rows(registered) == []
