"""Output versions: what a run freezes at submit, and which versions are no longer current."""

import pytest

from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.errors import Refused
from robin_run_manager.versions import (
    RELEASE_OUTPUT_VERSIONS,
    get_frozen_output_versions,
    get_outdated_versions,
)

from model_fixtures import read_pinned_model

OWL = read_pinned_model("owl-v4")
PERCH = read_pinned_model("perch-v8")
HEAD = read_pinned_model("songbird-head-v1")


def get_release_versions_without(model_id: str) -> tuple[ModelOutputVersion, ...]:
    return tuple(version for version in RELEASE_OUTPUT_VERSIONS if version.model != model_id)


def test_an_audio_run_freezes_its_model_version():
    assert get_frozen_output_versions([OWL.card], RELEASE_OUTPUT_VERSIONS) == {"model:owl/v4": 1}


def test_a_head_run_freezes_the_head_runtime_and_the_backbone():
    assert get_frozen_output_versions([HEAD.card, PERCH.card], RELEASE_OUTPUT_VERSIONS) == {
        "head_runtime:onnx": 1,
        "model:perch/v8": 1,
    }


def test_a_model_that_the_release_does_not_have_is_refused():
    with pytest.raises(Refused) as refused:
        get_frozen_output_versions([OWL.card], get_release_versions_without("model:owl/v4"))

    assert refused.value.reason == "unknown_output_version"
    assert refused.value.details == {"output_version_id": "model:owl/v4"}
    assert "model:owl/v4" in refused.value.message


def test_a_head_whose_backbone_the_release_does_not_have_is_refused():
    release_versions = get_release_versions_without("model:perch/v8")

    with pytest.raises(Refused) as refused:
        get_frozen_output_versions([HEAD.card, PERCH.card], release_versions)

    assert refused.value.reason == "unknown_output_version"
    assert refused.value.details == {"output_version_id": "model:perch/v8"}


def test_get_outdated_versions_lists_each_changed_and_missing_version():
    release_versions = (
        ModelOutputVersion(model="model:perch/v8", version=2),
        ModelOutputVersion(model="model:owl/v4", version=1),
    )
    run = {"model:perch/v8": 1, "head_runtime:onnx": 1, "model:owl/v4": 1}

    assert get_outdated_versions(run, release_versions) == {
        "model:perch/v8": {"run": 1, "release": 2},
        "head_runtime:onnx": {"run": 1, "release": None},
    }
    assert get_outdated_versions({"model:owl/v4": 1}, release_versions) == {}
