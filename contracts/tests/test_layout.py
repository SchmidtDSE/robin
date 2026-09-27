"""Where a published artifact sits below a writer's root."""

from typing import get_args
from urllib.parse import unquote

import pytest

from robin_contracts.layout import FILE_NAMES, artifact_path
from robin_contracts.results import ArtifactKind


def test_the_path_names_the_kind_the_recording_and_the_file():
    assert (
        artifact_path("scores", "soundhub", "42")
        == "scores/recording_namespace=soundhub/recording_value=42/scores.arrow"
    )


@pytest.mark.parametrize("awkward", ["a:b", "x/y", "k=v", "p%20q", "sp ace", "é", ".."])
def test_awkward_values_stay_inside_their_own_segment_and_decode_exactly(awkward):
    segments = artifact_path("embeddings", awkward, awkward).split("/")

    assert len(segments) == 4
    for segment, key in zip(segments[1:3], ["recording_namespace", "recording_value"]):
        name, _, encoded = segment.partition("=")
        assert name == key
        assert unquote(encoded) == awkward


def test_non_ascii_is_percent_encoded_as_utf8_and_unreserved_characters_pass_through():
    assert artifact_path("scores", "é", "-._~").split("/")[1:3] == [
        "recording_namespace=%C3%A9",
        "recording_value=-._~",
    ]


def test_every_kind_has_a_file_name():
    assert set(FILE_NAMES) == set(get_args(ArtifactKind))
    assert artifact_path("detections", "n", "v").endswith("/detections.parquet")
