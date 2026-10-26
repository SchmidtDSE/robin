import pytest
from pydantic import ValidationError

from robin_contracts.ports.catalog import (
    CatalogResult,
    RequestedRecording,
    serialize_source,
)

RECORDING = {"namespace": "ns", "value": "a", "audio_uri": "s3://b/a.wav"}


@pytest.mark.parametrize(
    "value", [float("nan"), {1, 2}, 10**400], ids=["nan", "set", "integer-outside-float-range"]
)
def test_catalog_result_refuses_a_source_that_is_not_json(value):
    with pytest.raises(ValidationError):
        CatalogResult(recordings=(), source={"key": value})


def test_source_text_over_64_kib_is_refused():
    assert serialize_source({"text": "a" * 1000})
    with pytest.raises(ValueError):
        serialize_source({"text": "a" * 64 * 1024})


def test_catalog_result_refuses_a_recording_with_embeddings():
    with pytest.raises(ValidationError):
        CatalogResult(recordings=({**RECORDING, "embeddings": "s3://b/a.parquet"},))


@pytest.mark.parametrize("field", ["namespace", "value", "audio_uri"])
@pytest.mark.parametrize("text", ["a\u0000", "a\ud800"], ids=["nul", "lone-surrogate"])
def test_requested_recording_refuses_text_that_postgresql_cannot_store(field, text):
    with pytest.raises(ValidationError):
        RequestedRecording(**{**RECORDING, field: text})
