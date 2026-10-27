import pytest
from pydantic import BaseModel, Field, ValidationError

from robin_contracts.ports.catalog import (
    CatalogResult,
    InvalidCatalogSettings,
    RequestedRecording,
    check_settings,
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


class _Settings(BaseModel, extra="forbid"):
    namespace: str = Field(min_length=1)
    token: int


def test_check_settings_returns_the_checked_settings():
    assert check_settings(_Settings, {"namespace": "lab", "token": 4}) == _Settings(
        namespace="lab", token=4
    )


def test_check_settings_names_each_field_but_not_its_value():
    with pytest.raises(InvalidCatalogSettings) as raised:
        check_settings(_Settings, {"namespace": "", "token": "hunter2", "sheet": "1"})

    message = str(raised.value)
    assert "namespace:" in message and "token:" in message and "sheet:" in message
    assert "hunter2" not in message
    assert raised.value.__context__ is None
