import dataclasses
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest

from robin_contracts.cards import ModelCard
from robin_contracts.protocols import (
    EmbeddingDtype,
    Model,
    ModelCapabilities,
    ModelContext,
    ScoreRetention,
    noop,
)
from robin_contracts.records import WindowOutput

CARD = ModelCard(
    model_name="owl",
    model_version="1",
    runtime="tensorflow",
    segment_duration=3.0,
    sample_rate=32000,
    min_detection_threshold=0.0,
)

CAPABILITIES = ModelCapabilities(emits_scores=True, emits_embeddings=False)


class Adapter:
    """A model that satisfies the protocol without reporting lost windows."""

    recipe = "a recipe"
    capabilities = CAPABILITIES

    def run(self, input) -> Iterator[WindowOutput]:
        yield WindowOutput(recording_index=0, start=0.0, end=3.0)

    def after_recording(self) -> None:
        return None

    def clean_up(self) -> None:
        return None


class AdapterWithoutRun:
    recipe = "a recipe"
    capabilities = CAPABILITIES

    def after_recording(self) -> None:
        return None

    def clean_up(self) -> None:
        return None


def build_context(**overrides) -> ModelContext:
    fields = {
        "card": CARD,
        "registry": None,
        "weights": {},
        "settings": {},
        "scratch_dir": Path("/tmp"),
        "emit_embeddings": False,
    }
    return ModelContext(**(fields | overrides))


def test_an_adapter_satisfying_the_protocol_needs_no_failed_windows():
    adapter = Adapter()

    assert not hasattr(adapter, "failed_windows")
    assert isinstance(adapter, Model)


def test_an_adapter_missing_run_does_not_satisfy_the_protocol():
    assert not isinstance(AdapterWithoutRun(), Model)


def test_noop_accepts_a_line_and_returns_none():
    assert noop("a line the caller wanted discarded") is None


def test_model_context_logs_nowhere_by_default():
    assert build_context().log is noop


def test_score_retention_declares_exactly_three_modes():
    assert typing.get_args(ScoreRetention) == ("full", "thresholded", "top_k")


def test_embedding_dtype_declares_exactly_the_two_widths():
    assert typing.get_args(EmbeddingDtype) == ("float16", "float32")


def test_capabilities_declare_no_embedding_dtype_unless_an_adapter_sets_one():
    assert CAPABILITIES.embedding_dtype is None


def test_capabilities_declare_no_top_k_cap_by_default():
    assert CAPABILITIES.native_top_k is None


def test_capabilities_have_no_vocabulary_field():
    with pytest.raises(TypeError):
        ModelCapabilities(emits_scores=True, emits_embeddings=False, vocabulary=("a",))


def test_a_capped_instance_supports_only_top_k_retention():
    capped = ModelCapabilities(
        emits_scores=True,
        emits_embeddings=False,
        supported_retention=frozenset({"top_k"}),
        native_top_k=5,
    )

    assert capped.native_top_k == 5


def test_an_uncapped_instance_supports_any_retention_but_top_k():
    uncapped = ModelCapabilities(
        emits_scores=True,
        emits_embeddings=False,
        supported_retention=frozenset({"full", "thresholded"}),
    )

    assert uncapped.native_top_k is None


@pytest.mark.parametrize(
    ("retention", "native_top_k"),
    [
        pytest.param(frozenset({"top_k"}), None, id="top_k_without_a_cap"),
        pytest.param(frozenset({"full", "top_k"}), None, id="top_k_among_others_without_a_cap"),
        pytest.param(frozenset({"full"}), 5, id="a_cap_without_top_k"),
        pytest.param(frozenset(), 5, id="a_cap_with_no_retention"),
        pytest.param(frozenset({"thresholded", "top_k"}), 5, id="a_cap_beside_other_retention"),
        pytest.param(frozenset({"top_k"}), 0, id="a_zero_cap"),
        pytest.param(frozenset({"top_k"}), -1, id="a_negative_cap"),
        pytest.param(frozenset({"top_k"}), True, id="a_boolean_cap"),
        pytest.param(frozenset({"top_k"}), 5.0, id="a_float_cap"),
    ],
)
def test_capabilities_refuse_a_cap_that_disagrees_with_retention(retention, native_top_k):
    with pytest.raises(ValueError, match="native_top_k"):
        ModelCapabilities(
            emits_scores=True,
            emits_embeddings=False,
            supported_retention=retention,
            native_top_k=native_top_k,
        )


def test_a_valid_instance_derives_a_configured_copy_with_replace():
    uncapped = ModelCapabilities(
        emits_scores=True,
        emits_embeddings=False,
        supported_retention=frozenset({"full"}),
    )

    capped = dataclasses.replace(
        uncapped, supported_retention=frozenset({"top_k"}), native_top_k=5
    )

    assert capped.native_top_k == 5
    assert uncapped.native_top_k is None
