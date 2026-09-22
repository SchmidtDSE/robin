import typing
from collections.abc import Iterator
from pathlib import Path

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
