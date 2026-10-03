import typing
from collections.abc import Iterator
from pathlib import Path

import pytest

from robin_contracts.cards import ModelCard
from robin_contracts.protocols import Model, ModelContext, ScoreRetention, noop
from robin_contracts.records import WindowOutput

REGISTRY_DIGEST = "sha256:" + "a" * 64

CARD = ModelCard.model_validate({
    "model_name": "owl",
    "model_version": "1",
    "runtime": "tensorflow",
    "window_duration": 3.0,
    "window_overlap": 0.0,
    "sample_rate": 32000,
    "min_detection_threshold": 0.0,
    "score_domain": "probability",
    "taxa_registry_digest": REGISTRY_DIGEST,
    "audio": {
        "downmix": "mean",
        "resampler": {"by": "runner", "algorithm": "soxr_hq"},
        "pad": "drop",
    },
    "backend": "tensorflow",
    "dtype": "float32",
})


class Adapter:
    """A model that satisfies the protocol with its three methods and nothing else."""

    def run(self, input) -> Iterator[WindowOutput]:
        yield WindowOutput(start=0.0, end=3.0)

    def after_recording(self) -> None:
        return None

    def clean_up(self) -> None:
        return None


class AdapterWithoutRun:
    def after_recording(self) -> None:
        return None

    def clean_up(self) -> None:
        return None


def build_context(**overrides) -> ModelContext:
    fields = {
        "card": CARD,
        "registry": None,
        "files": {},
        "settings": {},
        "resources": {},
        "scratch_dir": Path("/tmp"),
        "emit_embeddings": False,
    }
    return ModelContext(**(fields | overrides))


def test_an_adapter_satisfying_the_protocol_needs_no_failed_windows():
    adapter = Adapter()

    assert not hasattr(adapter, "failed_windows")
    assert isinstance(adapter, Model)


def test_an_adapter_reports_nothing_its_card_states():
    assert set(typing.get_type_hints(Model)) == set()


def test_an_adapter_missing_run_does_not_satisfy_the_protocol():
    assert not isinstance(AdapterWithoutRun(), Model)


def test_noop_accepts_a_line_and_returns_none():
    assert noop("a line the caller wanted discarded") is None


def test_model_context_logs_nowhere_by_default():
    assert build_context().log is noop


def test_model_context_holds_every_pinned_file_by_role():
    registry = Path("/tmp/taxa.csv")
    context = build_context(files={"weights": Path("/tmp/owl.h5"), "taxa_registry": registry})

    assert context.files["taxa_registry"] == registry


def test_score_retention_declares_exactly_three_modes():
    assert typing.get_args(ScoreRetention) == ("full", "thresholded", "top_k")


def test_model_context_carries_the_works_resources():
    resources = {"batch_size": 8, "device": "cpu"}

    assert build_context(resources=resources).resources == resources


def test_model_context_must_be_given_resources():
    fields = {
        "card": CARD,
        "registry": None,
        "files": {},
        "settings": {},
        "scratch_dir": Path("/tmp"),
        "emit_embeddings": False,
    }

    with pytest.raises(TypeError, match="resources"):
        ModelContext(**fields)
