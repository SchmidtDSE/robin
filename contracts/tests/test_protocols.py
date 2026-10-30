from pathlib import Path

import pytest

from robin_contracts.cards import ModelCard
from robin_contracts.protocols import Model, ModelContext

REGISTRY_DIGEST = "sha256:" + "a" * 64

CARD = ModelCard.model_validate({
    "model_name": "owl",
    "model_version": "1",
    "runtime": "tensorflow",
    "window_duration": 3.0,
    "sample_rate": 32000,
    "min_detection_threshold": 0.0,
    "score_domain": "sigmoid",
    "taxa_registry_digest": REGISTRY_DIGEST,
    "audio": {
        "downmix": "mean",
        "resampler": {"by": "runner", "algorithm": "soxr_hq"},
        "pad": "drop",
    },
    "backend": "tensorflow",
    "dtype": "float32",
})


class AdapterWithoutRun:
    def after_recording(self) -> None:
        return None

    def clean_up(self) -> None:
        return None


def test_an_adapter_missing_run_does_not_satisfy_the_protocol():
    assert not isinstance(AdapterWithoutRun(), Model)


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
