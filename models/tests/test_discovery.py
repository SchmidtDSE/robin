"""The installed distribution: what it registers, what listing imports, and its resources."""

import importlib.util
import subprocess
import sys
from importlib.resources import as_file, files

import pytest

from robin_contracts.cards import model_ref, read_card
from robin_contracts.protocols import ModelContext
from robin_inference_engine import errors
from robin_inference_engine.construct_model import construct_model, installed_models

RUNTIME_MODULES = ("robin_models.owl.adapter", "tensorflow", "sox_tensorflow")


def bundled_card():
    with as_file(files("robin_models.owl") / "resources" / "card.yaml") as path:
        return read_card(path)


def modules_loaded_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter and return which runtime modules it imported."""
    report = f"import sys; print(','.join(m for m in {RUNTIME_MODULES!r} if m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", f"{code}\n{report}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {name for name in result.stdout.strip().split(",") if name}


def test_owl_v4_is_installed():
    assert "owl/v4" in installed_models()


def test_listing_models_imports_no_adapter_and_no_runtime():
    code = (
        "from robin_inference_engine.construct_model import installed_models\n"
        "assert 'owl/v4' in installed_models()"
    )
    assert modules_loaded_after(code) == set()


def test_importing_the_owl_package_imports_no_adapter_and_no_runtime():
    assert modules_loaded_after("import robin_models.owl") == set()


def test_the_bundled_card_reads_as_owl_v4():
    assert model_ref(bundled_card()).id == "owl/v4"


runtime_missing = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is not None,
    reason="the OWL runtime is installed",
)


@runtime_missing
def test_importing_the_adapter_without_its_runtime_names_the_extra():
    with pytest.raises(ModuleNotFoundError, match=r"robin-models\[owl\]"):
        importlib.import_module("robin_models.owl.adapter")


@runtime_missing
def test_constructing_owl_without_its_runtime_is_an_unloadable_entry_point(tmp_path):
    card = bundled_card()
    context = ModelContext(
        card=card,
        registry=None,
        files={},
        settings={},
        resources={},
        scratch_dir=tmp_path,
        emit_embeddings=False,
    )
    with pytest.raises(errors.EngineError) as caught:
        construct_model(ref=model_ref(card), context=context)
    assert caught.value.code == errors.MODEL_ENTRY_POINT_UNLOADABLE
    assert caught.value.stage == errors.CONSTRUCT_MODEL
    assert "robin-models[owl]" in caught.value.detail
