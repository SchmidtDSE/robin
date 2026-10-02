"""BirdNET v2.4: its bundled card and registry."""

import csv
import hashlib
import subprocess
import sys
from collections import Counter
from importlib.resources import as_file, files

from robin_contracts.cards import model_ref, read_card
from robin_inference_engine.load_registry import load_registry

RESOURCES = files("robin_models.birdnet") / "resources"

with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)
    with _path.open(newline="", encoding="utf-8") as _file:
        REGISTRY_ROWS = list(csv.DictReader(_file))

REGISTRY_SHA256 = "b758a58bee475d45b1fb04a8f42cbaa13c794d2dc782f0e86968e6c1515d4c2d"
RUNTIME_MODULES = ("tensorflow", "scipy", "soundfile")


def runtime_modules_loaded_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter and return which runtime modules it imported."""
    report = f"import sys; print(','.join(m for m in {RUNTIME_MODULES!r} if m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", f"{code}\n{report}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {name for name in result.stdout.strip().split(",") if name}


def test_the_card_states_every_field():
    assert CARD.model_dump() == {
        "model_name": "birdnet",
        "model_version": "v2p4",
        "runtime": "tensorflow",
        "window_duration": 3.0,
        "window_overlap": 0.0,
        "sample_rate": 48000,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": "sha256:" + REGISTRY_SHA256,
        "spectrogram_shape": None,
        "audio": {
            "downmix": "mean",
            "resampler": {"by": "runner", "algorithm": "scipy_fft_per_window"},
            "pad": "centre_crop_end_pad",
        },
        "backend": "pb-fp32",
        "embedding_transform": {"kind": "identity"},
        "dtype": "float32",
        "inference_params": (),
        "can_emit_embeddings": True,
        "embedding_dim": 1024,
        "embedding_dtype": "float32",
    }


def test_the_card_reads_as_birdnet_v2p4():
    assert model_ref(CARD).id == "birdnet/v2p4"


def test_the_card_names_the_bundled_registry_by_its_digest():
    digest = hashlib.sha256((RESOURCES / "taxa_registry.csv").read_bytes()).hexdigest()

    assert digest == REGISTRY_SHA256
    assert CARD.taxa_registry_digest == "sha256:" + digest


def test_the_registry_loads_with_6522_entries_of_three_kinds():
    kinds = Counter(entry.label_kind for entry in REGISTRY.entries)

    assert len(REGISTRY.entries) == 6522
    assert kinds == {"taxon": 6510, "non_taxonomic": 11, "unresolved": 1}


def test_every_rows_label_kind_follows_from_its_kind():
    def expected(kind: str) -> str:
        return {"non_taxon": "non_taxonomic", "unresolved": "unresolved"}.get(kind, "taxon")

    for row in REGISTRY_ROWS:
        assert row["label_kind"] == expected(row["kind"]), row


def test_the_one_unresolved_class_is_the_glossy_backed_drongo_with_no_key():
    [entry] = [entry for entry in REGISTRY.entries if entry.label_kind == "unresolved"]

    assert entry.class_index == 1918
    assert entry.label == "Dicrurus divaricatus_Glossy-backed Drongo"
    assert entry.scientific_name == "Dicrurus divaricatus"
    assert entry.gbif_taxon_key is None


def test_the_non_taxonomic_classes_have_no_name_and_no_key():
    rows = [entry for entry in REGISTRY.entries if entry.label_kind == "non_taxonomic"]

    assert len(rows) == 11
    for entry in rows:
        assert entry.scientific_name is None
        assert entry.gbif_taxon_key is None


def test_importing_the_birdnet_package_imports_no_runtime():
    assert runtime_modules_loaded_after("import robin_models.birdnet") == set()
