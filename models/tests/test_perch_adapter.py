"""Perch v8: its bundled card and registry."""

import csv
import hashlib
import subprocess
import sys
from collections import Counter
from importlib.resources import as_file, files

from robin_contracts.cards import model_ref, read_card
from robin_inference_engine.load_registry import load_registry

RESOURCES = files("robin_models.perch") / "resources"

with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)
    with _path.open(newline="", encoding="utf-8") as _file:
        REGISTRY_ROWS = list(csv.DictReader(_file))

REGISTRY_SHA256 = "ca28f370cb5924af9966fee9fcd0a12d57632bba85a01a3f5908f9011ef99fd9"
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
        "model_name": "perch",
        "model_version": "v8",
        "runtime": "tensorflow",
        "window_duration": 5.0,
        "window_overlap": 0.0,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": "sha256:" + REGISTRY_SHA256,
        "spectrogram_shape": None,
        "audio": {
            "downmix": "mean",
            "resampler": {"by": "runner", "algorithm": "scipy_polyphase_with_context"},
            "pad": "centre_crop_end_pad",
        },
        "backend": "pb-fp32",
        "embedding_transform": {"kind": "identity"},
        "dtype": "float32",
        "inference_params": (),
        "can_emit_embeddings": True,
        "embedding_dim": 1280,
        "embedding_dtype": "float32",
    }


def test_the_card_reads_as_perch_v8():
    assert model_ref(CARD).id == "perch/v8"


def test_the_card_names_the_bundled_registry_by_its_digest():
    digest = hashlib.sha256((RESOURCES / "taxa_registry.csv").read_bytes()).hexdigest()

    assert digest == REGISTRY_SHA256
    assert CARD.taxa_registry_digest == "sha256:" + digest


def test_the_registry_loads_with_10932_entries_taxa_and_unresolved():
    kinds = Counter(entry.label_kind for entry in REGISTRY.entries)

    assert len(REGISTRY.entries) == 10932
    assert kinds == {"taxon": 10917, "unresolved": 15}


def test_every_rows_label_kind_follows_from_its_kind():
    for row in REGISTRY_ROWS:
        expected = "unresolved" if row["kind"] == "unresolved" else "taxon"
        assert row["label_kind"] == expected, row


def test_the_unresolved_classes_are_the_fifteen_with_no_key():
    unresolved = [entry for entry in REGISTRY.entries if entry.label_kind == "unresolved"]

    assert [(e.class_index, e.label, e.scientific_name) for e in unresolved] == [
        (1590, "braeme2", "Riccordia bracei bracei"),
        (1591, "braeme3", "Riccordia bracei elegans"),
        (3468, "fotdro4", "Dicrurus divaricatus"),
        (3641, "gnfhum2", "Leucolia viridifrons wagneri"),
        (3700, "goftyr4", "Zimmerius viridiflavus flavidifrons"),
        (3721, "gollor1", "Glossoptila goldiei"),
        (5545, "masyel3", "Geothlypis semiflava chiriquensis"),
        (7108, "rebwoo4", "Hylexetastes uniformis brigidai"),
        (7178, "reepar3", "Calamornis heudei"),
        (8941, "stisha2", "Leucocarbo chalconotus stewarti"),
        (9377, "thbvir2", "Vireo pallens approximans"),
        (9683, "vemdro1", "Dicrurus atactus"),
        (9684, "vemdro5", "Dicrurus atactus"),
        (9702, "verfly8", "Pyrocephalus nanus dubius"),
        (9934, "wetsab3", "Pampa curvipennis pampa"),
    ]
    for entry in unresolved:
        assert entry.gbif_taxon_key is None


def test_every_taxon_has_a_scientific_name_and_a_gbif_key():
    for entry in REGISTRY.entries:
        if entry.label_kind == "taxon":
            assert entry.scientific_name, entry
            assert entry.gbif_taxon_key is not None, entry


def test_importing_the_perch_package_imports_no_runtime():
    assert runtime_modules_loaded_after("import robin_models.perch") == set()
