import csv
import hashlib
from pathlib import Path

import pytest

from robin_contracts.registry import RegistryEntry
from robin_inference_engine.registry import load_registry

COLUMNS = [
    "class_index",
    "label",
    "label_kind",
    "scientific_name",
    "common_name",
    "gbif_taxon_key",
]


def taxon(class_index: int, label: str) -> dict:
    row = {
        "class_index": class_index,
        "label": label,
        "label_kind": "taxon",
        "scientific_name": f"Aegolius {label.lower()}",
        "common_name": f"{label} owl",
        "gbif_taxon_key": 5232207 + class_index,
    }
    return row


def write_registry(tmp_path: Path, rows: list[dict], columns: list[str] = COLUMNS) -> Path:
    path = tmp_path / "taxa_registry.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_registry_fingerprints_the_exact_bytes(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AEAC"), taxon(1, "BRCA")])

    registry = load_registry(path)

    assert registry.fingerprint == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    prefix, digest = registry.fingerprint.split(":")
    assert prefix == "sha256"
    assert len(digest) == 64


def test_load_registry_reads_entries_in_class_index_order(tmp_path):
    path = write_registry(tmp_path, [taxon(2, "CCCC"), taxon(0, "AAAA"), taxon(1, "BBBB")])

    registry = load_registry(path)

    assert [entry.class_index for entry in registry.entries] == [0, 1, 2]
    assert [entry.label for entry in registry.entries] == ["AAAA", "BBBB", "CCCC"]


def test_load_registry_rejects_a_gap_in_class_index(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA"), taxon(2, "CCCC")])

    with pytest.raises(ValueError, match="dense 0-based range"):
        load_registry(path)


def test_load_registry_rejects_a_duplicate_class_index(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA"), taxon(0, "BBBB")])

    with pytest.raises(ValueError, match="dense 0-based range"):
        load_registry(path)


def test_load_registry_rejects_a_duplicate_label(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA"), taxon(1, "AAAA")])

    with pytest.raises(ValueError, match="duplicate label"):
        load_registry(path)


def test_load_registry_rejects_an_empty_label(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA"), taxon(1, "BBBB") | {"label": ""}])

    with pytest.raises(ValueError, match="label must not be empty"):
        load_registry(path)


def test_load_registry_rejects_a_missing_label_kind_column(tmp_path):
    columns = [column for column in COLUMNS if column != "label_kind"]
    rows = [{key: value for key, value in taxon(0, "AAAA").items() if key != "label_kind"}]
    path = write_registry(tmp_path, rows, columns)

    with pytest.raises(ValueError, match="missing required column"):
        load_registry(path)


@pytest.mark.parametrize("column", COLUMNS)
def test_load_registry_rejects_duplicate_interpreted_columns(tmp_path, column):
    path = tmp_path / "taxa_registry.csv"
    row = taxon(0, "AAAA")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS + [column])
        writer.writerow([row[name] for name in COLUMNS] + [row[column]])

    with pytest.raises(ValueError, match=f"duplicate column {column}") as failure:
        load_registry(path)

    assert str(path) in str(failure.value)


def test_load_registry_rejects_an_unknown_label_kind(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA") | {"label_kind": "species"}])

    with pytest.raises(ValueError, match="label_kind must be"):
        load_registry(path)


def test_load_registry_does_not_infer_label_kind_from_a_gbif_key(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA") | {"label_kind": ""}])

    with pytest.raises(ValueError, match="label_kind must be"):
        load_registry(path)


def test_load_registry_requires_a_scientific_name_for_a_taxon(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA") | {"scientific_name": ""}])

    with pytest.raises(ValueError, match="scientific_name"):
        load_registry(path)


def test_load_registry_requires_a_gbif_key_for_a_taxon(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA") | {"gbif_taxon_key": ""}])

    with pytest.raises(ValueError, match="gbif_taxon_key"):
        load_registry(path)


def test_load_registry_rejects_a_non_integer_gbif_key(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA") | {"gbif_taxon_key": "5232207.0"}])

    with pytest.raises(ValueError, match="gbif_taxon_key must be an integer"):
        load_registry(path)


@pytest.mark.parametrize("field", ["scientific_name", "gbif_taxon_key"])
def test_load_registry_rejects_taxon_fields_on_a_non_taxonomic_row(tmp_path, field):
    row = taxon(0, "RAIN") | {
        "label_kind": "non_taxonomic",
        "scientific_name": "",
        "gbif_taxon_key": "",
    }
    path = write_registry(tmp_path, [row | {field: taxon(0, "RAIN")[field]}])

    with pytest.raises(ValueError, match="non_taxonomic"):
        load_registry(path)


def test_load_registry_accepts_an_unresolved_taxon(tmp_path):
    row = taxon(0, "AAAA") | {"label_kind": "unresolved", "gbif_taxon_key": ""}
    path = write_registry(tmp_path, [row])

    entry = load_registry(path).by_index(0)

    assert entry.label_kind == "unresolved"
    assert entry.scientific_name == "Aegolius aaaa"
    assert entry.gbif_taxon_key is None


def test_load_registry_requires_a_scientific_name_for_an_unresolved_taxon(tmp_path):
    row = taxon(0, "AAAA") | {
        "label_kind": "unresolved",
        "scientific_name": "",
        "gbif_taxon_key": "",
    }
    path = write_registry(tmp_path, [row])

    with pytest.raises(ValueError, match="unresolved taxon requires a scientific_name"):
        load_registry(path)


def test_load_registry_rejects_a_gbif_key_on_an_unresolved_taxon(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA") | {"label_kind": "unresolved"}])

    with pytest.raises(ValueError, match="must leave gbif_taxon_key empty"):
        load_registry(path)


def test_load_registry_tolerates_columns_it_does_not_read(tmp_path):
    columns = COLUMNS + ["model", "rank", "kind", "match_type"]
    row = taxon(0, "AEAC") | {
        "model": "owl",
        "rank": "SPECIES",
        "kind": "species",
        "match_type": "EXACT",
    }
    path = write_registry(tmp_path, [row], columns)

    registry = load_registry(path)

    assert registry.entries == (
        RegistryEntry(
            class_index=0,
            label="AEAC",
            label_kind="taxon",
            scientific_name="Aegolius aeac",
            common_name="AEAC owl",
            gbif_taxon_key=5232207,
        ),
    )


def test_load_registry_does_not_read_the_kind_column(tmp_path):
    columns = COLUMNS + ["kind"]
    path = write_registry(tmp_path, [taxon(0, "AEAC") | {"kind": "non_taxon"}], columns)

    registry = load_registry(path)

    assert registry.by_index(0).label_kind == "taxon"
    assert registry.by_index(0).gbif_taxon_key == 5232207


def test_load_registry_rejects_a_non_integer_class_index(tmp_path):
    path = write_registry(tmp_path, [taxon(0, "AAAA"), taxon(1, "BBBB") | {"class_index": "one"}])

    with pytest.raises(ValueError) as failure:
        load_registry(path)

    assert str(path) in str(failure.value)
    assert "row 2" in str(failure.value)
    assert "class_index" in str(failure.value)


def test_load_registry_rejects_a_short_row_as_a_value_error(tmp_path):
    path = tmp_path / "taxa_registry.csv"
    path.write_text(",".join(COLUMNS) + "\n0,AAAA,taxon,Aegolius aaaa,AAAA owl,5232207\nBBBB\n")

    with pytest.raises(ValueError) as failure:
        load_registry(path)

    assert str(path) in str(failure.value)
    assert "row 2" in str(failure.value)


def test_load_registry_rejects_an_empty_registry(tmp_path):
    path = write_registry(tmp_path, [])

    with pytest.raises(ValueError, match="no rows"):
        load_registry(path)
