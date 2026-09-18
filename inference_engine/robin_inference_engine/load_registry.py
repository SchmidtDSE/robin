"""Reading and verifying the CSV that binds a model's output positions to classes."""

import csv
import hashlib
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from robin_contracts.registry import RegistryEntry, TaxonRegistry

REQUIRED_COLUMNS = ("class_index", "label", "label_kind")
READ_COLUMNS = REQUIRED_COLUMNS + ("scientific_name", "common_name", "gbif_taxon_key")
LABEL_KINDS = ("taxon", "unresolved", "non_taxonomic")


def load_registry(path: Path) -> TaxonRegistry:
    """Read a registry CSV and return its verified entries with a digest of its bytes.

    Every check here refuses rather than repairs, because each failure it catches would
    otherwise reach a published table looking like data instead of a bug.
    """
    data = path.read_bytes()
    fingerprint = "sha256:" + hashlib.sha256(data).hexdigest()

    # utf-8-sig drops a byte-order mark, which would otherwise be read as part of
    # the first header name and report an encoding fault as a missing column.
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    _check_header(path, reader.fieldnames or ())

    rows = list(reader)
    if not rows:
        raise ValueError(f"registry {path}: has no rows, so no output position has a meaning")

    entries = [_read_row(path, position, row) for position, row in enumerate(rows, start=1)]
    entries.sort(key=lambda entry: entry.class_index)
    _check_labels_are_unique(path, entries)
    _check_class_index_is_dense(path, entries)
    return TaxonRegistry(fingerprint=fingerprint, entries=tuple(entries))


def _check_header(path: Path, columns: Sequence[str]) -> None:
    """The header decides what every row means, so it is checked before any row is read."""
    for column in READ_COLUMNS:
        if columns.count(column) > 1:
            raise ValueError(f"registry {path}: duplicate column {column}")

    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"registry {path}: missing required column {', '.join(missing)}")


def _read_row(path: Path, position: int, row: dict[str, str | None]) -> RegistryEntry:
    # A cell is absent whether it is empty or missing because the row is short, which
    # DictReader reports as '' and None respectively.
    values = {column: row.get(column) or None for column in READ_COLUMNS}

    class_index = _parse_class_index(path, position, values["class_index"])
    label_kind = _parse_label_kind(path, class_index, values["label_kind"])
    gbif_taxon_key = _parse_gbif_taxon_key(path, class_index, values["gbif_taxon_key"])
    scientific_name = values["scientific_name"]
    _check_kind_obligations(path, class_index, label_kind, scientific_name, gbif_taxon_key)

    label = values["label"]
    if label is None:
        raise ValueError(f"registry {path}: class_index {class_index}: label must not be empty")

    return RegistryEntry(
        class_index=class_index,
        label=label,
        label_kind=label_kind,
        scientific_name=scientific_name,
        common_name=values["common_name"],
        gbif_taxon_key=gbif_taxon_key,
    )


def _parse_class_index(path: Path, position: int, value: str | None) -> int:
    # The row cannot be named by its class_index here: that is the field that failed.
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"registry {path}: row {position}: class_index must be an integer, got {value!r}"
        ) from None


def _parse_label_kind(
    path: Path, class_index: int, value: str | None
) -> Literal["taxon", "unresolved", "non_taxonomic"]:
    if value not in LABEL_KINDS:
        raise ValueError(
            f"registry {path}: class_index {class_index}: label_kind must be "
            f"{' or '.join(repr(kind) for kind in LABEL_KINDS)}, got {value!r}"
        )
    return value


def _parse_gbif_taxon_key(path: Path, class_index: int, value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"registry {path}: class_index {class_index}: gbif_taxon_key must be an "
            f"integer, got {value!r}"
        ) from None


def _check_kind_obligations(
    path: Path,
    class_index: int,
    label_kind: Literal["taxon", "unresolved", "non_taxonomic"],
    scientific_name: str | None,
    gbif_taxon_key: int | None,
) -> None:
    # A common_name is allowed on every kind and is deliberately not checked here.
    where = f"registry {path}: class_index {class_index}"
    if label_kind == "taxon":
        if scientific_name is None:
            raise ValueError(f"{where}: a taxon requires a scientific_name")
        if gbif_taxon_key is None:
            raise ValueError(f"{where}: a taxon requires a gbif_taxon_key")
        return

    # An unresolved class is an organism the registry could not bind to a GBIF key.
    # It keeps its scientific name: that name is what a reader has, and dropping it
    # would make the row indistinguishable from a class that is not an organism.
    if label_kind == "unresolved":
        if scientific_name is None:
            raise ValueError(f"{where}: an unresolved taxon requires a scientific_name")
        if gbif_taxon_key is not None:
            raise ValueError(
                f"{where}: an unresolved taxon must leave gbif_taxon_key empty; "
                "a row carrying a key is a taxon"
            )
        return

    if scientific_name is not None or gbif_taxon_key is not None:
        raise ValueError(
            f"{where}: a non_taxonomic class must leave scientific_name and "
            "gbif_taxon_key empty"
        )


def _check_labels_are_unique(path: Path, entries: list[RegistryEntry]) -> None:
    # Counted over the rows, not over a label-keyed dict, which would already have
    # collapsed a duplicate pair into one key.
    seen: set[str] = set()
    for entry in entries:
        if entry.label in seen:
            raise ValueError(f"registry {path}: duplicate label {entry.label!r}")
        seen.add(entry.label)


def _check_class_index_is_dense(path: Path, entries: list[RegistryEntry]) -> None:
    """A gap or a duplicate silently relabels every class after it with its neighbour."""
    for expected, entry in enumerate(entries):
        if entry.class_index != expected:
            raise ValueError(
                f"registry {path}: class_index must be a dense 0-based range, expected "
                f"{expected} at position {expected}, got {entry.class_index}"
            )
