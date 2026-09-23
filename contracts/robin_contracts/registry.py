"""The label binding that gives a model's output positions their meaning."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One declared class: its position in the output vector, and what it means."""

    class_index: int
    label: str
    label_kind: Literal["taxon", "unresolved", "non_taxonomic"]
    scientific_name: str | None = None
    common_name: str | None = None
    gbif_taxon_key: int | None = None


@dataclass(frozen=True, slots=True)
class TaxonRegistry:
    """A verified registry: its entries in class_index order, and the digest of the
    bytes they were read from.

    Lookup tables are built once at construction. The label check runs per accepted
    score, where rebuilding them would be quadratic in a registry of several thousand
    entries.
    """

    fingerprint: str
    entries: tuple[RegistryEntry, ...]
    _by_label: dict[str, RegistryEntry] = field(init=False, repr=False, compare=False)
    _labels: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_label = {entry.label: entry for entry in self.entries}
        object.__setattr__(self, "_by_label", by_label)
        object.__setattr__(self, "_labels", frozenset(by_label))

    @property
    def labels(self) -> frozenset[str]:
        """Every declared label."""
        return self._labels

    def by_label(self, label: str) -> RegistryEntry:
        """The entry declaring `label`."""
        return self._by_label[label]

    def by_index(self, class_index: int) -> RegistryEntry:
        """The entry at output position `class_index`."""
        return self.entries[class_index]
