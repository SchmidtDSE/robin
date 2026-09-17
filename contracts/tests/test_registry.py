import pytest

from robin_contracts.registry import RegistryEntry, TaxonRegistry

FINGERPRINT = "sha256:" + "0" * 64


def build_registry(*labels: str) -> TaxonRegistry:
    entries = tuple(
        RegistryEntry(class_index=index, label=label, label_kind="non_taxonomic")
        for index, label in enumerate(labels)
    )
    return TaxonRegistry(fingerprint=FINGERPRINT, entries=entries)


def test_by_label_and_by_index_resolve_the_same_entry():
    registry = build_registry("rain", "wind", "noise")

    assert registry.by_label("wind") is registry.by_index(1)
    assert registry.by_index(1).class_index == 1


def test_labels_carries_every_declared_label():
    registry = build_registry("rain", "wind", "noise")

    assert registry.labels == frozenset({"rain", "wind", "noise"})


def test_by_label_raises_for_an_undeclared_label():
    registry = build_registry("rain")

    with pytest.raises(KeyError):
        registry.by_label("wind")


def test_equal_registries_are_equal_and_hash_alike():
    one = build_registry("rain", "wind")
    other = build_registry("rain", "wind")

    assert one == other
    assert hash(one) == hash(other)
