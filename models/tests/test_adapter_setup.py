"""The resource checks that every adapter shares."""

import pytest
from pydantic import PositiveInt

from robin_models._adapter_support.adapter_setup import Resources, get_resources


class ExampleResources(Resources):
    batch_size: PositiveInt = 8


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        pytest.param({}, ExampleResources(batch_size=8, device="cpu"), id="none_given"),
        pytest.param(
            {"batch_size": 2, "read_block_seconds": 5.0},
            ExampleResources(batch_size=2),
            id="another_adapters_key",
        ),
    ],
)
def test_the_resources_are_read_and_other_keys_are_ignored(resources, expected):
    assert get_resources(ExampleResources, resources, "model test/1") == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", True),
        ("batch_size", 8.0),
        ("batch_size", "8"),
        ("batch_size", None),
        ("device", "GPU"),
        ("device", "/GPU:0"),
        ("device", "CPU"),
        ("device", ""),
        ("device", 0),
        ("device", None),
    ],
    ids=repr,
)
def test_a_bad_resource_is_refused(name, value):
    with pytest.raises(ValueError, match=name):
        get_resources(ExampleResources, {name: value}, "model test/1")


def test_every_bad_resource_is_refused_at_once():
    with pytest.raises(ValueError) as caught:
        get_resources(ExampleResources, {"batch_size": 0, "device": "tpu"}, "model test/1")
    assert "batch_size" in str(caught.value)
    assert "device" in str(caught.value)
