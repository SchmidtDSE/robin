"""The checks that every adapter makes on its card and its resources.

Each check takes `owner`, the name the error gives, such as "model owl/v4". Importing this
module loads no ML runtime.
"""

from collections.abc import Mapping
from typing import Any

from robin_contracts.cards import ModelCard
from robin_contracts.protocols import JsonScalar

DEVICE = "cpu"


def refuse_a_card_the_model_does_not_follow(
    card: ModelCard, behaviour: Mapping[str, Any], owner: str
) -> None:
    """Refuse a card that gives another value for a field whose value the model fixes."""
    for field, does in behaviour.items():
        states = getattr(card, field)
        if states != does:
            raise ValueError(f"card field {field} is {states!r}, but {owner} does {does!r}")


def get_batch_size(resources: Mapping[str, JsonScalar], default: int) -> int:
    """The `batch_size` resource, or `default`."""
    size = resources.get("batch_size", default)
    # bool is an int subclass, and True is a type error, not a batch of one.
    if type(size) is not int or size < 1:
        raise ValueError(f"resource batch_size must be a positive int, got {size!r}")
    return size


def check_device(resources: Mapping[str, JsonScalar], owner: str) -> None:
    """Refuse a `device` resource other than the CPU, the only device that adapters use."""
    device = resources.get("device", DEVICE)
    if device != DEVICE:
        raise ValueError(f"{owner} runs only on resource device {DEVICE!r}, got {device!r}")
