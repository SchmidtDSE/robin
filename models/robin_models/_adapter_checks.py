"""The checks that every adapter makes on its card and its resources.

Each check takes `owner`, the name the error gives, such as "model owl/v4". Importing this
module loads no ML runtime.
"""

from collections.abc import Mapping
from typing import Any

from robin_contracts.cards import ModelCard
from robin_contracts.protocols import JsonScalar

DEVICE = "cpu"
GPU = "gpu"
# The TensorFlow device of each `device` resource that the TensorFlow adapters take.
_TF_DEVICES = {DEVICE: "/CPU:0", GPU: "/GPU:0"}


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


def get_tf_device(resources: Mapping[str, JsonScalar], owner: str, tf: Any) -> str:
    """Returns the TensorFlow device for the `device` resource: `/CPU:0` or `/GPU:0`.

    The resource is `cpu` (the default) or `gpu`. Raises ValueError for another value,
    and for `gpu` when TensorFlow sees no GPU: the model would then run on the CPU and
    give no error. `tf` is the TensorFlow module, which the adapter has imported.
    """
    device = resources.get("device", DEVICE)
    if device not in _TF_DEVICES:
        raise ValueError(
            f"{owner} runs on resource device {DEVICE!r} or {GPU!r}, got {device!r}"
        )
    if device == GPU and not tf.config.list_physical_devices("GPU"):
        raise ValueError(
            f"{owner} has resource device {GPU!r}, but TensorFlow sees no GPU. Check that "
            "the job runs the GPU image on a GPU instance."
        )
    return _TF_DEVICES[device]


def check_device(resources: Mapping[str, JsonScalar], owner: str) -> None:
    """Refuse a `device` resource other than the CPU, for a runtime that has only the CPU."""
    device = resources.get("device", DEVICE)
    if device != DEVICE:
        raise ValueError(f"{owner} runs only on resource device {DEVICE!r}, got {device!r}")
