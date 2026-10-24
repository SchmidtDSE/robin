"""What registering a model needs: the bundled cards and registries, and the file roles
each model reads.

Importing this module loads no ML runtime.
"""

from importlib.resources import as_file, files
from pathlib import Path

from robin_contracts.cards import HeadCard, ModelCard, model_ref, read_card
from robin_contracts.work import REGISTRY_ROLE

# The role names the adapters read. The adapters import them from here.
WEIGHTS_ROLE = "weights"
GRAPH_ROLE = "graph"
# Where each pinned file goes in the folder TensorFlow loads, for BirdNET and Perch.
SAVED_MODEL_LAYOUT = {
    "saved_model": Path("saved_model.pb"),
    "variables_data": Path("variables") / "variables.data-00000-of-00001",
    "variables_index": Path("variables") / "variables.index",
}

# The roles each model always reads. Every model also takes the taxa registry, which it
# needs only for scores.
_BUILT_IN = {
    "owl/v4": ("owl", frozenset({WEIGHTS_ROLE})),
    "birdnet/v2p4": ("birdnet", frozenset(SAVED_MODEL_LAYOUT)),
    "perch/v8": ("perch", frozenset(SAVED_MODEL_LAYOUT)),
}
_HEAD_RUNTIMES = {"onnx": frozenset({GRAPH_ROLE})}

# A built-in model's registry is pinned at `robin-models:<model_id>/taxa_registry.csv`.
# The worker reads it from its own installed robin-models. No other file is named this way.
BUNDLED_SCHEME = "robin-models:"
_REGISTRY_FILE = "taxa_registry.csv"


def get_built_in_model_ids() -> tuple[str, ...]:
    """The IDs of the models whose cards this package ships, such as `owl/v4`."""
    return tuple(_BUILT_IN)


def get_bundled_card(model_id: str) -> ModelCard:
    """The card this package ships for `model_id`. Raises KeyError for another model."""
    package, _ = _BUILT_IN[model_id]
    with as_file(files(f"robin_models.{package}") / "resources" / "card.yaml") as path:
        return read_card(path)


def get_bundled_registry_location(model_id: str) -> str:
    """The location of the registry this package ships for `model_id`.

    Raises KeyError for another model.
    """
    if model_id not in _BUILT_IN:
        raise KeyError(model_id)
    return f"{BUNDLED_SCHEME}{model_id}/{_REGISTRY_FILE}"


def is_bundled_location(uri: str) -> bool:
    """Whether `uri` names a file that this package ships."""
    return uri.startswith(BUNDLED_SCHEME)


def get_bundled_file(location: str) -> Path:
    """The installed file that a bundled registry location names.

    Raises ValueError for any other location, or if the file is not on disk.
    """
    model_id, _, name = location.removeprefix(BUNDLED_SCHEME).rpartition("/")
    if not is_bundled_location(location) or name != _REGISTRY_FILE or model_id not in _BUILT_IN:
        raise ValueError(f"{location!r} names no registry that robin-models ships")
    package, _ = _BUILT_IN[model_id]
    resource = files(f"robin_models.{package}") / "resources" / name
    # Robin installs robin-models from wheels, never as a zip, so each resource is a file.
    if not isinstance(resource, Path) or not resource.is_file():
        raise ValueError(f"robin-models has no file on disk for {location!r}")
    return resource


def get_head_runtime_roles() -> dict[str, frozenset[str]]:
    """The roles each head runtime always reads, by runtime name, such as `onnx`."""
    return dict(_HEAD_RUNTIMES)


def get_file_roles(card: ModelCard | HeadCard) -> frozenset[str] | None:
    """The roles the adapter for `card` always reads, or None if this package has no adapter.

    The taxa registry is not in the set: every model takes it, and needs it only for scores.
    """
    if isinstance(card, HeadCard):
        return _HEAD_RUNTIMES.get(card.runtime)
    built_in = _BUILT_IN.get(model_ref(card).id)
    if built_in is None:
        return None
    _, roles = built_in
    return roles
