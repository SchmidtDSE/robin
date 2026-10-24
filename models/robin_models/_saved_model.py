"""What the BirdNET and Perch adapters share: each loads a TensorFlow SavedModel from its
pinned files, and checks the signature that it calls.

The adapters load and call the SavedModel themselves. Each check takes `owner`, the name
the error gives, such as "model perch/v8". Importing this module loads no ML runtime.
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path

from robin_models.registration import SAVED_MODEL_LAYOUT


def require_roles(files: Mapping[str, Path], owner: str) -> None:
    """Refuse files that do not give each role of the SavedModel layout."""
    for role in SAVED_MODEL_LAYOUT:
        if role not in files:
            raise ValueError(
                f"{owner} needs a file under the role {role!r}, "
                f"and got the roles {sorted(files)}"
            )


def create_saved_model_folder(files: Mapping[str, Path], scratch_dir: Path, prefix: str) -> Path:
    """A new folder laid out as TensorFlow expects, linking to the pinned files."""
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=scratch_dir))
    for role, place in SAVED_MODEL_LAYOUT.items():
        link = folder / place
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(files[role])
    return folder


def get_signature(loaded, name: str, owner: str) -> object:
    """The loaded SavedModel's signature `name`."""
    signatures = loaded.signatures
    if name not in signatures:
        raise ValueError(
            f"{owner} needs the SavedModel signature {name!r}, and found {sorted(signatures)}"
        )
    return signatures[name]


def require_input(signature, name: str, width: int, owner: str) -> None:
    """Refuse a signature that does not take one input, `inputs`, of `width` samples."""
    inputs = signature.structured_input_signature[1]
    if list(inputs) != ["inputs"]:
        raise ValueError(
            f"{owner} needs the {name} signature to take one input named 'inputs', "
            f"and it takes {sorted(inputs)}"
        )
    shape = inputs["inputs"].shape.as_list()
    if shape != [None, width]:
        raise ValueError(
            f"{owner} needs the {name} signature's input shape to be "
            f"[None, {width}], and it is {shape}"
        )


def require_output(signature, name: str, output: str, width: int, owner: str) -> None:
    """Refuse a signature whose output `output` does not have `width` values for each window."""
    outputs = signature.structured_outputs
    shape = outputs[output].shape.as_list() if output in outputs else None
    if shape != [None, width]:
        raise ValueError(
            f"{owner} needs the {name} signature's output {output!r} to have "
            f"shape [None, {width}], and found {shape}"
        )
