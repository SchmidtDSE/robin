"""What every adapter uses to set itself up: it imports its runtime, checks its card and
files, and reads its resources.

Each function takes `owner`, the name the error gives, such as "model owl/v4". Importing
this module loads no ML runtime.
"""

from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError
from robin_contracts.cards import ModelCard
from robin_contracts.protocols import JsonScalar

# The devices that the `device` resource can name.
Device = Literal["cpu", "gpu"]
# The TensorFlow device of each `device` resource.
_TF_DEVICES = {"cpu": "/CPU:0", "gpu": "/GPU:0"}
# The frame count that soundfile gives for a FLAC whose header gives 0 total samples, which
# FLAC uses for "unknown". It is the largest 64-bit integer, not 0.
UNKNOWN_FRAMES = 2**63 - 1


@contextmanager
def explain_a_missing_runtime(owner: str, extra: str, modules: Collection[str]) -> Iterator[None]:
    """Name the extra to install when one of the runtime `modules` is not installed.

    An adapter imports its runtime modules inside this block. A missing module that is not
    in `modules` is raised unchanged, because no extra installs it.
    """
    try:
        yield
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] not in modules:
            raise
        raise ModuleNotFoundError(
            f"{owner} needs {error.name}, which is not installed; "
            f"install robin-models[{extra}]",
            name=error.name,
        ) from error


def refuse_a_card_the_model_does_not_follow(
    card: ModelCard, behaviour: Mapping[str, Any], owner: str
) -> None:
    """Refuse a card that gives another value for a field whose value the model fixes."""
    for field, fixed in behaviour.items():
        stated = getattr(card, field)
        if stated != fixed:
            raise ValueError(f"card field {field} is {stated!r}, but {owner} does {fixed!r}")


def require_roles(files: Mapping[str, Path], roles: Collection[str], owner: str) -> None:
    """Refuse `files` unless they give a file for each of `roles`."""
    for role in roles:
        if role not in files:
            raise ValueError(
                f"{owner} needs a file under the role {role!r}, "
                f"and got the roles {sorted(files)}"
            )


def refuse_audio_with_no_length(frames: int, path: Path, owner: str) -> None:
    """Refuse a file whose header gives no length.

    `frames` is the file's frame count from soundfile. Windows for that count would fill
    all memory, so the check must come before any window is made.
    """
    if frames == UNKNOWN_FRAMES:
        raise ValueError(f"{owner} cannot run {path}: its header gives no length")


class Resources(BaseModel):
    """The resources that every adapter reads from its work's `resources`.

    Each adapter makes a subclass that gives `batch_size` its default. A subclass can also
    add resources, or narrow `device` to the devices that the adapter can use. Keys that no
    field names are ignored. Strict types refuse a value of another type: True, 8.0 and "8"
    are not a batch size.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")

    batch_size: PositiveInt
    device: Device = "cpu"


ResourcesT = TypeVar("ResourcesT", bound=Resources)


def get_resources(
    kind: type[ResourcesT], resources: Mapping[str, JsonScalar], owner: str
) -> ResourcesT:
    """Return `resources` as an instance of `kind`.

    Raises ValueError that names each bad resource and its value.
    """
    try:
        return kind.model_validate(dict(resources))
    except ValidationError as error:
        problems = "; ".join(_describe_problem(problem) for problem in error.errors())
        raise ValueError(f"{owner} cannot use its resources: {problems}") from None


def _describe_problem(problem: Mapping[str, Any]) -> str:
    """Return one resource problem as "name: what is wrong, got value"."""
    name = ".".join(map(str, problem["loc"]))
    return f"{name}: {problem['msg']}, got {problem['input']!r}"


def get_tf_device(device: Device, tf: Any, owner: str) -> str:
    """Return the TensorFlow device for the `device` resource: `/CPU:0` or `/GPU:0`.

    Raises ValueError for `gpu` when TensorFlow sees no GPU: the model would then run on
    the CPU and give no error. `tf` is the TensorFlow module, which the adapter has
    imported.
    """
    if device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise ValueError(
            f"{owner} has resource device 'gpu', but TensorFlow sees no GPU. Check that "
            "the job runs the GPU image on a GPU instance."
        )
    return _TF_DEVICES[device]
