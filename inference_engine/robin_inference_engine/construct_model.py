"""Finding an installed model distribution's factory by model ref, and calling it.

A distribution registers its factory under the entry-point group below, named by the
ref's `name/version`. Listing reads installed metadata only, so asking what is
installed imports no model and loads no ML runtime.
"""

from importlib.metadata import EntryPoint, entry_points

from robin_contracts.cards import ModelRef
from robin_contracts.protocols import Model, ModelContext
from robin_inference_engine import errors

ENTRY_POINT_GROUP = "robin.models"


def installed_models() -> tuple[str, ...]:
    """The ref of every installed model, sorted. Imports nothing."""
    return tuple(sorted({entry.name for entry in entry_points(group=ENTRY_POINT_GROUP)}))


def construct_model(*, ref: ModelRef, context: ModelContext) -> Model:
    """Load the one factory registered for `ref`, call it, and check what it returned.

    The check is `isinstance` against the runtime-checkable protocol, which confirms
    the attributes exist and nothing about their signatures.
    """
    factory = _load(_entry_point(ref))
    try:
        model = factory(context)
    except Exception as exc:
        raise _refused(
            errors.MODEL_CONSTRUCTION_FAILED,
            f"the factory for {ref.id} raised {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(model, Model):
        raise _refused(
            errors.MODEL_PROTOCOL_UNSATISFIED,
            f"the factory for {ref.id} returned a {type(model).__name__}, which does "
            f"not have the attributes of a model",
        )
    return model


def _entry_point(ref: ModelRef) -> EntryPoint:
    matches = [entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name == ref.id]
    if not matches:
        installed = ", ".join(installed_models()) or "none"
        raise _refused(
            errors.MODEL_NOT_INSTALLED,
            f"no installed distribution registers {ref.id} under {ENTRY_POINT_GROUP}; "
            f"installed: {installed}",
        )
    # Choosing one of several would run a model the caller did not pick.
    if len(matches) > 1:
        registrations = "; ".join(sorted(_registration(entry) for entry in matches))
        raise _refused(
            errors.MODEL_REGISTERED_TWICE,
            f"{len(matches)} installed distributions register {ref.id}: {registrations}",
        )
    return matches[0]


def _registration(entry: EntryPoint) -> str:
    return f"{entry.dist.name} {entry.dist.version} ({entry.value})"


def _load(entry: EntryPoint) -> object:
    try:
        return entry.load()
    except Exception as exc:
        raise _refused(
            errors.MODEL_ENTRY_POINT_UNLOADABLE,
            f"entry point {entry.name} = {entry.value} could not be loaded: "
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _refused(code: str, detail: str) -> errors.EngineError:
    return errors.EngineError(code, errors.CONSTRUCT_MODEL, detail)
