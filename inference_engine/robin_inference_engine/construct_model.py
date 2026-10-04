"""Finding an installed distribution's factory for a model or a head, and calling it.

A backbone's factory is registered under `robin.models`, named by the ref's
`name/version`. A head's is registered under `robin.head_runtimes`, named by the runtime
its card names. Listing reads installed metadata only, so asking what is installed
imports no model and loads no ML runtime.
"""

from importlib.metadata import EntryPoint, entry_points

from robin_contracts.cards import ModelRef
from robin_contracts.protocols import Model, ModelContext
from robin_inference_engine import errors

ENTRY_POINT_GROUP = "robin.models"
HEAD_RUNTIME_GROUP = "robin.head_runtimes"


def installed_models() -> tuple[str, ...]:
    """The ref of every installed model, sorted. Imports nothing."""
    return _installed(ENTRY_POINT_GROUP)


def construct_model(*, ref: ModelRef, context: ModelContext) -> Model:
    """Load the one factory registered for `ref`, call it, and check what it returned.

    The check is `isinstance` against the runtime-checkable protocol, which confirms
    the attributes exist and nothing about their signatures.
    """
    return _construct(ENTRY_POINT_GROUP, ref.id, context)


def construct_head(*, runtime: str, context: ModelContext) -> Model:
    """Load the one factory registered for a head `runtime`, call it, and check what it
    returned, as `construct_model` does."""
    return _construct(HEAD_RUNTIME_GROUP, runtime, context)


def _installed(group: str) -> tuple[str, ...]:
    return tuple(sorted({entry.name for entry in entry_points(group=group)}))


def _construct(group: str, name: str, context: ModelContext) -> Model:
    factory = _load(_entry_point(group, name))
    try:
        model = factory(context)
    except Exception as exc:
        raise _refused(
            errors.MODEL_CONSTRUCTION_FAILED,
            f"the factory for {name} under {group} raised {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(model, Model):
        raise _refused(
            errors.MODEL_PROTOCOL_UNSATISFIED,
            f"the factory for {name} under {group} returned a {type(model).__name__}, "
            f"which does not have the attributes of a model",
        )
    return model


def _entry_point(group: str, name: str) -> EntryPoint:
    matches = [entry for entry in entry_points(group=group) if entry.name == name]
    if not matches:
        installed = ", ".join(_installed(group)) or "none"
        raise _refused(
            errors.MODEL_NOT_INSTALLED,
            f"no installed distribution registers {name} under {group}; "
            f"installed: {installed}",
        )
    # Choosing one of several would run a model the caller did not pick.
    if len(matches) > 1:
        registrations = "; ".join(sorted(_registration(entry) for entry in matches))
        raise _refused(
            errors.MODEL_REGISTERED_TWICE,
            f"{len(matches)} installed distributions register {name} under {group}: "
            f"{registrations}",
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
            f"entry point {entry.name} = {entry.value} under {entry.group} could not be loaded: "
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _refused(code: str, detail: str) -> errors.EngineError:
    return errors.EngineError(code, errors.CONSTRUCT_MODEL, detail)
