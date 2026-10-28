"""The composition root: it builds what the operations use from a profile.

This is the one module in `robin_run_manager` that may import adapters. It imports
only the adapters that the profile names, and only in the function that needs them, so
a command that only reads does not load the storage libraries.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from robin_contracts.ports.compute import ComputeAdapter
from robin_contracts.ports.storage import TaskStorage
from robin_contracts.tasks import SplitLocation
from robin_run_manager.database import Database, open_database
from robin_run_manager.profile import Profile


@dataclass(frozen=True)
class Services:
    """What the operations use.

    `compute` is None if the profile has no compute section, or names compute that this
    version has no adapter for. `task_storage` holds a task storage for each storage
    kind that this version can read and write task files in. `split_location` is the
    location rule of the storage that the worker writes with.
    """

    profile: Profile
    database: Database
    compute: ComputeAdapter | None = None
    task_storage: Mapping[str, TaskStorage] = field(default_factory=dict)
    split_location: SplitLocation | None = None


def compose(profile: Profile) -> Services:
    return Services(
        profile=profile,
        database=open_database(profile.database),
        compute=_get_compute(profile),
        task_storage=_get_task_storage(profile),
        split_location=_split_location,
    )


def _get_compute(profile: Profile) -> ComputeAdapter | None:
    # The Kubernetes adapter is not in this version.
    if profile.compute is None or profile.compute.kind != "local":
        return None
    from robin_adapters.compute.local import LocalCompute

    return LocalCompute()


def _get_task_storage(profile: Profile) -> dict[str, TaskStorage]:
    # Only local storage has a task storage in this version.
    if profile.storage is None or profile.storage.kind != "local":
        return {}
    from robin_adapters.storage.local import LocalTaskStorage

    return {"local": LocalTaskStorage()}


def _split_location(uri: str) -> tuple[str, str] | None:
    """The location rule of the storage adapters. It loads ondio only when it is first used."""
    from robin_adapters.storage.location import split_location

    return split_location(uri)
