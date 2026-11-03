"""The composition root: it builds what the operations use from a profile.

This is the one module in `robin_run_manager` that may import adapters. It imports
only the adapters that the profile names, and only in the function that needs them, so
a command that only reads does not load the storage libraries.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from robin_contracts.ports.compute import ComputeAdapter
from robin_contracts.ports.storage import TaskStorage
from robin_contracts.tasks import SplitLocation
from robin_run_manager.database import Database, open_database
from robin_run_manager.profile import Profile


@dataclass(frozen=True)
class Services:
    """What the operations use.

    `compute` holds a compute adapter for each compute kind that a setup of the profile
    uses and that this version has an adapter for. `task_storage` holds a task storage
    for each storage kind that this version can read and write task files in.
    `split_location` is the location rule of the storage that the worker writes with.
    `storage_env_vars` holds the environment variables that the worker needs to reach
    that storage, such as an S3 endpoint. The run manager copies them into each task
    that it gives a compute adapter.
    """

    profile: Profile
    database: Database
    compute: Mapping[str, ComputeAdapter] = field(default_factory=dict)
    task_storage: Mapping[str, TaskStorage] = field(default_factory=dict)
    split_location: SplitLocation | None = None
    storage_env_vars: Mapping[str, str] = field(default_factory=dict)


def compose(profile: Profile) -> Services:
    task_storage = _get_task_storage(profile)
    return Services(
        profile=profile,
        database=open_database(profile.database),
        compute=_get_compute(profile),
        task_storage=task_storage,
        split_location=_split_location,
        storage_env_vars=_get_storage_env_vars(task_storage),
    )


def _get_compute(profile: Profile) -> dict[str, ComputeAdapter]:
    # Only local compute has an adapter in this version.
    if profile.compute is None:
        return {}
    if "local" not in {setup.kind for setup in profile.compute.setups.values()}:
        return {}
    from robin_adapters.compute.local import LocalCompute

    return {"local": LocalCompute()}


def _get_task_storage(profile: Profile) -> dict[str, TaskStorage]:
    storage = profile.storage
    if storage is None:
        return {}
    if storage.kind == "local":
        from robin_adapters.storage.local import LocalTaskStorage

        return {"local": LocalTaskStorage()}
    from robin_adapters.storage.ondio import OndioTaskStorage

    options = {} if storage.endpoint is None else {"endpoint_url": str(storage.endpoint)}
    return {"s3": OndioTaskStorage(storage.root, **options)}


def _get_storage_env_vars(task_storage: Mapping[str, TaskStorage]) -> dict[str, str]:
    """Returns the environment variables that the worker needs to reach the task storage.

    Only S3 storage has any.
    """
    if "s3" not in task_storage:
        return {}
    from robin_adapters.storage.ondio import OndioTaskStorage

    return cast(OndioTaskStorage, task_storage["s3"]).get_worker_env_vars()


def _split_location(uri: str) -> tuple[str, str] | None:
    """The location rule of the storage adapters. It loads ondio only when it is first used."""
    from robin_adapters.storage.location import split_location

    return split_location(uri)
