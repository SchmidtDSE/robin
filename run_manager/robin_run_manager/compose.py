"""The composition root: it builds what the operations use from a profile.

This is the one module in `robin_run_manager` that may import adapters. It imports
only the adapters that the profile names, and only in the function that needs them, so
a command that only reads does not load the storage libraries.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from robin_contracts.ports.compute import ComputeAdapter
from robin_contracts.ports.storage import FileProvider, TaskStorage
from robin_contracts.tasks import SplitLocation
from robin_run_manager.database import Database, open_database
from robin_run_manager.models import RemoteFiles
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
    that it gives a compute adapter. `remote_files` reads a model's files on S3 at
    registration, at the storage endpoint when the profile has S3 storage.
    """

    profile: Profile
    database: Database
    compute: Mapping[str, ComputeAdapter] = field(default_factory=dict)
    task_storage: Mapping[str, TaskStorage] = field(default_factory=dict)
    split_location: SplitLocation | None = None
    storage_env_vars: Mapping[str, str] = field(default_factory=dict)
    remote_files: RemoteFiles | None = None


def compose(profile: Profile) -> Services:
    task_storage = _get_task_storage(profile)
    return Services(
        profile=profile,
        database=open_database(profile.database),
        compute=_get_compute(profile),
        task_storage=task_storage,
        split_location=_split_location,
        storage_env_vars=_get_storage_env_vars(task_storage),
        remote_files=_get_remote_files(profile),
    )


# ---------------------------------------------------------------------------
# Compute: an adapter for each compute kind that a setup uses.
# ---------------------------------------------------------------------------


def _get_compute(profile: Profile) -> dict[str, ComputeAdapter]:
    """A compute adapter for each kind that a setup uses and that has an adapter. The
    kubernetes kind has none in this version."""
    if profile.compute is None:
        return {}
    kinds = {setup.kind for setup in profile.compute.setups.values()}
    return {
        kind: get_adapter() for kind, get_adapter in _COMPUTE_ADAPTERS.items() if kind in kinds
    }


# Each function imports its adapter only when it is called, so a profile that does not
# use a kind does not load that adapter's packages, such as boto3.
def _get_local_compute() -> ComputeAdapter:
    from robin_adapters.compute.local.adapter import LocalCompute

    return LocalCompute()


def _get_aws_batch_compute() -> ComputeAdapter:
    from robin_adapters.compute.aws_batch.adapter import AwsBatchCompute

    return AwsBatchCompute()


_COMPUTE_ADAPTERS: dict[str, Callable[[], ComputeAdapter]] = {
    "local": _get_local_compute,
    "aws-batch": _get_aws_batch_compute,
}


# ---------------------------------------------------------------------------
# Storage: the task storage, what the worker needs to reach it, and its location rule.
# ---------------------------------------------------------------------------


def _get_task_storage(profile: Profile) -> dict[str, TaskStorage]:
    storage = profile.storage
    if storage is None:
        return {}
    if storage.kind == "local":
        from robin_adapters.storage.local.adapter import LocalTaskStorage

        return {"local": LocalTaskStorage()}
    from robin_adapters.storage.object_store.adapter import ObjectStoreTaskStorage

    return {"s3": ObjectStoreTaskStorage(storage.root, **storage.get_backend_options())}


def _get_storage_env_vars(task_storage: Mapping[str, TaskStorage]) -> dict[str, str]:
    """Returns the environment variables that the worker needs to reach the task storage.

    Only S3 storage has any.
    """
    if "s3" not in task_storage:
        return {}
    from robin_adapters.storage.object_store.adapter import ObjectStoreTaskStorage

    return cast(ObjectStoreTaskStorage, task_storage["s3"]).get_worker_env_vars()


def _get_remote_files(profile: Profile) -> RemoteFiles:
    """Returns the factory of the file provider that reads a model's files on S3.

    It uses the storage section's endpoint, if any. It loads ondio only when it is
    first called.
    """
    options = {} if profile.storage is None else profile.storage.get_backend_options()

    def get_remote_files(directory: Path) -> FileProvider:
        from robin_adapters.storage.object_store.adapter import ObjectStoreFileProvider

        return ObjectStoreFileProvider(directory, **options)

    return get_remote_files


def _split_location(uri: str) -> tuple[str, str] | None:
    """The location rule of the storage adapters. It loads ondio only when it is first used."""
    from robin_adapters.storage.location import split_location

    return split_location(uri)
