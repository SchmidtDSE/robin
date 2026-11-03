"""The settings class of each storage backend, which the profile's `storage` section uses.

Each class is in the `settings` module of its backend in `robin_adapters.storage`. A
`settings` module imports no adapter, so a profile loads without the packages that an
adapter needs, such as ondio.
"""

from collections.abc import Mapping
from typing import Annotated

from pydantic import Field

from robin_adapters.storage.local.settings import LocalStorageSettings
from robin_adapters.storage.object_store.settings import S3StorageSettings
from robin_contracts.ports.storage import StorageSettings

# A storage section of any kind, chosen by its `kind`.
Storage = Annotated[LocalStorageSettings | S3StorageSettings, Field(discriminator="kind")]

# The settings class of each storage kind, by its `kind`.
STORAGE_CLASSES: Mapping[str, type[StorageSettings]] = {
    "local": LocalStorageSettings,
    "s3": S3StorageSettings,
}
