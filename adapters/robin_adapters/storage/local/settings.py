"""The settings class of the `local` storage backend."""

from typing import Literal
from urllib.parse import unquote, urlsplit

from robin_contracts.local_files import get_local_path
from robin_contracts.ports.storage import StorageSettings


class LocalStorageSettings(StorageSettings):
    """Robin writes its work files, result files, and outputs under a local `root`."""

    kind: Literal["local"]

    @classmethod
    def get_path_in_store(cls, root: str) -> str:
        try:
            get_local_path(root)
        except ValueError:
            raise ValueError("give an absolute path or a local file:// URI") from None
        path = root if root.startswith("/") else unquote(urlsplit(root).path)
        return path.removeprefix("/")
