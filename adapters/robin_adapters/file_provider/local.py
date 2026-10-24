"""The `FileProvider` port over the local file system."""

from pathlib import Path

from robin_contracts.local_files import get_local_path


class LocalFileProvider:
    """Returns the local file a uri names, without copying it.

    `fetch` accepts the locations that `get_local_path` accepts, and raises `ValueError`
    for anything else. `release` does nothing.
    """

    def fetch(self, uri: str) -> Path:
        path = get_local_path(uri)
        if not path.is_file():
            raise FileNotFoundError(f"{uri!r} names {path}, which is not an existing file")
        return path

    def release(self, path: Path) -> None:
        pass

