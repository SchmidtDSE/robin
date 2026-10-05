"""The `FileProvider` port over any storage ondio supports, such as S3."""

import shutil
import uuid
from pathlib import Path
from typing import Any

import ondio


class OndioFileProvider:
    """Downloads the object at a uri into its own folder under `directory`.

    The file is named for the uri's last segment. ondio's errors propagate as raised,
    and a uri with no file name after its last `/` raises `ValueError`. `release`
    deletes the file and its folder, and refuses a path this provider didn't return.
    `backend_options` are passed to every ondio call, which passes them to its backend,
    such as `profile_name` for S3.
    """

    def __init__(self, directory: Path, **backend_options: Any) -> None:
        self._directory = Path(directory).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._backend_options = backend_options
        self._fetched: set[Path] = set()

    def fetch(self, uri: str) -> Path:
        name = uri.rsplit("/", 1)[-1]
        # `.` and `..` would name the folder or its parent rather than a file in it.
        if name in ("", ".", ".."):
            raise ValueError(f"{uri!r} has no file name after its last '/'")
        folder = self._directory / uuid.uuid4().hex
        folder.mkdir()
        path = folder / name
        try:
            ondio.download(uri, path, **self._backend_options)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self._fetched.add(path)
        return path

    def release(self, path: Path) -> None:
        if path not in self._fetched:
            raise ValueError(f"{path} was not fetched by this provider, or is already released")
        self._fetched.remove(path)
        shutil.rmtree(path.parent)
