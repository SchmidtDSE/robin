"""Storage for one task, chosen for each location from its scheme.

A local location is an absolute path or a `file://` URI with no host. An `s3://`
location goes through ondio, with credentials from the environment. The work file,
each model file, each input, and the output root can each use a different store.
"""

from pathlib import Path

import ondio
from robin_adapters.artifact_writer.local import LocalArtifactWriter
from robin_adapters.artifact_writer.ondio import OndioArtifactWriter
from robin_adapters.file_provider.local import LocalFileProvider, local_path
from robin_adapters.file_provider.ondio import OndioFileProvider
from robin_adapters.location import LOCAL_STORE, split_location
from robin_contracts.ports import ArtifactWriter, FileProvider


def has_location(uri: str) -> bool:
    """Whether the worker can read or write `uri`."""
    return split_location(uri) is not None


def read_bytes(uri: str) -> bytes:
    if _is_local(uri):
        return local_path(uri).read_bytes()
    return ondio.read(_supported(uri))


def create(uri: str, data: bytes) -> None:
    """Write `data` at `uri` only if nothing is there. A reader never sees part of it.

    Raises `FileExistsError` if something is there. A local file is written only in an
    existing folder, so a wrong path makes no folders.
    """
    if _is_local(uri):
        path = local_path(uri)
        if not path.parent.is_dir():
            raise FileNotFoundError(f"the folder {path.parent} does not exist")
        uri = str(path)
    try:
        ondio.create(_supported(uri), data)
    except ondio.ObjectExistsError as exc:
        raise FileExistsError(f"{uri} already exists") from exc


def artifact_writer(output_root: str) -> ArtifactWriter:
    if _is_local(output_root):
        return LocalArtifactWriter(local_path(output_root))
    return OndioArtifactWriter(_supported(output_root))


class TaskFiles:
    """The `FileProvider` for one task: local files as they are, S3 objects downloaded.

    Downloads go in their own folders under `scratch`. One local file can be fetched
    more than once, and each fetch is released once.
    """

    def __init__(self, scratch: Path) -> None:
        self._scratch = scratch
        self._local = LocalFileProvider()
        self._remote: OndioFileProvider | None = None
        self._held: dict[Path, tuple[FileProvider, int]] = {}

    def fetch(self, uri: str) -> Path:
        provider = self._provider_for(uri)
        path = provider.fetch(uri)
        _, count = self._held.get(path, (provider, 0))
        self._held[path] = (provider, count + 1)
        return path

    def release(self, path: Path) -> None:
        if path not in self._held:
            raise ValueError(f"{path} was not fetched for this task, or is already released")
        provider, count = self._held.pop(path)
        if count > 1:
            self._held[path] = (provider, count - 1)
        provider.release(path)

    def _provider_for(self, uri: str) -> FileProvider:
        if _is_local(uri):
            return self._local
        _supported(uri)
        if self._remote is None:
            self._remote = OndioFileProvider(self._scratch)
        return self._remote


def _is_local(uri: str) -> bool:
    place = split_location(uri)
    return place is not None and place[0] == LOCAL_STORE


def _supported(uri: str) -> str:
    if split_location(uri) is None:
        raise ValueError(
            f"{uri!r} is not an absolute local path, a local file:// URI, or an s3:// URI"
        )
    return uri
