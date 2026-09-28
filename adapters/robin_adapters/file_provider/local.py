"""The `FileProvider` port over the local file system."""

import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname


class LocalFileProvider:
    """Returns the local file a uri names, without copying it.

    `fetch` accepts an absolute path, used as written, or a `file:` uri with no host
    or `localhost`. Anything else raises `ValueError`. `release` does nothing.
    """

    def fetch(self, uri: str) -> Path:
        path = _local_path(uri)
        if not path.is_file():
            raise FileNotFoundError(f"{uri!r} names {path}, which is not an existing file")
        return path

    def release(self, path: Path) -> None:
        pass


def _local_path(uri: str) -> Path:
    # A bare path is checked first so that `%`, `?` and `#` in a file name are kept
    # as written rather than decoded or split off.
    if os.path.isabs(uri):
        return Path(uri)
    parts = urlsplit(uri)
    # In a `file:` uri a literal `?` or `#` can only start a query or fragment, which
    # name no file. urlsplit reports an empty one as absent, so the text is checked.
    if parts.scheme != "file" or parts.netloc not in ("", "localhost") or "?" in uri or "#" in uri:
        raise ValueError(f"{uri!r} is not an absolute path or a local file: uri")
    path = url2pathname(parts.path)
    if not os.path.isabs(path):
        raise ValueError(f"{uri!r} does not name an absolute path")
    return Path(path)
