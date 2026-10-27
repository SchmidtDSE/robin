"""Which locations name a local file.

The run manager checks a model's file locations with this rule when it registers the
model, and the worker reads files with it, so a registration never stores a location
that the worker refuses. The rules match ondio's.
"""

import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def has_uri_scheme(location: str) -> bool:
    """Whether `location` starts with a URI scheme, such as `s3:`. If not, it is a path."""
    return _SCHEME.match(location) is not None


def get_local_path(location: str) -> Path:
    """The absolute local path that `location` names.

    `location` is an absolute path, used as written, or a `file://` URI with no host or
    `localhost`. Raises `ValueError` for anything else.
    """
    # A bare path is checked first so that `%`, `?` and `#` in a file name are kept
    # as written rather than decoded or split off.
    if os.path.isabs(location):
        return Path(location)
    not_local = f"{location!r} is not an absolute path or a local file:// URI"
    try:
        parts = urlsplit(location)
    except ValueError:
        raise ValueError(not_local) from None
    # In a `file:` URI, `?` and `#` start a query or a fragment, which names no file.
    # urlsplit drops an empty query or fragment, so check the text itself.
    if (
        parts.scheme != "file"
        or not location[len("file:"):].startswith("//")
        or parts.netloc not in ("", "localhost")
        or "?" in location
        or "#" in location
    ):
        raise ValueError(not_local)
    path = url2pathname(parts.path)
    if not os.path.isabs(path):
        raise ValueError(f"{location!r} does not name an absolute path")
    return Path(path)


def get_file_uri(location: str) -> str:
    """The `file://` URI that Robin writes for `location`, an absolute path or a `file:` URI.

    A path becomes `file:///` and the path, percent-encoded. A `file:` URI must already
    have that spelling, so that one file always has one URI. Raises `ValueError`, with a
    reason that starts with "not", for anything else.
    """
    if location.startswith("/"):
        _check_path(location)
        return PurePosixPath(location).as_uri()
    if not location.lower().startswith("file:"):
        raise ValueError("not an absolute path or a file:/// URI")
    try:
        path = unquote(urlsplit(location).path)
    except ValueError:
        path = ""
    if not path.startswith("/"):
        raise ValueError("not a valid file URI; give an absolute path or a file:/// URI")
    _check_path(path)
    robins = PurePosixPath(path).as_uri()
    if location != robins:
        raise ValueError(f"not in the spelling that Robin uses; write it as {robins}")
    return location


def get_audio_uri(location: str, folder: Path | None = None) -> str:
    """The URI that Robin stores for the audio at `location`.

    A URI with a scheme other than `file:` is kept as written. An absolute path or a
    `file:` URI becomes Robin's spelling, as `get_file_uri` gives it. A relative path is
    read from `folder`. Raises `ValueError`, with a reason that starts with "not", for a
    relative path with no `folder` and for what `get_file_uri` refuses.
    """
    if not location.startswith("/") and not location.lower().startswith("file:"):
        if has_uri_scheme(location):
            return location
        if folder is None:
            raise ValueError("not an absolute path or a URI")
        location = str(PurePosixPath(folder, location))
    return get_file_uri(location)


def _check_path(path: str) -> None:
    if "\x00" in path:
        raise ValueError("not a valid path, because it holds a NUL character")
