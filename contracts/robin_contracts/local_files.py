"""Which locations name a local file.

The run manager checks a model's file locations with this rule when it registers the
model, and the worker reads files with it, so a registration never stores a location
that the worker refuses. The rules match ondio's.
"""

import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname


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
