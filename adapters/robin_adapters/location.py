"""Splits a location into its store and its path in that store.

The result check uses this to decide whether an artifact is under the task's output
root, so each kind of location follows the rule of the storage that writes it.
"""

import ondio

LOCAL_STORE = "file://"
_SUPPORTED_SCHEMES = frozenset({"file", "s3"})


def split_location(uri: str) -> tuple[str, str] | None:
    """The store and the path, with no leading `/`, that `uri` names, or None if unsupported."""
    try:
        store, path = ondio.locate(uri)
    except (ondio.OndioError, ValueError):
        return None
    if store.partition("://")[0] not in _SUPPORTED_SCHEMES:
        return None
    return store, path
