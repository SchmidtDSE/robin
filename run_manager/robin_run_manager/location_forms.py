"""The location forms that each compute kind can read, for each role of a location.

The roles are `recording`, `model_file`, `embeddings`, and `storage_root`. The check
looks at the form of a location only, not at whether the file exists or can be read.
"""

from collections.abc import Iterable
from typing import Any

from robin_contracts.local_files import get_local_path
from robin_models.registration import is_bundled_location

_LOCAL = "absolute path"
_FILE = "file://"
_S3 = "s3://"
_BUNDLED = "robin-models:"

# A bundled location names only a model's registry, so only `model_file` accepts it.
_ACCEPTED_FORMS = {
    "local": {
        "recording": (_LOCAL, _FILE, _S3),
        "model_file": (_LOCAL, _FILE, _S3, _BUNDLED),
        "embeddings": (_LOCAL, _FILE, _S3),
        "storage_root": (_LOCAL, _FILE, _S3),
    },
    "kubernetes": {
        "recording": (_S3,),
        "model_file": (_S3, _BUNDLED),
        "embeddings": (_S3,),
        "storage_root": (_S3,),
    },
}


def get_unsupported_locations(
    compute_kind: str, locations: Iterable[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Returns each `(role, location)` whose form `compute_kind` cannot read for the role.

    Each entry has `role`, `location`, and `accepted_forms`, in the order of `locations`.
    """
    unsupported = []
    for role, location in locations:
        accepted = _ACCEPTED_FORMS[compute_kind][role]
        if _get_form(location) not in accepted:
            unsupported.append({"role": role, "location": location, "accepted_forms": accepted})
    return unsupported


def is_s3_uri(location: str) -> bool:
    """Whether `location` is an `s3://` URI that names a bucket."""
    bucket, _, _ = location.removeprefix(_S3).partition("/")
    return location.startswith(_S3) and bucket != ""


def _get_form(location: str) -> str | None:
    if is_s3_uri(location):
        return _S3
    if is_bundled_location(location):
        return _BUNDLED
    try:
        get_local_path(location)
    except ValueError:
        return None
    return _FILE if location.startswith(_FILE) else _LOCAL
