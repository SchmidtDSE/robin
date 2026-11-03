"""The check that each location has a form that a compute kind can read, for its role.

Each compute kind's setup class gives the forms that it accepts (`ACCEPTED_FORMS`).
This module finds the form of a location. The check looks at the form only, not at
whether the file exists or can be read.
"""

from collections.abc import Iterable
from typing import Any

from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import LocationForm
from robin_contracts.ports.storage import is_s3_uri
from robin_models.registration import is_bundled_location
from robin_run_manager.compute_setups import SETUP_CLASSES

# The text that a refusal shows for each form.
_FORM_TEXT = {
    "path": "absolute path",
    "file": "file://",
    "s3": "s3://",
    "bundled": "robin-models:",
}


def get_unsupported_locations(
    compute_kind: str, locations: Iterable[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Returns each `(role, location)` whose form `compute_kind` cannot read for the role.

    Each entry has `role`, `location`, and `accepted_forms`, in the order of `locations`.
    """
    unsupported = []
    accepted_forms = SETUP_CLASSES[compute_kind].ACCEPTED_FORMS
    for role, location in locations:
        accepted = accepted_forms[role]
        if _get_form(location) not in accepted:
            shown = tuple(_FORM_TEXT[form] for form in accepted)
            unsupported.append({"role": role, "location": location, "accepted_forms": shown})
    return unsupported


def _get_form(location: str) -> LocationForm | None:
    if is_s3_uri(location):
        return "s3"
    if is_bundled_location(location):
        return "bundled"
    try:
        get_local_path(location)
    except ValueError:
        return None
    return "file" if location.startswith("file://") else "path"
