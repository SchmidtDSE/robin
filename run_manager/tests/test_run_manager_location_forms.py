from unittest.mock import ANY

import pytest

from robin_run_manager.location_forms import get_unsupported_locations

SAMPLES = {
    "path": "/data/a.wav",
    "file": "file:///data/a.wav",
    "s3": "s3://lab/a.wav",
    "s3-without-a-bucket": "s3:///a.wav",
    "s3-with-nothing": "s3://",
    "robin-models": "robin-models:owl/v4/taxa_registry.csv",
    "http": "http://lab.example/a.wav",
    "gs": "gs://lab/a.wav",
    "relative": "data/a.wav",
}

# The forms that each compute kind accepts for each role.
ACCEPTED = {
    "local": {
        "recording": {"path", "file", "s3"},
        "model_file": {"path", "file", "s3", "robin-models"},
        "embeddings": {"path", "file", "s3"},
        "storage_root": {"path", "file", "s3"},
    },
    "kubernetes": {
        "recording": {"s3"},
        "model_file": {"s3", "robin-models"},
        "embeddings": {"s3"},
        "storage_root": {"s3"},
    },
}

CELLS = [
    (kind, role, form, form in forms)
    for kind, roles in ACCEPTED.items()
    for role, forms in roles.items()
    for form in SAMPLES
]


@pytest.mark.parametrize(
    ("kind", "role", "form", "accepted"),
    CELLS,
    ids=[f"{kind}-{role}-{form}" for kind, role, form, _ in CELLS],
)
def test_get_unsupported_locations_follows_the_table(kind, role, form, accepted):
    location = SAMPLES[form]
    found = get_unsupported_locations(kind, [(role, location)])
    if accepted:
        assert found == []
    else:
        assert found == [{"role": role, "location": location, "accepted_forms": ANY}]


def test_get_unsupported_locations_gives_only_the_refused_ones():
    found = get_unsupported_locations(
        "kubernetes", [("recording", "s3://lab/a.wav"), ("storage_root", "/data")]
    )
    assert [entry["location"] for entry in found] == ["/data"]
