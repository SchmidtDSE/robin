"""`robin models register`, `list`, and `show`, and the PAGE options."""

import json
from pathlib import Path

import pytest

from robin_run_manager import models
from robin_run_manager.errors import Unavailable

FIXTURES = Path(__file__).parents[2] / "run_manager" / "tests" / "fixtures"
OWL = FIXTURES / "owl-v4.yaml"


@pytest.fixture
def migrated(robin):
    assert robin("db", "migrate").code == 0


def register_versions(tmp_path, robin, versions):
    for version in versions:
        path = tmp_path / f"owl-{version}.yaml"
        path.write_text(OWL.read_text().replace("model_version: v4", f"model_version: {version}"))
        assert robin("models", "register", str(path)).code == 0


def test_register_prints_the_model_as_json(robin, migrated):
    ran = robin("--json", "models", "register", str(OWL))

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert (body["model_id"], body["created"]) == ("owl/v4", True)
    assert body["card_digest"].startswith("sha256:v1:")


def test_register_says_what_happened(robin, migrated):
    assert robin("models", "register", str(OWL)).stdout == "Registered owl/v4.\n"
    assert robin("models", "register", str(OWL)).stdout == "owl/v4 was already registered.\n"


def test_a_conflicting_registration_exits_1(robin, migrated, tmp_path):
    robin("models", "register", str(OWL))
    changed = tmp_path / "owl.yaml"
    changed.write_text(OWL.read_text().replace("size_bytes: 1048576", "size_bytes: 7"))

    ran = robin("--json", "models", "register", str(changed))

    assert ran.code == 1
    assert ran.error()["reason"] == "model_conflict"
    assert ran.stdout == ""


def test_a_missing_file_exits_1_with_invalid_request(robin, migrated, tmp_path):
    ran = robin("--json", "models", "register", str(tmp_path / "absent.yaml"))

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_request"


def test_a_database_that_is_not_migrated_exits_1(robin):
    ran = robin("--json", "models", "list")

    assert ran.code == 1
    assert ran.error()["reason"] == "schema_not_current"


def test_list_prints_the_page_envelope(robin, migrated):
    robin("models", "register", str(OWL))

    ran = robin("--json", "models", "list")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert [item["model_id"] for item in body["items"]] == ["owl/v4"]
    assert body["next_cursor"] is None


def test_list_shows_the_next_cursor_on_stderr(robin, migrated, tmp_path):
    register_versions(tmp_path, robin, ["v4", "v5"])

    ran = robin("models", "list", "--limit", "1")

    assert ran.code == 0, ran.stderr
    assert ran.stdout.splitlines()[0].startswith("owl/v4")
    assert len(ran.stdout.splitlines()) == 1
    assert "--cursor" in ran.stderr


def test_a_cursor_gives_the_next_page(robin, migrated, tmp_path):
    register_versions(tmp_path, robin, ["v4", "v5"])
    cursor = json.loads(robin("--json", "models", "list", "--limit", "1").stdout)["next_cursor"]

    ran = robin("--json", "models", "list", "--limit", "1", "--cursor", cursor)

    assert [item["model_id"] for item in json.loads(ran.stdout)["items"]] == ["owl/v5"]


def test_all_follows_every_page_into_one_envelope(robin, migrated, tmp_path):
    register_versions(tmp_path, robin, ["v4", "v5", "v6", "v7", "v8"])

    ran = robin("--json", "models", "list", "--limit", "2", "--all")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert [item["model_id"] for item in body["items"]] == [
        f"owl/{v}" for v in ("v4", "v5", "v6", "v7", "v8")
    ]
    assert body["next_cursor"] is None


def test_cursor_with_all_is_a_syntax_error(robin):
    ran = robin("models", "list", "--cursor", "x", "--all")

    assert ran.code == 2


def test_a_limit_out_of_range_exits_1(robin, migrated):
    ran = robin("--json", "models", "list", "--limit", "0")

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_request"


def test_a_broken_cursor_exits_1(robin, migrated):
    ran = robin("--json", "models", "list", "--cursor", "broken")

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_cursor"


def test_show_prints_the_card_and_files(robin, migrated):
    robin("models", "register", str(OWL))

    ran = robin("--json", "models", "show", "owl/v4")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert body["card"]["model_name"] == "owl"
    assert body["files"]["weights"]["uri"] == "s3://robin-models/owl/v4/weights.h5"


def test_show_of_a_missing_model_exits_1(robin, migrated):
    ran = robin("--json", "models", "show", "owl/v9")

    assert ran.code == 1
    assert ran.error()["reason"] == "not_found"


def test_all_prints_nothing_when_a_later_page_fails(robin, migrated, tmp_path, monkeypatch):
    register_versions(tmp_path, robin, ["v4", "v5", "v6"])
    real = models.list_models

    def fail_after_the_first_page(database, *, limit, cursor):
        if cursor is not None:
            raise Unavailable("The database stopped answering.")
        return real(database, limit=limit, cursor=cursor)

    monkeypatch.setattr(models, "list_models", fail_after_the_first_page)

    ran = robin("--json", "models", "list", "--limit", "2", "--all")

    assert ran.code != 0
    assert ran.stdout == ""
    assert ran.error()["reason"] == "unavailable"
