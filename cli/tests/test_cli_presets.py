"""`robin presets create`, `list`, and `show`."""

import json
from pathlib import Path

import pytest

from cli_doubles import write_owl_files

FIXTURES = Path(__file__).parents[2] / "run_manager" / "tests" / "fixtures"
OWL_CONFIGURATION = FIXTURES / "configuration-owl-v4.yaml"


@pytest.fixture
def owl_registered(robin, tmp_path):
    assert robin("db", "migrate").code == 0
    assert robin("models", "register", "owl/v4", *write_owl_files(tmp_path / "owl")).code == 0


def create(robin, name, path=OWL_CONFIGURATION):
    return robin("--json", "presets", "create", name, "--configuration", str(path))


def test_create_prints_the_preset_as_json(robin, owl_registered):
    ran = create(robin, "owl-default")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert (body["name"], body["created"]) == ("owl-default", True)
    assert body["configuration_digest"].startswith("sha256:v1:")


def test_create_prints_the_digest(robin, owl_registered):
    digest = json.loads(create(robin, "owl-default").stdout)["configuration_digest"]

    ran = robin("presets", "create", "owl-default", "--configuration", str(OWL_CONFIGURATION))

    assert ran.code == 0
    assert digest in ran.stdout and "already" in ran.stdout


def test_a_conflicting_create_exits_1(robin, owl_registered, tmp_path):
    create(robin, "owl-default")
    other = tmp_path / "other.yaml"
    other.write_text(OWL_CONFIGURATION.read_text().replace("window_overlap: 0.0", "window_overlap: 6.0"))

    ran = create(robin, "owl-default", other)

    assert ran.code == 1
    assert ran.error()["reason"] == "preset_conflict"


def test_an_invalid_configuration_exits_1_with_its_code(robin, owl_registered, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(OWL_CONFIGURATION.read_text().replace("window_overlap: 0.0", "gain: 1.0"))

    ran = create(robin, "owl-default", bad)

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_configuration"
    assert ran.error()["details"]["code"] == "setting_undeclared"


def test_a_missing_configuration_file_exits_1_with_invalid_request(robin, owl_registered, tmp_path):
    ran = create(robin, "owl-default", tmp_path / "absent.yaml")

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_request"


def test_configuration_is_required(robin):
    assert robin("presets", "create", "owl-default").code == 2


def test_list_prints_the_page_envelope(robin, owl_registered):
    create(robin, "owl-b")
    create(robin, "owl-a")

    ran = robin("--json", "presets", "list")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert [item["name"] for item in body["items"]] == ["owl-a", "owl-b"]
    assert body["next_cursor"] is None


def test_all_follows_every_page(robin, owl_registered):
    for name in ("a", "b", "c", "d", "e"):
        create(robin, name)

    ran = robin("--json", "presets", "list", "--limit", "2", "--all")

    assert [item["name"] for item in json.loads(ran.stdout)["items"]] == list("abcde")


def test_show_prints_the_configuration(robin, owl_registered):
    create(robin, "owl-default")

    ran = robin("--json", "presets", "show", "owl-default")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert body["configuration"]["model"]["name"] == "owl"
    assert "s3://" not in ran.stdout


def test_show_of_a_missing_preset_exits_1(robin, owl_registered):
    ran = robin("--json", "presets", "show", "absent")

    assert ran.code == 1
    assert ran.error()["reason"] == "not_found"


def test_show_prints_a_list_value_as_json(robin, owl_registered):
    create(robin, "owl-default")

    ran = robin("presets", "show", "owl-default")

    assert ran.code == 0, ran.stderr
    assert "['scores']" not in ran.stdout
    assert 'output_kinds: [\n  "scores"\n]' in ran.stdout
