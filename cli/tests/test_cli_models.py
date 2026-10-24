"""`robin models register`, `list`, and `show`, and the PAGE options."""

import json
from importlib.resources import files

import pytest

from robin_run_manager import models
from robin_run_manager.errors import Unavailable

from cli_doubles import write_owl_files
from model_fixtures import read_pinned_model


@pytest.fixture
def migrated(robin):
    assert robin("db", "migrate").code == 0


@pytest.fixture
def owl(tmp_path):
    """`models register` arguments for OWL v4, with its files written."""
    return ["owl/v4", *write_owl_files(tmp_path / "owl")]


def register_versions(database, versions):
    # Only owl/v4 has an adapter, so other versions are stored directly, for paging.
    owl = read_pinned_model("owl-v4")
    for version in versions:
        card = owl.card.model_copy(update={"model_version": version})
        models.register_model(database, owl.model_copy(update={"card": card}), actor="ana")


def test_register_prints_the_model_as_json(robin, migrated, owl):
    ran = robin("--json", "models", "register", *owl)

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert (body["model_id"], body["created"]) == ("owl/v4", True)
    assert body["card_digest"].startswith("sha256:v1:")


def test_register_says_what_happened(robin, migrated, owl):
    assert robin("models", "register", *owl).stdout == "Registered owl/v4.\n"
    assert robin("models", "register", *owl).stdout == "owl/v4 was already registered.\n"


def test_a_changed_file_under_the_same_model_exits_1(robin, migrated, owl, tmp_path):
    robin("models", "register", *owl)
    (tmp_path / "owl" / "weights.h5").write_bytes(b"other weights")

    ran = robin("--json", "models", "register", *owl)

    assert ran.code == 1
    assert ran.error()["reason"] == "model_conflict"
    assert ran.stdout == ""


def test_a_card_file_registers_the_same_model(robin, migrated, tmp_path):
    card = tmp_path / "card.yaml"
    card.write_text((files("robin_models.owl") / "resources" / "card.yaml").read_text())

    ran = robin(
        "--json", "models", "register", "--card", str(card), *write_owl_files(tmp_path / "owl")
    )

    assert ran.code == 0, ran.stderr
    assert json.loads(ran.stdout)["model_id"] == "owl/v4"


def test_a_relative_path_is_stored_as_an_absolute_uri(robin, migrated, tmp_path, monkeypatch):
    write_owl_files(tmp_path / "owl")
    monkeypatch.chdir(tmp_path)

    ran = robin(
        "models", "register", "owl/v4",
        "--file", "weights=owl/weights.h5", "--file", "taxa_registry=./owl/taxa_registry.csv",
    )

    assert ran.code == 0, ran.stderr
    stored = json.loads(robin("--json", "models", "show", "owl/v4").stdout)["files"]
    assert stored["weights"]["uri"] == (tmp_path / "owl" / "weights.h5").as_uri()
    assert stored["taxa_registry"]["uri"] == (tmp_path / "owl" / "taxa_registry.csv").as_uri()


def test_a_built_in_model_without_a_registry_file_takes_the_one_robin_models_ships(
    robin, migrated, tmp_path
):
    weights, _ = write_owl_files(tmp_path / "owl")[1::2]

    ran = robin("models", "register", "owl/v4", "--file", weights)

    assert ran.code == 0, ran.stderr
    stored = json.loads(robin("--json", "models", "show", "owl/v4").stdout)["files"]
    assert stored["taxa_registry"]["uri"] == "robin-models:owl/v4/taxa_registry.csv"


def test_a_relative_symlink_is_stored_by_its_own_path(robin, migrated, tmp_path, monkeypatch):
    # The link, not its target, is the location: pointing it at a changed file is a conflict.
    write_owl_files(tmp_path / "owl")
    (tmp_path / "links").mkdir()
    (tmp_path / "links" / "weights.h5").symlink_to(tmp_path / "owl" / "weights.h5")
    monkeypatch.chdir(tmp_path)

    ran = robin(
        "models", "register", "owl/v4",
        "--file", "weights=links/weights.h5", "--file", "taxa_registry=owl/taxa_registry.csv",
    )

    assert ran.code == 0, ran.stderr
    stored = json.loads(robin("--json", "models", "show", "owl/v4").stdout)["files"]
    assert stored["weights"]["uri"] == (tmp_path / "links" / "weights.h5").as_uri()


def test_a_missing_file_exits_1_with_invalid_request(robin, migrated, owl, tmp_path):
    (tmp_path / "owl" / "weights.h5").unlink()

    ran = robin("--json", "models", "register", *owl)

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_request"


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["--file", "weights=/w.h5"], id="no_model_and_no_card"),
        pytest.param(["owl/v4", "--card", "card.yaml", "--file", "weights=/w.h5"],
                     id="a_model_and_a_card"),
        pytest.param(["owl/v4"], id="no_files"),
        pytest.param(["owl/v4", "--file", "/w.h5"], id="a_file_with_no_role"),
        pytest.param(["owl/v4", "--file", "weights=/a.h5", "--file", "weights=/b.h5"],
                     id="a_repeated_role"),
    ],
)
def test_register_arguments_that_do_not_fit_exit_2(robin, arguments):
    assert robin("models", "register", *arguments).code == 2


def test_a_database_that_is_not_migrated_exits_1(robin):
    ran = robin("--json", "models", "list")

    assert ran.code == 1
    assert ran.error()["reason"] == "schema_not_current"


def test_list_prints_the_page_envelope(robin, migrated, owl):
    robin("models", "register", *owl)

    ran = robin("--json", "models", "list")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert [item["model_id"] for item in body["items"]] == ["owl/v4"]
    assert body["next_cursor"] is None


def test_list_shows_the_next_cursor_on_stderr(robin, migrated, database):
    register_versions(database, ["v4", "v5"])

    ran = robin("models", "list", "--limit", "1")

    assert ran.code == 0, ran.stderr
    assert ran.stdout.splitlines()[0].startswith("owl/v4")
    assert len(ran.stdout.splitlines()) == 1
    assert "--cursor" in ran.stderr


def test_a_cursor_gives_the_next_page(robin, migrated, database):
    register_versions(database, ["v4", "v5"])
    cursor = json.loads(robin("--json", "models", "list", "--limit", "1").stdout)["next_cursor"]

    ran = robin("--json", "models", "list", "--limit", "1", "--cursor", cursor)

    assert [item["model_id"] for item in json.loads(ran.stdout)["items"]] == ["owl/v5"]


def test_all_follows_every_page_into_one_envelope(robin, migrated, database):
    register_versions(database, ["v4", "v5", "v6", "v7", "v8"])

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


def test_show_prints_the_card_and_files(robin, migrated, owl):
    robin("models", "register", *owl)

    ran = robin("--json", "models", "show", "owl/v4")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert body["card"]["model_name"] == "owl"
    assert body["files"]["weights"]["uri"].endswith("/owl/weights.h5")


def test_show_of_a_missing_model_exits_1(robin, migrated):
    ran = robin("--json", "models", "show", "owl/v9")

    assert ran.code == 1
    assert ran.error()["reason"] == "not_found"


def test_all_prints_nothing_when_a_later_page_fails(robin, migrated, database, monkeypatch):
    register_versions(database, ["v4", "v5", "v6"])
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
