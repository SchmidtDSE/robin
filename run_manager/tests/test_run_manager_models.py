"""Model registration: preparing a model from its files, then storing it once."""

import hashlib
import threading
from importlib.resources import files

import psycopg
import pytest
from psycopg.types.json import Json

from robin_contracts.cards import card_digest
from robin_run_manager.database import transaction
from robin_run_manager.errors import INVALID_REQUEST, MODEL_CONFLICT, NOT_FOUND, Refused
from robin_run_manager.models import (
    get_built_in_card,
    list_models,
    prepare_built_in_model,
    prepare_model,
    read_card_file,
    register_model,
    show_model,
)

from model_fixtures import read_pinned_model

OWL_REGISTRY = files("robin_models.owl") / "resources" / "taxa_registry.csv"
WEIGHTS = b"bytes that stand for the weights"


def connection_values(settings):
    return {
        "host": settings.host,
        "port": settings.port,
        "dbname": settings.dbname,
        "user": settings.user,
        "password": settings.password.get_secret_value(),
    }


def model_rows(database):
    with transaction(database, read_only=True) as connection:
        return connection.execute(
            "SELECT model_id, card_digest, files, registered_by FROM models ORDER BY model_id"
        ).fetchall()


def test_a_new_model_is_registered(migrated_database):
    model = read_pinned_model("owl-v4")

    registered = register_model(migrated_database, model, actor="ana")

    assert registered.model_id == "owl/v4"
    assert registered.card_digest == card_digest(model.card)
    assert registered.created
    [row] = model_rows(migrated_database)
    assert (row["model_id"], row["registered_by"]) == ("owl/v4", "ana")
    assert row["files"]["weights"]["uri"] == "s3://robin-models/owl/v4/weights.h5"


def test_the_same_file_again_returns_the_existing_row(migrated_database):
    model = read_pinned_model("owl-v4")
    first = register_model(migrated_database, model, actor="ana")

    again = register_model(migrated_database, model, actor="ben")

    assert again.model_id == first.model_id
    assert again.card_digest == first.card_digest
    assert not again.created
    [row] = model_rows(migrated_database)
    assert row["registered_by"] == "ana"


def test_a_moved_file_under_the_same_identity_is_a_conflict(migrated_database):
    model = read_pinned_model("owl-v4")
    register_model(migrated_database, model, actor="ana")
    files = dict(model.files)
    files["weights"] = files["weights"].model_copy(update={"uri": "s3://elsewhere/w.h5"})

    with pytest.raises(Refused) as refused:
        register_model(migrated_database, model.model_copy(update={"files": files}), actor="ana")

    assert refused.value.reason == MODEL_CONFLICT
    assert "owl/v4" in refused.value.message and "files" in refused.value.message
    assert len(model_rows(migrated_database)) == 1


def test_another_card_under_the_same_identity_is_a_conflict(migrated_database):
    model = read_pinned_model("owl-v4")
    register_model(migrated_database, model, actor="ana")
    card = model.card.model_copy(update={"window_duration": 6.0})

    with pytest.raises(Refused) as refused:
        register_model(migrated_database, model.model_copy(update={"card": card}), actor="ana")

    assert refused.value.reason == MODEL_CONFLICT
    assert "card" in refused.value.message
    [row] = model_rows(migrated_database)
    assert row["card_digest"] == card_digest(model.card)


def write_owl_card(tmp_path, change=lambda text: text):
    path = tmp_path / "card.yaml"
    card = (files("robin_models.owl") / "resources" / "card.yaml").read_text()
    path.write_text(change(card))
    return path


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda text: "model_name: [unclosed", id="not_yaml"),
        pytest.param(lambda text: "- a list\n", id="not_a_mapping"),
        pytest.param(lambda text: text + "sample_rate: 8000\n", id="a_repeated_key"),
        pytest.param(lambda text: text.replace("sample_rate: 8000", "sample_rate: -1"),
                     id="a_field_error"),
        pytest.param(lambda text: text + "extra: 1\n", id="an_unknown_field"),
    ],
)
def test_a_bad_card_file_is_an_invalid_request(tmp_path, change):
    path = write_owl_card(tmp_path, change)

    with pytest.raises(Refused) as refused:
        read_card_file(path)

    assert refused.value.reason == INVALID_REQUEST
    assert str(path) in refused.value.message


def test_a_missing_card_file_is_an_invalid_request(tmp_path):
    with pytest.raises(Refused) as refused:
        read_card_file(tmp_path / "absent.yaml")

    assert refused.value.reason == INVALID_REQUEST


@pytest.fixture
def owl_files(tmp_path):
    weights = tmp_path / "weights.h5"
    weights.write_bytes(WEIGHTS)
    registry = tmp_path / "taxa_registry.csv"
    registry.write_bytes(OWL_REGISTRY.read_bytes())
    return {"weights": str(weights), "taxa_registry": registry.as_uri()}


def test_preparation_pins_each_file_by_its_bytes(owl_files, tmp_path):
    model = prepare_model(get_built_in_card("owl/v4"), owl_files)

    weights = model.files["weights"]
    assert weights.uri == (tmp_path / "weights.h5").as_uri()
    assert weights.digest == "sha256:" + hashlib.sha256(WEIGHTS).hexdigest()
    assert weights.size_bytes == len(WEIGHTS)
    assert model.files["taxa_registry"].uri == owl_files["taxa_registry"]


def test_a_built_in_model_without_a_registry_pins_the_one_robin_models_ships(owl_files):
    model = prepare_built_in_model("owl/v4", {"weights": owl_files["weights"]})

    registry = model.files["taxa_registry"]
    assert registry.uri == "robin-models:owl/v4/taxa_registry.csv"
    assert registry.digest == "sha256:" + hashlib.sha256(OWL_REGISTRY.read_bytes()).hexdigest()
    assert registry.size_bytes == len(OWL_REGISTRY.read_bytes())


def test_a_built_in_model_keeps_a_registry_it_is_given(owl_files):
    model = prepare_built_in_model("owl/v4", owl_files)

    assert model.files["taxa_registry"].uri == owl_files["taxa_registry"]


# Only the default for a built-in model makes this location, so there is one way to say it.
@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(lambda files: prepare_built_in_model("owl/v4", files), id="built_in"),
        pytest.param(lambda files: prepare_model(get_built_in_card("owl/v4"), files), id="card"),
    ],
)
def test_a_bundled_location_given_with_a_file_is_refused(owl_files, prepare):
    location = "robin-models:owl/v4/taxa_registry.csv"

    with pytest.raises(Refused) as refused:
        prepare(owl_files | {"taxa_registry": location})

    assert refused.value.reason == INVALID_REQUEST
    assert refused.value.details == {"role": "taxa_registry", "location": location}


def test_an_unknown_built_in_model_is_refused_with_the_known_ones(owl_files):
    with pytest.raises(Refused) as refused:
        prepare_built_in_model("owl/v9", {"weights": owl_files["weights"]})

    assert refused.value.reason == INVALID_REQUEST
    assert "owl/v4" in refused.value.details["built_in_models"]


def test_a_registry_that_the_card_does_not_name_is_refused(owl_files, tmp_path):
    (tmp_path / "taxa_registry.csv").write_text("another registry\n")

    with pytest.raises(Refused) as refused:
        prepare_model(get_built_in_card("owl/v4"), owl_files)

    assert refused.value.reason == INVALID_REQUEST
    assert refused.value.details["role"] == "taxa_registry"


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        pytest.param(lambda files: {"taxa_registry": files["taxa_registry"]}, "missing_roles",
                     id="a_missing_role"),
        pytest.param(lambda files: files | {"extra": files["weights"]}, "unknown_roles",
                     id="an_unknown_role"),
    ],
)
def test_roles_that_do_not_suit_the_model_are_refused(owl_files, change, problem):
    with pytest.raises(Refused) as refused:
        prepare_model(get_built_in_card("owl/v4"), change(owl_files))

    assert refused.value.reason == INVALID_REQUEST
    assert refused.value.details[problem]


def test_a_model_without_its_score_roles_is_prepared(tmp_path):
    layout = {}
    for role in ("saved_model", "variables_data", "variables_index"):
        (tmp_path / role).write_bytes(role.encode())
        layout[role] = str(tmp_path / role)

    model = prepare_model(get_built_in_card("perch/v8"), layout)

    assert set(model.files) == set(layout)


def perch_scores_files(tmp_path) -> dict[str, str]:
    layout = {}
    for role in ("saved_model", "variables_data", "variables_index"):
        (tmp_path / role).write_bytes(role.encode())
        layout[role] = str(tmp_path / role)
    registry = tmp_path / "taxa_registry.csv"
    registry.write_bytes((files("robin_models.perch") / "resources" / "taxa_registry.csv").read_bytes())
    return layout | {"taxa_registry": str(registry)}


def test_a_registry_alone_gives_perch_its_scores(tmp_path):
    layout = perch_scores_files(tmp_path)

    model = prepare_model(get_built_in_card("perch/v8"), layout)

    assert set(model.files) == set(layout)


def test_a_labels_file_is_a_role_perch_does_not_read(tmp_path):
    (tmp_path / "labels.csv").write_bytes(b"ebird2021\n")
    layout = perch_scores_files(tmp_path) | {"labels": str(tmp_path / "labels.csv")}

    with pytest.raises(Refused) as refused:
        prepare_model(get_built_in_card("perch/v8"), layout)

    assert refused.value.reason == INVALID_REQUEST
    assert refused.value.details["unknown_roles"] == ["labels"]


def test_an_unreadable_file_is_refused_with_its_role(owl_files, tmp_path):
    (tmp_path / "weights.h5").unlink()

    with pytest.raises(Refused) as refused:
        prepare_model(get_built_in_card("owl/v4"), owl_files)

    assert refused.value.reason == INVALID_REQUEST
    assert refused.value.details["role"] == "weights"


# Each form names the real weights file, so only its form can be wrong. The contracts
# tests check each form that is not local; these check that registration refuses one,
# and refuses a local path that the system cannot open.
@pytest.mark.parametrize(
    "spell",
    [
        pytest.param(lambda path: f"file://host{path}", id="a_file_uri_with_a_host"),
        pytest.param(lambda path: f"{path.as_uri()}%00", id="a_null_byte"),
    ],
)
def test_a_location_robin_cannot_read_here_is_refused(owl_files, tmp_path, spell):
    location = spell(tmp_path / "weights.h5")

    with pytest.raises(Refused) as refused:
        prepare_model(get_built_in_card("owl/v4"), owl_files | {"weights": location})

    assert refused.value.reason == INVALID_REQUEST
    assert refused.value.details == {"role": "weights", "location": location}


def test_a_model_with_no_adapter_is_refused(owl_files):
    card = get_built_in_card("owl/v4").model_copy(update={"model_version": "v9"})

    with pytest.raises(Refused) as refused:
        prepare_model(card, owl_files)

    assert refused.value.reason == INVALID_REQUEST


def test_a_registration_waits_for_a_concurrent_one_and_returns_its_row(
    migrated_database, database_settings
):
    model = read_pinned_model("owl-v4")
    files = {role: pinned.model_dump(mode="json") for role, pinned in model.files.items()}
    results = []
    with psycopg.connect(**connection_values(database_settings)) as other:
        # The same registration, made by another session and not committed yet.
        other.execute(
            "INSERT INTO models (model_id, card, card_digest, files, registered_by) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("owl/v4", Json(model.card.model_dump(mode="json")), card_digest(model.card),
             Json(files), "ben"),
        )
        thread = threading.Thread(
            target=lambda: results.append(register_model(migrated_database, model, actor="ana"))
        )
        thread.start()
        thread.join(timeout=1)
        assert thread.is_alive(), "the registration did not wait for the other session"
        other.commit()
    thread.join()

    assert (results[0].model_id, results[0].created) == ("owl/v4", False)
    [row] = model_rows(migrated_database)
    assert row["registered_by"] == "ben"


def test_a_listed_model_shows_its_summary(migrated_database):
    register_model(migrated_database, read_pinned_model("owl-v4"), actor="ana")

    [item] = list_models(migrated_database).items

    assert item["model_id"] == "owl/v4"
    assert item["kind"] == "model"
    assert item["card_digest"].startswith("sha256:v1:")
    assert item["output_version_id"] == "model:owl/v4"
    assert item["registered_by"] == "ana"
    assert isinstance(item["registered_at"], str)
    assert "card" not in item and "files" not in item


def test_a_head_is_listed_as_a_head(migrated_database):
    register_model(migrated_database, read_pinned_model("songbird-head-v1"), actor="ana")

    [item] = list_models(migrated_database).items

    assert (item["kind"], item["output_version_id"]) == ("head", "head_runtime:onnx")


def test_showing_a_model_gives_its_card_and_files(migrated_database):
    model = read_pinned_model("owl-v4")
    register_model(migrated_database, model, actor="ana")

    shown = show_model(migrated_database, "owl/v4")

    assert shown["card"] == model.card.model_dump(mode="json")
    assert shown["files"]["weights"]["uri"] == "s3://robin-models/owl/v4/weights.h5"
    assert shown["model_id"] == "owl/v4"


def test_showing_a_missing_model_is_not_found(migrated_database):
    with pytest.raises(Refused) as refused:
        show_model(migrated_database, "owl/v9")

    assert refused.value.reason == NOT_FOUND


def test_a_field_problem_names_the_field_and_not_its_value(tmp_path):
    path = write_owl_card(tmp_path, lambda text: text.replace("sample_rate: 8000", "sample_rate: -12345"))

    with pytest.raises(Refused) as refused:
        read_card_file(path)

    fields = [problem["field"] for problem in refused.value.details["problems"]]
    assert any(field.endswith("sample_rate") for field in fields)
    assert "-12345" not in str(refused.value.details)
