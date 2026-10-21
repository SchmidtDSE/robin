"""Model registration: the same file twice returns one row; other content is refused."""

import threading
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Json

from robin_contracts.cards import card_digest
from robin_run_manager.database import transaction
from robin_run_manager.errors import INVALID_REQUEST, MODEL_CONFLICT, NOT_FOUND, Refused
from robin_run_manager.models import list_models, read_registration, register_model, show_model

FIXTURES = Path(__file__).parent / "fixtures"
OWL = FIXTURES / "owl-v4.yaml"


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
    model = read_registration(OWL)

    registered = register_model(migrated_database, model, actor="ana")

    assert registered.model_id == "owl/v4"
    assert registered.card_digest == card_digest(model.card)
    assert registered.created
    [row] = model_rows(migrated_database)
    assert (row["model_id"], row["registered_by"]) == ("owl/v4", "ana")
    assert row["files"]["weights"]["uri"] == "s3://robin-models/owl/v4/weights.h5"


def test_the_same_file_again_returns_the_existing_row(migrated_database):
    model = read_registration(OWL)
    first = register_model(migrated_database, model, actor="ana")

    again = register_model(migrated_database, model, actor="ben")

    assert again.model_id == first.model_id
    assert again.card_digest == first.card_digest
    assert not again.created
    [row] = model_rows(migrated_database)
    assert row["registered_by"] == "ana"


def test_a_moved_file_under_the_same_identity_is_a_conflict(migrated_database):
    model = read_registration(OWL)
    register_model(migrated_database, model, actor="ana")
    files = dict(model.files)
    files["weights"] = files["weights"].model_copy(update={"uri": "s3://elsewhere/w.h5"})

    with pytest.raises(Refused) as refused:
        register_model(migrated_database, model.model_copy(update={"files": files}), actor="ana")

    assert refused.value.reason == MODEL_CONFLICT
    assert "owl/v4" in refused.value.message and "files" in refused.value.message
    assert len(model_rows(migrated_database)) == 1


def test_another_card_under_the_same_identity_is_a_conflict(migrated_database):
    model = read_registration(OWL)
    register_model(migrated_database, model, actor="ana")
    card = model.card.model_copy(update={"window_duration": 6.0})

    with pytest.raises(Refused) as refused:
        register_model(migrated_database, model.model_copy(update={"card": card}), actor="ana")

    assert refused.value.reason == MODEL_CONFLICT
    assert "card" in refused.value.message
    [row] = model_rows(migrated_database)
    assert row["card_digest"] == card_digest(model.card)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("card: [unclosed", id="not_yaml"),
        pytest.param("- a list\n", id="not_a_mapping"),
        pytest.param(OWL.read_text() + "files: {}\n", id="a_repeated_key"),
        pytest.param(OWL.read_text().replace("sample_rate: 8000", "sample_rate: -1"),
                     id="a_field_error"),
        pytest.param(OWL.read_text() + "extra: 1\n", id="an_unknown_field"),
    ],
)
def test_a_bad_registration_file_is_an_invalid_request(tmp_path, text):
    path = tmp_path / "model.yaml"
    path.write_text(text)

    with pytest.raises(Refused) as refused:
        read_registration(path)

    assert refused.value.reason == INVALID_REQUEST
    assert str(path) in refused.value.message


def test_a_missing_registration_file_is_an_invalid_request(tmp_path):
    with pytest.raises(Refused) as refused:
        read_registration(tmp_path / "absent.yaml")

    assert refused.value.reason == INVALID_REQUEST


def test_a_registration_waits_for_a_concurrent_one_and_returns_its_row(
    migrated_database, database_settings
):
    model = read_registration(OWL)
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
    register_model(migrated_database, read_registration(OWL), actor="ana")

    [item] = list_models(migrated_database).items

    assert item["model_id"] == "owl/v4"
    assert item["kind"] == "model"
    assert item["card_digest"].startswith("sha256:v1:")
    assert item["output_version_id"] == "model:owl/v4"
    assert item["registered_by"] == "ana"
    assert isinstance(item["registered_at"], str)
    assert "card" not in item and "files" not in item


def test_a_head_is_listed_as_a_head(migrated_database):
    register_model(migrated_database, read_registration(FIXTURES / "songbird-head-v1.yaml"), actor="ana")

    [item] = list_models(migrated_database).items

    assert (item["kind"], item["output_version_id"]) == ("head", "head_runtime:onnx")


def test_showing_a_model_gives_its_card_and_files(migrated_database):
    model = read_registration(OWL)
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
    path = tmp_path / "model.yaml"
    path.write_text(OWL.read_text().replace("sample_rate: 8000", "sample_rate: -12345"))

    with pytest.raises(Refused) as refused:
        read_registration(path)

    fields = [problem["field"] for problem in refused.value.details["problems"]]
    assert any(field.endswith("sample_rate") for field in fields)
    assert "-12345" not in str(refused.value.details)
