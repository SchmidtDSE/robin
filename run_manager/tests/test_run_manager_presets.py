"""Presets: a name for a configuration. The same name and configuration return one row."""

import threading
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Json

from robin_contracts.cards import card_digest
from robin_contracts.work import AudioInput, PinnedModel
from robin_run_manager.configuration import (
    Configuration,
    UnresolvedConfiguration,
    get_configuration_digest,
    read_configuration,
    resolve_configuration,
)
from robin_run_manager.database import transaction
from robin_run_manager.errors import (
    INVALID_CONFIGURATION,
    INVALID_REQUEST,
    NOT_FOUND,
    PRESET_CONFLICT,
    UNKNOWN_CONFIGURATION,
    UNKNOWN_MODEL,
    Refused,
)
from robin_run_manager.models import register_model, show_model
from robin_run_manager.presets import create_preset, list_presets, show_preset

from model_fixtures import read_pinned_model

FIXTURES = Path(__file__).parent / "fixtures"
OWL_CONFIGURATION = read_configuration(FIXTURES / "configuration-owl-v4.yaml")
PERCH_CONFIGURATION = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
HEAD_CONFIGURATION = read_configuration(FIXTURES / "configuration-songbird-head-v1.yaml")


@pytest.fixture
def registered(migrated_database):
    for name in ("owl-v4", "perch-v8", "songbird-head-v1"):
        register_model(migrated_database, read_pinned_model(name), actor="ana")
    return migrated_database


def other_owl_configuration() -> UnresolvedConfiguration:
    return UnresolvedConfiguration.model_validate(
        OWL_CONFIGURATION.model_dump() | {"settings": {"window_overlap": 6.0}}
    )


def connection_values(settings):
    return {
        "host": settings.host,
        "port": settings.port,
        "dbname": settings.dbname,
        "user": settings.user,
        "password": settings.password.get_secret_value(),
    }


def count(database, table: str) -> int:
    with transaction(database, read_only=True) as connection:
        return connection.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


def test_a_new_preset_is_created(registered):
    created = create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana")

    assert created.name == "owl-default"
    assert created.configuration_digest.startswith("sha256:v1:")
    assert created.created
    assert (count(registered, "presets"), count(registered, "configurations")) == (1, 1)


@pytest.mark.parametrize("name", ["", "  "])
def test_an_empty_name_is_an_invalid_request(registered, name):
    with pytest.raises(Refused) as refused:
        create_preset(registered, name, OWL_CONFIGURATION, actor="ana")

    assert refused.value.reason == INVALID_REQUEST
    assert count(registered, "presets") == 0


def test_the_same_name_and_configuration_return_the_existing_preset(registered):
    first = create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana")

    again = create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ben")

    assert again.configuration_digest == first.configuration_digest
    assert not again.created
    with transaction(registered, read_only=True) as connection:
        row = connection.execute("SELECT created_by FROM presets").fetchone()
    assert row["created_by"] == "ana"


def test_the_same_name_with_another_configuration_is_a_conflict(registered):
    create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana")

    with pytest.raises(Refused) as refused:
        create_preset(registered, "owl-default", other_owl_configuration(), actor="ana")

    assert refused.value.reason == PRESET_CONFLICT
    assert "owl-default" in refused.value.message
    # Nothing changed: not even the other configuration was stored.
    assert (count(registered, "presets"), count(registered, "configurations")) == (1, 1)


def test_two_names_for_one_configuration_share_one_configuration_row(registered):
    one = create_preset(registered, "owl-a", OWL_CONFIGURATION, actor="ana")
    two = create_preset(registered, "owl-b", OWL_CONFIGURATION, actor="ana")

    assert one.configuration_digest == two.configuration_digest
    assert (count(registered, "presets"), count(registered, "configurations")) == (2, 1)


def test_a_head_preset_names_the_backbone_presets_configuration(registered):
    backbone = create_preset(registered, "perch-embeddings", PERCH_CONFIGURATION, actor="ana")
    assert HEAD_CONFIGURATION.input.backbone_configuration == backbone.configuration_digest

    head = create_preset(registered, "songbird", HEAD_CONFIGURATION, actor="ana")

    assert head.created


def test_a_head_preset_from_a_backbone_without_embeddings_is_refused(registered):
    scores_only = UnresolvedConfiguration.model_validate(
        PERCH_CONFIGURATION.model_dump() | {"outputs": [PERCH_CONFIGURATION.model_dump()["outputs"][0]]}
    )
    backbone = create_preset(registered, "perch-scores", scores_only, actor="ana")
    head = UnresolvedConfiguration.model_validate(
        HEAD_CONFIGURATION.model_dump()
        | {"input": {"kind": "embedding_artifact",
                     "backbone_configuration": backbone.configuration_digest}}
    )

    with pytest.raises(Refused) as refused:
        create_preset(registered, "songbird", head, actor="ana")

    assert refused.value.reason == INVALID_CONFIGURATION
    assert refused.value.details["code"] == "backbone_requests_no_embeddings"


def test_a_configuration_naming_an_unregistered_model_is_refused(migrated_database):
    with pytest.raises(Refused) as refused:
        create_preset(migrated_database, "owl-default", OWL_CONFIGURATION, actor="ana")

    assert refused.value.reason == UNKNOWN_MODEL
    assert "owl/v4" in refused.value.message


def test_a_head_naming_an_unknown_backbone_configuration_is_refused(registered):
    with pytest.raises(Refused) as refused:
        create_preset(registered, "songbird", HEAD_CONFIGURATION, actor="ana")

    assert refused.value.reason == UNKNOWN_CONFIGURATION
    assert HEAD_CONFIGURATION.input.backbone_configuration in refused.value.message


def test_a_presets_digest_is_the_digest_of_the_resolved_file(registered):
    owl = read_pinned_model("owl-v4")

    created = create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana")

    assert created.configuration_digest == get_configuration_digest(
        resolve_configuration(OWL_CONFIGURATION, owl, AudioInput())
    )


def test_a_create_waits_for_a_concurrent_one_and_then_conflicts(registered, database_settings):
    owl = read_pinned_model("owl-v4")
    other_configuration = resolve_configuration(other_owl_configuration(), owl, AudioInput())
    other_digest = get_configuration_digest(other_configuration)
    outcomes = []

    def create():
        try:
            outcomes.append(create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana"))
        except Refused as refused:
            outcomes.append(refused.reason)

    with psycopg.connect(**connection_values(database_settings)) as other:
        # The same name for another configuration, made by another session and not
        # committed yet.
        other.execute(
            "INSERT INTO configurations (digest, model_id, configuration) VALUES (%s, %s, %s)",
            (other_digest, "owl/v4", Json(other_configuration.model_dump(mode="json"))),
        )
        other.execute(
            "INSERT INTO presets (name, configuration_digest, created_by) VALUES (%s, %s, %s)",
            ("owl-default", other_digest, "ben"),
        )
        thread = threading.Thread(target=create)
        thread.start()
        thread.join(timeout=1)
        assert thread.is_alive(), "the create did not wait for the other session"
        other.commit()
    thread.join()

    assert outcomes == [PRESET_CONFLICT]
    # The losing create stored nothing, not even its configuration.
    assert (count(registered, "presets"), count(registered, "configurations")) == (1, 1)


def test_a_listed_preset_shows_its_summary_and_no_file_locations(registered):
    created = create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana")

    [item] = list_presets(registered).items

    assert item == {
        "name": "owl-default",
        "configuration_digest": created.configuration_digest,
        "model_name": "owl",
        "model_version": "v4",
        "output_kinds": ["scores"],
        "created_by": "ana",
        "created_at": item["created_at"],
    }
    assert isinstance(item["created_at"], str)


def test_showing_a_preset_gives_its_resolved_configuration(registered):
    create_preset(registered, "owl-default", OWL_CONFIGURATION, actor="ana")

    shown = show_preset(registered, "owl-default")

    assert shown["configuration"]["model"]["name"] == "owl"
    assert shown["configuration"]["settings"] == {"window_overlap": 0.0}
    assert "s3://" not in str(shown)


def test_showing_a_missing_preset_is_not_found(registered):
    with pytest.raises(Refused) as refused:
        show_preset(registered, "absent")

    assert refused.value.reason == NOT_FOUND


def test_stored_documents_read_back_with_the_same_digests(migrated_database):
    # A float of 1e16 or more is where a stored number could come back as an int.
    owl = read_pinned_model("owl-v4")
    card = owl.card.model_copy(update={"model_name": "wide", "window_duration": 1e17})
    registered = register_model(migrated_database, owl.model_copy(update={"card": card}), actor="ana")
    file = UnresolvedConfiguration.model_validate(
        OWL_CONFIGURATION.model_dump() | {"model": "wide/v4", "settings": {"window_overlap": 1e16}}
    )
    created = create_preset(migrated_database, "wide", file, actor="ana")

    shown_model = show_model(migrated_database, "wide/v4")
    shown_preset = show_preset(migrated_database, "wide")

    stored = PinnedModel.model_validate({"card": shown_model["card"], "files": shown_model["files"]})
    assert card_digest(stored.card) == registered.card_digest
    configuration = Configuration.model_validate(shown_preset["configuration"])
    assert get_configuration_digest(configuration) == created.configuration_digest
