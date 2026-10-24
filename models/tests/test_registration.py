"""What registration needs from robin-models: bundled cards, and the file roles each model reads."""

from importlib.resources import files

import pytest

from robin_contracts.cards import HeadCard, model_ref
from robin_models import registration


@pytest.mark.parametrize("model_id", registration.get_built_in_model_ids())
def test_each_built_in_card_is_the_model_it_is_named_for(model_id):
    assert model_ref(registration.get_bundled_card(model_id)).id == model_id


def test_a_head_reads_the_roles_of_its_runtime():
    head = HeadCard.model_validate(
        {
            "model_name": "songbird-head",
            "model_version": "v1",
            "runtime": "onnx",
            "backbone": {"name": "perch", "version": "v8", "digest": "sha256:v1:" + "8" * 64},
            "embedding_dim": 1280,
            "min_detection_threshold": 0.0,
            "score_domain": "sigmoid",
            "taxa_registry_digest": "sha256:" + "3" * 64,
        }
    )

    assert registration.get_file_roles(head) == {registration.GRAPH_ROLE}


def test_a_model_with_no_adapter_has_no_roles():
    card = registration.get_bundled_card("owl/v4").model_copy(update={"model_version": "v9"})

    assert registration.get_file_roles(card) is None


@pytest.mark.parametrize("model_id", registration.get_built_in_model_ids())
def test_a_bundled_registry_location_names_the_registry_this_package_ships(model_id):
    location = registration.get_bundled_registry_location(model_id)
    name = model_id.split("/")[0]
    shipped = files(f"robin_models.{name}") / "resources" / "taxa_registry.csv"

    assert registration.get_bundled_file(location).read_bytes() == shipped.read_bytes()


@pytest.mark.parametrize(
    "location",
    [
        pytest.param("robin-models:owl/v9/taxa_registry.csv", id="an_unknown_version"),
        pytest.param("robin-models:owl/v4/card.yaml", id="another_file"),
        pytest.param("robin-models:owl/v4/../owl/v4/taxa_registry.csv", id="a_relative_part"),
        pytest.param("robin-models:owl/v4", id="no_file"),
        pytest.param("robin-model:owl/v4/taxa_registry.csv", id="another_scheme"),
    ],
)
def test_a_location_that_names_no_bundled_registry_is_refused(location):
    with pytest.raises(ValueError, match="robin-models"):
        registration.get_bundled_file(location)
