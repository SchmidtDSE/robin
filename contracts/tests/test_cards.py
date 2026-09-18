import json

import pytest
from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes, sha256_v1
from robin_contracts.cards import HeadCard, ModelCard, ModelRef, card_digest
from robin_contracts.embedding_transforms import L2Norm

CARD_DIGEST = "sha256:v1:" + "a" * 64


def _head_card() -> HeadCard:
    return HeadCard(
        model_name="nutria",
        model_version="v1",
        backbone=ModelRef(name="perch", version="v8", digest=CARD_DIGEST),
        weights_uri="weights.onnx",
        classes=("nutria",),
        required_embedding_transform=L2Norm(),
        taxa_registry_uri="taxa.csv",
    )


def _model_card() -> ModelCard:
    return ModelCard(
        model_name="perch",
        model_version="v8",
        runtime="tf-saved-model",
        segment_duration=5.0,
        sample_rate=32000,
        min_detection_threshold=0.01,
    )


def test_model_ref_has_a_stable_human_identifier():
    assert ModelRef(name="perch", version="v8", digest=CARD_DIGEST).id == "perch/v8"


@pytest.mark.parametrize(
    "digest",
    ["not-a-digest", "a" * 16, "a" * 64, "sha256:" + "a" * 64, "sha256:v1:" + "a" * 40],
)
def test_model_ref_refuses_a_digest_it_could_not_have_produced(digest):
    with pytest.raises(ValidationError) as exc:
        ModelRef(name="perch", version="v8", digest=digest)

    assert exc.value.errors()[0]["loc"] == ("digest",)


def test_model_ref_accepts_what_card_digest_produces():
    assert ModelRef(
        name="perch", version="v8", digest=card_digest(_model_card())
    ).digest.startswith("sha256:v1:")


def test_head_card_requires_an_embedding_transform():
    with pytest.raises(ValidationError):
        HeadCard.model_validate({
            "model_name": "nutria",
            "model_version": "v1",
            "backbone": {"name": "perch", "version": "v8", "digest": "a" * 16},
            "weights_uri": "weights.onnx",
            "classes": ["nutria"],
            "taxa_registry_uri": "taxa.csv",
        })


def test_head_card_serializes_its_required_transform():
    card = _head_card()

    assert card.model_dump(mode="json")["required_embedding_transform"] == {
        "kind": "l2",
    }


def test_model_card_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ModelCard.model_validate({
            "model_name": "perch",
            "model_version": "v8",
            "runtime": "tf-saved-model",
            "segment_duration": 5.0,
            "sample_rate": 32000,
            "min_detection_threshold": 0.01,
            "unknown": True,
        })


def test_model_card_canonical_output_declares_every_field():
    """A card's digest is how the bank addresses it. Dropping null or default
    fields from the encoding would move every digest at once, silently."""
    encoded = json.loads(canonical_json_bytes(_model_card()))

    assert set(encoded) == set(ModelCard.model_fields)


def test_card_digest_delegates_to_the_canonical_digest():
    head = _head_card()

    assert card_digest(head) == sha256_v1(head)
