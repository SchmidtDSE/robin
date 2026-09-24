import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes, sha256_v1
from robin_contracts.cards import (
    HeadCard,
    ModelCard,
    ModelRef,
    card_digest,
    read_card,
    write_card,
)
from robin_contracts.embedding_transforms import L2Norm

CARD_DIGEST = "sha256:v1:" + "a" * 64


def _head_card() -> HeadCard:
    return HeadCard(
        model_name="nutria",
        model_version="v1",
        backbone=ModelRef(name="perch", version="v8", digest=CARD_DIGEST),
        classes=("nutria",),
        required_embedding_transform=L2Norm(),
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
            "backbone": {"name": "perch", "version": "v8", "digest": CARD_DIGEST},
            "classes": ["nutria"],
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


MODEL_CARD_FIELDS = {
    "model_name": "perch",
    "model_version": "v8",
    "runtime": "tf-saved-model",
    "segment_duration": 5.0,
    "sample_rate": 32000,
    "min_detection_threshold": 0.01,
}


# --- Card files ---------------------------------------------------------------


def _write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "card.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("card", [_model_card(), _head_card()], ids=["model", "head"])
def test_a_written_card_reads_back_as_the_same_card(tmp_path, card):
    path = tmp_path / "card.yaml"

    write_card(card, path)
    read = read_card(path)

    assert type(read) is type(card)
    assert read == card
    assert card_digest(read) == card_digest(card)


def test_a_written_card_follows_the_field_order(tmp_path):
    path = tmp_path / "card.yaml"

    write_card(_model_card(), path)
    keys = [line.split(":")[0] for line in path.read_text().splitlines() if line[:1].isalpha()]

    assert keys == list(ModelCard.model_fields)


def test_reading_a_card_keeps_comments_and_layout_out_of_its_digest(tmp_path):
    path = _write_text(
        tmp_path,
        "# a hand-written backbone card\n"
        "sample_rate: 32000   # Hz\n"
        "model_version: v8\n"
        "model_name: perch\n"
        "runtime: tf-saved-model\n"
        "segment_duration: 5.0\n"
        "min_detection_threshold: 0.01\n",
    )

    assert card_digest(read_card(path)) == card_digest(_model_card())


def test_the_same_card_in_two_folders_has_one_digest(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    write_card(_head_card(), first / "card.yaml")
    write_card(_head_card(), second / "card.yaml")

    assert card_digest(read_card(first / "card.yaml")) == card_digest(
        read_card(second / "card.yaml")
    )


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        pytest.param("model_version: 1.10\n", "a float", id="unquoted_float"),
        pytest.param("model_version: no\n", "a bool", id="unquoted_no"),
    ],
)
def test_a_loose_yaml_value_is_refused_not_coerced(tmp_path, text, reason):
    body = "".join(
        f"{key}: {value}\n" for key, value in MODEL_CARD_FIELDS.items() if key != "model_version"
    )
    path = _write_text(tmp_path, body + text)

    with pytest.raises(ValueError, match="model_version") as exc:
        read_card(path)

    assert str(path) in str(exc.value), reason


def test_a_quoted_version_is_text(tmp_path):
    body = "".join(
        f"{key}: {value}\n" for key, value in MODEL_CARD_FIELDS.items() if key != "model_version"
    )
    path = _write_text(tmp_path, body + 'model_version: "1.10"\n')

    assert read_card(path).model_version == "1.10"


def test_a_repeated_key_is_refused(tmp_path):
    body = "".join(f"{key}: {value}\n" for key, value in MODEL_CARD_FIELDS.items())
    path = _write_text(tmp_path, body + "model_version: v9\n")

    with pytest.raises(ValueError, match="model_version") as exc:
        read_card(path)

    assert str(path) in str(exc.value)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("- model_name: perch\n", id="list"),
        pytest.param("perch\n", id="bare_string"),
    ],
)
def test_a_file_that_is_not_a_mapping_is_refused(tmp_path, text):
    path = _write_text(tmp_path, text)

    with pytest.raises(ValueError) as exc:
        read_card(path)

    assert str(path) in str(exc.value)


def test_a_card_with_an_unknown_field_is_refused(tmp_path):
    body = "".join(f"{key}: {value}\n" for key, value in MODEL_CARD_FIELDS.items())
    path = _write_text(tmp_path, body + "colour: brown\n")

    with pytest.raises(ValueError, match="colour") as exc:
        read_card(path)

    assert str(path) in str(exc.value)
    assert isinstance(exc.value.__cause__, ValidationError)


def test_malformed_yaml_is_refused_with_its_cause_kept(tmp_path):
    import yaml

    path = _write_text(tmp_path, "model_name: [perch\n")

    with pytest.raises(ValueError) as exc:
        read_card(path)

    assert str(path) in str(exc.value)
    assert isinstance(exc.value.__cause__, yaml.YAMLError)


def test_a_card_cannot_construct_a_python_object(tmp_path):
    path = _write_text(tmp_path, "model_name: !!python/object/apply:os.getcwd []\n")

    with pytest.raises(ValueError) as exc:
        read_card(path)

    assert str(path) in str(exc.value)


def test_a_missing_card_file_is_a_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_card(tmp_path / "absent.yaml")


def test_a_card_that_is_not_utf8_is_refused_naming_its_path(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_bytes(b"model_name: owl\n\xff\n")

    with pytest.raises(ValueError) as exc:
        read_card(path)

    assert str(path) in str(exc.value)
    assert isinstance(exc.value.__cause__, UnicodeDecodeError)
