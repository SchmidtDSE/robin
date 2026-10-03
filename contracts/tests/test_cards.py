import json
import typing
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes, sha256_v1
from robin_contracts.cards import (
    BackendResampled,
    EmbeddingDtype,
    HeadCard,
    ModelCard,
    ModelRef,
    RunnerResampled,
    card_digest,
    read_card,
    write_card,
)
from robin_contracts.embedding_transforms import L2Norm

CARD_DIGEST = "sha256:v1:" + "a" * 64
REGISTRY_DIGEST = "sha256:" + "a" * 64

MODEL_CARD_FIELDS = {
    "model_name": "perch",
    "model_version": "v8",
    "runtime": "tf-saved-model",
    "window_duration": 5.0,
    "window_overlap": 0.0,
    "sample_rate": 32000,
    "min_detection_threshold": 0.01,
    "score_domain": "probability",
    "taxa_registry_digest": REGISTRY_DIGEST,
    "audio": {
        "downmix": "mean",
        "resampler": {"by": "runner", "algorithm": "soxr_hq"},
        "pad": "centre_crop_end_pad",
    },
    "backend": "tf-saved-model",
    "dtype": "float32",
}


def _head_card() -> HeadCard:
    return HeadCard(
        model_name="nutria",
        model_version="v1",
        backbone=ModelRef(name="perch", version="v8", digest=CARD_DIGEST),
        classes=("nutria",),
        required_embedding_transform=L2Norm(),
    )


def _model_card(**overrides) -> ModelCard:
    return ModelCard.model_validate(MODEL_CARD_FIELDS | overrides)


def _yaml(fields: dict) -> str:
    return yaml.safe_dump(fields, sort_keys=False)


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
        ModelCard.model_validate(MODEL_CARD_FIELDS | {"unknown": True})


# --- What a model emits -------------------------------------------------------


@pytest.mark.parametrize("domain", ["probability", None])
def test_a_card_states_its_score_domain_or_that_it_emits_no_scores(domain):
    digest = REGISTRY_DIGEST if domain else None
    card = _model_card(score_domain=domain, taxa_registry_digest=digest)

    assert card.score_domain == domain


def test_a_card_must_state_its_score_domain():
    fields = {key: value for key, value in MODEL_CARD_FIELDS.items() if key != "score_domain"}

    with pytest.raises(ValidationError) as exc:
        ModelCard.model_validate(fields)

    assert exc.value.errors()[0]["loc"] == ("score_domain",)


def test_a_score_domain_the_engine_does_not_know_is_refused():
    with pytest.raises(ValidationError) as exc:
        _model_card(score_domain="logit")

    assert exc.value.errors()[0]["loc"] == ("score_domain",)


@pytest.mark.parametrize("form", ["omitted", "null"])
def test_a_card_that_emits_scores_must_declare_its_registry_digest(form):
    fields = {
        key: value for key, value in MODEL_CARD_FIELDS.items() if key != "taxa_registry_digest"
    }
    if form == "null":
        fields["taxa_registry_digest"] = None

    with pytest.raises(ValidationError, match="must declare taxa_registry_digest"):
        ModelCard.model_validate(fields)


def test_a_card_that_emits_no_scores_refuses_a_registry_digest():
    with pytest.raises(ValidationError, match="must not declare taxa_registry_digest"):
        _model_card(score_domain=None)


@pytest.mark.parametrize("form", ["omitted", "null"])
def test_a_card_that_emits_no_scores_needs_no_registry_digest(form):
    fields = {
        key: value for key, value in MODEL_CARD_FIELDS.items() if key != "taxa_registry_digest"
    }
    if form == "null":
        fields["taxa_registry_digest"] = None

    card = ModelCard.model_validate(fields | {"score_domain": None})

    assert card.taxa_registry_digest is None


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param("sha256:" + "A" * 64, id="uppercase"),
        pytest.param("sha256:" + "a" * 63, id="63_characters"),
        pytest.param("sha256:v1:" + "a" * 64, id="canonical_json_digest"),
    ],
)
def test_a_malformed_registry_digest_is_refused(digest):
    with pytest.raises(ValidationError) as exc:
        _model_card(taxa_registry_digest=digest)

    assert f"expected 'sha256:' and 64 hex characters, got {digest!r}" in str(exc.value)


def test_embedding_dtype_declares_exactly_the_two_widths():
    assert typing.get_args(EmbeddingDtype) == ("float16", "float32")


def test_a_card_that_emits_embeddings_states_their_width_and_precision():
    card = _model_card(can_emit_embeddings=True, embedding_dim=1280, embedding_dtype="float16")

    assert (card.embedding_dim, card.embedding_dtype) == (1280, "float16")


@pytest.mark.parametrize(
    "missing",
    [{"embedding_dim": 1280}, {"embedding_dtype": "float32"}],
    ids=["no_dtype", "no_dim"],
)
def test_a_card_that_emits_embeddings_without_their_width_or_precision_is_refused(missing):
    with pytest.raises(ValidationError, match="can_emit_embeddings"):
        _model_card(can_emit_embeddings=True, **missing)


def test_an_embedding_precision_the_engine_does_not_store_is_refused():
    with pytest.raises(ValidationError) as exc:
        _model_card(can_emit_embeddings=True, embedding_dim=1280, embedding_dtype="float64")

    assert exc.value.errors()[0]["loc"] == ("embedding_dtype",)


@pytest.mark.parametrize(
    "stray",
    [{"embedding_dim": 1280}, {"embedding_dtype": "float32"}],
    ids=["dim", "dtype"],
)
def test_a_card_that_emits_no_embeddings_refuses_their_width_or_precision(stray):
    with pytest.raises(ValidationError, match="can_emit_embeddings false"):
        _model_card(**stray)


def test_a_card_that_emits_no_embeddings_needs_no_embedding_precision():
    assert _model_card().embedding_dtype is None


@pytest.mark.parametrize(("source", "storage"), [("float32", "float16"), ("float16", "float32")])
def test_a_cards_emitted_precision_may_differ_from_its_storage_precision(source, storage):
    card = _model_card(
        can_emit_embeddings=True, embedding_dim=4, embedding_dtype=source, dtype=storage
    )

    assert (card.embedding_dtype, card.dtype) == (source, storage)


# --- The recipe facts ---------------------------------------------------------


def test_a_cards_audio_handling_reads_as_the_recipe_types():
    audio = _model_card().audio

    assert audio.downmix == "mean"
    assert audio.resampler == RunnerResampled(algorithm="soxr_hq")
    assert audio.pad == "centre_crop_end_pad"


def test_a_card_can_state_that_its_library_resamples():
    resampler = {"by": "backend", "library": "birdnet", "version": "2.4"}
    card = _model_card(audio=MODEL_CARD_FIELDS["audio"] | {"resampler": resampler})

    assert card.audio.resampler == BackendResampled(library="birdnet", version="2.4")


def test_a_card_file_can_state_that_the_runner_resamples_each_window_with_scipy(tmp_path):
    resampler = {"by": "runner", "algorithm": "scipy_fft_per_window"}
    audio = MODEL_CARD_FIELDS["audio"] | {"resampler": resampler}
    path = _write_text(tmp_path, _yaml(MODEL_CARD_FIELDS | {"audio": audio}))

    assert read_card(path).audio.resampler == RunnerResampled(algorithm="scipy_fft_per_window")


def test_a_card_file_can_state_that_the_runner_resamples_with_scipy_polyphase_and_context(
    tmp_path,
):
    resampler = {"by": "runner", "algorithm": "scipy_polyphase_with_context"}
    audio = MODEL_CARD_FIELDS["audio"] | {"resampler": resampler}
    path = _write_text(tmp_path, _yaml(MODEL_CARD_FIELDS | {"audio": audio}))

    assert read_card(path).audio.resampler == RunnerResampled(
        algorithm="scipy_polyphase_with_context"
    )


def test_a_card_file_naming_a_runner_algorithm_the_contract_does_not_is_refused(tmp_path):
    resampler = {"by": "runner", "algorithm": "sinc_best"}
    audio = MODEL_CARD_FIELDS["audio"] | {"resampler": resampler}
    path = _write_text(tmp_path, _yaml(MODEL_CARD_FIELDS | {"audio": audio}))

    with pytest.raises(ValueError, match="algorithm") as exc:
        read_card(path)

    assert str(path) in str(exc.value)


@pytest.mark.parametrize(
    "field", ["window_overlap", "dtype", "audio", "backend"]
)
def test_a_card_file_missing_a_recipe_fact_is_refused_when_read(tmp_path, field):
    fields = {key: value for key, value in MODEL_CARD_FIELDS.items() if key != field}
    path = _write_text(tmp_path, _yaml(fields))

    with pytest.raises(ValueError, match=field) as exc:
        read_card(path)

    assert str(path) in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("downmix", "average"), ("pad", "zero_pad"), ("resampler", "soxr_hq")],
)
def test_a_card_file_with_audio_handling_no_recipe_names_is_refused_when_read(
    tmp_path, field, value
):
    audio = MODEL_CARD_FIELDS["audio"] | {field: value}
    path = _write_text(tmp_path, _yaml(MODEL_CARD_FIELDS | {"audio": audio}))

    with pytest.raises(ValueError, match=field) as exc:
        read_card(path)

    assert str(path) in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("window_duration", 0.0), ("window_duration", -1.0), ("window_overlap", -1.0)],
)
def test_a_card_whose_window_is_not_positive_or_overlap_is_negative_is_refused(field, value):
    with pytest.raises(ValidationError) as exc:
        _model_card(**{field: value})

    assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", ["window_duration", "window_overlap"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_card_whose_window_or_overlap_is_not_finite_is_refused(field, value):
    with pytest.raises(ValidationError) as exc:
        _model_card(**{field: value})

    assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("overlap", [5.0, 6.0])
def test_a_card_whose_windows_would_not_advance_is_refused(overlap):
    with pytest.raises(ValidationError, match="must be less than window_duration"):
        _model_card(window_duration=5.0, window_overlap=overlap)


def test_a_storage_precision_the_engine_does_not_store_is_refused():
    with pytest.raises(ValidationError) as exc:
        _model_card(dtype="float64")

    assert exc.value.errors()[0]["loc"] == ("dtype",)


def test_model_card_canonical_output_declares_every_field():
    """A card's digest is how the bank addresses it. Dropping null or default
    fields from the encoding would move every digest at once, silently."""
    encoded = json.loads(canonical_json_bytes(_model_card()))

    assert set(encoded) == set(ModelCard.model_fields)


def test_card_digest_delegates_to_the_canonical_digest():
    head = _head_card()

    assert card_digest(head) == sha256_v1(head)


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


def test_a_card_file_carrying_a_registry_digest_reads_back_with_it(tmp_path):
    path = _write_text(tmp_path, _yaml(MODEL_CARD_FIELDS))

    assert read_card(path).taxa_registry_digest == REGISTRY_DIGEST


def test_a_written_card_keeps_its_registry_digest(tmp_path):
    path = tmp_path / "card.yaml"

    write_card(_model_card(), path)

    assert read_card(path).taxa_registry_digest == REGISTRY_DIGEST


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
        "window_duration: 5.0\n"
        "window_overlap: 0.0\n"
        "min_detection_threshold: 0.01\n"
        "score_domain: probability\n"
        f"taxa_registry_digest: {REGISTRY_DIGEST}\n"
        "audio:\n"
        "  pad: centre_crop_end_pad   # keys in any order\n"
        "  downmix: mean\n"
        "  resampler: {by: runner, algorithm: soxr_hq}\n"
        "backend: tf-saved-model\n"
        "dtype: float32\n",
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
    body = _yaml({key: value for key, value in MODEL_CARD_FIELDS.items() if key != "model_version"})
    path = _write_text(tmp_path, body + text)

    with pytest.raises(ValueError, match="model_version") as exc:
        read_card(path)

    assert str(path) in str(exc.value), reason


def test_a_quoted_version_is_text(tmp_path):
    body = _yaml({key: value for key, value in MODEL_CARD_FIELDS.items() if key != "model_version"})
    path = _write_text(tmp_path, body + 'model_version: "1.10"\n')

    assert read_card(path).model_version == "1.10"


def test_a_repeated_key_is_refused(tmp_path):
    body = _yaml(MODEL_CARD_FIELDS)
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
    body = _yaml(MODEL_CARD_FIELDS)
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
