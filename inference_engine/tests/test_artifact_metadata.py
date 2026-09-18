"""The provenance every artifact carries, and the reader check that it is all there."""

import hashlib
import json

import pytest

from robin_contracts.cards import ModelRef
from robin_contracts.embedding_transforms import L2Norm
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled
from robin_contracts.work import (
    AudioInput,
    FileDigest,
    InferenceWork,
    ModelSelection,
    RecordingRef,
)
from robin_inference_engine import errors
from robin_inference_engine.artifacts.metadata import (
    REGISTRY_KEYS,
    REQUIRED_KEYS,
    decode_metadata,
    require_metadata_keys,
    required_metadata,
    score_metadata,
)

HEX = "0" * 64
RECORD_DIGEST = f"sha256:v1:{HEX}"
FILE_DIGEST = f"sha256:{HEX}"
OTHER_FILE_DIGEST = "sha256:" + "b" * 64
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64
MAP_CHECKSUM = "sha256:" + "c" * 64

MODEL_REF = ModelRef(name="owl", version="1", digest=RECORD_DIGEST)


def build_recipe(**overrides) -> Recipe:
    fields = {
        "model": MODEL_REF,
        "backend": "tensorflow",
        "audio": AudioSpec(
            sample_rate=32000,
            window=12.0,
            hop=6.0,
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        "embedding_transform": L2Norm(),
        "dtype": "float32",
    }
    return Recipe(**(fields | overrides))


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (
            RecordingRef(index=0, namespace="soundhub", value="42", audio_uri="s3://b/42.wav"),
        ),
        "model": ModelSelection(
            ref=MODEL_REF,
            card_digest=RECORD_DIGEST,
            files=(
                FileDigest(
                    role="weights", uri="s3://b/owl.h5", digest=FILE_DIGEST, size_bytes=8
                ),
            ),
            registry_fingerprint=REGISTRY_FINGERPRINT,
        ),
        "input": AudioInput(),
        "settings": {},
        "resources": {},
        "outputs": (ScoresRequest(contract_id="robin.scores.arrow/1", retention="full"),),
    }
    return InferenceWork(**(fields | overrides))


def build_metadata(**overrides) -> dict[bytes, bytes]:
    fields = {
        "contract_id": "robin.scores.arrow/1",
        "work": build_work(),
        "recipe": build_recipe(),
        "registry_uri": "s3://b/registry.csv",
        "registry_fingerprint": REGISTRY_FINGERPRINT,
        "recording_map_uri": "s3://b/recording-map.json",
        "recording_map_checksum": MAP_CHECKSUM,
    }
    return required_metadata(**(fields | overrides))


def build_scores_request(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def test_every_shared_key_is_present():
    decoded = decode_metadata(build_metadata())

    assert set(decoded) == set(REQUIRED_KEYS) | set(REGISTRY_KEYS)
    assert len(REQUIRED_KEYS) == 9
    assert len(REGISTRY_KEYS) == 2


def test_a_work_with_no_registry_writes_every_key_required_of_every_artifact():
    # An embeddings-only work needs no registry, and what it writes must still satisfy
    # the check every reader makes. The registry pair is conditional, so it lives apart.
    decoded = decode_metadata(
        build_metadata(registry_uri=None, registry_fingerprint=None)
    )

    require_metadata_keys(decoded, REQUIRED_KEYS, contract_id="robin.scores.arrow/1")


def test_the_recipe_fingerprint_is_the_digest_of_the_recipe_it_ships_with():
    recipe = build_recipe()
    raw = build_metadata(recipe=recipe)

    shipped = raw[b"robin.recipe"]
    assert (
        "sha256:v1:" + hashlib.sha256(shipped).hexdigest()
        == decode_metadata(raw)["robin.recipe_fingerprint"]
    )
    assert decode_metadata(raw)["robin.recipe_fingerprint"] == recipe.id


def test_the_shipped_recipe_carries_the_windowing_a_reader_needs():
    # The recipe is the only thing that says how the audio was cut up, so a reader
    # holding the artifact and nothing else recovers the windowing from it alone.
    recipe = build_recipe()

    audio = json.loads(decode_metadata(build_metadata(recipe=recipe))["robin.recipe"])["audio"]

    assert audio["sample_rate"] == recipe.audio.sample_rate
    assert audio["window"] == recipe.audio.window
    assert audio["hop"] == recipe.audio.hop
    assert audio["pad"] == recipe.audio.pad


def test_model_file_digests_carry_role_digest_and_size_and_no_uri():
    work = build_work(
        model=ModelSelection(
            ref=MODEL_REF,
            card_digest=RECORD_DIGEST,
            files=(
                FileDigest(
                    role="weights", uri="s3://b/owl.h5", digest=FILE_DIGEST, size_bytes=8
                ),
                FileDigest(
                    role="labels", uri="s3://b/labels.csv", digest=OTHER_FILE_DIGEST,
                    size_bytes=16,
                ),
            ),
        )
    )

    digests = json.loads(
        decode_metadata(build_metadata(work=work))["robin.model_file_digests"]
    )

    assert digests == [
        {"role": "weights", "digest": FILE_DIGEST, "size_bytes": 8},
        {"role": "labels", "digest": OTHER_FILE_DIGEST, "size_bytes": 16},
    ]


def test_model_ref_is_name_slash_version():
    assert decode_metadata(build_metadata())["robin.model_ref"] == "owl/1"


def test_the_work_digest_ignores_resources():
    plain = build_metadata(work=build_work(resources={}))
    resourced = build_metadata(work=build_work(resources={"gpus": 4}))

    assert (
        decode_metadata(plain)["robin.work_digest"]
        == decode_metadata(resourced)["robin.work_digest"]
    )


def test_an_absent_registry_omits_its_keys():
    decoded = decode_metadata(build_metadata(registry_uri=None, registry_fingerprint=None))

    assert "robin.registry_uri" not in decoded
    assert "robin.registry_fingerprint" not in decoded


def test_score_metadata_records_the_requests_floor():
    request = build_scores_request(retention="thresholded", min_score=0.005)

    decoded = decode_metadata(score_metadata(request, score_domain="probability"))

    assert decoded["robin.score_domain"] == "probability"
    assert decoded["robin.score_retention"] == "thresholded"
    assert decoded["robin.score_floor"] == "0.005"
    assert "robin.score_top_k" not in decoded


def test_score_metadata_records_the_requests_top_k():
    request = build_scores_request(retention="top_k", min_score=0.005, top_k=5)

    decoded = decode_metadata(score_metadata(request, score_domain="probability"))

    assert decoded["robin.score_retention"] == "top_k"
    assert decoded["robin.score_floor"] == "0.005"
    assert decoded["robin.score_top_k"] == "5"


def test_full_retention_carries_no_floor_and_no_top_k():
    decoded = decode_metadata(
        score_metadata(build_scores_request(), score_domain="probability")
    )

    assert set(decoded) == {"robin.score_domain", "robin.score_retention"}
    assert decoded["robin.score_retention"] == "full"


def test_a_floor_round_trips_through_its_repr():
    floor = 0.1 + 0.2
    request = build_scores_request(retention="thresholded", min_score=floor)

    stored = decode_metadata(score_metadata(request, score_domain="probability"))

    assert float(stored["robin.score_floor"]) == floor


def test_every_key_and_value_is_utf8_bytes():
    raw = build_metadata()

    assert all(isinstance(key, bytes) for key in raw)
    assert all(isinstance(value, bytes) for value in raw.values())
    assert all(isinstance(value, str) for value in decode_metadata(raw).values())
    assert decode_metadata(None) == {}


def test_a_value_that_is_not_utf8_is_refused_as_a_malformed_artifact():
    raw = build_metadata()
    raw[b"robin.recipe"] = b"\xff\xfe"

    with pytest.raises(errors.EngineError) as exc:
        decode_metadata(raw)

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    assert "robin.recipe" in exc.value.detail


def test_a_key_that_is_not_utf8_is_refused_as_a_malformed_artifact():
    with pytest.raises(errors.EngineError) as exc:
        decode_metadata({b"\xff\xfe": b"anything"})

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_missing_key_check_names_every_absent_key():
    decoded = decode_metadata(build_metadata())
    absent = ("robin.recipe", "robin.model_ref", "robin.model_file_digests")
    for key in absent:
        del decoded[key]

    with pytest.raises(errors.EngineError) as exc:
        require_metadata_keys(decoded, REQUIRED_KEYS, contract_id="robin.scores.arrow/1")

    assert exc.value.code == errors.ARTIFACT_METADATA_INCOMPLETE
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    for key in absent:
        assert key in exc.value.detail
