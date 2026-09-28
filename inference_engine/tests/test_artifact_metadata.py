"""The provenance every artifact carries, and the reader check that it is all there."""

import hashlib
import json

import pytest

from robin_contracts.cards import HeadCard, ModelCard, ModelRef, card_digest, model_ref
from robin_contracts.embedding_transforms import L2Norm
from robin_contracts.output_contracts import ScoresRequest, ThresholdPolicy, TopKPolicy
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
    recording_work_digest,
)
from robin_inference_engine import errors
from robin_inference_engine.artifacts.metadata import (
    EMBEDDING_KEYS,
    REGISTRY_KEYS,
    REQUIRED_KEYS,
    decode_metadata,
    detection_metadata,
    embedding_metadata,
    require_metadata_keys,
    required_metadata,
    score_metadata,
)

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
OTHER_FILE_DIGEST = "sha256:" + "b" * 64
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64

CARD = ModelCard(
    model_name="owl",
    model_version="1",
    runtime="tensorflow",
    segment_duration=12.0,
    sample_rate=32000,
    min_detection_threshold=0.0,
)
WEIGHTS = PinnedFile(uri="s3://b/owl.h5", digest=FILE_DIGEST, size_bytes=8)
REGISTRY_FILE = PinnedFile(uri="s3://b/registry.csv", digest=REGISTRY_FINGERPRINT, size_bytes=16)


def build_model(card: ModelCard | HeadCard = CARD, **overrides) -> PinnedModel:
    fields = {
        "card": card,
        "files": {"weights": WEIGHTS, REGISTRY_ROLE: REGISTRY_FILE},
    }
    return PinnedModel(**(fields | overrides))


def build_recipe(**overrides) -> Recipe:
    fields = {
        "model": model_ref(CARD),
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


SOUNDHUB_42 = RecordingRef(namespace="soundhub", value="42", audio_uri="s3://b/42.wav")
SOUNDHUB_43 = RecordingRef(namespace="soundhub", value="43", audio_uri="s3://b/43.wav")


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (SOUNDHUB_42,),
        "model": build_model(),
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
        "recording": SOUNDHUB_42,
        "recipe": build_recipe(),
        "registry_uri": "s3://b/registry.csv",
        "registry_fingerprint": REGISTRY_FINGERPRINT,
    }
    return required_metadata(**(fields | overrides))


def build_scores_request(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def test_every_shared_key_is_present():
    decoded = decode_metadata(build_metadata())

    assert set(decoded) == set(REQUIRED_KEYS) | set(REGISTRY_KEYS)
    assert len(REQUIRED_KEYS) == 7
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


def test_model_file_digests_carry_each_roles_digest_and_size_and_no_uri():
    digests = json.loads(decode_metadata(build_metadata())["robin.model_file_digests"])

    assert digests == {
        "weights": {"digest": FILE_DIGEST, "size_bytes": 8},
        REGISTRY_ROLE: {"digest": REGISTRY_FINGERPRINT, "size_bytes": 16},
    }


def test_model_file_digests_do_not_depend_on_file_order():
    forwards = build_work(model=build_model(files={"weights": WEIGHTS, REGISTRY_ROLE: REGISTRY_FILE}))
    backwards = build_work(model=build_model(files={REGISTRY_ROLE: REGISTRY_FILE, "weights": WEIGHTS}))

    assert (
        decode_metadata(build_metadata(work=forwards))["robin.model_file_digests"]
        == decode_metadata(build_metadata(work=backwards))["robin.model_file_digests"]
    )


def test_model_ref_is_name_slash_version():
    assert decode_metadata(build_metadata())["robin.model_ref"] == "owl/1"


def test_the_recording_work_digest_ignores_resources():
    plain = build_metadata(work=build_work(resources={}))
    resourced = build_metadata(work=build_work(resources={"gpus": 4}))

    assert (
        decode_metadata(plain)["robin.recording_work_digest"]
        == decode_metadata(resourced)["robin.recording_work_digest"]
    )


def test_the_header_carries_the_digest_of_the_work_narrowed_to_its_recording():
    work = build_work(recordings=(SOUNDHUB_42, SOUNDHUB_43))

    headers = {
        recording.value: decode_metadata(build_metadata(work=work, recording=recording))
        for recording in work.recordings
    }

    for recording in work.recordings:
        assert headers[recording.value]["robin.recording_work_digest"] == (
            recording_work_digest(work, recording)
        )
    assert (
        headers["42"]["robin.recording_work_digest"]
        != headers["43"]["robin.recording_work_digest"]
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


def test_detection_metadata_records_the_policy_and_the_scores_file_it_selected_from():
    policy = ThresholdPolicy(min_score=0.1 + 0.2)

    decoded = decode_metadata(
        detection_metadata(
            policy, source_contract_id="robin.scores.arrow/1", source_checksum=FILE_DIGEST
        )
    )

    assert set(decoded) == {"robin.detection_policy", "robin.source_artifacts"}
    assert ThresholdPolicy.model_validate_json(decoded["robin.detection_policy"]) == policy
    # One scores file, named by its contract and bytes but not by where it was published.
    assert json.loads(decoded["robin.source_artifacts"]) == [
        {"contract_id": "robin.scores.arrow/1", "checksum": FILE_DIGEST}
    ]


def test_a_top_k_policy_with_no_floor_records_the_floor_as_null():
    decoded = decode_metadata(
        detection_metadata(
            TopKPolicy(k=3), source_contract_id="robin.scores.arrow/1", source_checksum=FILE_DIGEST
        )
    )

    assert json.loads(decoded["robin.detection_policy"]) == {
        "kind": "top_k",
        "k": 3,
        "min_score": None,
    }


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


def test_embedding_metadata_carries_every_key_a_vector_cannot_imply():
    decoded = decode_metadata(
        embedding_metadata(
            build_work(), dim=1280, source_dtype="float32", storage_dtype="float16"
        )
    )

    assert set(decoded) == set(EMBEDDING_KEYS)
    assert len(EMBEDDING_KEYS) == 5
    assert decoded["robin.embedding_dim"] == "1280"
    assert decoded["robin.embedding_storage_dtype"] == "float16"


@pytest.mark.parametrize("source_dtype", ["float32", "float16"])
@pytest.mark.parametrize("storage_dtype", ["float32", "float16"])
def test_the_source_dtype_is_the_one_the_instance_declared(source_dtype, storage_dtype):
    # A declaration rather than a reading: the header is stamped when the file is
    # created, before any window has arrived to take a dtype from.
    decoded = decode_metadata(
        embedding_metadata(
            build_work(), dim=8, source_dtype=source_dtype, storage_dtype=storage_dtype
        )
    )

    assert decoded["robin.embedding_source_dtype"] == source_dtype
    assert decoded["robin.embedding_storage_dtype"] == storage_dtype


@pytest.mark.parametrize(
    ("registry_uri", "registry_fingerprint"),
    [("s3://b/registry.csv", None), (None, REGISTRY_FINGERPRINT)],
)
def test_half_a_registry_binding_is_refused(registry_uri, registry_fingerprint):
    # A uri without a fingerprint names a file nothing pins, and the reverse pins a
    # file nothing names. Neither is an artifact that bound no registry.
    with pytest.raises(RuntimeError) as exc:
        build_metadata(
            registry_uri=registry_uri, registry_fingerprint=registry_fingerprint
        )

    assert "robin.registry_uri" in str(exc.value)
    assert "robin.registry_fingerprint" in str(exc.value)


def test_a_backbones_own_run_names_itself_as_the_backbone():
    decoded = decode_metadata(
        embedding_metadata(build_work(), dim=8, source_dtype="float32", storage_dtype="float32")
    )
    shared = decode_metadata(build_metadata())

    assert decoded["robin.backbone_ref"] == shared["robin.model_ref"]
    assert decoded["robin.backbone_card_digest"] == shared["robin.model_card_digest"]


HEAD = HeadCard(
    model_name="amy-head",
    model_version="1",
    backbone=ModelRef(name="perch", version="8", digest="sha256:v1:" + "d" * 64),
    classes=("owl",),
    required_embedding_transform=L2Norm(),
)


def test_a_head_names_the_backbone_its_card_declares():
    work = build_work(model=build_model(HEAD))

    decoded = decode_metadata(
        embedding_metadata(work, dim=8, source_dtype="float32", storage_dtype="float32")
    )

    assert decoded["robin.backbone_ref"] == "perch/8"
    assert decoded["robin.backbone_card_digest"] == HEAD.backbone.digest


@pytest.mark.parametrize("card", [CARD, HEAD], ids=["backbone", "head"])
def test_the_model_keys_come_from_the_works_card(card):
    work = build_work(model=build_model(card))

    shared = decode_metadata(build_metadata(work=work, recipe=build_recipe(model=model_ref(card))))

    assert shared["robin.model_ref"] == f"{card.model_name}/{card.model_version}"
    assert shared["robin.model_card_digest"] == card_digest(card)
