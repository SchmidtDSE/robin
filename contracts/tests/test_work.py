"""The work a caller hands the engine, its scientific digest, and batch partitioning."""

import pytest
from pydantic import ValidationError

from robin_contracts.cards import AudioGeometry, HeadCard, ModelCard, ModelRef, RunnerResampled
from robin_contracts.output_contracts import DetectionsRequest, ScoresRequest, ThresholdPolicy
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    EmbeddingArtifactInput,
    InferenceWork,
    InputArtifact,
    PinnedFile,
    PinnedModel,
    RecordingRef,
    partition,
    recording_work_digest,
    work_digest,
)

REGISTRY_DIGEST = "sha256:" + "a" * 64

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
RECORD_DIGEST = f"sha256:v1:{HEX}"

CARD = ModelCard(
    model_name="owl",
    model_version="1",
    runtime="tensorflow",
    window_duration=3.0,
    sample_rate=32000,
    min_detection_threshold=0.0,
    score_domain="sigmoid",
    taxa_registry_digest=REGISTRY_DIGEST,
    audio=AudioGeometry(
        downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
    ),
    backend="tensorflow",
    dtype="float32",
)

HEAD = HeadCard(
    model_name="nutria",
    model_version="1",
    runtime="onnx",
    backbone=ModelRef(name="perch", version="8", digest=RECORD_DIGEST),
    embedding_dim=1280,
    min_detection_threshold=0.0,
    score_domain="sigmoid",
    taxa_registry_digest=REGISTRY_DIGEST,
)

EMBEDDINGS_INPUT = EmbeddingArtifactInput(contract_id="robin.embeddings.parquet/1", backbone=CARD)


def build_recording(**overrides) -> RecordingRef:
    fields = {
        "namespace": "soundhub",
        "value": "42",
        "audio_uri": "s3://bucket/42.wav",
    }
    return RecordingRef(**(fields | overrides))


def build_file(**overrides) -> PinnedFile:
    fields = {
        "uri": "s3://bucket/owl.tflite",
        "digest": FILE_DIGEST,
        "size_bytes": 1024,
    }
    return PinnedFile(**(fields | overrides))


def build_pinned_model(**overrides) -> PinnedModel:
    fields = {
        "card": CARD,
        "files": {"weights": build_file()},
    }
    return PinnedModel(**(fields | overrides))


def build_scores(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.parquet/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (build_recording(),),
        "model": build_pinned_model(),
        "input": AudioInput(),
        "settings": {},
        "resources": {},
        "outputs": (build_scores(),),
    }
    return InferenceWork(**(fields | overrides))


def test_a_minimal_work_round_trips():
    work = build_work()

    assert InferenceWork(**work.model_dump(mode="json")) == work
    assert isinstance(work.outputs, tuple)


def test_work_has_no_run_identity_fields():
    assert set(InferenceWork.model_fields) == {
        "schema_version",
        "recordings",
        "model",
        "input",
        "settings",
        "resources",
        "outputs",
    }


def test_unknown_fields_are_forbidden():
    with pytest.raises(ValidationError):
        build_work(location="s3://bucket/run-7/")


def test_recordings_must_be_non_empty():
    with pytest.raises(ValidationError):
        build_work(recordings=())


def test_the_same_value_in_two_namespaces_is_two_recordings():
    work = build_work(
        recordings=(
            build_recording(namespace="soundhub", value="42"),
            build_recording(namespace="arbimon", value="42"),
        )
    )

    assert [(one.namespace, one.value) for one in work.recordings] == [
        ("soundhub", "42"),
        ("arbimon", "42"),
    ]


def test_duplicate_namespace_and_value_is_refused():
    with pytest.raises(ValidationError, match="repeated"):
        build_work(
            recordings=(
                build_recording(namespace="soundhub", value="42"),
                build_recording(namespace="soundhub", value="42", audio_uri="s3://b/other.wav"),
            )
        )


def test_outputs_must_be_non_empty_and_must_not_repeat_a_kind():
    with pytest.raises(ValidationError):
        build_work(outputs=())

    with pytest.raises(ValidationError):
        build_work(outputs=(build_scores(), build_scores()))


def test_detections_without_scores_is_refused():
    detections = DetectionsRequest(
        contract_id="robin.detections.parquet/1",
        policy=ThresholdPolicy(min_score=0.1),
    )

    with pytest.raises(ValidationError):
        build_work(outputs=(detections,))

    assert len(build_work(outputs=(build_scores(), detections)).outputs) == 2


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2]])
def test_settings_reject_a_nested_value(value):
    with pytest.raises(ValidationError):
        build_work(settings={"a": value})


def test_settings_keep_bool_and_int_distinct():
    work = build_work(settings={"a": True, "b": 1})

    assert work.settings["a"] is True
    assert isinstance(work.settings["b"], int) and not isinstance(work.settings["b"], bool)


@pytest.mark.parametrize("field", ["settings", "resources"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_a_non_finite_float(field, value):
    with pytest.raises(ValidationError):
        build_work(**{field: {"a": value}})


def test_work_digest_ignores_resources():
    quiet = build_work(resources={"device": "cpu"})
    loud = build_work(resources={"device": "gpu", "batch_size": 64})

    assert work_digest(quiet) == work_digest(loud)

    assert work_digest(build_work(settings={"sensitivity": 1.0})) != work_digest(
        build_work(settings={"sensitivity": 1.5})
    )


def test_work_digest_ignores_settings_key_order():
    one = build_work(settings={"a": 1, "b": 2})
    other = build_work(settings={"b": 2, "a": 1})

    assert list(one.settings) != list(other.settings)
    assert work_digest(one) == work_digest(other)


@pytest.mark.parametrize("field", ["namespace", "value"])
def test_a_recording_refuses_an_empty_identity(field):
    with pytest.raises(ValidationError):
        build_recording(**{field: ""})


@pytest.mark.parametrize("card", [CARD, HEAD], ids=["model", "head"])
def test_a_pinned_model_round_trips_as_the_kind_of_card_it_holds(card):
    pinned = build_pinned_model(card=card)

    rebuilt = PinnedModel.model_validate(pinned.model_dump(mode="json"))

    assert type(rebuilt.card) is type(card)
    assert rebuilt == pinned


def build_embeddings_file(**overrides) -> InputArtifact:
    fields = {"uri": "s3://bucket/42/embeddings.parquet", "checksum": FILE_DIGEST}
    return InputArtifact(**(fields | overrides))


def build_head_work(**overrides) -> InferenceWork:
    fields = {
        "recordings": (build_recording(embeddings=build_embeddings_file()),),
        "model": build_pinned_model(card=HEAD),
        "input": EMBEDDINGS_INPUT,
    }
    return build_work(**(fields | overrides))


def test_a_head_work_over_embeddings_round_trips_as_a_head():
    work = build_head_work()

    rebuilt = InferenceWork.model_validate(work.model_dump(mode="json"))

    assert isinstance(rebuilt.model.card, HeadCard)
    assert rebuilt.input.backbone == CARD
    assert work_digest(rebuilt) == work_digest(work)


def test_an_embeddings_input_names_no_backbone_settings_by_default():
    assert EMBEDDINGS_INPUT.backbone_settings == {}


def test_an_embeddings_input_with_backbone_settings_round_trips():
    source = EmbeddingArtifactInput(
        contract_id="robin.embeddings.parquet/1",
        backbone=CARD,
        backbone_settings={"window_overlap": 1.0},
    )

    rebuilt = EmbeddingArtifactInput.model_validate(source.model_dump(mode="json"))

    assert rebuilt == source
    assert rebuilt.backbone_settings == {"window_overlap": 1.0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_backbone_settings_reject_a_non_finite_float(value):
    with pytest.raises(ValidationError):
        EmbeddingArtifactInput(
            contract_id="robin.embeddings.parquet/1",
            backbone=CARD,
            backbone_settings={"window_overlap": value},
        )


def test_a_recording_naming_its_embeddings_file_round_trips():
    recording = build_recording(embeddings=build_embeddings_file())

    assert RecordingRef.model_validate(recording.model_dump(mode="json")) == recording
    assert recording.embeddings.checksum == FILE_DIGEST


def test_a_work_over_embeddings_with_a_recording_naming_no_file_is_refused():
    named = build_recording(value="42", embeddings=build_embeddings_file())
    unnamed = build_recording(namespace="archive", value="43")

    with pytest.raises(ValidationError, match="'archive'.*'43'"):
        build_head_work(recordings=(named, unnamed))


def test_a_work_over_audio_with_a_recording_naming_an_embeddings_file_is_refused():
    named = build_recording(namespace="archive", value="43", embeddings=build_embeddings_file())

    with pytest.raises(ValidationError, match="'archive'.*'43'"):
        build_work(recordings=(build_recording(), named))


def test_a_head_recording_work_digest_depends_on_its_own_embeddings_file_only():
    def recordings(r_checksum: str, s_checksum: str) -> tuple[RecordingRef, RecordingRef]:
        return (
            build_recording(value="42", embeddings=build_embeddings_file(checksum=r_checksum)),
            build_recording(value="43", embeddings=build_embeddings_file(checksum=s_checksum)),
        )

    other = f"sha256:{'1' * 64}"
    base = build_head_work(recordings=recordings(FILE_DIGEST, FILE_DIGEST))
    r_changed = build_head_work(recordings=recordings(other, FILE_DIGEST))
    s_changed = build_head_work(recordings=recordings(FILE_DIGEST, other))

    assert recording_work_digest(base, base.recordings[0]) != recording_work_digest(
        r_changed, r_changed.recordings[0]
    )
    assert recording_work_digest(base, base.recordings[0]) == recording_work_digest(
        s_changed, s_changed.recordings[0]
    )


def test_the_registry_role_is_named_once():
    assert REGISTRY_ROLE == "taxa_registry"


def test_file_order_does_not_change_the_work_digest():
    weights = build_file()
    registry = build_file(uri="s3://bucket/taxa.csv", digest=f"sha256:{'1' * 64}")
    one = build_work(model=build_pinned_model(files={"weights": weights, REGISTRY_ROLE: registry}))
    other = build_work(model=build_pinned_model(files={REGISTRY_ROLE: registry, "weights": weights}))

    assert list(one.model.files) != list(other.model.files)
    assert work_digest(one) == work_digest(other)


def test_the_card_is_part_of_the_work_digest():
    other_card = CARD.model_copy(update={"min_detection_threshold": 0.1})

    assert work_digest(build_work()) != work_digest(
        build_work(model=build_pinned_model(card=other_card))
    )


R = build_recording(value="42", audio_uri="s3://bucket/42.wav")
S = build_recording(value="43", audio_uri="s3://bucket/43.wav")
T = build_recording(value="44", audio_uri="s3://bucket/44.wav")


def test_a_recording_work_digest_is_the_digest_of_the_work_holding_only_that_recording():
    work = build_work(recordings=(R, S), settings={"sensitivity": 1.0})

    assert recording_work_digest(work, S) == work_digest(
        build_work(recordings=(S,), settings={"sensitivity": 1.0})
    )


def test_a_recording_work_digest_does_not_depend_on_the_other_recordings():
    one = build_work(recordings=(R, S))
    other = build_work(recordings=(T, R))

    assert work_digest(one) != work_digest(other)
    assert recording_work_digest(one, R) == recording_work_digest(other, R)


def test_two_recordings_of_one_work_have_different_recording_work_digests():
    work = build_work(recordings=(R, S))

    assert recording_work_digest(work, R) != recording_work_digest(work, S)


def test_a_recording_work_digest_changes_with_the_audio_uri():
    moved = R.model_copy(update={"audio_uri": "s3://bucket/moved.wav"})

    assert recording_work_digest(build_work(recordings=(R,)), R) != recording_work_digest(
        build_work(recordings=(moved,)), moved
    )


def test_a_recording_work_digest_changes_with_settings():
    one = build_work(recordings=(R,), settings={"sensitivity": 1.0})
    other = build_work(recordings=(R,), settings={"sensitivity": 1.5})

    assert recording_work_digest(one, R) != recording_work_digest(other, R)


def test_a_recording_work_digest_ignores_resources():
    quiet = build_work(recordings=(R, S), resources={"device": "cpu"})
    loud = build_work(recordings=(R, S), resources={"device": "gpu", "batch_size": 64})

    assert recording_work_digest(quiet, R) == recording_work_digest(loud, R)


def test_a_recording_work_digest_refuses_a_recording_the_work_does_not_name():
    work = build_work(recordings=(R, S))
    # Same identity, different audio: not the recording this work ran.
    moved = R.model_copy(update={"audio_uri": "s3://bucket/moved.wav"})

    with pytest.raises(ValueError):
        recording_work_digest(work, T)
    with pytest.raises(ValueError):
        recording_work_digest(work, moved)


def test_digest_fields_reject_the_wrong_family():
    with pytest.raises(ValidationError):
        build_file(digest=RECORD_DIGEST)

    # An unlabelled digest is refused too: nothing in it says what was hashed.
    with pytest.raises(ValidationError):
        build_file(digest=HEX)

    short = "0" * 40
    with pytest.raises(ValidationError):
        build_file(digest=f"sha256:{short}")
    with pytest.raises(ValidationError):
        build_embeddings_file(checksum=f"sha256:{short}")
    with pytest.raises(ValidationError):
        build_embeddings_file(checksum=RECORD_DIGEST)


@pytest.mark.parametrize(("count", "batch_size"), [(7, 3), (6, 3), (1, 1), (3, 10)])
def test_partition_covers_every_index_exactly_once(count, batch_size):
    batches = partition(count, batch_size)

    assert [index for batch in batches for index in batch] == list(range(count))
    assert all(len(batch) <= batch_size for batch in batches)


def test_partition_of_nothing_is_no_batches():
    assert partition(0, 10) == ()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_partition_refuses_a_non_positive_batch_size(batch_size):
    with pytest.raises(ValueError):
        partition(10, batch_size)

    with pytest.raises(ValueError):
        partition(-1, 10)
