"""The work a caller hands the engine, its scientific digest, and batch partitioning."""

import pytest
from pydantic import ValidationError

from robin_contracts.cards import ModelRef
from robin_contracts.output_contracts import DetectionsRequest, ScoresRequest, ThresholdPolicy
from robin_contracts.work import (
    AudioInput,
    EmbeddingArtifactInput,
    FileDigest,
    InferenceWork,
    ModelSelection,
    RecordingRef,
    partition,
    work_digest,
)

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
RECORD_DIGEST = f"sha256:v1:{HEX}"

MODEL_REF = ModelRef(name="owl", version="1", digest=RECORD_DIGEST)


def build_recording(**overrides) -> RecordingRef:
    fields = {
        "index": 0,
        "namespace": "soundhub",
        "value": "42",
        "audio_uri": "s3://bucket/42.wav",
    }
    return RecordingRef(**(fields | overrides))


def build_file(**overrides) -> FileDigest:
    fields = {
        "role": "weights",
        "uri": "s3://bucket/owl.tflite",
        "digest": FILE_DIGEST,
        "size_bytes": 1024,
    }
    return FileDigest(**(fields | overrides))


def build_selection(**overrides) -> ModelSelection:
    fields = {
        "ref": MODEL_REF,
        "files": (build_file(),),
    }
    return ModelSelection(**(fields | overrides))


def build_scores(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (build_recording(),),
        "model": build_selection(),
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


def test_duplicate_recording_index_is_refused():
    with pytest.raises(ValidationError):
        build_work(
            recordings=(
                build_recording(index=0, value="42"),
                build_recording(index=0, value="43"),
            )
        )


def test_the_same_value_in_two_namespaces_is_two_recordings():
    work = build_work(
        recordings=(
            build_recording(index=0, namespace="soundhub", value="42"),
            build_recording(index=1, namespace="arbimon", value="42"),
        )
    )

    assert len(work.recordings) == 2


def test_duplicate_namespace_and_value_is_refused():
    with pytest.raises(ValidationError):
        build_work(
            recordings=(
                build_recording(index=0, namespace="soundhub", value="42"),
                build_recording(index=1, namespace="soundhub", value="42"),
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


def test_a_selection_carries_no_card_digest_beside_its_ref():
    with pytest.raises(ValidationError, match="card_digest"):
        build_selection(card_digest=RECORD_DIGEST)


def test_digest_fields_reject_the_wrong_family():
    with pytest.raises(ValidationError):
        build_selection(registry_fingerprint=RECORD_DIGEST)

    with pytest.raises(ValidationError):
        build_file(digest=RECORD_DIGEST)

    # An unlabelled digest is refused too: nothing in it says what was hashed.
    with pytest.raises(ValidationError):
        build_file(digest=HEX)

    short = "0" * 40
    with pytest.raises(ValidationError):
        build_file(digest=f"sha256:{short}")
    with pytest.raises(ValidationError):
        build_selection(registry_fingerprint=f"sha256:{short}")
    with pytest.raises(ValidationError):
        build_recording(audio_digest=f"sha256:{short}")
    with pytest.raises(ValidationError):
        EmbeddingArtifactInput(
            contract_id="robin.embeddings.arrow/1",
            uri="s3://bucket/embeddings.arrow",
            checksum=f"sha256:{short}",
            recording_map_uri="s3://bucket/recording-map.json",
            recording_map_checksum=FILE_DIGEST,
        )


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
