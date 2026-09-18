"""What a recording_index means, written down so a reader needs nothing else."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.cards import ModelRef
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.work import (
    AudioInput,
    FileDigest,
    InferenceWork,
    ModelSelection,
    RecordingRef,
    work_digest,
)
from robin_inference_engine import errors
from robin_inference_engine.artifacts.recording_map import (
    CONTRACT_ID,
    DECLARED_KEYS,
    RecordingMap,
    read_recording_map,
    write_recording_map,
)
from robin_inference_engine.artifacts.staging import checksum_file

HEX = "0" * 64
RECORD_DIGEST = f"sha256:v1:{HEX}"
FILE_DIGEST = f"sha256:{HEX}"
AUDIO_DIGEST = "sha256:" + "d" * 64

MODEL_REF = ModelRef(name="owl", version="1", digest=RECORD_DIGEST)

REPOSITORY = Path(__file__).resolve().parents[2]


def build_recording(index: int, **overrides) -> RecordingRef:
    fields = {
        "index": index,
        "namespace": "soundhub",
        "value": str(40 + index),
        "source_revision": f"etag:{index}",
        "audio_uri": f"s3://b/{40 + index}.wav",
        "audio_digest": AUDIO_DIGEST,
        "duration_seconds": 3600.0,
    }
    return RecordingRef(**(fields | overrides))


def build_work(*recordings: RecordingRef) -> InferenceWork:
    return work_over(recordings or (build_recording(0), build_recording(1)))


def work_over(recordings: tuple[RecordingRef, ...]) -> InferenceWork:
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=recordings,
        model=ModelSelection(
            ref=MODEL_REF,
            card_digest=RECORD_DIGEST,
            files=(
                FileDigest(
                    role="weights", uri="s3://b/owl.h5", digest=FILE_DIGEST, size_bytes=8
                ),
            ),
        ),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=(ScoresRequest(contract_id="robin.scores.arrow/1", retention="full"),),
    )


def stage(tmp_path, payload: bytes, name: str = "hand-written.json") -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(payload)
    return path, checksum_file(path)


def hand_written(recordings: list[dict], **overrides) -> bytes:
    document = {
        "schema_version": CONTRACT_ID,
        "work_digest": f"sha256:v1:{HEX}",
        "recordings": recordings,
    }
    return canonical_json_bytes(document | overrides)


# A work validates a request and the map reader validates stored bytes, so the same
# three rules are written in both places: they raise different errors for different
# callers and cannot share code. Relaxing one side alone fails here.
CONFLICTING_RECORDINGS = (
    pytest.param((), id="no_recordings"),
    pytest.param(
        (build_recording(0), build_recording(0, value="99")), id="repeated_index"
    ),
    pytest.param(
        (build_recording(0), build_recording(1, value="40")), id="repeated_identity"
    ),
)


@pytest.mark.parametrize("recordings", CONFLICTING_RECORDINGS)
def test_a_work_and_the_map_reader_refuse_the_same_recordings(recordings, tmp_path):
    with pytest.raises(ValidationError):
        work_over(recordings)

    payload = hand_written([one.model_dump(mode="json") for one in recordings])
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID


def test_a_map_round_trips(tmp_path):
    work = build_work()

    staged = write_recording_map(work, path=tmp_path / "map.json")
    read = read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert read.recordings == work.recordings


def test_the_map_declares_its_contract_identifier(tmp_path):
    staged = write_recording_map(build_work(), path=tmp_path / "map.json")

    assert staged.contract_id == CONTRACT_ID
    assert staged.kind == "recording_map"
    assert json.loads(staged.path.read_bytes())["schema_version"] == CONTRACT_ID

    declaring = REPOSITORY / "contracts/robin_contracts/output_contracts.py"
    assert CONTRACT_ID in declaring.read_text()
    spelled_in_the_engine = [
        module
        for module in (REPOSITORY / "inference_engine/robin_inference_engine").rglob("*.py")
        if CONTRACT_ID in module.read_text()
    ]
    assert spelled_in_the_engine == []


def test_the_writer_writes_exactly_the_keys_this_contract_declares(tmp_path):
    staged = write_recording_map(build_work(), path=tmp_path / "map.json")

    assert set(json.loads(staged.path.read_bytes())) == DECLARED_KEYS


def test_a_key_this_contract_does_not_declare_is_refused(tmp_path):
    payload = hand_written(
        [build_recording(0).model_dump(mode="json")], note="added by hand"
    )
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "note" in exc.value.detail


def test_a_row_that_is_not_a_valid_reference_is_refused(tmp_path):
    payload = hand_written([{"index": 0, "namespace": "soundhub"}])
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "position 0" in exc.value.detail


def test_a_map_that_names_no_recording_is_refused(tmp_path):
    payload = hand_written([])
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID


def test_a_map_built_directly_refuses_a_repeated_index():
    with pytest.raises(errors.EngineError) as exc:
        RecordingMap(
            work_digest=RECORD_DIGEST,
            recordings=(build_recording(0), build_recording(0, value="99")),
        )

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID


def test_a_map_built_directly_refuses_a_repeated_identity():
    with pytest.raises(errors.EngineError) as exc:
        RecordingMap(
            work_digest=RECORD_DIGEST,
            recordings=(build_recording(0), build_recording(1, value="40")),
        )

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID


def test_the_map_carries_the_works_digest(tmp_path):
    work = build_work()

    staged = write_recording_map(work, path=tmp_path / "map.json")
    read = read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert read.work_digest == work_digest(work)


def test_a_map_with_no_work_digest_is_refused(tmp_path):
    payload = canonical_json_bytes(
        {
            "schema_version": CONTRACT_ID,
            "recordings": [build_recording(0).model_dump(mode="json")],
        }
    )
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "no work_digest" in exc.value.detail


def test_a_work_digest_that_is_not_text_is_reported_as_what_it_is(tmp_path):
    payload = hand_written([build_recording(0).model_dump(mode="json")], work_digest=17)
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "int" in exc.value.detail


def test_a_work_digest_that_is_not_a_canonical_digest_is_refused(tmp_path):
    payload = hand_written(
        [build_recording(0).model_dump(mode="json")], work_digest="not-a-digest"
    )
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "not-a-digest" in exc.value.detail


def test_rows_is_the_number_of_recordings(tmp_path):
    work = build_work(build_recording(0), build_recording(1), build_recording(2))

    staged = write_recording_map(work, path=tmp_path / "map.json")

    assert staged.rows == 3


def test_the_same_value_in_two_namespaces_is_two_recordings(tmp_path):
    work = build_work(
        build_recording(0, namespace="soundhub", value="42"),
        build_recording(1, namespace="arbimon", value="42"),
    )

    staged = write_recording_map(work, path=tmp_path / "map.json")
    read = read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert len(read.recordings) == 2
    assert read.by_index(0).namespace == "soundhub"
    assert read.by_index(1).namespace == "arbimon"


def test_a_duplicate_namespace_value_pair_is_refused_by_the_reader(tmp_path):
    payload = hand_written(
        [
            build_recording(0, namespace="soundhub", value="42").model_dump(mode="json"),
            build_recording(1, namespace="soundhub", value="42").model_dump(mode="json"),
        ]
    )
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "soundhub" in exc.value.detail


def test_a_duplicate_index_is_refused_by_the_reader(tmp_path):
    payload = hand_written(
        [
            build_recording(0, value="42").model_dump(mode="json"),
            build_recording(0, value="43").model_dump(mode="json"),
        ]
    )
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "0" in exc.value.detail


def test_a_null_source_revision_is_never_filled_in(tmp_path):
    work = build_work(build_recording(0, source_revision=None))

    staged = write_recording_map(work, path=tmp_path / "map.json")
    read = read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert json.loads(staged.path.read_bytes())["recordings"][0]["source_revision"] is None
    assert read.by_index(0).source_revision is None

    invented = hand_written(
        [build_recording(0).model_dump(mode="json") | {"source_revision": ""}]
    )
    path, checksum = stage(tmp_path, invented, name="invented.json")
    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)
    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID


def test_a_checksum_mismatch_is_refused(tmp_path):
    staged = write_recording_map(build_work(), path=tmp_path / "map.json")
    bytes_on_disk = bytearray(staged.path.read_bytes())
    bytes_on_disk[-2] ^= 0x20
    staged.path.write_bytes(bytes(bytes_on_disk))

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert exc.value.code == errors.ARTIFACT_CHECKSUM_MISMATCH
    assert staged.checksum in exc.value.detail


def test_a_wrong_schema_version_is_refused(tmp_path):
    payload = hand_written(
        [build_recording(0).model_dump(mode="json")],
        schema_version="robin.recording-map.json/2",
    )
    path, checksum = stage(tmp_path, payload)

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED


def test_bytes_that_are_not_json_are_refused_with_a_typed_error(tmp_path):
    path, checksum = stage(tmp_path, b"{not json at all", name="garbage.json")

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_bytes_that_are_not_utf8_are_refused_with_a_typed_error(tmp_path):
    path, checksum = stage(tmp_path, b'\x80{"a": 1}', name="not-utf8.json")

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(path, expected_checksum=checksum)

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_file_that_is_not_there_is_refused_with_a_typed_error(tmp_path):
    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(tmp_path / "absent.json", expected_checksum=FILE_DIGEST)

    assert exc.value.code == errors.ARTIFACT_UNREADABLE
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_short_write_is_caught_when_the_map_is_read_back(tmp_path, monkeypatch):
    # The checksum a writer stages must describe the bytes it meant to write, so a
    # write that lands incomplete fails verification instead of certifying itself.
    whole_write = Path.write_bytes
    monkeypatch.setattr(Path, "write_bytes", lambda self, data: whole_write(self, data[:-4]))
    staged = write_recording_map(build_work(), path=tmp_path / "map.json")
    monkeypatch.undo()

    with pytest.raises(errors.EngineError) as exc:
        read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert exc.value.code == errors.ARTIFACT_CHECKSUM_MISMATCH


def test_by_index_resolves_every_recording(tmp_path):
    work = build_work(build_recording(3), build_recording(7))

    staged = write_recording_map(work, path=tmp_path / "map.json")
    read = read_recording_map(staged.path, expected_checksum=staged.checksum)

    assert read.by_index(3).value == "43"
    assert read.by_index(7).value == "47"
    with pytest.raises(KeyError):
        read.by_index(4)


def test_a_map_replays_byte_identically(tmp_path):
    work = build_work()

    first = write_recording_map(work, path=tmp_path / "first.json")
    second = write_recording_map(work, path=tmp_path / "second.json")

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.checksum == second.checksum
