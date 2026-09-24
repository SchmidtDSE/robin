"""One row per window and label: what the writer stages, and what the reader refuses."""

from pathlib import Path

import pyarrow as pa
import pytest

from robin_contracts.cards import ModelCard, model_ref
from robin_contracts.embedding_transforms import L2Norm
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.records import ClassScore
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts.metadata import (
    REGISTRY_KEYS,
    REQUIRED_KEYS,
    required_metadata,
    score_metadata,
)
from robin_inference_engine.artifacts import scores as scores_module
from robin_inference_engine.artifacts.scores import (
    CONTRACT_ID,
    SCORE_BATCH_ROWS,
    SCORES_SCHEMA,
    ScoresWriter,
    read_scores,
)
from robin_inference_engine.artifacts.staging import checksum_file

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64

CARD = ModelCard(
    model_name="owl",
    model_version="1",
    runtime="tensorflow",
    segment_duration=12.0,
    sample_rate=32000,
    min_detection_threshold=0.0,
)

def a_recording(namespace: str, value: str) -> RecordingRef:
    return RecordingRef(namespace=namespace, value=value, audio_uri=f"s3://b/{value}.wav")


SOUNDHUB_42 = a_recording("soundhub", "42")

REPOSITORY = Path(__file__).resolve().parents[2]

TOP_K_KEYS = (
    *REQUIRED_KEYS,
    *REGISTRY_KEYS,
    "robin.score_domain",
    "robin.score_retention",
    "robin.score_floor",
    "robin.score_top_k",
)


def build_recipe() -> Recipe:
    return Recipe(
        model=model_ref(CARD),
        backend="tensorflow",
        audio=AudioSpec(
            sample_rate=32000,
            window=12.0,
            hop=6.0,
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        embedding_transform=L2Norm(),
        dtype="float32",
    )


def build_work() -> InferenceWork:
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=(
            RecordingRef(namespace="soundhub", value="42", audio_uri="s3://b/42.wav"),
        ),
        model=PinnedModel(
            card=CARD,
            files={
                "weights": PinnedFile(uri="s3://b/owl.h5", digest=FILE_DIGEST, size_bytes=8),
                REGISTRY_ROLE: PinnedFile(
                    uri="s3://b/registry.csv", digest=REGISTRY_FINGERPRINT, size_bytes=8
                ),
            },
            registry_fingerprint=REGISTRY_FINGERPRINT,
        ),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=(ScoresRequest(contract_id="robin.scores.arrow/1", retention="full"),),
    )


def build_request(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_metadata(request: ScoresRequest | None = None, **overrides) -> dict[bytes, bytes]:
    shared = required_metadata(
        contract_id=overrides.pop("contract_id", CONTRACT_ID),
        work=build_work(),
        recipe=build_recipe(),
        registry_uri="s3://b/registry.csv",
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )
    scores = score_metadata(request or build_request(), score_domain="probability")
    return shared | scores


def build_window(
    start: float = 0.0,
    *,
    recording: RecordingRef = SOUNDHUB_42,
    labels=("owl", "wren"),
    scores=None,
):
    values = scores if scores is not None else [0.5] * len(labels)
    return AcceptedWindow(
        recording=recording,
        start=start,
        end=start + 12.0,
        scores=tuple(
            ClassScore(label=label, score=score) for label, score in zip(labels, values)
        ),
        embedding=None,
    )


def write_artifact(path, windows, *, metadata=None):
    with ScoresWriter(path, metadata=metadata or build_metadata()) as writer:
        for window in windows:
            writer.write(window)
        return writer.close()


def write_raw_stream(path, schema, batches=()):
    with pa.OSFile(str(path), "wb") as handle:
        with pa.ipc.new_stream(handle, schema) as stream:
            for batch in batches:
                stream.write_batch(batch)
    return checksum_file(path)


def read_rows(path, checksum):
    with read_scores(path, expected_checksum=checksum) as stream:
        return [batch.to_pylist() for batch in stream.batches]


# --- writing ---------------------------------------------------------------


def test_a_window_writes_one_row_per_label(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window(labels=("a", "b", "c"))])

    assert staged.rows == 3
    assert staged.kind == "scores"
    assert staged.contract_id == CONTRACT_ID


def test_a_window_with_no_scores_writes_no_rows(tmp_path):
    staged = write_artifact(
        tmp_path / "scores.arrow", [build_window(0.0, labels=()), build_window(6.0, labels=())]
    )

    assert staged.rows == 0
    assert read_rows(staged.path, staged.checksum) == []


def test_the_schema_is_the_declared_five_fields_with_declared_types(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window()])

    with read_scores(staged.path, expected_checksum=staged.checksum):
        pass

    assert SCORES_SCHEMA.names == [
        "recording_namespace", "recording_value", "window_start_s", "window_end_s", "label",
        "score",
    ]
    assert [field.type for field in SCORES_SCHEMA] == [
        pa.string(), pa.string(), pa.float64(), pa.float64(), pa.string(), pa.float64()
    ]
    assert all(not field.nullable for field in SCORES_SCHEMA)


def test_rows_survive_a_round_trip(tmp_path):
    recording = a_recording("arbimon", "rec:7")
    window = build_window(6.0, recording=recording, labels=("owl", "wren"), scores=[0.0, 1.0])

    staged = write_artifact(tmp_path / "scores.arrow", [window])

    assert read_rows(staged.path, staged.checksum) == [
        [
            {"recording_namespace": "arbimon", "recording_value": "rec:7",
             "window_start_s": 6.0, "window_end_s": 18.0, "label": "owl", "score": 0.0},
            {"recording_namespace": "arbimon", "recording_value": "rec:7",
             "window_start_s": 6.0, "window_end_s": 18.0, "label": "wren", "score": 1.0},
        ]
    ]


def test_row_order_follows_the_windows_it_was_given(tmp_path):
    other = a_recording("arbimon", "42")
    windows = [
        build_window(0.0, labels=("b", "a")),
        build_window(6.0, labels=("a", "b")),
        build_window(0.0, recording=other, labels=("a", "b")),
    ]

    staged = write_artifact(tmp_path / "scores.arrow", windows)

    rows = [row for batch in read_rows(staged.path, staged.checksum) for row in batch]
    assert [
        (row["recording_namespace"], row["recording_value"], row["window_start_s"], row["label"])
        for row in rows
    ] == [
        ("soundhub", "42", 0.0, "b"), ("soundhub", "42", 0.0, "a"),
        ("soundhub", "42", 6.0, "a"), ("soundhub", "42", 6.0, "b"),
        ("arbimon", "42", 0.0, "a"), ("arbimon", "42", 0.0, "b"),
    ]


def test_the_stream_is_written_in_bounded_batches(tmp_path):
    labels = tuple(f"label-{index}" for index in range(100))
    windows = [build_window(float(n) * 12.0, labels=labels) for n in range(200)]

    staged = write_artifact(tmp_path / "scores.arrow", windows)

    batches = read_rows(staged.path, staged.checksum)
    assert len(batches) > 1
    assert sum(len(batch) for batch in batches) == staged.rows == 20000
    # A window is never split, so a batch overshoots the bound by less than one window.
    assert all(len(batch) <= SCORE_BATCH_ROWS + len(labels) for batch in batches)


def test_the_writer_places_each_value_under_its_own_column_name(tmp_path, monkeypatch):
    # window_start_s and window_end_s share a type, so a writer that matched
    # columns by position would swap them under a reordered schema and raise no
    # error.
    swapped = pa.schema(
        [
            SCORES_SCHEMA.field("recording_namespace"),
            SCORES_SCHEMA.field("recording_value"),
            SCORES_SCHEMA.field("window_end_s"),
            SCORES_SCHEMA.field("window_start_s"),
            SCORES_SCHEMA.field("label"),
            SCORES_SCHEMA.field("score"),
        ]
    )
    monkeypatch.setattr(scores_module, "SCORES_SCHEMA", swapped)

    staged = write_artifact(tmp_path / "scores.arrow", [build_window(6.0)])

    row = read_rows(staged.path, staged.checksum)[0][0]
    assert row["window_start_s"] == 6.0
    assert row["window_end_s"] == 18.0


def test_the_file_grows_before_close(tmp_path):
    labels = tuple(f"label-{index}" for index in range(100))
    path = tmp_path / "scores.arrow"

    with ScoresWriter(path, metadata=build_metadata()) as writer:
        for n in range(120):
            writer.write(build_window(float(n) * 12.0, labels=labels))
        assert path.stat().st_size > 0
        writer.close()


def test_a_writer_refuses_a_header_declaring_another_contract(tmp_path):
    # The staged record names this contract unconditionally, so a header naming a
    # different one would produce a file whose header and staged record disagree.
    path = tmp_path / "scores.arrow"

    with pytest.raises(RuntimeError) as exc:
        ScoresWriter(path, metadata=build_metadata(contract_id="robin.embeddings.arrow/1"))

    assert "robin.embeddings.arrow/1" in str(exc.value)
    assert not path.exists()


def test_a_writer_refuses_a_header_missing_a_required_key(tmp_path):
    metadata = {
        key: value
        for key, value in build_metadata().items()
        if key != b"robin.recipe"
    }
    path = tmp_path / "scores.arrow"

    with pytest.raises(RuntimeError) as exc:
        ScoresWriter(path, metadata=metadata)

    assert "robin.recipe" in str(exc.value)
    assert not path.exists()


def test_a_writer_refuses_a_header_missing_a_key_its_retention_requires(tmp_path):
    request = build_request(retention="top_k", min_score=0.005, top_k=5)
    metadata = {
        key: value
        for key, value in build_metadata(request).items()
        if key != b"robin.score_top_k"
    }
    path = tmp_path / "scores.arrow"

    with pytest.raises(RuntimeError) as exc:
        ScoresWriter(path, metadata=metadata)

    assert "robin.score_top_k" in str(exc.value)
    assert not path.exists()


def test_a_writer_refuses_a_retention_the_contract_does_not_declare(tmp_path):
    metadata = build_metadata() | {b"robin.score_retention": b"banana"}
    path = tmp_path / "scores.arrow"

    with pytest.raises(RuntimeError) as exc:
        ScoresWriter(path, metadata=metadata)

    assert "banana" in str(exc.value)
    assert not path.exists()


def test_a_staged_artifact_reads_back_under_the_contract_it_was_staged_as(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window()])

    with read_scores(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.contract"] == staged.contract_id


# --- staging and lifecycle -------------------------------------------------


def test_a_schema_only_stream_reads_back(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [])

    assert staged.rows == 0
    with read_scores(staged.path, expected_checksum=staged.checksum) as stream:
        assert list(stream.batches) == []
        assert stream.metadata["robin.contract"] == CONTRACT_ID


def test_the_writer_leaves_its_file_for_the_caller(tmp_path):
    path = tmp_path / "scores.arrow"

    with ScoresWriter(path, metadata=build_metadata()) as writer:
        writer.write(build_window())
        staged = writer.close()
        assert staged.path.exists()

    assert path.exists()
    assert checksum_file(path) == staged.checksum


def test_write_after_close_is_refused(tmp_path):
    writer = ScoresWriter(tmp_path / "scores.arrow", metadata=build_metadata())
    writer.close()

    with pytest.raises(RuntimeError):
        writer.write(build_window())


def test_close_twice_is_refused(tmp_path):
    writer = ScoresWriter(tmp_path / "scores.arrow", metadata=build_metadata())
    writer.close()

    with pytest.raises(RuntimeError):
        writer.close()


def test_a_writer_released_before_close_refuses_to_stage(tmp_path):
    # Releasing discards whatever was still pending, so staging here would report
    # a row count and a checksum for a file that was never finished.
    path = tmp_path / "scores.arrow"
    with pytest.raises(ZeroDivisionError):
        with ScoresWriter(path, metadata=build_metadata()) as writer:
            writer.write(build_window())
            raise ZeroDivisionError

    with pytest.raises(RuntimeError) as exc:
        writer.close()

    assert "released" in str(exc.value)


# --- reading ---------------------------------------------------------------


def test_the_artifact_declares_its_contract_identifier(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window()])

    with read_scores(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.contract"] == CONTRACT_ID

    declaring = REPOSITORY / "contracts/robin_contracts/output_contracts.py"
    assert CONTRACT_ID in declaring.read_text()
    spelled_in_the_engine = [
        module
        for module in (REPOSITORY / "inference_engine/robin_inference_engine").rglob("*.py")
        if CONTRACT_ID in module.read_text()
    ]
    assert spelled_in_the_engine == []


def test_every_required_metadata_key_is_present_on_the_stream(tmp_path):
    request = build_request(retention="top_k", min_score=0.005, top_k=5)
    staged = write_artifact(
        tmp_path / "scores.arrow", [build_window()], metadata=build_metadata(request)
    )

    with read_scores(staged.path, expected_checksum=staged.checksum) as stream:
        assert set(stream.metadata) == set(TOP_K_KEYS)
        assert len(TOP_K_KEYS) == 13


@pytest.mark.parametrize("absent", TOP_K_KEYS)
def test_a_reader_refuses_an_artifact_missing_any_required_key(tmp_path, absent):
    request = build_request(retention="top_k", min_score=0.005, top_k=5)
    metadata = {
        key: value
        for key, value in build_metadata(request).items()
        if key != absent.encode("utf-8")
    }
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, SCORES_SCHEMA.with_metadata(metadata))

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_reader_refuses_a_scores_artifact_that_names_no_registry(tmp_path):
    # Scores are label positions, and only a registry says what a label means, so a
    # scores artifact carries the registry pair even though other artifacts need not.
    metadata = required_metadata(
        contract_id=CONTRACT_ID,
        work=build_work(),
        recipe=build_recipe(),
        registry_uri=None,
        registry_fingerprint=None,
    ) | score_metadata(build_request(), score_domain="probability")
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, SCORES_SCHEMA.with_metadata(metadata))

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_METADATA_INCOMPLETE
    for key in REGISTRY_KEYS:
        assert key in exc.value.detail


def test_a_reader_refuses_a_header_that_is_not_utf8(tmp_path):
    # Bytes this engine did not write are refused by code like every other distrust
    # on this path, rather than escaping as a decoding error from the standard library.
    metadata = build_metadata() | {b"robin.recipe": b"\xff\xfe"}
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, SCORES_SCHEMA.with_metadata(metadata))

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_reader_refuses_a_stream_carrying_no_metadata_at_all(tmp_path):
    # The right five columns and an empty header is what every tool but this writer
    # produces, and Arrow reports that header as absent rather than as empty.
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, SCORES_SCHEMA)

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED


def test_a_reader_refuses_a_checksum_mismatch(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window()])
    bytes_on_disk = bytearray(staged.path.read_bytes())
    bytes_on_disk[-4] ^= 0xFF
    staged.path.write_bytes(bytes(bytes_on_disk))

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(staged.path, expected_checksum=staged.checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_CHECKSUM_MISMATCH


def test_a_reader_refuses_a_file_that_is_not_there(tmp_path):
    with pytest.raises(errors.EngineError) as exc:
        with read_scores(tmp_path / "absent.arrow", expected_checksum=FILE_DIGEST):
            pass

    assert exc.value.code == errors.ARTIFACT_UNREADABLE
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_reader_refuses_a_missing_field(tmp_path):
    narrowed = pa.schema(
        [field for field in SCORES_SCHEMA if field.name != "window_end_s"]
    ).with_metadata(build_metadata())
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, narrowed)

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "window_end_s" in exc.value.detail


def test_a_reader_refuses_an_artifact_missing_the_recording_namespace(tmp_path):
    narrowed = pa.schema(
        [field for field in SCORES_SCHEMA if field.name != "recording_namespace"]
    ).with_metadata(build_metadata())
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, narrowed)

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "recording_namespace" in exc.value.detail


def test_a_reader_refuses_a_recording_value_stored_as_a_number(tmp_path):
    # A value is text, so a number column is refused: "042" would lose its leading zero.
    numbered = pa.schema(
        [
            pa.field(field.name, pa.int64() if field.name == "recording_value" else field.type,
                     nullable=False)
            for field in SCORES_SCHEMA
        ]
    ).with_metadata(build_metadata())
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, numbered)

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "recording_value" in exc.value.detail


def test_a_reader_refuses_a_narrowed_score_type(tmp_path):
    narrowed = pa.schema(
        [
            pa.field(field.name, pa.float32() if field.name == "score" else field.type,
                     nullable=False)
            for field in SCORES_SCHEMA
        ]
    ).with_metadata(build_metadata())
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, narrowed)

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "score" in exc.value.detail


def test_a_reader_matches_fields_by_name_not_position(tmp_path):
    reordered = pa.schema(list(reversed(list(SCORES_SCHEMA)))).with_metadata(build_metadata())
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, reordered)

    with read_scores(path, expected_checksum=checksum) as stream:
        assert stream.metadata["robin.contract"] == CONTRACT_ID


def test_a_reader_refuses_another_contracts_artifact(tmp_path):
    metadata = build_metadata(contract_id="robin.embeddings.arrow/1")
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, SCORES_SCHEMA.with_metadata(metadata))

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED
    assert "robin.embeddings.arrow/1" in exc.value.detail


def test_a_thresholded_artifact_declares_the_requested_floor(tmp_path):
    request = build_request(retention="thresholded", min_score=0.005)
    staged = write_artifact(
        tmp_path / "thresholded.arrow", [build_window()], metadata=build_metadata(request)
    )

    with read_scores(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.score_retention"] == "thresholded"
        assert stream.metadata["robin.score_floor"] == "0.005"
        assert "robin.score_top_k" not in stream.metadata

    full = write_artifact(
        tmp_path / "full.arrow", [build_window()], metadata=build_metadata(build_request())
    )
    with read_scores(full.path, expected_checksum=full.checksum) as stream:
        assert stream.metadata["robin.score_retention"] == "full"
        assert "robin.score_floor" not in stream.metadata
        assert "robin.score_top_k" not in stream.metadata


def test_a_reader_refuses_a_schema_that_names_one_column_twice(tmp_path):
    # Arrow permits a repeated field name, and a lookup by that name then has no
    # single answer, so the column this contract declares is not identifiable.
    duplicated = pa.schema(
        list(SCORES_SCHEMA) + [pa.field("score", pa.string(), nullable=False)]
    ).with_metadata(build_metadata())
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, duplicated)

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    assert "score" in exc.value.detail


@pytest.mark.parametrize("declared", ["banana", "", "FULL", "top-k"])
def test_a_reader_refuses_a_retention_the_contract_does_not_declare(tmp_path, declared):
    # The reader picks which keys to require from this value, so a value outside
    # the declared set would match no branch and let the header through unchecked.
    metadata = build_metadata() | {
        b"robin.score_retention": declared.encode("utf-8")
    }
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, SCORES_SCHEMA.with_metadata(metadata))

    with pytest.raises(errors.EngineError) as exc:
        with read_scores(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


@pytest.mark.parametrize("retention", ["full", "thresholded", "top_k"])
def test_a_reader_accepts_every_retention_the_contract_declares(tmp_path, retention):
    fields = {"full": {}, "thresholded": {"min_score": 0.005},
              "top_k": {"min_score": 0.005, "top_k": 5}}[retention]
    request = build_request(retention=retention, **fields)
    staged = write_artifact(
        tmp_path / f"{retention}.arrow", [build_window()],
        metadata=build_metadata(request),
    )

    with read_scores(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.score_retention"] == retention


def test_a_reader_accepts_additional_columns(tmp_path):
    widened = pa.schema(
        list(SCORES_SCHEMA) + [pa.field("provenance", pa.string(), nullable=False)]
    ).with_metadata(build_metadata())
    rows = [{
        "recording_namespace": "soundhub",
        "recording_value": "42",
        "window_start_s": 0.0,
        "window_end_s": 12.0,
        "label": "owl",
        "score": 0.5,
        "provenance": "external producer",
    }]
    batch = pa.RecordBatch.from_pylist(rows, schema=widened)
    path = tmp_path / "scores.arrow"
    checksum = write_raw_stream(path, widened, (batch,))

    with read_scores(path, expected_checksum=checksum) as stream:
        assert [row for batch in stream.batches for row in batch.to_pylist()] == rows


def test_a_reader_reports_a_truncated_batch_as_a_typed_error(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window()])
    # Remove the stream terminator and part of the batch body, leaving its header intact.
    staged.path.write_bytes(staged.path.read_bytes()[:-20])
    checksum = checksum_file(staged.path)

    with read_scores(staged.path, expected_checksum=checksum) as stream:
        with pytest.raises(errors.EngineError) as exc:
            list(stream.batches)

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    assert staged.path.name in exc.value.detail


def test_a_reader_does_not_translate_errors_from_the_consumer(tmp_path):
    staged = write_artifact(tmp_path / "scores.arrow", [build_window()])
    failure = OSError("consumer failed")

    with pytest.raises(OSError) as exc:
        with read_scores(staged.path, expected_checksum=staged.checksum):
            raise failure

    assert exc.value is failure
