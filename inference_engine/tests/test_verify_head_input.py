"""Reading a head's input: one recording's embeddings file, checked against the head and
the recording before any row is read."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from robin_contracts.canonical import checksum_file
from robin_contracts.cards import (
    AudioGeometry,
    HeadCard,
    ModelCard,
    RunnerResampled,
    model_ref,
)
from robin_contracts.output_contracts import EmbeddingsRequest
from robin_contracts.specs import recipe
from robin_contracts.work import (
    AudioInput,
    InferenceWork,
    InputArtifact,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts import embeddings as embeddings_module
from robin_inference_engine.artifacts.embeddings import CONTRACT_ID, EmbeddingsWriter
from robin_inference_engine.artifacts.metadata import embedding_metadata, required_metadata
from robin_inference_engine.verify_head_input import read_head_input

DIM = 4
REGISTRY_DIGEST = "sha256:" + "a" * 64


def build_backbone(**overrides) -> ModelCard:
    fields = {
        "model_name": "test-backbone",
        "model_version": "1",
        "runtime": "none",
        "window_duration": 3.0,
        "window_overlap": 0.0,
        "sample_rate": 16000,
        "min_detection_threshold": 0.0,
        "score_domain": None,
        "taxa_registry_digest": None,
        "audio": AudioGeometry(
            downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
        ),
        "backend": "none",
        "dtype": "float32",
        "can_emit_embeddings": True,
        "embedding_dim": DIM,
        "embedding_dtype": "float32",
    }
    return ModelCard(**(fields | overrides))


BACKBONE = build_backbone()


def head_reading(backbone: ModelCard) -> HeadCard:
    return HeadCard(
        model_name="test-head",
        model_version="1",
        runtime="test-runtime",
        backbone=model_ref(backbone),
        embedding_dim=backbone.embedding_dim,
        min_detection_threshold=0.0,
        score_domain="probability",
        taxa_registry_digest=REGISTRY_DIGEST,
    )


HEAD = head_reading(BACKBONE)

# Holds a "/" and an "=", which a value differing only after either must not hide.
RECORDING = RecordingRef(namespace="soundhub", value="site=a/42", audio_uri="s3://b/42.wav")


def backbone_work(card: ModelCard, recording: RecordingRef) -> InferenceWork:
    weights = PinnedFile(uri="s3://b/weights.bin", digest="sha256:" + "0" * 64, size_bytes=8)
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=(recording,),
        model=PinnedModel(card=card, files={"weights": weights}),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=(EmbeddingsRequest(contract_id=CONTRACT_ID),),
    )


def stored_rows(rows: int, dim: int = DIM) -> np.ndarray:
    """Values float16 cannot hold exactly, so a narrowed file differs from them."""
    return (np.arange(rows * dim, dtype=np.float32).reshape(rows, dim) + 1) / 3


def write_embeddings(
    path: Path,
    *,
    backbone: ModelCard = BACKBONE,
    recording: RecordingRef = RECORDING,
    rows: int = 3,
    header: Mapping[str, str | None] | None = None,
) -> np.ndarray:
    """Write `rows` windows as the backbone would, then set (or, for None, drop) each
    `header` value. Returns the vectors given to the writer."""
    work = backbone_work(backbone, recording)
    dim, storage = backbone.embedding_dim, backbone.dtype
    metadata = required_metadata(
        contract_id=CONTRACT_ID,
        work=work,
        recording=recording,
        recipe=recipe(backbone),
        registry_uri=None,
        registry_fingerprint=None,
    ) | embedding_metadata(
        work, dim=dim, source_dtype=backbone.embedding_dtype, storage_dtype=storage
    )
    vectors = stored_rows(rows, dim)
    with EmbeddingsWriter(
        path, recording=recording, dim=dim, storage_dtype=storage, metadata=metadata
    ) as writer:
        for row, vector in enumerate(vectors):
            start = 3.0 * row
            writer.write(
                AcceptedWindow(
                    recording=recording,
                    start=start,
                    end=start + 3.0,
                    scores=(),
                    embedding=vector,
                )
            )
        writer.close()
    if header:
        rewrite_header(path, header)
    return vectors


def rewrite_header(path: Path, header: Mapping[str, str | None]) -> None:
    with pa.ipc.open_stream(path) as reader:
        schema, batches = reader.schema, list(reader)
    metadata = {key.decode(): value.decode() for key, value in schema.metadata.items()}
    for key, value in header.items():
        if value is None:
            metadata.pop(key)
        else:
            metadata[key] = value
    schema = schema.with_metadata(metadata)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
        for batch in batches:
            writer.write_batch(batch.replace_schema_metadata(metadata))


def read_stored(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The file's starts, ends and vectors as stored, read with Arrow alone."""
    with pa.ipc.open_stream(path) as reader:
        table = reader.read_all()
    vectors = table.column("embedding").combine_chunks()
    dim = vectors.type.list_size
    return (
        table.column("window_start_s").to_numpy(),
        table.column("window_end_s").to_numpy(),
        vectors.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim),
    )


def naming(path: Path, recording: RecordingRef = RECORDING) -> RecordingRef:
    """`recording`, naming the file at `path` by its true checksum."""
    embeddings = InputArtifact(uri=path.as_uri(), checksum=checksum_file(path))
    return RecordingRef(**(recording.model_dump() | {"embeddings": embeddings}))


def refusal(path: Path, *, card: HeadCard = HEAD, recording: RecordingRef | None = None):
    recording = recording or naming(path)
    with pytest.raises(errors.EngineError) as caught:
        read_head_input(path, card=card, recording=recording)
    error = caught.value
    assert error.stage == errors.READ_INPUT_ARTIFACT
    assert error.recording == recording
    assert errors.named(recording) in error.detail
    return error


# ---------------------------------------------------------------------------
# The header, against the head card and the recording.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("robin.backbone_ref", "other-backbone/1"),
        ("robin.backbone_card_digest", "sha256:v1:" + "e" * 64),
    ],
)
def test_a_file_from_another_backbone_than_the_heads_is_refused(tmp_path, key, value):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path, header={key: value})

    error = refusal(path)

    assert error.code == errors.HEAD_INPUT_BACKBONE_MISMATCH
    assert value in error.detail
    expected = HEAD.backbone.id if key == "robin.backbone_ref" else HEAD.backbone.digest
    assert expected in error.detail


def test_a_file_of_another_width_than_the_head_takes_is_refused(tmp_path):
    path = tmp_path / "embeddings.arrow"
    wider = build_backbone(embedding_dim=DIM + 1)
    # Named as the head's backbone, so only the width disagrees.
    write_embeddings(
        path,
        backbone=wider,
        header={
            "robin.backbone_ref": HEAD.backbone.id,
            "robin.backbone_card_digest": HEAD.backbone.digest,
        },
    )

    error = refusal(path)

    assert error.code == errors.HEAD_INPUT_WIDTH_MISMATCH
    assert str(DIM + 1) in error.detail
    assert str(DIM) in error.detail


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("robin.recording_namespace", "other"),
        ("robin.recording_value", "site=a/43"),
        ("robin.recording_value", "site=b/42"),
    ],
    ids=["namespace", "value_after_a_slash", "value_after_an_equals_sign"],
)
def test_a_file_holding_another_recording_is_refused(tmp_path, key, value):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path, header={key: value})

    error = refusal(path)

    assert error.code == errors.HEAD_INPUT_RECORDING_MISMATCH
    assert value in error.detail
    expected = RECORDING.namespace if key == "robin.recording_namespace" else RECORDING.value
    assert expected in error.detail


# ---------------------------------------------------------------------------
# The reader's own refusals, carrying the recording.
# ---------------------------------------------------------------------------


def test_a_file_whose_checksum_is_not_the_recordings_is_refused(tmp_path):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path)
    embeddings = InputArtifact(uri=path.as_uri(), checksum="sha256:" + "1" * 64)
    recording = RecordingRef(**(RECORDING.model_dump() | {"embeddings": embeddings}))

    error = refusal(path, recording=recording)

    assert error.code == errors.ARTIFACT_CHECKSUM_MISMATCH


def test_a_file_whose_header_names_no_recording_value_is_refused(tmp_path):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path, header={"robin.recording_value": None})

    error = refusal(path)

    assert error.code == errors.ARTIFACT_METADATA_INCOMPLETE
    assert "robin.recording_value" in error.detail


def test_a_file_with_a_null_window_start_is_refused(tmp_path):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path)
    with pa.ipc.open_stream(path) as reader:
        schema, [batch] = reader.schema, list(reader)
    starts = batch.column("window_start_s")
    nulled = pc.if_else(pa.array([False, True, False]), pa.nulls(3, starts.type), starts)
    columns = [nulled if name == "window_start_s" else batch.column(name) for name in schema.names]
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(pa.RecordBatch.from_arrays(columns, schema=schema))

    error = refusal(path)

    assert error.code == errors.ARTIFACT_MALFORMED


# ---------------------------------------------------------------------------
# The rows, as the head is given them.
# ---------------------------------------------------------------------------


def test_a_float32_file_is_read_as_stored(tmp_path):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path)
    starts, ends, stored = read_stored(path)

    given = read_head_input(path, card=HEAD, recording=naming(path))

    assert given.kind == "embeddings"
    assert given.starts.dtype == np.float64 and given.ends.dtype == np.float64
    assert np.array_equal(given.starts, starts)
    assert np.array_equal(given.ends, ends)
    assert given.values.dtype == np.float32
    assert given.values.shape == (3, DIM)
    assert given.values.flags["C_CONTIGUOUS"]
    assert np.array_equal(given.values, stored)


def test_a_float16_file_is_widened_to_float32_exactly(tmp_path):
    path = tmp_path / "embeddings.arrow"
    narrow = build_backbone(dtype="float16")
    written = write_embeddings(path, backbone=narrow)
    _, _, stored = read_stored(path)
    assert stored.dtype == np.float16
    assert not np.array_equal(stored.astype(np.float32), written)

    given = read_head_input(path, card=head_reading(narrow), recording=naming(path))

    assert given.values.dtype == np.float32
    assert given.values.flags["C_CONTIGUOUS"]
    assert np.array_equal(given.values, stored.astype(np.float32))


def test_every_row_of_a_file_written_in_several_batches_is_read_in_order(
    tmp_path, monkeypatch
):
    # Two rows a batch.
    monkeypatch.setattr(embeddings_module, "EMBEDDING_BATCH_BYTES", 2 * DIM * 4)
    path = tmp_path / "embeddings.arrow"
    written = write_embeddings(path, rows=7)
    with pa.ipc.open_stream(path) as reader:
        assert len(list(reader)) > 1

    given = read_head_input(path, card=HEAD, recording=naming(path))

    assert np.array_equal(given.values, written)
    assert np.array_equal(given.starts, 3.0 * np.arange(7))


def test_reading_leaves_the_file_unchanged(tmp_path):
    path = tmp_path / "embeddings.arrow"
    write_embeddings(path)
    before = checksum_file(path)

    read_head_input(path, card=HEAD, recording=naming(path))

    assert checksum_file(path) == before
