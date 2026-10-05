"""One row per window: what the embeddings writer stages, and what the reader refuses."""

import hashlib
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
from robin_contracts.specs import AudioSpec, Recipe
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
from robin_inference_engine.artifacts.embeddings import (
    CONTRACT_ID,
    EMBEDDING_BATCH_BYTES,
    EmbeddingsWriter,
    embeddings_schema,
    read_embeddings,
)
from robin_inference_engine.artifacts.metadata import (
    EMBEDDING_KEYS,
    REGISTRY_KEYS,
    REQUIRED_KEYS,
    embedding_metadata,
    required_metadata,
)

HEX = "0" * 64
FILE_DIGEST = f"sha256:{HEX}"
BACKBONE_DIGEST = "sha256:v1:" + "d" * 64
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64

CARD = ModelCard(
    model_name="perch",
    model_version="8",
    runtime="tensorflow",
    window_duration=5.0,
    sample_rate=32000,
    min_detection_threshold=0.0,
    can_emit_embeddings=True,
    embedding_dim=4,
    window_overlap=0.0,
    score_domain="probability",
    taxa_registry_digest=REGISTRY_FINGERPRINT,
    audio=AudioGeometry(
        downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
    ),
    backend="tensorflow",
    dtype="float32",
    embedding_dtype="float32",
)


def a_recording(namespace: str, value: str) -> RecordingRef:
    return RecordingRef(namespace=namespace, value=value, audio_uri=f"s3://b/{value}.wav")


SOUNDHUB_42 = a_recording("soundhub", "42")


def build_model(card: ModelCard | HeadCard = CARD) -> PinnedModel:
    return PinnedModel(
        card=card,
        files={
            "weights": PinnedFile(uri="s3://b/perch.tf", digest=FILE_DIGEST, size_bytes=8),
            REGISTRY_ROLE: PinnedFile(
                uri="s3://b/registry.csv", digest=REGISTRY_FINGERPRINT, size_bytes=8
            ),
        },
    )

REPOSITORY = Path(__file__).resolve().parents[2]

DIM = 4


def build_recipe(**overrides) -> Recipe:
    fields = {
        "model": model_ref(CARD),
        "backend": "tensorflow",
        "audio": AudioSpec(
            sample_rate=32000,
            window_duration=5.0,
            window_overlap=0.0,
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        "dtype": "float32",
    }
    return Recipe(**(fields | overrides))


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (
            RecordingRef(namespace="soundhub", value="42", audio_uri="s3://b/42.wav"),
        ),
        "model": build_model(),
        "input": AudioInput(),
        "settings": {},
        "resources": {},
        "outputs": (EmbeddingsRequest(contract_id="robin.embeddings.arrow/1"),),
    }
    return InferenceWork(**(fields | overrides))


def build_metadata(
    *,
    dim: int = DIM,
    source_dtype: str = "float32",
    storage_dtype: str = "float32",
    **overrides,
):
    fields = {
        "contract_id": CONTRACT_ID,
        "work": build_work(),
        "recording": SOUNDHUB_42,
        "recipe": build_recipe(dtype=storage_dtype),
        "registry_uri": "s3://b/registry.csv",
        "registry_fingerprint": REGISTRY_FINGERPRINT,
    }
    fields |= overrides
    return required_metadata(**fields) | embedding_metadata(
        fields["work"], dim=dim, source_dtype=source_dtype, storage_dtype=storage_dtype
    )


HEADER_KEYS = (*REQUIRED_KEYS, *REGISTRY_KEYS, *EMBEDDING_KEYS)


def build_window(
    start: float = 0.0,
    *,
    recording: RecordingRef = SOUNDHUB_42,
    values=None,
    dim: int = DIM,
    dtype="float32",
):
    # The writer takes the array it is handed; the acceptance boundary is what refuses
    # a dtype the instance did not declare, so a float16 vector is legitimate here.
    vector = (
        None
        if values is False
        else np.arange(dim, dtype=dtype) if values is None
        else np.asarray(values, dtype=dtype)
    )
    return AcceptedWindow(
        recording=recording,
        start=start,
        end=start + 5.0,
        scores=(),
        embedding=vector,
    )


def write_artifact(
    path, windows, *, metadata=None, dim=DIM, source_dtype="float32", storage_dtype="float32"
):
    header = metadata or build_metadata(
        dim=dim, source_dtype=source_dtype, storage_dtype=storage_dtype
    )
    with EmbeddingsWriter(
        path, recording=SOUNDHUB_42, dim=dim, storage_dtype=storage_dtype, metadata=header
    ) as writer:
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
    with read_embeddings(path, expected_checksum=checksum) as stream:
        return [batch.to_pylist() for batch in stream.batches]


# --- the schema ------------------------------------------------------------


def test_the_schema_is_the_declared_three_fields_with_declared_types():
    schema = embeddings_schema(DIM, "float32")

    assert schema.names == ["window_start_s", "window_end_s", "embedding"]
    assert [field.type for field in schema][:2] == [pa.float64(), pa.float64()]
    assert schema.field("embedding").type == pa.list_(pa.float32(), DIM)
    assert all(not field.nullable for field in schema)


def test_the_embedding_column_is_a_fixed_size_list_and_not_a_variable_length_one():
    # A variable-length list carries no width, so a head holding the schema alone
    # could not reject a wrong-width artifact before it loads a graph.
    embedding = embeddings_schema(DIM, "float32").field("embedding").type

    assert pa.types.is_fixed_size_list(embedding)
    assert embedding.list_size == DIM
    assert embedding != pa.list_(pa.float32())


def test_the_storage_dtype_chooses_the_lists_value_type():
    assert embeddings_schema(DIM, "float16").field("embedding").type == pa.list_(
        pa.float16(), DIM
    )


# --- writing and round-tripping --------------------------------------------


def test_a_window_writes_one_row(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])

    assert staged.rows == 1
    assert staged.kind == "embeddings"
    assert staged.contract_id == CONTRACT_ID
    assert staged.recording == SOUNDHUB_42


def test_a_window_with_no_embedding_writes_no_row(tmp_path):
    # The acceptance boundary owns whether an absent vector is a defect; this writer
    # counts what it was given rather than holding a second opinion.
    staged = write_artifact(
        tmp_path / "embeddings.arrow",
        [build_window(0.0), build_window(5.0, values=False)],
    )

    assert staged.rows == 1
    assert read_rows(staged.path, staged.checksum)[0] == [
        {
            "window_start_s": 0.0,
            "window_end_s": 5.0,
            "embedding": [0.0, 1.0, 2.0, 3.0],
        }
    ]


def test_vectors_survive_a_float32_round_trip_exactly(tmp_path):
    # The extremes of the type, so a narrowing or a widening anywhere on the path
    # shows up as an inequality rather than as a rounding nobody notices.
    source = np.asarray([0.0, -1.5, np.finfo(np.float32).max, np.finfo(np.float32).tiny],
                        dtype=np.float32)

    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window(values=source)])

    stored = read_rows(staged.path, staged.checksum)[0][0]["embedding"]
    assert np.asarray(stored, dtype=np.float32).tolist() == source.tolist()
    assert stored == source.astype(float).tolist()


def test_row_order_follows_the_windows_it_was_given(tmp_path):
    windows = [
        build_window(0.0, values=[0, 0, 0, 0]),
        build_window(5.0, values=[1, 1, 1, 1]),
        build_window(10.0, values=[2, 2, 2, 2]),
    ]

    staged = write_artifact(tmp_path / "embeddings.arrow", windows)

    rows = [row for batch in read_rows(staged.path, staged.checksum) for row in batch]
    assert [(row["window_start_s"], row["embedding"][0]) for row in rows] == [
        (0.0, 0.0),
        (5.0, 1.0),
        (10.0, 2.0),
    ]


def test_a_staged_artifact_reads_back_under_the_contract_it_was_staged_as(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.contract"] == staged.contract_id


def test_the_artifact_declares_its_contract_identifier(tmp_path):
    # The identifier is spelled once, in the contracts package, and imported everywhere.
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.contract"] == CONTRACT_ID

    declaring = REPOSITORY / "contracts/robin_contracts/output_contracts.py"
    assert CONTRACT_ID in declaring.read_text()
    spelled_in_the_engine = [
        module
        for module in (REPOSITORY / "inference_engine/robin_inference_engine").rglob("*.py")
        if CONTRACT_ID in module.read_text()
    ]
    assert spelled_in_the_engine == []


# --- staging and lifecycle -------------------------------------------------


def test_the_writer_leaves_its_file_for_the_caller(tmp_path):
    path = tmp_path / "embeddings.arrow"

    with EmbeddingsWriter(
        path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32", metadata=build_metadata()
    ) as writer:
        writer.write(build_window())
        staged = writer.close()
        assert staged.path.exists()

    assert path.exists()
    assert checksum_file(path) == staged.checksum


def test_write_after_close_is_refused(tmp_path):
    writer = EmbeddingsWriter(
        tmp_path / "embeddings.arrow", recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32",
        metadata=build_metadata(),
    )
    writer.close()

    with pytest.raises(RuntimeError):
        writer.write(build_window())


def test_close_twice_is_refused(tmp_path):
    writer = EmbeddingsWriter(
        tmp_path / "embeddings.arrow", recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32",
        metadata=build_metadata(),
    )
    writer.close()

    with pytest.raises(RuntimeError):
        writer.close()


def test_a_writer_released_before_close_refuses_to_stage(tmp_path):
    # Releasing discards whatever was still pending, so staging here would report a
    # row count and a checksum for a file that was never finished.
    path = tmp_path / "embeddings.arrow"
    with pytest.raises(ZeroDivisionError):
        with EmbeddingsWriter(
            path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32", metadata=build_metadata()
        ) as writer:
            writer.write(build_window())
            raise ZeroDivisionError

    with pytest.raises(RuntimeError) as exc:
        writer.close()

    assert "released" in str(exc.value)


def test_a_schema_only_stream_reads_back(tmp_path):
    # A work that completes no window still writes a requested kind, so the artifact
    # must be valid and correctly typed before any row exists.
    staged = write_artifact(tmp_path / "embeddings.arrow", [])

    assert staged.rows == 0
    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert list(stream.batches) == []
        assert stream.metadata["robin.contract"] == CONTRACT_ID


# --- the width -------------------------------------------------------------


def test_the_width_is_in_the_type_of_a_stream_that_holds_no_vector(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [], dim=1280)

    with pa.OSFile(str(staged.path), "rb") as handle:
        schema = pa.ipc.open_stream(handle).schema

    assert schema.field("embedding").type == pa.list_(pa.float32(), 1280)


def test_a_vector_of_another_width_is_an_engine_defect(tmp_path):
    # The acceptance boundary has already refused every other width, so a wrong one
    # reaching the writer is the engine contradicting itself, not an untrusted input.
    writer = EmbeddingsWriter(
        tmp_path / "embeddings.arrow", recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32",
        metadata=build_metadata(),
    )

    with pytest.raises(RuntimeError) as exc:
        writer.write(build_window(values=[1.0, 2.0]))

    assert str(DIM) in str(exc.value) and "2" in str(exc.value)


# --- batching --------------------------------------------------------------


WIDE = 1280
# Enough vectors to cross the writer's buffer more than once, whatever it is set to.
ENOUGH_TO_FLUSH = 3 * EMBEDDING_BATCH_BYTES // (WIDE * np.dtype(np.float32).itemsize)


def test_the_stream_is_written_in_bounded_batches(tmp_path):
    # The contract is that it batches at all, not where the boundaries fall: a
    # reader must accept any batching, including none.
    windows = [build_window(float(n) * 5.0, dim=WIDE) for n in range(ENOUGH_TO_FLUSH)]

    staged = write_artifact(tmp_path / "embeddings.arrow", windows, dim=WIDE)

    batches = read_rows(staged.path, staged.checksum)
    assert len(batches) > 1
    assert max(len(batch) for batch in batches) < len(windows)
    assert sum(len(batch) for batch in batches) == staged.rows == len(windows)


def test_the_file_grows_before_close(tmp_path):
    path = tmp_path / "embeddings.arrow"

    with EmbeddingsWriter(
        path, recording=SOUNDHUB_42, dim=WIDE, storage_dtype="float32", metadata=build_metadata(dim=WIDE)
    ) as writer:
        for n in range(ENOUGH_TO_FLUSH):
            writer.write(build_window(float(n) * 5.0, dim=WIDE))
        assert path.stat().st_size > 0
        writer.close()


# --- the header ------------------------------------------------------------


def test_every_required_key_is_present_on_the_stream(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert set(stream.metadata) == set(HEADER_KEYS)
        assert len(HEADER_KEYS) == 16


@pytest.mark.parametrize("absent", HEADER_KEYS)
def test_a_reader_refuses_an_artifact_missing_any_required_key(tmp_path, absent):
    metadata = {
        key: value
        for key, value in build_metadata().items()
        if key != absent.encode("utf-8")
    }
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, embeddings_schema(DIM, "float32").with_metadata(metadata)
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    if absent == "robin.contract":
        assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED
    else:
        # Naming the one key removed means each case fails on its own key, not on
        # some other check that happens to share the code.
        assert exc.value.code == errors.ARTIFACT_METADATA_INCOMPLETE
        assert exc.value.detail.endswith(f": {absent}")


def test_a_writer_refuses_a_header_declaring_another_contract(tmp_path):
    path = tmp_path / "embeddings.arrow"

    with pytest.raises(RuntimeError) as exc:
        EmbeddingsWriter(
            path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32",
            metadata=build_metadata(contract_id="robin.scores.arrow/1"),
        )

    assert "robin.scores.arrow/1" in str(exc.value)
    assert not path.exists()


def test_a_writer_refuses_a_header_missing_a_required_key(tmp_path):
    metadata = {
        key: value
        for key, value in build_metadata().items()
        if key != b"robin.backbone_ref"
    }
    path = tmp_path / "embeddings.arrow"

    with pytest.raises(RuntimeError) as exc:
        EmbeddingsWriter(path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32", metadata=metadata)

    assert "robin.backbone_ref" in str(exc.value)
    assert not path.exists()


def test_a_writer_refuses_a_header_that_contradicts_the_stream_it_would_write(tmp_path):
    path = tmp_path / "embeddings.arrow"

    with pytest.raises(RuntimeError) as exc:
        EmbeddingsWriter(
            path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32", metadata=build_metadata(dim=DIM + 1)
        )
    assert "robin.embedding_dim" in str(exc.value)

    with pytest.raises(RuntimeError) as exc:
        EmbeddingsWriter(
            path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32",
            metadata=build_metadata(storage_dtype="float16"),
        )
    assert "robin.embedding_storage_dtype" in str(exc.value)
    assert not path.exists()


@pytest.mark.parametrize("dim", [0, -1, -4, 2147483648])
def test_a_writer_refuses_an_invalid_dimension_before_creating_a_file(tmp_path, dim):
    path = tmp_path / "embeddings.arrow"

    with pytest.raises(RuntimeError) as exc:
        EmbeddingsWriter(
            path, recording=SOUNDHUB_42, dim=dim, storage_dtype="float32", metadata=build_metadata(dim=dim)
        )

    assert "robin.embedding_dim" in str(exc.value)
    assert not path.exists()


def test_a_writer_refuses_a_header_that_is_not_utf8(tmp_path):
    metadata = build_metadata() | {b"robin.backbone_ref": b"\xff\xfe"}
    path = tmp_path / "embeddings.arrow"

    with pytest.raises(RuntimeError) as exc:
        EmbeddingsWriter(path, recording=SOUNDHUB_42, dim=DIM, storage_dtype="float32", metadata=metadata)

    assert "UTF-8" in str(exc.value)
    assert not path.exists()


def test_a_work_that_bound_no_registry_writes_neither_registry_key(tmp_path):
    # These rows carry no labels, so a backbone whose card names no registry has
    # nothing to write here, and the artifact is complete without the pair.
    metadata = build_metadata(registry_uri=None, registry_fingerprint=None)

    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()], metadata=metadata)

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert set(stream.metadata) == set(REQUIRED_KEYS) | set(EMBEDDING_KEYS)
        for key in REGISTRY_KEYS:
            assert key not in stream.metadata


def test_a_work_that_bound_a_registry_writes_both_registry_keys(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.registry_uri"] == "s3://b/registry.csv"
        assert stream.metadata["robin.registry_fingerprint"] == REGISTRY_FINGERPRINT


@pytest.mark.parametrize("present", REGISTRY_KEYS)
def test_a_reader_refuses_a_header_carrying_half_a_registry_binding(tmp_path, present):
    metadata = {
        key: value
        for key, value in build_metadata().items()
        if key.decode("utf-8") not in REGISTRY_KEYS or key.decode("utf-8") == present
    }
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, embeddings_schema(DIM, "float32").with_metadata(metadata)
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_METADATA_INCOMPLETE
    assert present not in exc.value.detail
    assert any(key in exc.value.detail for key in REGISTRY_KEYS)


def test_a_backbone_run_names_itself_as_its_backbone(tmp_path):
    own = write_artifact(tmp_path / "backbone.arrow", [build_window()])

    with read_embeddings(own.path, expected_checksum=own.checksum) as stream:
        assert stream.metadata["robin.backbone_ref"] == stream.metadata["robin.model_ref"]
        assert (
            stream.metadata["robin.backbone_card_digest"]
            == stream.metadata["robin.model_card_digest"]
        )


def test_the_shipped_recipe_hashes_to_the_fingerprint_beside_it(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        shipped = stream.metadata["robin.recipe"].encode("utf-8")
        assert (
            "sha256:v1:" + hashlib.sha256(shipped).hexdigest()
            == stream.metadata["robin.recipe_fingerprint"]
        )
        assert stream.metadata["robin.recipe_fingerprint"] == build_recipe().id


# --- the storage dtype -----------------------------------------------------


def test_a_float16_recipe_narrows_once_at_write_time_and_the_header_says_so(tmp_path):
    source = np.asarray([0.5, -0.25, 1.0, 0.1], dtype=np.float32)

    staged = write_artifact(
        tmp_path / "embeddings.arrow", [build_window(values=source)], storage_dtype="float16"
    )

    with pa.OSFile(str(staged.path), "rb") as handle:
        assert pa.ipc.open_stream(handle).schema.field("embedding").type == pa.list_(
            pa.float16(), DIM
        )

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.embedding_storage_dtype"] == "float16"
        stored = [batch.to_pylist() for batch in stream.batches][0][0]["embedding"]

    assert np.asarray(stored, dtype=np.float16).tolist() == source.astype(np.float16).tolist()


@pytest.mark.parametrize("source_dtype", ["float32", "float16"])
@pytest.mark.parametrize("storage_dtype", ["float32", "float16"])
def test_the_header_records_the_declared_source_dtype_whatever_is_stored(
    tmp_path, source_dtype, storage_dtype
):
    # The two are independent: what the adapter declared it emits, and what the recipe
    # asked to be stored. Neither is read off the other.
    staged = write_artifact(
        tmp_path / "embeddings.arrow",
        [build_window(dtype=source_dtype)],
        source_dtype=source_dtype,
        storage_dtype=storage_dtype,
    )

    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.embedding_source_dtype"] == source_dtype
        assert stream.metadata["robin.embedding_storage_dtype"] == storage_dtype
        stored = [batch.to_pylist() for batch in stream.batches][0][0]["embedding"]

    assert stored == np.arange(DIM, dtype=storage_dtype).tolist()


@pytest.mark.parametrize("source_dtype", ["float32", "float16"])
def test_a_work_completing_no_window_still_records_the_declared_source_dtype(
    tmp_path, source_dtype
):
    # The case the declaration exists for: the header is stamped at creation, so no
    # array is ever available to read the dtype from.
    staged = write_artifact(
        tmp_path / "embeddings.arrow", [], source_dtype=source_dtype
    )

    assert staged.rows == 0
    with read_embeddings(staged.path, expected_checksum=staged.checksum) as stream:
        assert stream.metadata["robin.embedding_source_dtype"] == source_dtype


def test_a_subnormal_that_underflows_to_zero_is_written(tmp_path):
    # Ordinary precision loss, and the declared consequence of asking for float16.
    # Refusing it would make the narrow width unusable.
    underflows = float(np.finfo(np.float16).smallest_subnormal) / 2
    source = np.asarray([underflows] * DIM, dtype=np.float32)

    staged = write_artifact(
        tmp_path / "embeddings.arrow", [build_window(values=source)], storage_dtype="float16"
    )

    assert staged.rows == 1
    assert read_rows(staged.path, staged.checksum)[0][0]["embedding"] == [0.0] * DIM


@pytest.mark.parametrize("storage_dtype", ["float32", "float16"])
def test_an_all_zero_embedding_is_written_unchanged_and_counted(tmp_path, storage_dtype):
    # The value most likely to be mistaken for absent data. It is a row like any other.
    staged = write_artifact(
        tmp_path / "embeddings.arrow",
        [build_window(values=[0.0] * DIM)],
        storage_dtype=storage_dtype,
    )

    assert staged.rows == 1
    assert read_rows(staged.path, staged.checksum)[0][0]["embedding"] == [0.0] * DIM


def test_a_finite_value_that_does_not_survive_narrowing_is_refused(tmp_path):
    # Narrowing happens after the finiteness check, so a value that passed it can
    # become infinite under a header declaring a clean narrowing.
    overflows = float(np.finfo(np.float16).max) * 2
    source = np.asarray([1.0, overflows, 1.0, 1.0], dtype=np.float32)
    writer = EmbeddingsWriter(
        tmp_path / "embeddings.arrow", recording=SOUNDHUB_42, dim=DIM, storage_dtype="float16",
        metadata=build_metadata(storage_dtype="float16"),
    )

    with pytest.raises(errors.EngineError) as exc:
        writer.write(
            build_window(5.0, recording=a_recording("arbimon", "3"), values=source)
        )

    assert exc.value.code == errors.EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE
    assert exc.value.stage == errors.WRITE_ARTIFACT
    assert exc.value.recording == a_recording("arbimon", "3")
    assert exc.value.window_start_s == 5.0
    assert "float16" in exc.value.detail


def test_a_file_a_refusal_interrupted_still_closes_and_reads_back(tmp_path):
    # The refusal happens mid-write, so the row it refused must leave no trace: closing
    # afterwards has to produce a readable artifact holding exactly the good window.
    overflows = float(np.finfo(np.float16).max) * 2
    writer = EmbeddingsWriter(
        tmp_path / "embeddings.arrow", recording=SOUNDHUB_42, dim=DIM, storage_dtype="float16",
        metadata=build_metadata(storage_dtype="float16"),
    )
    writer.write(build_window())

    with pytest.raises(errors.EngineError) as exc:
        writer.write(build_window(5.0, values=[overflows] * DIM))
    assert exc.value.code == errors.EMBEDDING_VALUE_OUT_OF_STORAGE_DTYPE_RANGE

    staged = writer.close()

    assert staged.rows == 1
    assert staged.checksum == checksum_file(staged.path)
    assert read_rows(staged.path, staged.checksum)[0] == [
        {
            "window_start_s": 0.0,
            "window_end_s": 5.0,
            "embedding": [0.0, 1.0, 2.0, 3.0],
        }
    ]


def test_a_value_the_narrow_width_holds_is_not_refused(tmp_path):
    # The refusal is about what narrowing destroys, not about magnitude: the largest
    # value float16 represents is written like any other.
    largest = float(np.finfo(np.float16).max)
    source = np.asarray([largest, -largest, 0.0, 1.0], dtype=np.float32)

    staged = write_artifact(
        tmp_path / "embeddings.arrow", [build_window(values=source)], storage_dtype="float16"
    )

    assert staged.rows == 1
    assert read_rows(staged.path, staged.checksum)[0][0]["embedding"][0] == largest


def test_a_reader_refuses_an_artifact_whose_declared_width_is_not_its_list_width(tmp_path):
    # The artifact contradicting itself. A consumer that trusted the header would
    # size its input from one number and receive rows of another.
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, embeddings_schema(DIM * 2, "float32").with_metadata(build_metadata(dim=DIM))
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert str(DIM) in exc.value.detail and str(DIM * 2) in exc.value.detail


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param("wide", id="not-a-number"),
        pytest.param("²", id="unicode-digit"),
        pytest.param("9" * 4301, id="overlong-integer"),
        pytest.param("2147483648", id="above-arrow-limit"),
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
    ],
)
def test_a_reader_refuses_an_invalid_declared_width(tmp_path, declared):
    metadata = build_metadata() | {b"robin.embedding_dim": declared.encode("utf-8")}
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, embeddings_schema(DIM, "float32").with_metadata(metadata)
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    assert "robin.embedding_dim" in exc.value.detail


def test_a_reader_refuses_an_artifact_storing_a_width_its_header_does_not_declare(tmp_path):
    # The same self-contradiction one field along: a header promising float16 over
    # float32 rows would have a consumer read twice the bytes it expected.
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path,
        embeddings_schema(DIM, "float16").with_metadata(build_metadata(storage_dtype="float32")),
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "float16" in exc.value.detail and "float32" in exc.value.detail


def test_a_reader_refuses_a_variable_length_list(tmp_path):
    # The donor's type. It is invisible until a head tries to reject a wrong-width
    # artifact and finds the schema carries no width to reject it by.
    unsized = pa.schema(
        [
            *list(embeddings_schema(DIM, "float32"))[:3],
            pa.field("embedding", pa.list_(pa.float32()), nullable=False),
        ]
    ).with_metadata(build_metadata())
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, unsized)

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "embedding" in exc.value.detail


# --- reading bytes this engine did not write -------------------------------


def test_a_reader_refuses_a_checksum_mismatch(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])
    bytes_on_disk = bytearray(staged.path.read_bytes())
    bytes_on_disk[-4] ^= 0xFF
    staged.path.write_bytes(bytes(bytes_on_disk))

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(staged.path, expected_checksum=staged.checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_CHECKSUM_MISMATCH


def test_a_reader_refuses_a_file_that_is_not_there(tmp_path):
    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(tmp_path / "absent.arrow", expected_checksum=FILE_DIGEST):
            pass

    assert exc.value.code == errors.ARTIFACT_UNREADABLE
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT


def test_a_reader_refuses_bytes_that_are_not_an_arrow_stream(tmp_path):
    path = tmp_path / "embeddings.arrow"
    path.write_bytes(b"not an arrow stream")

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum_file(path)):
            pass

    assert exc.value.code == errors.ARTIFACT_MALFORMED


def test_a_reader_reports_a_truncated_batch_as_a_typed_error(tmp_path):
    staged = write_artifact(tmp_path / "embeddings.arrow", [build_window()])
    # Remove the stream terminator and part of the batch body, header intact.
    staged.path.write_bytes(staged.path.read_bytes()[:-20])
    checksum = checksum_file(staged.path)

    with read_embeddings(staged.path, expected_checksum=checksum) as stream:
        with pytest.raises(errors.EngineError) as exc:
            list(stream.batches)

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert staged.path.name in exc.value.detail


def test_a_reader_refuses_a_null_embedding_row(tmp_path):
    staged = write_artifact(
        tmp_path / "embeddings.arrow", [build_window(5.0 * row) for row in range(3)]
    )
    with pa.ipc.open_stream(staged.path) as reader:
        schema, [batch] = reader.schema, list(reader)
    vectors = batch.column("embedding")
    nulled = pc.if_else(pa.array([True, False, False]), pa.nulls(3, vectors.type), vectors)
    columns = [nulled if name == "embedding" else batch.column(name) for name in schema.names]
    # The schema still declares the column not nullable; Arrow does not enforce it.
    checksum = write_raw_stream(
        staged.path, schema, (pa.RecordBatch.from_arrays(columns, schema=schema),)
    )

    with read_embeddings(staged.path, expected_checksum=checksum) as stream:
        with pytest.raises(errors.EngineError) as exc:
            list(stream.batches)

    assert exc.value.code == errors.ARTIFACT_MALFORMED
    assert "'embedding'" in exc.value.detail


def test_a_reader_refuses_another_contracts_artifact(tmp_path):
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path,
        embeddings_schema(DIM, "float32").with_metadata(
            build_metadata(contract_id="robin.scores.arrow/1")
        ),
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED
    assert "robin.scores.arrow/1" in exc.value.detail


def test_a_reader_refuses_a_stream_carrying_no_metadata_at_all(tmp_path):
    # The right three columns and an empty header is what every tool but this writer
    # produces, and Arrow reports that header as absent rather than as empty.
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, embeddings_schema(DIM, "float32"))

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED


def test_a_reader_refuses_a_header_that_is_not_utf8(tmp_path):
    metadata = build_metadata() | {b"robin.recipe": b"\xff\xfe"}
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, embeddings_schema(DIM, "float32").with_metadata(metadata)
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_MALFORMED


def test_a_reader_refuses_a_storage_dtype_the_contract_does_not_declare(tmp_path):
    # The schema to compare against is chosen from this value, so one outside the
    # declared set would leave the rows unchecked.
    metadata = build_metadata() | {b"robin.embedding_storage_dtype": b"float64"}
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, embeddings_schema(DIM, "float32").with_metadata(metadata)
    )

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "float64" in exc.value.detail


def test_a_reader_refuses_a_missing_field(tmp_path):
    narrowed = pa.schema(
        [field for field in embeddings_schema(DIM, "float32") if field.name != "window_end_s"]
    ).with_metadata(build_metadata())
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, narrowed)

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "window_end_s" in exc.value.detail


def test_a_reader_refuses_a_narrowed_bound_type(tmp_path):
    narrowed = pa.schema(
        [
            pa.field(
                field.name,
                pa.float32() if field.name == "window_start_s" else field.type,
                nullable=False,
            )
            for field in embeddings_schema(DIM, "float32")
        ]
    ).with_metadata(build_metadata())
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, narrowed)

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "window_start_s" in exc.value.detail


def test_a_reader_refuses_a_nullable_column(tmp_path):
    relaxed = pa.schema(
        [
            pa.field(field.name, field.type, nullable=field.name == "embedding")
            for field in embeddings_schema(DIM, "float32")
        ]
    ).with_metadata(build_metadata())
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, relaxed)

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "nullable" in exc.value.detail


def test_a_reader_refuses_a_schema_that_names_one_column_twice(tmp_path):
    duplicated = pa.schema(
        list(embeddings_schema(DIM, "float32"))
        + [pa.field("embedding", pa.string(), nullable=False)]
    ).with_metadata(build_metadata())
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, duplicated)

    with pytest.raises(errors.EngineError) as exc:
        with read_embeddings(path, expected_checksum=checksum):
            pass

    assert exc.value.code == errors.ARTIFACT_SCHEMA_INVALID
    assert "embedding" in exc.value.detail


def test_a_reader_matches_fields_by_name_not_position(tmp_path):
    reordered = pa.schema(
        list(reversed(list(embeddings_schema(DIM, "float32"))))
    ).with_metadata(build_metadata())
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(path, reordered)

    with read_embeddings(path, expected_checksum=checksum) as stream:
        assert stream.metadata["robin.contract"] == CONTRACT_ID


def test_a_reader_accepts_additional_columns(tmp_path):
    widened = pa.schema(
        list(embeddings_schema(DIM, "float32"))
        + [pa.field("provenance", pa.string(), nullable=False)]
    ).with_metadata(build_metadata())
    rows = [
        {
            "window_start_s": 0.0,
            "window_end_s": 5.0,
            "embedding": [0.0, 1.0, 2.0, 3.0],
            "provenance": "external producer",
        }
    ]
    path = tmp_path / "embeddings.arrow"
    checksum = write_raw_stream(
        path, widened, (pa.RecordBatch.from_pylist(rows, schema=widened),)
    )

    with read_embeddings(path, expected_checksum=checksum) as stream:
        assert [row for batch in stream.batches for row in batch.to_pylist()] == rows
