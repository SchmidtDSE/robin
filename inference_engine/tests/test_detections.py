"""Detections: which scores a policy selects, how they rank, and the file they are written to."""

import math
import random
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robin_contracts.canonical import checksum_file, sha256_v1
from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled, model_ref
from robin_contracts.output_contracts import (
    DetectionsRequest,
    ScoresRequest,
    ThresholdPolicy,
    TopKPolicy,
)
from robin_contracts.records import ClassScore
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.specs import AudioSpec, Recipe
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
    recording_work_digest,
)
from robin_inference_engine import detections, errors
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts import scores as scores_module
from robin_inference_engine.artifacts.metadata import (
    decode_metadata,
    detection_metadata,
    required_metadata,
    score_metadata,
)
from robin_inference_engine.artifacts.scores import ScoresStream, ScoresWriter
from robin_inference_engine.artifacts.staging import StagedArtifact, malformed
from robin_inference_engine.detections import (
    DETECTIONS_SCHEMA,
    build_detections_sql,
    registry_table,
    write_detections,
)

RECIPE_FINGERPRINT = "sha256:v1:" + "c" * 64
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64


def a_registry(*entries: RegistryEntry) -> TaxonRegistry:
    return TaxonRegistry(fingerprint=REGISTRY_FINGERPRINT, entries=entries)


def sounds(*labels: str) -> TaxonRegistry:
    """A registry declaring each label as a non-taxonomic class."""
    return a_registry(
        *(
            RegistryEntry(class_index=index, label=label, label_kind="non_taxonomic")
            for index, label in enumerate(labels)
        )
    )


BARRED_OWL = RegistryEntry(
    class_index=0,
    label="STVA",
    label_kind="taxon",
    scientific_name="Strix varia",
    common_name="Barred Owl",
    gbif_taxon_key=2497921,
)
UNRESOLVED_OWL = RegistryEntry(
    class_index=1,
    label="STOC",
    label_kind="unresolved",
    scientific_name="Strix occidentalis caurina",
    common_name="Northern Spotted Owl",
)
RAIN = RegistryEntry(class_index=2, label="Rain", label_kind="non_taxonomic")


def scores_table(rows) -> pa.Table:
    """`(window_start_s, label, score)` rows, each window three seconds long."""
    return pa.table(
        {
            "window_start_s": pa.array([start for start, _, _ in rows], pa.float64()),
            "window_end_s": pa.array([start + 3.0 for start, _, _ in rows], pa.float64()),
            "label": pa.array([label for _, label, _ in rows], pa.string()),
            "score": pa.array([score for _, _, score in rows], pa.float64()),
        }
    )


def select(policy, rows, registry: TaxonRegistry) -> pa.Table:
    """Run the detections query over constructed scores on an in-memory connection."""
    sql, parameters = build_detections_sql(policy, recipe_fingerprint=RECIPE_FINGERPRINT)
    with duckdb.connect() as connection:
        connection.register("scores", scores_table(rows))
        connection.register("registry", registry_table(registry))
        return connection.execute(sql, parameters).to_arrow_table()


def selected(table: pa.Table, *columns: str) -> list[tuple]:
    return list(zip(*(table.column(name).to_pylist() for name in columns)))


# --- Thresholds ---------------------------------------------------------------


@pytest.mark.parametrize(
    "floor",
    [
        pytest.param(0.3, id="decimal"),
        # The double after 0.1, which has no short decimal form.
        pytest.param(0.1 + 2**-56, id="no-decimal-spelling"),
    ],
)
def test_a_score_equal_to_the_floor_is_kept_and_the_one_below_is_dropped(floor):
    below = math.nextafter(floor, 0.0)

    table = select(
        ThresholdPolicy(min_score=floor),
        [(0.0, "at", floor), (0.0, "below", below)],
        sounds("at", "below"),
    )

    assert selected(table, "label", "score") == [("at", floor)]


def test_top_k_keeps_each_windows_k_highest():
    rows = [
        (0.0, "a", 0.9),
        (0.0, "b", 0.8),
        (0.0, "c", 0.7),
        (3.0, "a", 0.1),
        (3.0, "b", 0.2),
        (3.0, "c", 0.3),
    ]

    table = select(TopKPolicy(k=2), rows, sounds("a", "b", "c"))

    assert selected(table, "window_start_s", "label", "rank") == [
        (0.0, "a", 1),
        (0.0, "b", 2),
        (3.0, "c", 1),
        (3.0, "b", 2),
    ]


def test_a_top_k_floor_applies_before_the_k_are_chosen():
    rows = [(0.0, "a", 0.9), (0.0, "b", 0.4), (0.0, "c", 0.3)]

    table = select(TopKPolicy(k=2, min_score=0.5), rows, sounds("a", "b", "c"))

    assert selected(table, "label", "rank") == [("a", 1)]


# --- Ranking ------------------------------------------------------------------


def test_tied_scores_rank_by_label_in_code_point_order():
    labels = ["é", "a", "Z", "e", "B"]
    random.Random(7).shuffle(labels)
    rows = [(0.0, label, 0.5) for label in labels]
    policy = ThresholdPolicy(min_score=0.1)
    registry = sounds(*labels)

    first = select(policy, rows, registry)
    second = select(policy, list(reversed(rows)), registry)

    assert selected(first, "label", "rank") == [
        ("B", 1),
        ("Z", 2),
        ("a", 3),
        ("e", 4),
        ("é", 5),
    ]
    assert first.equals(second)


def test_rank_counts_only_the_scores_that_survive_the_floor():
    rows = [(0.0, "a", 0.3), (0.0, "b", 0.2), (0.0, "c", 0.7), (0.0, "d", 0.5)]

    table = select(ThresholdPolicy(min_score=0.4), rows, sounds("a", "b", "c", "d"))

    # Ranked among the two that survive, not among all four.
    assert selected(table, "label", "rank") == [("c", 1), ("d", 2)]


def test_rank_restarts_in_each_window():
    rows = [(start, label, 0.5) for start in (0.0, 3.0) for label in ("a", "b")]

    table = select(ThresholdPolicy(min_score=0.1), rows, sounds("a", "b"))

    assert selected(table, "window_start_s", "rank") == [
        (0.0, 1),
        (0.0, 2),
        (3.0, 1),
        (3.0, 2),
    ]
    assert table.schema.field("rank").type == pa.int32()


# --- The registry join ----------------------------------------------------------


def test_each_row_carries_its_registry_entry():
    rows = [(0.0, "STVA", 0.9), (0.0, "STOC", 0.8), (0.0, "Rain", 0.7)]

    table = select(
        ThresholdPolicy(min_score=0.5), rows, a_registry(BARRED_OWL, UNRESOLVED_OWL, RAIN)
    )

    assert selected(
        table, "label", "label_kind", "scientific_name", "common_name", "gbif_taxon_key"
    ) == [
        ("STVA", "taxon", "Strix varia", "Barred Owl", 2497921),
        ("STOC", "unresolved", "Strix occidentalis caurina", "Northern Spotted Owl", None),
        ("Rain", "non_taxonomic", None, None, None),
    ]
    assert set(table.column("recipe_fingerprint").to_pylist()) == {RECIPE_FINGERPRINT}


def test_resolved_taxa_are_the_rows_whose_kind_is_taxon():
    rows = [(0.0, "STVA", 0.9), (0.0, "STOC", 0.8), (0.0, "Rain", 0.7)]

    table = select(
        ThresholdPolicy(min_score=0.5), rows, a_registry(BARRED_OWL, UNRESOLVED_OWL, RAIN)
    )
    kinds = dict(selected(table, "label", "label_kind"))
    keys = dict(selected(table, "label", "gbif_taxon_key"))

    # An unresolved organism and a sound both have no key; only the kind tells them apart.
    assert [label for label, kind in kinds.items() if kind == "taxon"] == ["STVA"]
    assert keys["STOC"] is None and keys["Rain"] is None
    assert kinds["STOC"] != kinds["Rain"]


# --- Row order ------------------------------------------------------------------


def test_rows_are_ordered_by_window_start_then_rank():
    rows = [
        (6.0, "a", 0.2),
        (0.0, "b", 0.3),
        (3.0, "a", 0.9),
        (0.0, "a", 0.8),
        (6.0, "b", 0.7),
        (3.0, "b", 0.1),
    ]

    table = select(ThresholdPolicy(min_score=0.0), rows, sounds("a", "b"))

    assert selected(table, "window_start_s", "rank", "label") == [
        (0.0, 1, "a"),
        (0.0, 2, "b"),
        (3.0, 1, "a"),
        (3.0, 2, "b"),
        (6.0, 1, "b"),
        (6.0, 2, "a"),
    ]


def test_the_query_returns_the_declared_columns_but_the_id():
    table = select(ThresholdPolicy(min_score=0.0), [(0.0, "a", 0.5)], sounds("a"))

    assert table.column_names == [
        name for name in DETECTIONS_SCHEMA.names if name != "detection_id"
    ]


# ---------------------------------------------------------------------------
# The file: one recording's scores file in, its detections file out.
# ---------------------------------------------------------------------------

CARD = ModelCard(
    model_name="test-model",
    model_version="1",
    runtime="none",
    window_duration=3.0,
    sample_rate=16000,
    min_detection_threshold=0.0,
    window_overlap=0.0,
    score_domain="probability",
    taxa_registry_digest=REGISTRY_FINGERPRINT,
    audio=AudioGeometry(
        downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
    ),
    backend="none",
    dtype="float32",
)
RECIPE = Recipe(
    model=model_ref(CARD),
    backend="none",
    audio=AudioSpec(
        sample_rate=16000,
        window_duration=3.0,
        window_overlap=0.0,
        downmix="mean",
        resampler=RunnerResampled(algorithm="soxr_hq"),
        pad="drop",
    ),
    dtype="float32",
)
SCORES_REQUEST = ScoresRequest(contract_id="robin.scores.arrow/1", retention="full")
REGISTRY_URI = "s3://b/registry.csv"
OWLS = a_registry(BARRED_OWL, UNRESOLVED_OWL, RAIN)


def a_recording(namespace: str = "soundhub", value: str = "42") -> RecordingRef:
    return RecordingRef(namespace=namespace, value=value, audio_uri=f"s3://b/{value}.wav")


def a_work(*recordings: RecordingRef, policy=ThresholdPolicy(min_score=0.5)) -> InferenceWork:
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=recordings or (a_recording(),),
        model=PinnedModel(
            card=CARD,
            files={
                "weights": PinnedFile(
                    uri="s3://b/model.bin", digest="sha256:" + "0" * 64, size_bytes=8
                ),
                REGISTRY_ROLE: PinnedFile(
                    uri=REGISTRY_URI, digest=REGISTRY_FINGERPRINT, size_bytes=8
                ),
            },
        ),
        input=AudioInput(),
        settings={},
        resources={},
        outputs=(
            SCORES_REQUEST,
            DetectionsRequest(contract_id="robin.detections.parquet/1", policy=policy),
        ),
    )


def header(work, recording, contract_id):
    return required_metadata(
        contract_id=contract_id,
        work=work,
        recording=recording,
        recipe=RECIPE,
        registry_uri=REGISTRY_URI,
        registry_fingerprint=REGISTRY_FINGERPRINT,
    ) | score_metadata(SCORES_REQUEST, score_domain="probability")


def stage_scores(folder: Path, work, recording, windows) -> StagedArtifact:
    """`windows` is a list of `(start, {label: score})`; each window lasts three seconds."""
    folder.mkdir(parents=True, exist_ok=True)
    metadata = header(work, recording, "robin.scores.arrow/1")
    with ScoresWriter(folder / "scores.arrow", recording=recording, metadata=metadata) as writer:
        for start, scores in windows:
            writer.write(
                AcceptedWindow(
                    recording=recording,
                    start=start,
                    end=start + 3.0,
                    scores=tuple(ClassScore(label, score) for label, score in scores.items()),
                    embedding=None,
                )
            )
        return writer.close()


def detect(folder: Path, work, staged: StagedArtifact, policy=ThresholdPolicy(min_score=0.5)):
    return write_detections(
        staged,
        folder / "detections.parquet",
        work=work,
        recipe=RECIPE,
        policy=policy,
        registry=registry_table(OWLS),
        metadata=header(work, staged.recording, "robin.detections.parquet/1")
        | detection_metadata(
            policy, source_contract_id=staged.contract_id, source_checksum=staged.checksum
        ),
        temporary=folder,
    )


OWL_WINDOWS = [
    (0.0, {"STVA": 0.9, "STOC": 0.6, "Rain": 0.1}),
    (3.0, {"STVA": 0.2, "STOC": 0.7, "Rain": 0.8}),
]


def written(
    tmp_path, windows=OWL_WINDOWS, *, recording=None, policy=ThresholdPolicy(min_score=0.5)
):
    recording = recording or a_recording()
    work = a_work(recording)
    staged = stage_scores(tmp_path, work, recording, windows)
    return work, staged, detect(tmp_path, work, staged, policy)


def test_the_file_holds_the_declared_schema_and_the_header_it_was_given(tmp_path):
    work, scores, staged = written(tmp_path)

    stored = pq.read_schema(staged.path)
    header_keys = decode_metadata(stored.metadata)

    assert stored.remove_metadata().equals(DETECTIONS_SCHEMA, check_metadata=False)
    assert [field.nullable for field in stored] == [field.nullable for field in DETECTIONS_SCHEMA]
    expected = decode_metadata(
        header(work, scores.recording, "robin.detections.parquet/1")
        | detection_metadata(
            ThresholdPolicy(min_score=0.5),
            source_contract_id=scores.contract_id,
            source_checksum=scores.checksum,
        )
    )
    assert {key: header_keys[key] for key in expected} == expected
    assert header_keys["robin.contract"] == "robin.detections.parquet/1"


def test_the_staged_file_is_named_by_its_bytes_and_rows(tmp_path):
    _, scores, staged = written(tmp_path)

    assert (staged.kind, staged.contract_id) == ("detections", "robin.detections.parquet/1")
    assert staged.recording == scores.recording
    assert staged.checksum == checksum_file(staged.path)
    assert staged.rows == pq.read_metadata(staged.path).num_rows == 4
    rows = pq.read_table(staged.path).to_pylist()
    assert [(row["window_start_s"], row["label"], row["rank"]) for row in rows] == [
        (0.0, "STVA", 1),
        (0.0, "STOC", 2),
        (3.0, "Rain", 1),
        (3.0, "STOC", 2),
    ]


def test_a_detection_id_hashes_the_recording_its_window_and_its_rank(tmp_path):
    work, scores, staged = written(tmp_path)
    recording = scores.recording

    rows = pq.read_table(staged.path).to_pylist()

    assert [row["detection_id"] for row in rows] == [
        sha256_v1(
            {
                "recording_work_digest": recording_work_digest(work, recording),
                "recipe_fingerprint": RECIPE.id,
                "recording_namespace": recording.namespace,
                "recording_value": recording.value,
                "window_start_s": row["window_start_s"],
                "rank": row["rank"],
            }
        )
        for row in rows
    ]


def test_ids_keep_apart_what_joining_with_a_colon_would_merge(tmp_path):
    ids = []
    for namespace, value in (("a:b", "c"), ("a", "b:c")):
        recording = a_recording(namespace, value)
        _, _, staged = written(tmp_path / namespace, recording=recording)
        ids.append(pq.read_table(staged.path).column("detection_id").to_pylist())

    assert not set(ids[0]) & set(ids[1])


def test_ids_keep_apart_starts_less_than_a_millisecond_apart(tmp_path):
    windows = [(1.0, {"STVA": 0.9}), (1.0004, {"STVA": 0.9})]

    _, _, staged = written(tmp_path, windows)
    ids = pq.read_table(staged.path).column("detection_id").to_pylist()

    assert len(ids) == len(set(ids)) == 2


def test_row_groups_do_not_depend_on_how_duckdb_hands_over_rows(tmp_path, monkeypatch):
    windows = [(3.0 * n, {"STVA": 0.9, "STOC": 0.8}) for n in range(5)]
    monkeypatch.setattr(detections, "ROW_GROUP_ROWS", 3)

    monkeypatch.setattr(detections, "DUCKDB_BATCH_ROWS", 1)
    _, _, one_at_a_time = written(tmp_path / "one", windows)
    monkeypatch.undo()
    monkeypatch.setattr(detections, "ROW_GROUP_ROWS", 3)
    _, _, default = written(tmp_path / "default", windows)

    assert one_at_a_time.path.read_bytes() == default.path.read_bytes()
    assert [
        pq.ParquetFile(default.path).metadata.row_group(index).num_rows for index in range(4)
    ] == [3, 3, 3, 1]


def test_a_policy_selecting_nothing_leaves_no_file(tmp_path):
    _, _, staged = written(tmp_path, policy=ThresholdPolicy(min_score=0.99))

    assert staged is None
    assert not (tmp_path / "detections.parquet").exists()


def many_windows_then(last: dict[str, float]):
    """Three windows of declared labels, then `last`.

    With a small batch size they span several batches.
    """
    return [(3.0 * n, {"STVA": 0.9, "STOC": 0.8}) for n in range(3)] + [(9.0, last)]


def refused(call) -> errors.EngineError:
    with pytest.raises(errors.EngineError) as raised:
        call()
    return raised.value


def test_a_label_the_registry_lacks_fails_even_below_the_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(scores_module, "SCORE_BATCH_ROWS", 2)
    recording = a_recording()
    work = a_work(recording)
    staged = stage_scores(tmp_path, work, recording, many_windows_then({"HAWK": 0.01}))

    error = refused(lambda: detect(tmp_path, work, staged))

    assert (error.code, error.stage) == (errors.UNMATCHED_LABELS, errors.AGGREGATE)
    assert error.recording == recording
    assert "HAWK" in error.detail


def test_a_scores_file_changed_after_staging_fails_at_aggregation(tmp_path):
    recording = a_recording()
    work = a_work(recording)
    staged = stage_scores(tmp_path, work, recording, OWL_WINDOWS)
    data = bytearray(staged.path.read_bytes())
    data[-1] ^= 0xFF
    staged.path.write_bytes(bytes(data))

    error = refused(lambda: detect(tmp_path, work, staged))

    assert (error.code, error.stage) == (errors.ARTIFACT_CHECKSUM_MISMATCH, errors.AGGREGATE)
    assert error.recording == recording


def failing_after_one_batch(monkeypatch, failure: Exception) -> None:
    """The scores reader yields its first batch, then raises `failure`."""
    real = detections.read_scores

    @contextmanager
    def reading(path, *, expected_checksum):
        with real(path, expected_checksum=expected_checksum) as stream:

            def batches():
                yield next(stream.batches)
                raise failure

            yield ScoresStream(metadata=stream.metadata, batches=batches())

    monkeypatch.setattr(detections, "read_scores", reading)


def test_a_batch_the_reader_refuses_keeps_the_readers_code(tmp_path, monkeypatch):
    monkeypatch.setattr(scores_module, "SCORE_BATCH_ROWS", 2)
    recording = a_recording()
    work = a_work(recording)
    staged = stage_scores(tmp_path, work, recording, many_windows_then({"STVA": 0.9}))
    failing_after_one_batch(monkeypatch, malformed("scores.arrow has an unreadable batch"))

    error = refused(lambda: detect(tmp_path, work, staged))

    assert (error.code, error.stage) == (errors.ARTIFACT_MALFORMED, errors.AGGREGATE)
    assert error.recording == recording


def test_an_engine_defect_in_the_stream_is_not_reported_as_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(scores_module, "SCORE_BATCH_ROWS", 2)
    recording = a_recording()
    work = a_work(recording)
    staged = stage_scores(tmp_path, work, recording, many_windows_then({"STVA": 0.9}))
    defect = RuntimeError("the engine contradicted itself")
    failing_after_one_batch(monkeypatch, defect)

    with pytest.raises(RuntimeError) as raised:
        detect(tmp_path, work, staged)

    assert raised.value is defect


def test_spilled_rows_stay_inside_the_folder_it_is_given(tmp_path, monkeypatch):
    connected = []
    real = duckdb.connect

    def connect(*args, **kwargs):
        connected.append(kwargs.get("config", {}))
        return real(*args, **kwargs)

    monkeypatch.setattr(detections.duckdb, "connect", connect)
    written(tmp_path)

    (config,) = connected
    assert Path(config["temp_directory"]).parent == tmp_path
