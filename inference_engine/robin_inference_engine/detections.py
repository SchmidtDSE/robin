"""One recording's detections: the scores a policy selects, ranked, and named by the registry."""

from collections.abc import Iterator, Mapping
from dataclasses import fields
from pathlib import Path
from typing import get_args

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from robin_contracts.canonical import sha256_v1
from robin_contracts.output_contracts import DetectionPolicy, DetectionsContractId, TopKPolicy
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.specs import Recipe
from robin_contracts.work import InferenceWork, RecordingRef, recording_work_digest
from robin_inference_engine import errors
from robin_inference_engine.artifacts.scores import SCORES_SCHEMA, read_scores
from robin_inference_engine.artifacts.staging import StagedArtifact, checksum_file

CONTRACT_ID: DetectionsContractId = get_args(DetectionsContractId)[0]

# Rows per Parquet row group. Groups are cut by count, so the file's bytes do not depend
# on how many rows DuckDB hands over at a time. Changing this changes every detections
# file's bytes, and the writer then refuses a replay into a root written before.
ROW_GROUP_ROWS = 2**16

# Rows asked of DuckDB per Arrow batch, which bounds the rows held in Python while writing.
DUCKDB_BATCH_ROWS = 2**16

DETECTION_ID = "detection_id"

DETECTIONS_SCHEMA = pa.schema(
    [
        pa.field(DETECTION_ID, pa.string(), nullable=False),
        pa.field("window_start_s", pa.float64(), nullable=False),
        pa.field("window_end_s", pa.float64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("label_kind", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False),
        pa.field("scientific_name", pa.string(), nullable=True),
        pa.field("common_name", pa.string(), nullable=True),
        pa.field("gbif_taxon_key", pa.int64(), nullable=True),
        pa.field("recipe_fingerprint", pa.string(), nullable=False),
    ]
)

_REGISTRY_ENTRY_FIELDS = frozenset(field.name for field in fields(RegistryEntry))

# The detections columns a registry entry supplies, typed as the detections file declares
# them, so joined columns come out with the declared types.
REGISTRY_SCHEMA = pa.schema(
    [field for field in DETECTIONS_SCHEMA if field.name in _REGISTRY_ENTRY_FIELDS]
)


def registry_table(registry: TaxonRegistry) -> pa.Table:
    """The registry as the relation the detections query joins to."""
    return pa.Table.from_pylist(
        [
            {name: getattr(entry, name) for name in REGISTRY_SCHEMA.names}
            for entry in registry.entries
        ],
        schema=REGISTRY_SCHEMA,
    )


def build_detections_sql(
    policy: DetectionPolicy, *, recipe_fingerprint: str
) -> tuple[str, dict[str, object]]:
    """The detections query over the relations `scores` and `registry`, and its parameters.

    It returns every detections column but the id, ordered by window start and rank.
    """
    # Values are bound, never formatted in: DuckDB would read a literal 0.3 as a DECIMAL
    # and compare against the rounded value instead of the stored double.
    parameters: dict[str, object] = {"recipe_fingerprint": recipe_fingerprint}
    floor = ""
    if policy.min_score is not None:
        floor = "WHERE score >= $min_score"
        parameters["min_score"] = policy.min_score
    top_k = ""
    if isinstance(policy, TopKPolicy):
        top_k = "WHERE rank <= $k"
        parameters["k"] = policy.k
    columns = ", ".join(
        f'"{name}"' for name in DETECTIONS_SCHEMA.names if name != DETECTION_ID
    )
    # Ranks restart in each window. The file holds one recording, so grouping rows by start
    # time groups them by window. Equal scores rank by label, so ranks never depend on row order.
    sql = f"""
        WITH kept AS (
            SELECT window_start_s, window_end_s, label, score
            FROM scores
            {floor}
        ),
        ranked AS (
            SELECT
                *,
                CAST(
                    ROW_NUMBER() OVER (
                        PARTITION BY window_start_s ORDER BY score DESC, label ASC
                    ) AS INTEGER
                ) AS rank
            FROM kept
        ),
        selected AS (
            SELECT *, $recipe_fingerprint AS recipe_fingerprint
            FROM ranked
            {top_k}
        )
        SELECT {columns}
        FROM selected
        INNER JOIN registry USING (label)
        ORDER BY window_start_s, rank
    """
    return sql, parameters


def write_detections(
    scores: StagedArtifact,
    path: Path,
    *,
    work: InferenceWork,
    recipe: Recipe,
    policy: DetectionPolicy,
    registry: pa.Table,
    metadata: Mapping[bytes, bytes],
    temporary: Path,
) -> StagedArtifact | None:
    """Select the detections in one recording's staged scores file and write them to `path`.

    `registry` is the table `registry_table` returns, and `metadata` the file's whole header.
    DuckDB spills under `temporary`. Returns None, leaving no file, when nothing is
    selected. An expected failure is an `EngineError` at the aggregate stage naming the
    recording.
    """
    recording = scores.recording
    try:
        rows = _select_and_write(
            scores,
            path,
            policy=policy,
            registry=registry,
            metadata=metadata,
            temporary=temporary,
            identity=_recording_identity(work, recording, recipe),
            recipe_fingerprint=recipe.id,
        )
    except errors.EngineError as error:
        raise _at_aggregate(error, recording) from error
    if not rows:
        path.unlink()
        return None
    return StagedArtifact(
        kind="detections",
        contract_id=CONTRACT_ID,
        recording=recording,
        path=path,
        checksum=checksum_file(path),
        rows=rows,
    )


def _select_and_write(
    scores: StagedArtifact,
    path: Path,
    *,
    policy: DetectionPolicy,
    registry: pa.Table,
    metadata: Mapping[bytes, bytes],
    temporary: Path,
    identity: Mapping[str, str],
    recipe_fingerprint: str,
) -> int:
    """Run the query over the verified scores and write what it selects; return the row count."""
    sql, parameters = build_detections_sql(policy, recipe_fingerprint=recipe_fingerprint)
    with read_scores(scores.path, expected_checksum=scores.checksum) as stream:
        source = _CheckedScores(stream.batches, registry.column("label"), scores.recording)
        config = {"temp_directory": str(temporary / "duckdb")}
        with duckdb.connect(config=config) as connection:
            connection.register("registry", registry)
            connection.register("scores", source.reader())
            try:
                selected = connection.execute(sql, parameters).to_arrow_reader(
                    DUCKDB_BATCH_ROWS
                )
                with_ids = (_with_ids(batch, identity) for batch in selected)
                return _write_parquet(path, with_ids, metadata)
            except duckdb.Error as exc:
                source.raise_recorded()
                raise errors.EngineError(
                    errors.AGGREGATION_FAILED,
                    errors.AGGREGATE,
                    f"selecting the detections of recording {errors.named(scores.recording)} "
                    f"raised {type(exc).__name__}: {exc}",
                    recording=scores.recording,
                ) from exc


class _CheckedScores:
    """The verified score batches DuckDB reads, with every label checked against the registry.

    DuckDB reports an exception raised inside this stream as its own error, keeping only
    the message. The first one is kept here so the caller can raise it as it was.
    """

    def __init__(
        self, batches: Iterator[pa.RecordBatch], labels: pa.ChunkedArray, recording: RecordingRef
    ) -> None:
        self._batches = batches
        self._labels = labels.combine_chunks()
        self._recording = recording
        self._raised: BaseException | None = None

    def reader(self) -> pa.RecordBatchReader:
        return pa.RecordBatchReader.from_batches(SCORES_SCHEMA, self._checked())

    def raise_recorded(self) -> None:
        """Raise the exception the stream raised, if it raised one."""
        if self._raised is not None:
            raise self._raised from None

    def _checked(self) -> Iterator[pa.RecordBatch]:
        try:
            for batch in self._batches:
                columns = batch.select(SCORES_SCHEMA.names)
                self._require_declared_labels(columns.column("label"))
                yield pa.RecordBatch.from_arrays(columns.columns, schema=SCORES_SCHEMA)
        except GeneratorExit:
            # Raised when the stream is closed before its end; that is not a failure to record.
            raise
        except BaseException as exc:
            self._raised = exc
            raise

    def _require_declared_labels(self, labels: pa.Array) -> None:
        # Every label, not only those selected: a label the registry no longer declares
        # means the registry changed after the scores were written, whatever it scored.
        unmatched = pc.filter(labels, pc.invert(pc.is_in(labels, value_set=self._labels)))
        if len(unmatched):
            raise errors.EngineError(
                errors.UNMATCHED_LABELS,
                errors.AGGREGATE,
                f"recording {errors.named(self._recording)} scored label "
                f"{unmatched[0].as_py()!r}, which the registry does not declare",
                recording=self._recording,
            )


def _recording_identity(
    work: InferenceWork, recording: RecordingRef, recipe: Recipe
) -> dict[str, str]:
    """The part of each `detection_id` that is the same for every row of one recording."""
    return {
        "recording_work_digest": recording_work_digest(work, recording),
        "recipe_fingerprint": recipe.id,
        "recording_namespace": recording.namespace,
        "recording_value": recording.value,
    }


def _with_ids(batch: pa.RecordBatch, identity: Mapping[str, str]) -> pa.RecordBatch:
    """`batch` with its `detection_id` column added in front."""
    # Hashed in Python, because canonical JSON writes each float as Python's `repr` does.
    ids = [
        sha256_v1({**identity, "window_start_s": start, "rank": rank})
        for start, rank in zip(
            batch.column("window_start_s").to_pylist(), batch.column("rank").to_pylist()
        )
    ]
    return pa.RecordBatch.from_arrays(
        [pa.array(ids, pa.string()), *batch.columns], schema=DETECTIONS_SCHEMA
    )


def _write_parquet(
    path: Path, batches: Iterator[pa.RecordBatch], metadata: Mapping[bytes, bytes]
) -> int:
    """Write `batches` in row groups of `ROW_GROUP_ROWS`, and return the rows written."""
    schema = DETECTIONS_SCHEMA.with_metadata(dict(metadata))
    written = 0
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    with pq.ParquetWriter(path, schema) as writer:
        for batch in batches:
            pending.append(batch)
            pending_rows += batch.num_rows
            while pending_rows >= ROW_GROUP_ROWS:
                table = pa.Table.from_batches(pending, schema=DETECTIONS_SCHEMA)
                writer.write_table(table.slice(0, ROW_GROUP_ROWS), row_group_size=ROW_GROUP_ROWS)
                written += ROW_GROUP_ROWS
                pending = table.slice(ROW_GROUP_ROWS).to_batches()
                pending_rows -= ROW_GROUP_ROWS
        if pending_rows:
            writer.write_table(
                pa.Table.from_batches(pending, schema=DETECTIONS_SCHEMA),
                row_group_size=ROW_GROUP_ROWS,
            )
            written += pending_rows
    return written


def _at_aggregate(error: errors.EngineError, recording: RecordingRef) -> errors.EngineError:
    """The same failure, reported at the aggregate stage with the same code."""
    return errors.EngineError(error.code, errors.AGGREGATE, error.detail, recording=recording)
