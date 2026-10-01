"""OWL v4 run end to end through `run_work` on a real recording, against the retired
SoundHub model runner's scores for the same audio.

The weights and the recording are not in the repository. Each is named by an
environment variable, and the tests skip when either is missing.
"""

import hashlib
import os
import shutil
from collections import defaultdict
from importlib.resources import as_file, files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("sox_tensorflow")

from robin_adapters.artifact_writer.local import LocalArtifactWriter
from robin_adapters.file_provider.local import LocalFileProvider
from robin_contracts.cards import read_card
from robin_contracts.layout import artifact_path
from robin_contracts.output_contracts import DetectionsRequest, ScoresRequest, ThresholdPolicy
from robin_contracts.results import InferenceSuccess
from robin_contracts.specs import recipe
from robin_contracts.work import AudioInput, InferenceWork, PinnedFile, PinnedModel, RecordingRef
from robin_inference_engine.artifacts.scores import read_scores
from robin_inference_engine.detections import DETECTIONS_SCHEMA
from robin_inference_engine.engine import run_work
from robin_inference_engine.load_registry import load_registry

WEIGHTS_VARIABLE = "ROBIN_OWL_V4_WEIGHTS"
AUDIO_VARIABLE = "ROBIN_OWL_BASELINE_AUDIO"

WEIGHTS_SHA256 = "b35e445fa294e90c55330a4efbc230fe6ae2ed3f1c7a3d804db6c86fba28eaa5"
WEIGHTS_SIZE = 11_144_960
AUDIO_SHA256 = "94d3acd8e3562ffaade43e3c0df4bc7a7241ff8ceadcb7d42409af14e8f651bd"
REGISTRY_SHA256 = "bb172a45a873cada2a0bd5a2846f12053b1cf249eb58eab5eacdbb0efa6b19ed"

EXPECTED_SCORES = Path(__file__).with_name("owl_v4_expected_scores.arrow")

# The expected scores come from a separate environment with other TensorFlow, soxr and
# NumPy versions. Its spectrograms differ from robin's by one grey level in a few
# pixels, which moves scores by up to about 2.3e-4.
SCORE_TOLERANCE = 1e-3

BATCH_SIZE = 64
WINDOW_S = 12.0
WINDOWS = 75
LABELS = 51
NON_TAXONOMIC = {"DOG", "DRUM", "FLY", "FROG", "HOSA", "SHOT", "YARD"}
GENERA = {"POEC", "SITT", "SPRU", "TAMI", "WHIS"}
NAMESPACE, VALUE = "baseline", "owl-v4-898s"

RESOURCES = files("robin_models.owl") / "resources"
with as_file(RESOURCES / "card.yaml") as _path:
    CARD = read_card(_path)
with as_file(RESOURCES / "taxa_registry.csv") as _path:
    REGISTRY = load_registry(_path)


def sha256(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned_input(variable: str, sha: str) -> Path:
    """The file `variable` names: skip if it is unset or absent, fail if it is another file."""
    named = os.environ.get(variable)
    if not named:
        pytest.skip(f"{variable} is not set")
    path = Path(named).resolve()
    if not path.is_file():
        pytest.skip(f"{variable} names {path}, which does not exist")
    actual = sha256(path)
    if actual != sha:
        pytest.fail(f"{path} has sha256 {actual}, but the pinned sha256 is {sha}")
    return path


def owl_work(weights: Path, registry: Path, audio: Path, outputs: tuple) -> InferenceWork:
    return InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=(
            RecordingRef(
                namespace=NAMESPACE, value=VALUE, audio_uri=str(audio), duration_seconds=898.0
            ),
        ),
        model=PinnedModel(
            card=CARD,
            files={
                "weights": PinnedFile(
                    uri=str(weights), digest=f"sha256:{WEIGHTS_SHA256}", size_bytes=WEIGHTS_SIZE
                ),
                "taxa_registry": PinnedFile(
                    uri=str(registry),
                    digest=f"sha256:{REGISTRY_SHA256}",
                    size_bytes=registry.stat().st_size,
                ),
            },
        ),
        input=AudioInput(),
        settings={},
        resources={"batch_size": BATCH_SIZE},
        outputs=outputs,
    )


class Run:
    """One work's result, and its artifacts read back from where the writer published them."""

    def __init__(self, work: InferenceWork, root: Path) -> None:
        self.root = root
        self.result = run_work(
            work,
            model_files=LocalFileProvider(),
            audio=LocalFileProvider(),
            artifacts=LocalArtifactWriter(root),
        )

    def record(self, kind: str):
        (record,) = [record for record in self.result.artifacts if record.kind == kind]
        return record

    def path(self, kind: str) -> Path:
        return self.root / artifact_path(kind, NAMESPACE, VALUE)

    def scores(self) -> tuple[dict[str, str], pa.Table]:
        with read_scores(
            self.path("scores"), expected_checksum=self.record("scores").checksum
        ) as stream:
            return dict(stream.metadata), pa.Table.from_batches(list(stream.batches))

    def detections(self) -> pa.Table:
        # Without partitioning=None, the path's segments come back as extra columns.
        return pq.read_table(self.path("detections"), partitioning=None)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Run]:
    weights = pinned_input(WEIGHTS_VARIABLE, WEIGHTS_SHA256)
    if weights.stat().st_size != WEIGHTS_SIZE:
        pytest.fail(f"{weights} is {weights.stat().st_size} bytes, not {WEIGHTS_SIZE}")
    audio = pinned_input(AUDIO_VARIABLE, AUDIO_SHA256)
    registry = tmp_path_factory.mktemp("registry") / "taxa_registry.csv"
    with as_file(RESOURCES / "taxa_registry.csv") as packaged:
        shutil.copyfile(packaged, registry)

    full = owl_work(
        weights,
        registry,
        audio,
        (
            ScoresRequest(contract_id="robin.scores.arrow/1", retention="full"),
            DetectionsRequest(
                contract_id="robin.detections.parquet/1", policy=ThresholdPolicy(min_score=0.0)
            ),
        ),
    )
    top_five = owl_work(
        weights,
        registry,
        audio,
        (
            ScoresRequest(
                contract_id="robin.scores.arrow/1", retention="top_k", min_score=0.0, top_k=5
            ),
        ),
    )
    return {
        "full": Run(full, tmp_path_factory.mktemp("full")),
        "top_five": Run(top_five, tmp_path_factory.mktemp("top_five")),
    }


def by_window(table: pa.Table) -> dict[tuple[float, float], list[dict]]:
    windows = defaultdict(list)
    for row in table.to_pylist():
        windows[(row["window_start_s"], row["window_end_s"])].append(row)
    return dict(windows)


def ranked(rows: list[dict]) -> list[dict]:
    """Rows in the order scores are ranked: score descending, then label ascending."""
    return sorted(rows, key=lambda row: (-row["score"], row["label"]))


def keyed_scores(table: pa.Table) -> dict[tuple[float, float, str], float]:
    keyed = {}
    for row in table.to_pylist():
        key = (row["window_start_s"], row["window_end_s"], row["label"])
        assert key not in keyed, f"{key} appears twice"
        keyed[key] = row["score"]
    return keyed


def string_cells(table: pa.Table) -> list[str]:
    return [
        cell
        for column, field in zip(table.columns, table.schema)
        if pa.types.is_string(field.type)
        for cell in column.to_pylist()
        if cell is not None
    ]


# The full run's result.


def test_the_full_run_succeeds_with_one_scores_and_one_detections_artifact(runs):
    result = runs["full"].result
    assert isinstance(result, InferenceSuccess), result
    assert sorted(record.kind for record in result.artifacts) == ["detections", "scores"]


def test_the_full_runs_coverage_accounts_for_all_75_windows(runs):
    result = runs["full"].result
    (coverage,) = result.coverage
    assert (coverage.namespace, coverage.value) == (NAMESPACE, VALUE)
    assert coverage.windows_completed == WINDOWS
    assert coverage.score_rows == WINDOWS * LABELS
    assert coverage.detection_rows == WINDOWS * LABELS
    assert coverage.embedding_rows == 0
    assert coverage.first_window_start_s == 0.0
    assert coverage.last_window_end_s == 900.0
    assert coverage.zero_window_reason is None
    assert runs["full"].record("scores").rows == coverage.score_rows
    assert runs["full"].record("detections").rows == coverage.detection_rows


def test_the_result_states_the_cards_window_geometry(runs):
    assert runs["full"].result.window_geometry == recipe(CARD).audio.geometry


# The full run's scores.


def test_the_full_scores_header_says_every_score_was_kept(runs):
    metadata, _ = runs["full"].scores()
    assert metadata["robin.score_retention"] == "full"
    assert metadata["robin.score_domain"] == "probability"
    assert "robin.score_floor" not in metadata
    assert "robin.score_top_k" not in metadata


def test_every_window_starts_on_a_multiple_of_12_s_and_scores_every_label_once(runs):
    _, table = runs["full"].scores()
    windows = by_window(table)
    assert list(windows) == [(i * WINDOW_S, i * WINDOW_S + WINDOW_S) for i in range(WINDOWS)]
    assert list(windows)[-1] == (888.0, 900.0)
    registry_labels = sorted(entry.label for entry in REGISTRY.entries)
    for rows in windows.values():
        assert sorted(row["label"] for row in rows) == registry_labels


def test_the_scores_match_the_retired_runners_within_the_tolerance(runs):
    expected = keyed_scores(pa.ipc.open_file(EXPECTED_SCORES).read_all())
    _, table = runs["full"].scores()
    actual = keyed_scores(table)
    assert actual.keys() == expected.keys()
    differences = {key: abs(actual[key] - expected[key]) for key in expected}
    worst = max(differences, key=differences.get)
    assert differences[worst] <= SCORE_TOLERANCE, (
        f"largest difference {differences[worst]:.3g} at {worst}"
    )


# The full run's detections.


def test_threshold_zero_detects_every_score_in_the_declared_schema(runs):
    table = runs["full"].detections()
    assert table.num_rows == WINDOWS * LABELS
    assert table.schema.equals(DETECTIONS_SCHEMA)


def test_ranks_follow_score_descending_then_label_ascending(runs):
    for rows in by_window(runs["full"].detections()).values():
        by_rank = sorted(rows, key=lambda row: row["rank"])
        assert [row["rank"] for row in by_rank] == list(range(1, LABELS + 1))
        assert by_rank == ranked(rows)


def test_non_taxonomic_detections_carry_no_scientific_name_and_no_key(runs):
    rows = [
        row for row in runs["full"].detections().to_pylist() if row["label"] in NON_TAXONOMIC
    ]
    assert len(rows) == len(NON_TAXONOMIC) * WINDOWS
    for row in rows:
        assert row["label_kind"] == "non_taxonomic"
        assert row["scientific_name"] is None
        assert row["gbif_taxon_key"] is None
        assert row["common_name"] == REGISTRY.by_label(row["label"]).common_name


def test_taxon_detections_carry_a_scientific_name_and_a_key_including_genera(runs):
    rows = [
        row for row in runs["full"].detections().to_pylist() if row["label_kind"] == "taxon"
    ]
    assert len(rows) == (LABELS - len(NON_TAXONOMIC)) * WINDOWS
    assert GENERA <= {row["label"] for row in rows}
    for row in rows:
        assert row["scientific_name"]
        assert row["gbif_taxon_key"] is not None


def test_no_name_is_substituted_in_the_scores_or_the_detections(runs):
    _, scores = runs["full"].scores()
    for cell in string_cells(scores) + string_cells(runs["full"].detections()):
        assert "unknown" not in cell.lower(), cell


# The top-five run.


def test_the_top_five_run_succeeds_with_five_scores_per_window_and_no_detections(runs):
    result = runs["top_five"].result
    assert isinstance(result, InferenceSuccess), result
    (coverage,) = result.coverage
    assert coverage.score_rows == WINDOWS * 5
    assert coverage.detection_rows == 0


def test_the_top_five_scores_header_says_what_was_kept(runs):
    metadata, _ = runs["top_five"].scores()
    assert metadata["robin.score_retention"] == "top_k"
    assert metadata["robin.score_top_k"] == "5"
    assert metadata["robin.score_floor"] == "0.0"


def test_each_windows_top_five_are_the_first_five_of_its_full_scores(runs):
    _, full = runs["full"].scores()
    _, top_five = runs["top_five"].scores()
    full_windows = by_window(full)
    top_windows = by_window(top_five)
    assert list(top_windows) == list(full_windows)
    for window, rows in top_windows.items():
        assert ranked(rows) == ranked(full_windows[window])[:5]
