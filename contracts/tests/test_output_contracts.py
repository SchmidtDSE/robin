"""Requested outputs, detection policies, and the output contract identifier table."""

import pathlib
from typing import get_args

import pytest
from pydantic import ValidationError

from robin_contracts.output_contracts import (
    DetectionsContractId,
    EmbeddingsContractId,
    EmbeddingsRequest,
    ResultContractId,
    ScoresContractId,
    ScoresRequest,
    ThresholdPolicy,
    TopKPolicy,
    WorkContractId,
)

SCORES = get_args(ScoresContractId)[0]
EMBEDDINGS = get_args(EmbeddingsContractId)[0]

RETIRED_IDENTIFIERS = (
    "window-embedding/1",
    "inference-execution-spec/1",
    "inference-request/1",
    "inference-manifest/1",
)


def build_scores(**overrides) -> ScoresRequest:
    fields = {"contract_id": SCORES, "retention": "thresholded", "min_score": 0.1}
    return ScoresRequest(**(fields | overrides))


def test_each_alias_spells_its_specified_identifier():
    # Transcribed, not derived from the aliases under test, so a changed alias fails.
    assert get_args(ScoresContractId)[0] == "robin.scores.parquet/1"
    assert get_args(EmbeddingsContractId)[0] == "robin.embeddings.parquet/1"
    assert get_args(DetectionsContractId)[0] == "robin.detections.parquet/1"
    assert get_args(WorkContractId)[0] == "robin.inference-work/1"
    assert get_args(ResultContractId)[0] == "robin.inference-result/1"


def test_a_request_refuses_a_foreign_contract_id():
    with pytest.raises(ValidationError) as exc:
        build_scores(contract_id=EMBEDDINGS)

    assert exc.value.errors()[0]["loc"] == ("contract_id",)


def test_threshold_policy_requires_a_min_score():
    with pytest.raises(ValidationError) as exc:
        ThresholdPolicy()

    assert exc.value.errors()[0]["loc"] == ("min_score",)

    assert ThresholdPolicy(min_score=0.25).min_score == 0.25


def test_top_k_policy_floor_is_optional():
    assert TopKPolicy(k=5).min_score is None
    assert TopKPolicy(k=5, min_score=0.1).min_score == 0.1


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_policy_refuses_a_non_positive_k(k):
    with pytest.raises(ValidationError) as exc:
        TopKPolicy(k=k)

    assert exc.value.errors()[0]["loc"] == ("k",)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_min_score_is_refused(value):
    with pytest.raises(ValidationError):
        ThresholdPolicy(min_score=value)

    with pytest.raises(ValidationError):
        TopKPolicy(k=5, min_score=value)

    with pytest.raises(ValidationError):
        build_scores(min_score=value)


@pytest.mark.parametrize("retention", ["thresholded", "top_k"])
def test_reduced_retention_requires_a_floor(retention):
    extra = {"top_k": 5} if retention == "top_k" else {}

    with pytest.raises(ValidationError):
        build_scores(retention=retention, min_score=None, **extra)

    assert build_scores(retention=retention, min_score=0.1, **extra).min_score == 0.1


@pytest.mark.parametrize("top_k", [None, 0, -1])
def test_top_k_retention_requires_a_positive_top_k(top_k):
    with pytest.raises(ValidationError):
        build_scores(retention="top_k", top_k=top_k)

    assert build_scores(retention="top_k", top_k=1).top_k == 1
    assert build_scores(retention="top_k", top_k=5).top_k == 5

    # The "iff": a top_k on any other retention is a request that means two things.
    with pytest.raises(ValidationError):
        build_scores(retention="thresholded", top_k=5)


def test_full_retention_refuses_a_floor_or_a_top_k():
    assert build_scores(retention="full", min_score=None).min_score is None

    with pytest.raises(ValidationError):
        build_scores(retention="full", min_score=0.1)

    with pytest.raises(ValidationError):
        build_scores(retention="full", min_score=None, top_k=5)


def test_an_embeddings_request_may_leave_its_storage_width_unnamed():
    assert EmbeddingsRequest(contract_id=EMBEDDINGS).storage_dtype is None


@pytest.mark.parametrize("width", ["float32", "float16"])
def test_an_embeddings_request_may_name_either_storage_width(width):
    assert EmbeddingsRequest(contract_id=EMBEDDINGS, storage_dtype=width).storage_dtype == width


def test_no_retired_identifier_appears_in_the_source_tree():
    root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "pixi.toml").exists():
        pytest.skip("no checkout to walk; the contracts wheel excludes the test tree")

    skipped_directories = {".git", ".pixi", ".wheelcheck", "_NOTES"}
    offenders = []
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".toml", ".md"} or not path.is_file():
            continue
        if skipped_directories & set(path.relative_to(root).parts):
            continue
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        offenders += [
            (path.relative_to(root).as_posix(), identifier)
            for identifier in RETIRED_IDENTIFIERS
            if identifier in text
        ]

    assert offenders == []
