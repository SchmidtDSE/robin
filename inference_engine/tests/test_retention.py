"""Keeping the scores a request asks for, from a window the boundary accepted."""

import numpy as np
import pytest

from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_inference_engine.retention import get_label_rank, retain_scores

CONTRACT = "robin.scores.parquet/1"

LABELS = ("owl", "rain", "wind", "hawk")
WINDOW = (0.2, 0.7, 0.5, 0.0)


def build_registry(labels) -> TaxonRegistry:
    entries = tuple(
        RegistryEntry(class_index=index, label=label, label_kind="non_taxonomic")
        for index, label in enumerate(labels)
    )
    return TaxonRegistry(fingerprint="sha256:" + "0" * 64, entries=entries)


def keep(request: ScoresRequest, values=WINDOW, labels=LABELS) -> list[tuple[str, float]]:
    """Retain one window's row, and return what was kept as (label, score) pairs."""
    scores = np.array(values, dtype=np.float64)
    indices = np.arange(len(scores), dtype=np.intp)
    rank = get_label_rank(build_registry(labels))
    kept, positions = retain_scores(scores, indices, request, rank)
    assert kept.dtype == np.float64 and positions.dtype == np.intp
    return [(labels[position], float(score)) for position, score in zip(positions, kept, strict=True)]


def test_full_keeps_the_row_and_its_positions_as_they_are():
    request = ScoresRequest(contract_id=CONTRACT, retention="full")
    scores = np.array(WINDOW)
    indices = np.arange(len(WINDOW), dtype=np.intp)

    kept, positions = retain_scores(scores, indices, request, get_label_rank(build_registry(LABELS)))

    assert kept is scores and positions is indices


def test_thresholded_keeps_scores_at_or_above_the_floor_in_registry_order():
    request = ScoresRequest(contract_id=CONTRACT, retention="thresholded", min_score=0.5)

    assert keep(request) == [("rain", 0.7), ("wind", 0.5)]


def test_top_k_applies_the_floor_then_keeps_the_highest_k():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.1, top_k=2)

    assert keep(request) == [("rain", 0.7), ("wind", 0.5)]


def test_top_k_orders_what_it_keeps_by_score_descending():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=4)

    assert keep(request) == [("rain", 0.7), ("wind", 0.5), ("owl", 0.2), ("hawk", 0.0)]


def test_a_tie_at_the_k_boundary_goes_to_the_label_first_by_code_point():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=2)

    kept = keep(request, values=(0.9, 0.5, 0.5, 0.5), labels=("Z", "b", "B", "a"))

    assert kept == [("Z", 0.9), ("B", 0.5)]


def test_top_k_keeps_fewer_than_k_when_the_floor_leaves_fewer():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.5, top_k=3)

    assert keep(request) == [("rain", 0.7), ("wind", 0.5)]


def test_top_k_of_a_window_with_fewer_than_k_scores_keeps_them_all():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=10)

    assert len(keep(request)) == len(WINDOW)


def test_an_empty_window_stays_empty():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=2)

    assert keep(request, values=(), labels=()) == []


def test_the_label_rank_is_each_positions_place_in_code_point_order():
    rank = get_label_rank(build_registry(("b", "c", "a")))

    assert rank.tolist() == [1, 2, 0]


@pytest.mark.parametrize("k", [1, 2, 5, 9, 30])
def test_top_k_matches_a_full_sort_for_ties_at_the_kth_place(k):
    # The labels are not in code-point order, so a rank that is not inverted fails.
    rng = np.random.default_rng(k)
    labels = tuple(rng.permutation([f"{chr(97 + n % 26)}{n}" for n in range(30)]).tolist())
    rank = get_label_rank(build_registry(labels))
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.25, top_k=k)
    for _ in range(200):
        # Five values only, so most rows have ties at the k-th place.
        scores = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=len(labels))
        candidates = [(-score, labels[n], n) for n, score in enumerate(scores) if score >= 0.25]
        expected = [n for _, _, n in sorted(candidates)[:k]]

        kept, positions = retain_scores(
            scores, np.arange(len(labels), dtype=np.intp), request, rank
        )

        assert positions.tolist() == expected
        assert kept.tolist() == scores[expected].tolist()
