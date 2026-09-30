"""Keeping the scores a request asks for, from a window the boundary accepted."""

from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.records import ClassScore
from robin_inference_engine.retention import retain_scores

CONTRACT = "robin.scores.arrow/1"


def scored(*pairs: tuple[str, float]) -> tuple[ClassScore, ...]:
    return tuple(ClassScore(label=label, score=score) for label, score in pairs)


WINDOW = scored(("owl", 0.2), ("rain", 0.7), ("wind", 0.5), ("hawk", 0.0))


def test_full_keeps_every_score_in_the_order_it_arrived():
    request = ScoresRequest(contract_id=CONTRACT, retention="full")

    assert retain_scores(WINDOW, request) == WINDOW


def test_thresholded_keeps_scores_at_or_above_the_floor_in_the_order_they_arrived():
    request = ScoresRequest(contract_id=CONTRACT, retention="thresholded", min_score=0.5)

    assert retain_scores(WINDOW, request) == scored(("rain", 0.7), ("wind", 0.5))


def test_top_k_applies_the_floor_then_keeps_the_highest_k():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.1, top_k=2)

    assert retain_scores(WINDOW, request) == scored(("rain", 0.7), ("wind", 0.5))


def test_top_k_orders_what_it_keeps_by_score_descending():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=4)

    assert retain_scores(WINDOW, request) == scored(
        ("rain", 0.7), ("wind", 0.5), ("owl", 0.2), ("hawk", 0.0)
    )


def test_a_tie_at_the_k_boundary_goes_to_the_label_first_by_code_point():
    window = scored(("Z", 0.9), ("b", 0.5), ("B", 0.5), ("a", 0.5))
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=2)

    assert retain_scores(window, request) == scored(("Z", 0.9), ("B", 0.5))


def test_top_k_keeps_fewer_than_k_when_the_floor_leaves_fewer():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.5, top_k=3)

    assert retain_scores(WINDOW, request) == scored(("rain", 0.7), ("wind", 0.5))


def test_top_k_of_a_window_with_fewer_than_k_scores_keeps_them_all():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=10)

    assert len(retain_scores(WINDOW, request)) == len(WINDOW)


def test_an_empty_window_stays_empty():
    request = ScoresRequest(contract_id=CONTRACT, retention="top_k", min_score=0.0, top_k=2)

    assert retain_scores((), request) == ()
