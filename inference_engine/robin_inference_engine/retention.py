"""Keeping the scores a request asks for from each accepted window."""

from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.records import ClassScore


def retain_scores(
    scores: tuple[ClassScore, ...], request: ScoresRequest
) -> tuple[ClassScore, ...]:
    """The window's scores that `request` keeps.

    `full` keeps every score and `thresholded` those at or above the floor, both in the
    order the model produced them. `top_k` applies the floor, then keeps the first k by
    score descending, ties going to the label first by code point, the order detections
    rank in.
    """
    if request.retention == "full":
        return scores
    kept = [score for score in scores if score.score >= request.min_score]
    if request.retention == "thresholded":
        return tuple(kept)
    kept.sort(key=lambda item: (-item.score, item.label))
    return tuple(kept[: request.top_k])
