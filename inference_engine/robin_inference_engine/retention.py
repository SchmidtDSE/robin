"""Keeping the scores a request asks for from each accepted window."""

import numpy as np

from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.registry import TaxonRegistry


def get_label_rank(registry: TaxonRegistry) -> np.ndarray:
    """Return each registry position's place when the labels are sorted by code point.

    `top_k` breaks a tie between equal scores by this rank. It is computed once for
    each registry, not once for each window.
    """
    labels = np.array([entry.label for entry in registry.entries], dtype=object)
    rank = np.empty(len(labels), dtype=np.intp)
    # argsort gives the positions in sorted order. Its inverse gives each position's place.
    rank[np.argsort(labels, kind="stable")] = np.arange(len(labels), dtype=np.intp)
    return rank


def retain_scores(
    scores: np.ndarray,
    indices: np.ndarray,
    request: ScoresRequest,
    label_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scores of one window that `request` keeps, and their registry positions.

    `scores` and `indices` are the window's whole row and its positions. `full` keeps
    them as they are. `thresholded` keeps the scores at or above the floor, in registry
    order. `top_k` applies the floor, then keeps the first k by score descending, ties
    going to the label first by code point, the order detections rank in.
    """
    if request.retention == "full":
        return scores, indices
    kept = scores >= request.min_score
    if request.retention == "thresholded":
        return scores[kept], indices[kept]
    candidates = np.flatnonzero(kept)
    k = request.top_k
    if len(candidates) > k:
        # Every candidate at or above the k-th largest value, so a tie at the k-th
        # place keeps all its members until the label decides.
        values = scores[candidates]
        kth = np.partition(values, len(values) - k)[len(values) - k]
        candidates = candidates[values >= kth]
    values = scores[candidates]
    # lexsort sorts by the last key first: score descending, then label rank.
    order = np.lexsort((label_rank[indices[candidates]], -values))[:k]
    chosen = candidates[order]
    return scores[chosen], indices[chosen]
