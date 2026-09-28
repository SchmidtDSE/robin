"""Find each kind of requested output in a work."""

from robin_contracts.output_contracts import DetectionsRequest, EmbeddingsRequest, ScoresRequest
from robin_contracts.work import InferenceWork


def scores_request(work: InferenceWork) -> ScoresRequest | None:
    """The work's scores request, or None if it asks for no scores."""
    return next((one for one in work.outputs if isinstance(one, ScoresRequest)), None)


def embeddings_request(work: InferenceWork) -> EmbeddingsRequest | None:
    """The work's embeddings request, or None if it asks for no embeddings."""
    return next((one for one in work.outputs if isinstance(one, EmbeddingsRequest)), None)


def detections_request(work: InferenceWork) -> DetectionsRequest | None:
    """The work's detections request, or None if it asks for no detections."""
    return next((one for one in work.outputs if isinstance(one, DetectionsRequest)), None)
