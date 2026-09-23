"""What a caller asks for, and the one place an output contract identifier is spelled."""

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

from robin_contracts.protocols import ScoreRetention

ScoresContractId = Literal["robin.scores.arrow/1"]
EmbeddingsContractId = Literal["robin.embeddings.arrow/1"]
RecordingMapContractId = Literal["robin.recording-map.json/1"]
DetectionsContractId = Literal["robin.detections.parquet/1"]
WorkContractId = Literal["robin.inference-work/1"]
ResultContractId = Literal["robin.inference-result/1"]


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("a score floor must be finite")
    return value


_FiniteFloor = Annotated[float, AfterValidator(_finite)]


class ThresholdPolicy(BaseModel, frozen=True, extra="forbid"):
    """Report every score at or above an explicitly chosen floor."""

    kind: Literal["threshold"] = "threshold"
    min_score: _FiniteFloor


class TopKPolicy(BaseModel, frozen=True, extra="forbid"):
    """Report the highest-ranked k scores per window, optionally above a floor."""

    kind: Literal["top_k"] = "top_k"
    k: Annotated[int, Field(gt=0)]
    min_score: _FiniteFloor | None = None


DetectionPolicy = Annotated[ThresholdPolicy | TopKPolicy, Field(discriminator="kind")]


class ScoresRequest(BaseModel, frozen=True, extra="forbid"):
    """The score stream, and how much of the vocabulary it keeps."""

    kind: Literal["scores"] = "scores"
    contract_id: ScoresContractId
    retention: ScoreRetention
    min_score: _FiniteFloor | None = None
    top_k: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _retention_carries_its_own_parameters(self) -> "ScoresRequest":
        # A full request carrying either parameter is rejected rather than ignored:
        # accepting it silently is how a request that meant thresholded is published
        # as full.
        if self.retention == "full":
            if self.min_score is not None or self.top_k is not None:
                raise ValueError(
                    "full retention keeps every label and takes no min_score or top_k"
                )
            return self
        if self.min_score is None:
            raise ValueError(f"{self.retention} retention requires a min_score")
        if self.retention == "top_k":
            if self.top_k is None:
                raise ValueError("top_k retention requires a top_k")
        elif self.top_k is not None:
            raise ValueError("top_k belongs only to top_k retention")
        return self


class EmbeddingsRequest(BaseModel, frozen=True, extra="forbid"):
    """The embedding stream, and the width its values are stored at."""

    kind: Literal["embeddings"] = "embeddings"
    contract_id: EmbeddingsContractId
    storage_dtype: Literal["float32", "float16"] | None = None  # None: the recipe's width


class DetectionsRequest(BaseModel, frozen=True, extra="forbid"):
    """The selected detections, and the policy that selects them."""

    kind: Literal["detections"] = "detections"
    contract_id: DetectionsContractId
    policy: DetectionPolicy


OutputRequest = Annotated[
    ScoresRequest | EmbeddingsRequest | DetectionsRequest, Field(discriminator="kind")
]
