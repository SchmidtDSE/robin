"""What the engine hands back: what it produced, what it finished, or why it stopped."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from robin_contracts.output_contracts import (
    DetectionPolicy,
    DetectionsContractId,
    EmbeddingsContractId,
    RecordingMapContractId,
    ResultContractId,
    ScoresContractId,
    ScoresRequest,
)
from robin_contracts.specs import Recipe, WindowGeometry
from robin_contracts.work import (
    BytesDigest,
    CanonicalDigest,
    ModelSelection,
    NonEmptyText,
)

ArtifactContractId = (
    ScoresContractId | EmbeddingsContractId | DetectionsContractId | RecordingMapContractId
)
ArtifactKind = Literal["scores", "embeddings", "detections", "recording_map"]
ZeroWindowReason = Literal["shorter_than_window", "no_input_windows"]
FailureStage = Literal[
    "validate_request",
    "resolve_model",
    "load_registry",
    "acquire_audio",
    "read_input_artifact",
    "construct_model",
    "infer",
    "accept_window",
    "write_artifact",
    "aggregate",
]


_Seconds = Annotated[float, Field(allow_inf_nan=False)]
_Count = Annotated[int, Field(ge=0)]


class ArtifactRecord(BaseModel, frozen=True, extra="forbid"):
    """One artifact this work wrote: what it is, where it is, what it hashes to."""

    kind: ArtifactKind
    contract_id: ArtifactContractId
    uri: NonEmptyText
    checksum: BytesDigest
    size_bytes: _Count
    # Zero is a real value. For the recording map this is the number of recordings.
    rows: _Count


class RecordingCoverage(BaseModel, frozen=True, extra="forbid"):
    """What one recording finished, counted in windows; rows only check them."""

    recording_index: _Count
    windows_completed: _Count
    score_rows: _Count
    embedding_rows: _Count
    first_window_start_s: _Seconds | None = None
    last_window_end_s: _Seconds | None = None
    zero_window_reason: ZeroWindowReason | None = None

    @model_validator(mode="after")
    def _validate_window_bounds(self) -> "RecordingCoverage":
        start, end = self.first_window_start_s, self.last_window_end_s
        if (start is None) != (end is None):
            raise ValueError("a window range carries both of its bounds or neither")
        if (start is None) != (self.windows_completed == 0):
            raise ValueError(
                f"{self.windows_completed} completed windows and a window range of "
                f"{start} to {end} disagree about whether anything ran"
            )
        if start is None:
            return self
        if start < 0:
            raise ValueError(f"a window starts at or after zero, got {start}")
        if end <= start:
            raise ValueError(f"a window range must span time, got {start} to {end}")
        return self

    @model_validator(mode="after")
    def _validate_zero_window_coverage(self) -> "RecordingCoverage":
        if (self.windows_completed == 0) != (self.zero_window_reason is not None):
            raise ValueError(
                "a recording that completed no window declares why, and one that "
                "completed windows declares no reason"
            )
        if self.windows_completed == 0 and (self.score_rows or self.embedding_rows):
            raise ValueError(
                f"recording {self.recording_index} completed no window yet carries "
                f"{self.score_rows} score and {self.embedding_rows} embedding rows"
            )
        return self

    @model_validator(mode="after")
    def _validate_embedding_row_count(self) -> "RecordingCoverage":
        if self.embedding_rows > self.windows_completed:
            raise ValueError(
                f"{self.embedding_rows} embedding rows from "
                f"{self.windows_completed} completed windows"
            )
        return self


class FailureReport(BaseModel, frozen=True, extra="forbid"):
    """Why a work stopped: a stable code, the stage that raised it, and where."""

    code: NonEmptyText
    stage: FailureStage
    recording_index: _Count | None = None
    window_start_s: _Seconds | None = None
    detail: str


class InferenceSuccess(BaseModel, frozen=True, extra="forbid"):
    """A completed work: what it ran, what it wrote, and what it finished."""

    schema_version: ResultContractId
    outcome: Literal["success"] = "success"
    work_digest: CanonicalDigest
    recipe: Recipe
    model: ModelSelection
    registry_uri: NonEmptyText | None = None
    registry_fingerprint: BytesDigest | None = None
    window_geometry: WindowGeometry
    resolved_detection_policy: DetectionPolicy | None = None
    resolved_scores_request: ScoresRequest | None = None
    artifacts: tuple[ArtifactRecord, ...]
    recording_map: ArtifactRecord
    coverage: tuple[RecordingCoverage, ...]

    @model_validator(mode="after")
    def _validate_coverage_order(self) -> "InferenceSuccess":
        if not self.coverage:
            raise ValueError("a success carries one coverage row per recording")
        indices = [row.recording_index for row in self.coverage]
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(
                f"coverage ascends by recording_index without repeating, got {indices}"
            )
        return self

    @model_validator(mode="after")
    def _validate_registry_binding(self) -> "InferenceSuccess":
        if (self.registry_uri is None) != (self.registry_fingerprint is None):
            raise ValueError(
                f"registry_uri {self.registry_uri!r} and registry_fingerprint "
                f"{self.registry_fingerprint!r} are both set or both absent"
            )
        return self

    @model_validator(mode="after")
    def _validate_artifact_kinds(self) -> "InferenceSuccess":
        if self.recording_map.kind != "recording_map":
            raise ValueError(
                f"the recording map field holds a {self.recording_map.kind!r} artifact"
            )
        kinds = [artifact.kind for artifact in self.artifacts]
        if "recording_map" in kinds:
            raise ValueError("the recording map has its own field and is not an artifact")
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"a work writes one artifact per kind, got {kinds}")
        return self

    @model_validator(mode="after")
    def _validate_artifact_row_counts(self) -> "InferenceSuccess":
        written = {artifact.kind: artifact.rows for artifact in self.artifacts}
        counted = {
            "scores": sum(row.score_rows for row in self.coverage),
            "embeddings": sum(row.embedding_rows for row in self.coverage),
        }
        for kind, total in counted.items():
            if kind not in written:
                if total:
                    raise ValueError(
                        f"coverage counts {total} {kind} rows with no artifact to hold them"
                    )
                continue
            if total != written[kind]:
                raise ValueError(
                    f"coverage counts {total} {kind} rows; the artifact holds {written[kind]}"
                )
        return self

    @model_validator(mode="after")
    def _validate_resolved_requests(self) -> "InferenceSuccess":
        kinds = {artifact.kind for artifact in self.artifacts}
        if (self.resolved_scores_request is not None) != ("scores" in kinds):
            raise ValueError(
                "resolved_scores_request is present exactly when a scores artifact is"
            )
        if (self.resolved_detection_policy is not None) != ("detections" in kinds):
            raise ValueError(
                "resolved_detection_policy is present exactly when a detections artifact is"
            )
        return self


class InferenceFailure(BaseModel, frozen=True, extra="forbid"):
    """A work that failed whole: it names no artifact, even where bytes were written."""

    schema_version: ResultContractId
    outcome: Literal["failure"] = "failure"
    work_digest: CanonicalDigest
    failure: FailureReport


InferenceResult = Annotated[
    InferenceSuccess | InferenceFailure, Field(discriminator="outcome")
]
