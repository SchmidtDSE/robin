"""What the engine hands back: what it produced, what it finished, or why it stopped."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from robin_contracts.output_contracts import (
    DetectionPolicy,
    DetectionsContractId,
    EmbeddingsContractId,
    ResultContractId,
    ScoresContractId,
    ScoresRequest,
)
from robin_contracts.specs import Recipe, WindowGeometry
from robin_contracts.work import (
    BytesDigest,
    CanonicalDigest,
    NonEmptyText,
    PinnedModel,
)

ArtifactContractId = ScoresContractId | EmbeddingsContractId | DetectionsContractId
ArtifactKind = Literal["scores", "embeddings", "detections"]
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


def _named(namespace: str, value: str) -> str:
    # Quoted as a pair: any separator could also appear inside a value.
    return f"({namespace!r}, {value!r})"


class ArtifactRecord(BaseModel, frozen=True, extra="forbid"):
    """One artifact this work wrote: what it is, where it is, what it hashes to."""

    kind: ArtifactKind
    contract_id: ArtifactContractId
    uri: NonEmptyText
    checksum: BytesDigest
    size_bytes: _Count
    rows: _Count  # zero is a real value


class RecordingCoverage(BaseModel, frozen=True, extra="forbid"):
    """What one recording finished, counted in windows; rows only check them.

    The recording is named by its `namespace` and `value`, as in every artifact row.
    `audio_digest` is the work's, copied exactly, `null` included.
    """

    namespace: NonEmptyText
    value: NonEmptyText
    audio_digest: BytesDigest | None
    windows_completed: _Count
    score_rows: _Count
    embedding_rows: _Count
    first_window_start_s: _Seconds | None = None
    last_window_end_s: _Seconds | None = None
    zero_window_reason: ZeroWindowReason | None = None

    def _name(self) -> str:
        return _named(self.namespace, self.value)

    @model_validator(mode="after")
    def _validate_window_bounds(self) -> "RecordingCoverage":
        start, end = self.first_window_start_s, self.last_window_end_s
        if (start is None) != (end is None):
            raise ValueError("a window range carries both of its bounds or neither")
        if (start is None) != (self.windows_completed == 0):
            raise ValueError(
                f"recording {self._name()}: {self.windows_completed} completed windows and "
                f"a window range of {start} to {end} disagree about whether anything ran"
            )
        if start is None:
            return self
        if start < 0:
            raise ValueError(
                f"recording {self._name()}: a window starts at or after zero, got {start}"
            )
        if end <= start:
            raise ValueError(
                f"recording {self._name()}: a window range must span time, "
                f"got {start} to {end}"
            )
        return self

    @model_validator(mode="after")
    def _validate_zero_window_coverage(self) -> "RecordingCoverage":
        if (self.windows_completed == 0) != (self.zero_window_reason is not None):
            raise ValueError(
                f"recording {self._name()}: a recording that completed no window "
                "declares why, and one that completed windows declares no reason"
            )
        if self.windows_completed == 0 and (self.score_rows or self.embedding_rows):
            raise ValueError(
                f"recording {self._name()} completed no window yet carries "
                f"{self.score_rows} score and {self.embedding_rows} embedding rows"
            )
        return self

    @model_validator(mode="after")
    def _validate_embedding_row_count(self) -> "RecordingCoverage":
        if self.embedding_rows > self.windows_completed:
            raise ValueError(
                f"recording {self._name()}: {self.embedding_rows} embedding rows from "
                f"{self.windows_completed} completed windows"
            )
        return self


class FailureReport(BaseModel, frozen=True, extra="forbid"):
    """Why a work stopped: a stable code, the stage that raised it, and where."""

    code: NonEmptyText
    stage: FailureStage
    namespace: NonEmptyText | None = None
    value: NonEmptyText | None = None
    window_start_s: _Seconds | None = None
    detail: str

    @model_validator(mode="after")
    def _names_a_whole_recording_or_none(self) -> "FailureReport":
        if (self.namespace is None) != (self.value is None):
            raise ValueError(
                f"a failure names its recording by namespace and value together, got "
                f"{self.namespace!r} and {self.value!r}"
            )
        return self


class InferenceSuccess(BaseModel, frozen=True, extra="forbid"):
    """A completed work: what it ran, what it wrote, and what it finished."""

    schema_version: ResultContractId
    outcome: Literal["success"] = "success"
    work_digest: CanonicalDigest
    recipe: Recipe
    model: PinnedModel
    registry_uri: NonEmptyText | None = None
    registry_fingerprint: BytesDigest | None = None
    window_geometry: WindowGeometry
    resolved_detection_policy: DetectionPolicy | None = None
    resolved_scores_request: ScoresRequest | None = None
    artifacts: tuple[ArtifactRecord, ...]
    coverage: tuple[RecordingCoverage, ...]

    @model_validator(mode="after")
    def _validate_coverage_names_each_recording_once(self) -> "InferenceSuccess":
        # The order is the work's, which this record does not hold; the engine checks it.
        if not self.coverage:
            raise ValueError("a success carries one coverage row per recording")
        seen: set[tuple[str, str]] = set()
        for row in self.coverage:
            identity = (row.namespace, row.value)
            if identity in seen:
                raise ValueError(f"recording {_named(*identity)} appears twice in coverage")
            seen.add(identity)
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
        kinds = [artifact.kind for artifact in self.artifacts]
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
