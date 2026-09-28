"""Each recording's output files: its window artifacts, then the detections selected from them."""

from pathlib import Path
from typing import get_args

import pyarrow as pa

from robin_contracts.output_contracts import (
    DetectionsContractId,
    EmbeddingsContractId,
    ScoresContractId,
)
from robin_contracts.protocols import Model
from robin_contracts.registry import TaxonRegistry
from robin_contracts.results import ArtifactContractId
from robin_contracts.work import REGISTRY_ROLE, InferenceWork, RecordingRef
from robin_inference_engine.accept_window import AcceptedWindow
from robin_inference_engine.artifacts.embeddings import EmbeddingsWriter
from robin_inference_engine.artifacts.metadata import (
    detection_metadata,
    embedding_metadata,
    required_metadata,
    score_metadata,
)
from robin_inference_engine.artifacts.scores import ScoresWriter
from robin_inference_engine.artifacts.staging import StagedArtifact
from robin_inference_engine.detections import registry_table, write_detections
from robin_inference_engine.requested_outputs import (
    detections_request,
    embeddings_request,
    scores_request,
)

SCORES_CONTRACT_ID: ScoresContractId = get_args(ScoresContractId)[0]
EMBEDDINGS_CONTRACT_ID: EmbeddingsContractId = get_args(EmbeddingsContractId)[0]
DETECTIONS_CONTRACT_ID: DetectionsContractId = get_args(DetectionsContractId)[0]


class RecordingOutputs:
    """Opens each recording's window writers and stages its detections, with the headers all its
    files share. Built once per work.
    """

    def __init__(
        self,
        work: InferenceWork,
        *,
        model: Model,
        registry: TaxonRegistry | None,
        staging: Path,
    ) -> None:
        self._work = work
        self._model = model
        self._staging = staging
        self._scores = scores_request(work)
        self._embeddings = embeddings_request(work)
        self.registry_uri = work.model.files[REGISTRY_ROLE].uri if registry is not None else None
        self.registry_fingerprint = registry.fingerprint if registry is not None else None
        self.detections = detections_request(work)
        self._registry_table: pa.Table | None = None
        if self.detections is not None:
            if registry is None or self._scores is None:
                raise RuntimeError(
                    "detections were requested without a registry or the scores they select from"
                )
            self._registry_table = registry_table(registry)

    def open_window_writers(self, position: int, recording: RecordingRef) -> "_WindowWriters":
        """The recording's window writers, in a staging folder of its own."""
        folder = self._staging / str(position)
        folder.mkdir()
        writers = _WindowWriters()
        try:
            if self._scores is not None:
                writers.add(self._scores_writer(folder, recording))
            if self._embeddings is not None:
                writers.add(self._embeddings_writer(folder, recording))
        except BaseException:
            writers.release()
            raise
        return writers

    def stage_detections(self, staged: tuple[StagedArtifact, ...]) -> StagedArtifact | None:
        """The detections file selected from a recording's staged scores file.

        Returns None when detections were not requested, the recording has no scores
        file, or the policy selects nothing.
        """
        scores = next((one for one in staged if one.kind == "scores"), None)
        if self.detections is None or self._registry_table is None or scores is None:
            return None
        folder = scores.path.parent
        return write_detections(
            scores,
            folder / "detections.parquet",
            work=self._work,
            recipe=self._model.recipe,
            policy=self.detections.policy,
            registry=self._registry_table,
            metadata=self._detections_header(scores),
            temporary=folder,
        )

    def _scores_writer(self, folder: Path, recording: RecordingRef) -> ScoresWriter:
        return ScoresWriter(
            folder / "scores.arrow",
            recording=recording,
            metadata=self._header(SCORES_CONTRACT_ID, recording) | self._score_keys(),
        )

    def _embeddings_writer(self, folder: Path, recording: RecordingRef) -> EmbeddingsWriter:
        capabilities = self._model.capabilities
        # Stored at the recipe's width: it is inside the recipe fingerprint, and a
        # request naming a different one was refused before inference.
        storage_dtype = self._model.recipe.dtype
        return EmbeddingsWriter(
            folder / "embeddings.arrow",
            recording=recording,
            dim=capabilities.embedding_dim,
            storage_dtype=storage_dtype,
            metadata=self._header(EMBEDDINGS_CONTRACT_ID, recording)
            | embedding_metadata(
                self._work,
                dim=capabilities.embedding_dim,
                source_dtype=capabilities.embedding_dtype,
                storage_dtype=storage_dtype,
            ),
        )

    def _detections_header(self, scores: StagedArtifact) -> dict[bytes, bytes]:
        if self.detections is None:
            raise RuntimeError("no detections were requested, so there is no header for them")
        return (
            self._header(DETECTIONS_CONTRACT_ID, scores.recording)
            # These describe the scores the detections were ranked among, not the selection.
            | self._score_keys()
            | detection_metadata(
                self.detections.policy,
                source_contract_id=scores.contract_id,
                source_checksum=scores.checksum,
            )
        )

    def _header(
        self, contract_id: ArtifactContractId, recording: RecordingRef
    ) -> dict[bytes, bytes]:
        return required_metadata(
            contract_id=contract_id,
            work=self._work,
            recording=recording,
            recipe=self._model.recipe,
            registry_uri=self.registry_uri,
            registry_fingerprint=self.registry_fingerprint,
        )

    def _score_keys(self) -> dict[bytes, bytes]:
        if self._scores is None:
            raise RuntimeError("no scores were requested, so there are no score keys")
        return score_metadata(self._scores, score_domain=self._model.capabilities.score_domain)


# ---------------------------------------------------------------------------
# The two window artifacts, written as windows are accepted.
# ---------------------------------------------------------------------------


class _WindowWriters:
    """One recording's open writers, one per requested window kind."""

    def __init__(self) -> None:
        self._writers: list[ScoresWriter | EmbeddingsWriter] = []

    def add(self, writer: ScoresWriter | EmbeddingsWriter) -> None:
        self._writers.append(writer)

    def write(self, window: AcceptedWindow) -> None:
        for writer in self._writers:
            writer.write(window)

    def finish(self) -> tuple[StagedArtifact, ...]:
        """Finish every file, and keep only those holding rows.

        A recording with no rows of a kind has no file, so an empty one is deleted here
        and never reaches the writer port.
        """
        staged = tuple(writer.close() for writer in self._writers)
        for empty in (one for one in staged if not one.rows):
            empty.path.unlink()
        return tuple(one for one in staged if one.rows)

    def release(self) -> None:
        """Close every file handle, finished or not."""
        for writer in self._writers:
            writer.__exit__(None, None, None)

    def __enter__(self) -> "_WindowWriters":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
