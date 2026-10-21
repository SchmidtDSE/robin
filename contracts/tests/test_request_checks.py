"""The in-memory request checks, shared by the engine and the control plane."""

import pytest

from robin_contracts import request_checks
from robin_contracts.cards import (
    AudioGeometry,
    HeadCard,
    InferenceParam,
    ModelCard,
    RunnerResampled,
    model_ref,
)
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
)
from robin_contracts.request_checks import RequestProblem, check_request_for_problem
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    EmbeddingArtifactInput,
    PinnedFile,
    PinnedModel,
)

REGISTRY_DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64
WEIGHTS = PinnedFile(uri="s3://b/owl.tflite", digest="sha256:" + "0" * 64, size_bytes=8)
REGISTRY_FILE = PinnedFile(uri="s3://b/registry.csv", digest=REGISTRY_DIGEST, size_bytes=8)


def build_card(**overrides) -> ModelCard:
    fields = {
        "model_name": "owl",
        "model_version": "1",
        "runtime": "tensorflow",
        "window_duration": 3.0,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
        "score_domain": "sigmoid",
        "taxa_registry_digest": REGISTRY_DIGEST,
        "audio": AudioGeometry(
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        "backend": "tensorflow",
        "dtype": "float32",
        "inference_params": (InferenceParam(name="window_overlap", type="float"),),
    }
    return ModelCard(**(fields | overrides))


BACKBONE = build_card(
    model_name="perch",
    model_version="8",
    can_emit_embeddings=True,
    embedding_dim=1280,
    embedding_dtype="float32",
)


def build_head(**overrides) -> HeadCard:
    fields = {
        "model_name": "amy-head",
        "model_version": "1",
        "runtime": "onnx",
        "backbone": model_ref(BACKBONE),
        "embedding_dim": 1280,
        "min_detection_threshold": 0.0,
        "score_domain": "sigmoid",
        "taxa_registry_digest": REGISTRY_DIGEST,
    }
    return HeadCard(**(fields | overrides))


def pinned(card, files=None) -> PinnedModel:
    files = {"weights": WEIGHTS, REGISTRY_ROLE: REGISTRY_FILE} if files is None else files
    return PinnedModel(card=card, files=files)


def head_input(backbone: ModelCard = BACKBONE, settings=None) -> EmbeddingArtifactInput:
    return EmbeddingArtifactInput(
        contract_id="robin.embeddings.parquet/1",
        backbone=backbone,
        backbone_settings=settings or {},
    )


def scores(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.parquet/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def embeddings(**overrides) -> EmbeddingsRequest:
    return EmbeddingsRequest(**({"contract_id": "robin.embeddings.parquet/1"} | overrides))


def detections(min_score: float) -> DetectionsRequest:
    return DetectionsRequest(
        contract_id="robin.detections.parquet/1", policy=ThresholdPolicy(min_score=min_score)
    )


def problem(model, input=None, settings=None, outputs=None) -> RequestProblem | None:
    return check_request_for_problem(
        model,
        AudioInput() if input is None else input,
        settings or {},
        (scores(),) if outputs is None else outputs,
    )


HEAD = pinned(build_head())
THRESHOLDED = pinned(build_card(min_detection_threshold=0.005))

CASES = {
    request_checks.INPUT_KIND_DISAGREES_WITH_CARD: lambda: problem(HEAD),
    request_checks.HEAD_BACKBONE_DISAGREES: lambda: problem(
        HEAD, head_input(build_card(model_name="birdnet", can_emit_embeddings=True,
                                    embedding_dim=1280, embedding_dtype="float32"))
    ),
    request_checks.HEAD_EMITS_NO_EMBEDDINGS: lambda: problem(
        HEAD, head_input(), outputs=(scores(), embeddings())
    ),
    request_checks.EMBEDDINGS_NOT_EMITTED: lambda: problem(
        pinned(build_card()), outputs=(embeddings(),)
    ),
    request_checks.EMBEDDING_DTYPE_DISAGREES: lambda: problem(
        pinned(BACKBONE), outputs=(embeddings(storage_dtype="float16"),)
    ),
    request_checks.REGISTRY_REQUIRED: lambda: problem(
        pinned(build_card(), files={"weights": WEIGHTS})
    ),
    request_checks.REGISTRY_DISAGREES_WITH_CARD: lambda: problem(
        pinned(build_card(), files={
            "weights": WEIGHTS,
            REGISTRY_ROLE: PinnedFile(uri="s3://b/o.csv", digest=OTHER_DIGEST, size_bytes=8),
        })
    ),
    request_checks.SETTING_UNDECLARED: lambda: problem(
        pinned(build_card()), settings={"gain": 1.0}
    ),
    request_checks.SETTING_TYPE_MISMATCH: lambda: problem(
        pinned(build_card()), settings={"window_overlap": 1}
    ),
    request_checks.WINDOW_OVERLAP_INVALID: lambda: problem(
        pinned(build_card()), settings={"window_overlap": 3.0}
    ),
    request_checks.SCORES_NOT_EMITTED: lambda: problem(
        pinned(build_card(score_domain=None, taxa_registry_digest=None))
    ),
    request_checks.FULL_RETENTION_REDUCED: lambda: problem(THRESHOLDED),
    request_checks.MIN_SCORE_OUT_OF_DOMAIN: lambda: problem(
        pinned(build_card()), outputs=(scores(), detections(1.5))
    ),
    request_checks.SCORE_FLOOR_BELOW_MODEL_FLOOR: lambda: problem(
        THRESHOLDED, outputs=(scores(retention="thresholded", min_score=0.001),)
    ),
}


@pytest.mark.parametrize("code", sorted(CASES))
def test_each_problem_is_reported_with_its_code(code):
    found = CASES[code]()

    assert isinstance(found, RequestProblem)
    assert found.code == code
    assert found.detail


def test_every_code_has_a_case():
    assert set(CASES) == set(request_checks.REQUEST_PROBLEM_CODES)


def test_a_head_backbone_setting_is_checked_against_the_backbone():
    found = problem(HEAD, head_input(settings={"window_overlap": 3.0}))

    assert found.code == request_checks.WINDOW_OVERLAP_INVALID
    assert "perch/8" in found.detail


def test_a_valid_backbone_request_has_no_problem():
    outputs = (scores(), embeddings(), detections(0.5))

    assert problem(pinned(BACKBONE), settings={"window_overlap": 1.5}, outputs=outputs) is None


def test_a_valid_head_request_has_no_problem():
    assert problem(HEAD, head_input(settings={"window_overlap": 1.5})) is None


def test_the_score_range_is_zero_to_one():
    assert request_checks.SCORE_RANGE == (0.0, 1.0)
