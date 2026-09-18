"""What the engine declines to attempt, decided entirely from in-memory values."""

import pytest

from robin_contracts.cards import HeadCard, ModelCard, ModelRef
from robin_contracts.embedding_transforms import L2Norm
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
    TopKPolicy,
)
from robin_contracts.protocols import ModelCapabilities
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.work import (
    AudioInput,
    FileDigest,
    InferenceWork,
    ModelSelection,
    RecordingRef,
)
from robin_inference_engine import errors
from robin_inference_engine.validate_request import refuse_instance, refuse_request

HEX = "0" * 64
RECORD_DIGEST = f"sha256:v1:{HEX}"
FILE_DIGEST = f"sha256:{HEX}"
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64
OTHER_FINGERPRINT = "sha256:" + "b" * 64

MODEL_REF = ModelRef(name="owl", version="1", digest=RECORD_DIGEST)


def build_selection(**overrides) -> ModelSelection:
    fields = {
        "ref": MODEL_REF,
        "card_digest": RECORD_DIGEST,
        "files": (
            FileDigest(role="weights", uri="s3://b/owl.tflite", digest=FILE_DIGEST, size_bytes=8),
        ),
        "registry_fingerprint": REGISTRY_FINGERPRINT,
    }
    return ModelSelection(**(fields | overrides))


def build_scores(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_embeddings() -> EmbeddingsRequest:
    return EmbeddingsRequest(contract_id="robin.embeddings.arrow/1")


def build_detections(policy) -> DetectionsRequest:
    return DetectionsRequest(contract_id="robin.detections.parquet/1", policy=policy)


def build_work(**overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (
            RecordingRef(index=0, namespace="soundhub", value="42", audio_uri="s3://b/4.wav"),
        ),
        "model": build_selection(),
        "input": AudioInput(),
        "settings": {},
        "resources": {},
        "outputs": (build_scores(),),
    }
    return InferenceWork(**(fields | overrides))


def build_card(**overrides) -> ModelCard:
    fields = {
        "model_name": "owl",
        "model_version": "1",
        "runtime": "tensorflow",
        "segment_duration": 3.0,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
        "taxa_registry_uri": "s3://b/registry.csv",
    }
    return ModelCard(**(fields | overrides))


def build_head(**overrides) -> HeadCard:
    fields = {
        "model_name": "amy-head",
        "model_version": "1",
        "backbone": MODEL_REF,
        "weights_uri": "s3://b/head.keras",
        "classes": ("owl",),
        "required_embedding_transform": L2Norm(),
        "taxa_registry_uri": "s3://b/registry.csv",
    }
    return HeadCard(**(fields | overrides))


def build_registry(*, fingerprint: str = REGISTRY_FINGERPRINT, labels=("owl",)):
    return TaxonRegistry(
        fingerprint=fingerprint,
        entries=tuple(
            RegistryEntry(class_index=index, label=label, label_kind="non_taxonomic")
            for index, label in enumerate(labels)
        ),
    )


def build_capabilities(**overrides) -> ModelCapabilities:
    fields = {
        "emits_scores": True,
        "emits_embeddings": False,
        "score_domain": "probability",
        "supported_retention": frozenset({"full", "thresholded", "top_k"}),
        "native_score_floor": None,
        "embedding_dim": None,
    }
    return ModelCapabilities(**(fields | overrides))


def test_scores_from_an_adapter_that_emits_none_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(
                emits_scores=False, score_domain=None, supported_retention=frozenset()
            ),
            card=build_card(),
        )

    assert exc.value.code == errors.SCORES_NOT_EMITTED


def test_retention_outside_supported_retention_is_refused():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.1),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(supported_retention=frozenset({"full"})),
            card=build_card(),
        )

    assert exc.value.code == errors.RETENTION_UNSUPPORTED


def test_full_retention_with_an_excluding_floor_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.05),
            card=build_card(),
        )

    assert exc.value.code == errors.FULL_RETENTION_REDUCED

    # A floor at the domain minimum excludes nothing. This is OWL's configuration and
    # refusing it would fail the stage that runs it.
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.0),
            card=build_card(),
        )
        is None
    )


def test_full_retention_is_refused_when_the_instance_declares_only_top_k():
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(supported_retention=frozenset({"top_k"})),
            card=build_card(),
        )

    assert exc.value.code == errors.RETENTION_UNSUPPORTED


def test_a_thresholded_request_at_the_instance_floor_is_accepted():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.005),))

    assert (
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=0.005),
            card=build_card(),
        )
        is None
    )


def test_a_thresholded_request_above_the_instance_floor_is_refused():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.5),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=0.005),
            card=build_card(),
        )

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES
    assert "0.5" in exc.value.detail and "0.005" in exc.value.detail


def test_a_thresholded_request_below_the_instance_floor_is_refused():
    # The over-claiming direction: a reader would conclude that an absent label scored
    # under 0.001 when the boundary the rows were produced at was 0.005.
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.001),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=0.005),
            card=build_card(),
        )

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES


def test_a_reduced_request_against_an_instance_with_no_floor_is_refused():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.005),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=None),
            card=build_card(),
        )

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES


def test_a_top_k_request_checks_its_floor_too():
    work = build_work(
        outputs=(build_scores(retention="top_k", min_score=0.5, top_k=5),)
    )

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=0.005),
            card=build_card(),
        )

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES

    at_the_floor = build_work(
        outputs=(build_scores(retention="top_k", min_score=0.005, top_k=5),)
    )
    assert (
        refuse_instance(
            at_the_floor,
            capabilities=build_capabilities(native_score_floor=0.005),
            card=build_card(),
        )
        is None
    )


def test_a_full_request_is_unaffected_by_the_floor_check():
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.0),
            card=build_card(),
        )
        is None
    )

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.005),
            card=build_card(),
        )

    assert exc.value.code == errors.FULL_RETENTION_REDUCED


@pytest.mark.parametrize("floor", [1.5, -0.1])
def test_a_min_score_outside_the_score_domain_is_refused(floor):
    from_request = build_work(
        outputs=(build_scores(retention="thresholded", min_score=floor),)
    )
    # The instance declares the same floor, so what refuses is the domain and not the
    # disagreement: a request the instance does not impose is a different defect.
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            from_request,
            capabilities=build_capabilities(native_score_floor=floor),
            card=build_card(),
        )
    assert exc.value.code == errors.MIN_SCORE_OUT_OF_DOMAIN

    from_policy = build_work(
        outputs=(build_scores(), build_detections(ThresholdPolicy(min_score=floor)))
    )
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(from_policy, capabilities=build_capabilities(), card=build_card())
    assert exc.value.code == errors.MIN_SCORE_OUT_OF_DOMAIN


def test_embeddings_from_an_adapter_that_emits_none_is_refused():
    work = build_work(outputs=(build_embeddings(),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(emits_embeddings=False),
            card=build_card(can_emit_embeddings=True, embedding_dim=1024),
        )

    assert exc.value.code == errors.EMBEDDINGS_NOT_EMITTED


def test_embeddings_forbidden_by_the_card_is_refused():
    work = build_work(outputs=(build_embeddings(),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_request(
            work, card=build_card(can_emit_embeddings=False), registry=build_registry()
        )

    assert exc.value.code == errors.EMBEDDINGS_NOT_EMITTED

    # The card is the half that refuses here: the instance says it can.
    assert (
        refuse_instance(
            work,
            capabilities=build_capabilities(emits_embeddings=True, embedding_dim=1024),
            card=build_card(can_emit_embeddings=True, embedding_dim=1024),
        )
        is None
    )


def test_an_embedding_dim_disagreeing_with_the_card_is_refused():
    work = build_work(outputs=(build_embeddings(),))
    card = build_card(can_emit_embeddings=True, embedding_dim=1024)

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(emits_embeddings=True, embedding_dim=512),
            card=card,
        )
    assert exc.value.code == errors.EMBEDDING_DIM_DISAGREES

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(emits_embeddings=True, embedding_dim=None),
            card=card,
        )
    assert exc.value.code == errors.EMBEDDING_DIM_DISAGREES


@pytest.mark.parametrize("dimension", [None, 512])
def test_scores_only_work_refuses_inconsistent_embedding_dimensions(dimension):
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(
                emits_embeddings=True, embedding_dim=dimension
            ),
            card=build_card(can_emit_embeddings=True, embedding_dim=1024),
        )

    assert exc.value.code == errors.EMBEDDING_DIM_DISAGREES
    assert exc.value.stage == errors.VALIDATE_REQUEST


def test_scores_only_work_refuses_an_emission_its_card_forbids():
    # The card and the instance contradict each other. Which one is wrong is unknowable
    # here, and a work asking only for scores would otherwise carry the defect silently.
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dim=512),
            card=build_card(can_emit_embeddings=False),
        )

    assert exc.value.code == errors.EMBEDDING_EMISSION_DISAGREES
    assert exc.value.stage == errors.VALIDATE_REQUEST


def test_a_card_forbidding_embeddings_accepts_an_instance_that_emits_none():
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(emits_embeddings=False),
            card=build_card(can_emit_embeddings=False),
        )
        is None
    )


def test_scores_only_work_accepts_consistent_embedding_dimensions():
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dim=1024),
            card=build_card(can_emit_embeddings=True, embedding_dim=1024),
        )
        is None
    )


def test_scores_from_a_card_with_no_registry_uri_are_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_request(
            build_work(), card=build_card(taxa_registry_uri=None), registry=build_registry()
        )

    assert exc.value.code == errors.REGISTRY_REQUIRED


def test_a_registry_fingerprint_differing_from_the_pinned_one_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_request(
            build_work(),
            card=build_card(),
            registry=build_registry(fingerprint=OTHER_FINGERPRINT),
        )
    assert exc.value.code == errors.REGISTRY_FINGERPRINT_MISMATCH

    with pytest.raises(errors.EngineError) as exc:
        refuse_request(build_work(), card=build_card(), registry=None)
    assert exc.value.code == errors.REGISTRY_FINGERPRINT_MISMATCH


def test_a_head_class_outside_the_registry_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_request(
            build_work(),
            card=build_head(classes=("owl", "barred-owl")),
            registry=build_registry(labels=("owl",)),
        )

    assert exc.value.code == errors.HEAD_CLASS_NOT_IN_REGISTRY


def test_a_satisfiable_work_is_not_refused():
    work = build_work()

    assert refuse_request(work, card=build_card(), registry=build_registry()) is None
    assert (
        refuse_instance(work, capabilities=build_capabilities(), card=build_card()) is None
    )


def test_an_embeddings_only_work_needs_no_registry():
    work = build_work(
        model=build_selection(registry_fingerprint=None),
        outputs=(build_embeddings(),),
    )
    card = build_card(
        taxa_registry_uri=None, can_emit_embeddings=True, embedding_dim=1280
    )

    assert refuse_request(work, card=card, registry=None) is None
    assert (
        refuse_instance(
            work,
            capabilities=build_capabilities(
                emits_scores=False,
                emits_embeddings=True,
                score_domain=None,
                supported_retention=frozenset(),
                embedding_dim=1280,
            ),
            card=card,
        )
        is None
    )


REFUSING_CALLS = {
    "scores_not_emitted": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(
            emits_scores=False, score_domain=None, supported_retention=frozenset()
        ),
        card=build_card(),
    ),
    "retention_unsupported": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(supported_retention=frozenset({"top_k"})),
        card=build_card(),
    ),
    "full_retention_reduced": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(native_score_floor=0.05),
        card=build_card(),
    ),
    "score_floor_disagrees": lambda: refuse_instance(
        build_work(outputs=(build_scores(retention="thresholded", min_score=0.5),)),
        capabilities=build_capabilities(native_score_floor=0.005),
        card=build_card(),
    ),
    "min_score_out_of_domain": lambda: refuse_instance(
        build_work(outputs=(build_scores(retention="thresholded", min_score=1.5),)),
        capabilities=build_capabilities(native_score_floor=1.5),
        card=build_card(),
    ),
    "min_score_out_of_domain_from_a_top_k_policy": lambda: refuse_instance(
        build_work(outputs=(build_scores(), build_detections(TopKPolicy(k=5, min_score=2.0)))),
        capabilities=build_capabilities(),
        card=build_card(),
    ),
    "embeddings_not_emitted_by_the_instance": lambda: refuse_instance(
        build_work(outputs=(build_embeddings(),)),
        capabilities=build_capabilities(emits_embeddings=False),
        card=build_card(can_emit_embeddings=True, embedding_dim=8),
    ),
    "embeddings_not_emitted_by_the_card": lambda: refuse_request(
        build_work(outputs=(build_embeddings(),)),
        card=build_card(can_emit_embeddings=False),
        registry=build_registry(),
    ),
    "embedding_emission_disagrees": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(emits_embeddings=True, embedding_dim=512),
        card=build_card(can_emit_embeddings=False),
    ),
    "embedding_dim_disagrees": lambda: refuse_instance(
        build_work(outputs=(build_embeddings(),)),
        capabilities=build_capabilities(emits_embeddings=True, embedding_dim=512),
        card=build_card(can_emit_embeddings=True, embedding_dim=1024),
    ),
    "registry_required": lambda: refuse_request(
        build_work(), card=build_card(taxa_registry_uri=None), registry=build_registry()
    ),
    "registry_fingerprint_mismatch": lambda: refuse_request(
        build_work(), card=build_card(), registry=build_registry(fingerprint=OTHER_FINGERPRINT)
    ),
    "head_class_not_in_registry": lambda: refuse_request(
        build_work(),
        card=build_head(classes=("owl", "barred-owl")),
        registry=build_registry(labels=("owl",)),
    ),
}


@pytest.mark.parametrize("name", sorted(REFUSING_CALLS))
def test_every_refusal_raises_engine_error_with_the_validate_request_stage(name):
    with pytest.raises(errors.EngineError) as exc:
        REFUSING_CALLS[name]()

    assert exc.value.code in errors.VALIDATE_REQUEST_FAILURES
    assert exc.value.stage == errors.VALIDATE_REQUEST


def test_every_refusal_code_is_reachable():
    raised = set()
    for refuse in REFUSING_CALLS.values():
        with pytest.raises(errors.EngineError) as exc:
            refuse()
        raised.add(exc.value.code)

    assert raised == set(errors.VALIDATE_REQUEST_FAILURES)
