"""What the engine declines to attempt, decided entirely from in-memory values."""

import pytest

from robin_contracts.cards import HeadCard, ModelCard, ModelRef, model_ref
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
from robin_contracts.specs import AudioSpec, Recipe, RunnerResampled
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)
from robin_inference_engine import errors
from robin_inference_engine.validate_request import refuse_instance, refuse_request

HEX = "0" * 64
RECORD_DIGEST = f"sha256:v1:{HEX}"
FILE_DIGEST = f"sha256:{HEX}"
REGISTRY_FINGERPRINT = "sha256:" + "a" * 64
OTHER_FINGERPRINT = "sha256:" + "b" * 64

BACKBONE_REF = ModelRef(name="perch", version="8", digest=f"sha256:v1:{'c' * 64}")

WEIGHTS = PinnedFile(uri="s3://b/owl.tflite", digest=FILE_DIGEST, size_bytes=8)
REGISTRY_FILE = PinnedFile(uri="s3://b/registry.csv", digest=REGISTRY_FINGERPRINT, size_bytes=8)


def build_card(**overrides) -> ModelCard:
    fields = {
        "model_name": "owl",
        "model_version": "1",
        "runtime": "tensorflow",
        "segment_duration": 3.0,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
    }
    return ModelCard(**(fields | overrides))


EMBEDDING_CARD = build_card(can_emit_embeddings=True, embedding_dim=1024)


def build_head(**overrides) -> HeadCard:
    fields = {
        "model_name": "amy-head",
        "model_version": "1",
        "backbone": BACKBONE_REF,
        "classes": ("owl",),
        "required_embedding_transform": L2Norm(),
    }
    return HeadCard(**(fields | overrides))


def build_pinned_model(card: ModelCard | HeadCard | None = None, **overrides) -> PinnedModel:
    fields = {
        "card": build_card() if card is None else card,
        "files": {"weights": WEIGHTS, REGISTRY_ROLE: REGISTRY_FILE},
        "registry_fingerprint": REGISTRY_FINGERPRINT,
    }
    return PinnedModel(**(fields | overrides))


def build_recipe(card: ModelCard | HeadCard | None = None, **overrides) -> Recipe:
    """A recipe naming the model `card` describes, the default card unless given."""
    fields = {
        "model": model_ref(build_card() if card is None else card),
        "backend": "tensorflow",
        "audio": AudioSpec(
            sample_rate=32000,
            window=12.0,
            hop=6.0,
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        "embedding_transform": L2Norm(),
        "dtype": "float32",
    }
    return Recipe(**(fields | overrides))


RECIPE = build_recipe()


def build_scores(**overrides) -> ScoresRequest:
    fields = {"contract_id": "robin.scores.arrow/1", "retention": "full"}
    return ScoresRequest(**(fields | overrides))


def build_embeddings(**overrides) -> EmbeddingsRequest:
    fields = {"contract_id": "robin.embeddings.arrow/1"}
    return EmbeddingsRequest(**(fields | overrides))


def build_detections(policy) -> DetectionsRequest:
    return DetectionsRequest(contract_id="robin.detections.parquet/1", policy=policy)


def build_work(card: ModelCard | HeadCard | None = None, **overrides) -> InferenceWork:
    fields = {
        "schema_version": "robin.inference-work/1",
        "recordings": (RecordingRef(namespace="soundhub", value="42", audio_uri="s3://b/4.wav"),),
        "model": build_pinned_model(card),
        "input": AudioInput(),
        "settings": {},
        "resources": {},
        "outputs": (build_scores(),),
    }
    return InferenceWork(**(fields | overrides))


def build_head_work(card: HeadCard | None = None, **overrides) -> InferenceWork:
    """A work running the head build_head returns, unless given another."""
    return build_work(card=build_head() if card is None else card, **overrides)


def with_card(work: InferenceWork, card: ModelCard | HeadCard) -> InferenceWork:
    """The same work, running another card."""
    model = work.model.model_dump() | {"card": card.model_dump()}
    return InferenceWork.model_validate(work.model_dump() | {"model": model})


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
        "supported_retention": frozenset({"full", "thresholded"}),
        "native_score_floor": None,
        "embedding_dim": None,
        "embedding_dtype": None,
    }
    return ModelCapabilities(**(fields | overrides))


def test_scores_from_an_adapter_that_emits_none_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(
                emits_scores=False, score_domain=None, supported_retention=frozenset()
            ),
            recipe=RECIPE,
        )

    assert exc.value.code == errors.SCORES_NOT_EMITTED


def test_retention_outside_supported_retention_is_refused():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.1),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(supported_retention=frozenset({"full"})),
            recipe=RECIPE,
        )

    assert exc.value.code == errors.RETENTION_UNSUPPORTED


def test_full_retention_with_an_excluding_floor_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.05),
            recipe=RECIPE,
        )

    assert exc.value.code == errors.FULL_RETENTION_REDUCED

    # A floor at the domain minimum excludes nothing. This is OWL's configuration and
    # refusing it would fail the stage that runs it.
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.0),
            recipe=RECIPE,
        )
        is None
    )


def test_full_retention_is_refused_when_the_instance_declares_only_top_k():
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(
                supported_retention=frozenset({"top_k"}), native_top_k=5
            ),
            recipe=RECIPE,
        )

    assert exc.value.code == errors.RETENTION_UNSUPPORTED


def test_a_thresholded_request_at_the_instance_floor_is_accepted():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.005),))

    assert (
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=0.005),
            recipe=RECIPE,
        )
        is None
    )


def test_a_thresholded_request_above_the_instance_floor_is_refused():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.5),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=0.005),
            recipe=RECIPE,
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
            recipe=RECIPE,
        )

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES


def test_a_reduced_request_against_an_instance_with_no_floor_is_refused():
    work = build_work(outputs=(build_scores(retention="thresholded", min_score=0.005),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            work,
            capabilities=build_capabilities(native_score_floor=None),
            recipe=RECIPE,
        )

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES


def build_capped(*, k: int, floor: float = 0.005) -> ModelCapabilities:
    return build_capabilities(
        supported_retention=frozenset({"top_k"}), native_top_k=k, native_score_floor=floor
    )


CAPPED_AT_FIVE = build_capped(k=5)


def test_a_top_k_request_checks_its_floor_too():
    work = build_work(
        outputs=(build_scores(retention="top_k", min_score=0.5, top_k=5),)
    )

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(work, capabilities=CAPPED_AT_FIVE, recipe=RECIPE)

    assert exc.value.code == errors.SCORE_FLOOR_DISAGREES

    at_the_floor = build_work(
        outputs=(build_scores(retention="top_k", min_score=0.005, top_k=5),)
    )
    assert (
        refuse_instance(at_the_floor, capabilities=CAPPED_AT_FIVE, recipe=RECIPE)
        is None
    )


def test_a_full_request_is_unaffected_by_the_floor_check():
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.0),
            recipe=RECIPE,
        )
        is None
    )

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(native_score_floor=0.005),
            recipe=RECIPE,
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
            recipe=RECIPE,
        )
    assert exc.value.code == errors.MIN_SCORE_OUT_OF_DOMAIN

    from_policy = build_work(
        outputs=(build_scores(), build_detections(ThresholdPolicy(min_score=floor)))
    )
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(from_policy, capabilities=build_capabilities(), recipe=RECIPE)
    assert exc.value.code == errors.MIN_SCORE_OUT_OF_DOMAIN


def test_embeddings_from_an_adapter_that_emits_none_is_refused():
    work = build_work(outputs=(build_embeddings(),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            with_card(work, EMBEDDING_CARD),
            capabilities=build_capabilities(emits_embeddings=False),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )

    assert exc.value.code == errors.EMBEDDINGS_NOT_EMITTED


def test_embeddings_forbidden_by_the_card_is_refused():
    work = build_work(outputs=(build_embeddings(),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_request(
            with_card(work, build_card(can_emit_embeddings=False)),
            registry=build_registry(),
        )

    assert exc.value.code == errors.EMBEDDINGS_NOT_EMITTED

    # The card is the half that refuses here: the instance says it can.
    assert (
        refuse_instance(
            with_card(work, EMBEDDING_CARD),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=1024),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )
        is None
    )


def test_an_embedding_dim_disagreeing_with_the_card_is_refused():
    work = build_work(outputs=(build_embeddings(),))
    card = EMBEDDING_CARD

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            with_card(work, card),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=512),
            recipe=build_recipe(card=card),
        )
    assert exc.value.code == errors.EMBEDDING_DIM_DISAGREES

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            with_card(work, card),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=None),
            recipe=build_recipe(card=card),
        )
    assert exc.value.code == errors.EMBEDDING_DIM_DISAGREES


@pytest.mark.parametrize("dimension", [None, 512])
def test_scores_only_work_refuses_inconsistent_embedding_dimensions(dimension):
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(card=EMBEDDING_CARD),
            capabilities=build_capabilities(
                emits_embeddings=True, embedding_dtype="float32", embedding_dim=dimension
            ),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )

    assert exc.value.code == errors.EMBEDDING_DIM_DISAGREES
    assert exc.value.stage == errors.VALIDATE_REQUEST


def test_scores_only_work_refuses_an_emission_its_card_forbids():
    # The card and the instance contradict each other. Which one is wrong is unknowable
    # here, and a work asking only for scores would otherwise carry the defect silently.
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(card=build_card(can_emit_embeddings=False)),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=512),
            recipe=build_recipe(card=build_card(can_emit_embeddings=False)),
        )

    assert exc.value.code == errors.EMBEDDING_EMISSION_DISAGREES
    assert exc.value.stage == errors.VALIDATE_REQUEST


def test_a_card_forbidding_embeddings_accepts_an_instance_that_emits_none():
    assert (
        refuse_instance(
            build_work(card=build_card(can_emit_embeddings=False)),
            capabilities=build_capabilities(emits_embeddings=False),
            recipe=build_recipe(card=build_card(can_emit_embeddings=False)),
        )
        is None
    )


def test_scores_only_work_accepts_consistent_embedding_dimensions():
    assert (
        refuse_instance(
            build_work(card=EMBEDDING_CARD),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=1024),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )
        is None
    )


def build_outputs(*, embeddings: bool):
    """A work asking for scores, optionally for embeddings too."""
    return (build_scores(), build_embeddings()) if embeddings else (build_scores(),)


@pytest.mark.parametrize("embeddings", [False, True])
def test_an_emitting_instance_that_declares_no_precision_is_refused(embeddings):
    # The header records the declared precision, so an instance that declares none
    # cannot have one written for it, whether or not this work asks for embeddings.
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(card=EMBEDDING_CARD, outputs=build_outputs(embeddings=embeddings)),
            capabilities=build_capabilities(
                emits_embeddings=True, embedding_dim=1024, embedding_dtype=None
            ),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )

    assert exc.value.code == errors.EMBEDDING_SOURCE_DTYPE_INVALID
    assert exc.value.stage == errors.VALIDATE_REQUEST
    assert "emits_embeddings" in exc.value.detail
    assert "float16" in exc.value.detail and "float32" in exc.value.detail


@pytest.mark.parametrize("embeddings", [False, True])
def test_an_emitting_instance_declaring_an_unsupported_precision_is_refused(embeddings):
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(card=EMBEDDING_CARD, outputs=build_outputs(embeddings=embeddings)),
            capabilities=build_capabilities(
                emits_embeddings=True, embedding_dim=1024, embedding_dtype="float64"
            ),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )

    assert exc.value.code == errors.EMBEDDING_SOURCE_DTYPE_INVALID
    assert "float64" in exc.value.detail
    assert "float16" in exc.value.detail and "float32" in exc.value.detail


@pytest.mark.parametrize("embeddings", [False, True])
def test_a_non_emitting_instance_that_declares_a_precision_is_refused(embeddings):
    # A precision it will not emit is the same contradiction as a width it will not
    # emit: one of the two declarations is wrong and neither side can say which.
    work = build_work(outputs=build_outputs(embeddings=embeddings))
    capabilities = build_capabilities(emits_embeddings=False, embedding_dtype="float32")

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(work, capabilities=capabilities, recipe=RECIPE)

    assert exc.value.code == errors.EMBEDDING_SOURCE_DTYPE_INVALID
    assert "emits_embeddings" in exc.value.detail
    assert "float32" in exc.value.detail


def test_a_non_emitting_instance_declaring_no_precision_is_accepted():
    # Only without an embeddings request: asking a non-emitting instance for embeddings
    # is already refused, and would hide whether this check let the instance through.
    assert (
        refuse_instance(
            build_work(outputs=build_outputs(embeddings=False)),
            capabilities=build_capabilities(
                emits_embeddings=False, embedding_dtype=None
            ),
            recipe=RECIPE,
        )
        is None
    )


@pytest.mark.parametrize("declared", ["float16", "float32"])
def test_an_emitting_instance_declaring_either_width_is_accepted(declared):
    assert (
        refuse_instance(
            build_work(card=EMBEDDING_CARD, outputs=(build_scores(), build_embeddings())),
            capabilities=build_capabilities(
                emits_embeddings=True, embedding_dim=1024, embedding_dtype=declared
            ),
            recipe=build_recipe(card=EMBEDDING_CARD),
        )
        is None
    )


UNLABELLED = {"files": {"weights": WEIGHTS}, "registry_fingerprint": None}


@pytest.mark.parametrize("card", [build_card(), build_head()], ids=["backbone", "head"])
def test_scores_from_a_work_pinning_no_registry_file_are_refused(card):
    work = build_work(card=card, model=build_pinned_model(card, **UNLABELLED))

    with pytest.raises(errors.EngineError) as exc:
        refuse_request(work, registry=None)

    assert exc.value.code == errors.REGISTRY_REQUIRED
    assert REGISTRY_ROLE in exc.value.detail


def test_a_registry_fingerprint_differing_from_the_pinned_one_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_request(build_work(), registry=build_registry(fingerprint=OTHER_FINGERPRINT))
    assert exc.value.code == errors.REGISTRY_FINGERPRINT_MISMATCH

    with pytest.raises(errors.EngineError) as exc:
        refuse_request(build_work(), registry=None)
    assert exc.value.code == errors.REGISTRY_FINGERPRINT_MISMATCH


def test_a_head_class_outside_the_registry_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        refuse_request(
            build_head_work(card=build_head(classes=("owl", "barred-owl"))),
            registry=build_registry(labels=("owl",)),
        )

    assert exc.value.code == errors.HEAD_CLASS_NOT_IN_REGISTRY


def test_a_satisfiable_work_is_not_refused():
    work = build_work()

    assert refuse_request(work, registry=build_registry()) is None
    assert (
        refuse_instance(work, capabilities=build_capabilities(), recipe=RECIPE)
        is None
    )


def test_an_embeddings_only_work_needs_no_registry():
    card = build_card(can_emit_embeddings=True, embedding_dim=1280)
    work = build_work(
        card=card,
        model=build_pinned_model(card, **UNLABELLED),
        outputs=(build_embeddings(),),
    )

    assert refuse_request(work, registry=None) is None
    assert (
        refuse_instance(
            work,
            capabilities=build_capabilities(
                emits_scores=False,
                emits_embeddings=True, embedding_dtype="float32",
                score_domain=None,
                supported_retention=frozenset(),
                embedding_dim=1280,
            ),
            recipe=build_recipe(card=card),
        )
        is None
    )


def test_an_embeddings_request_at_the_recipes_width_is_accepted():
    work = build_work(outputs=(build_embeddings(storage_dtype="float16"),))

    assert (
        refuse_instance(
            with_card(work, EMBEDDING_CARD),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=1024),
            recipe=build_recipe(card=EMBEDDING_CARD, dtype="float16"),
        )
        is None
    )


@pytest.mark.parametrize(
    ("requested", "declared"), [("float16", "float32"), ("float32", "float16")]
)
def test_an_embeddings_request_at_another_width_is_refused(requested, declared):
    # The recipe's dtype is inside the recipe fingerprint, so honouring the request
    # would let two works sharing one fingerprint produce different bytes.
    work = build_work(outputs=(build_embeddings(storage_dtype=requested),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            with_card(work, EMBEDDING_CARD),
            capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=1024),
            recipe=build_recipe(card=EMBEDDING_CARD, dtype=declared),
        )

    assert exc.value.code == errors.EMBEDDING_DTYPE_DISAGREES
    assert requested in exc.value.detail and declared in exc.value.detail


def test_a_work_requesting_no_embeddings_is_unaffected_by_the_width_check():
    assert (
        refuse_instance(
            build_work(),
            capabilities=build_capabilities(),
            recipe=build_recipe(dtype="float16"),
        )
        is None
    )


def test_a_satisfiable_head_work_is_not_refused():
    assert refuse_request(build_head_work(), registry=build_registry()) is None


@pytest.mark.parametrize(
    "named",
    [
        pytest.param(ModelRef(name="birdnet", version="2.4", digest=RECIPE.model.digest), id="another_name"),
        pytest.param(model_ref(build_card(min_detection_threshold=0.5)), id="another_digest"),
    ],
)
def test_a_recipe_naming_another_model_is_refused(named):
    # The same name with another digest is a recipe built from a stale copy of the card.
    work = build_work()

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(work, capabilities=build_capabilities(), recipe=build_recipe(model=named))

    assert exc.value.code == errors.RECIPE_MODEL_DISAGREES
    assert exc.value.stage == errors.VALIDATE_REQUEST
    assert named.digest in exc.value.detail
    assert model_ref(work.model.card).digest in exc.value.detail


def test_a_recipe_naming_a_heads_own_model_is_accepted():
    head = build_head()

    assert (
        refuse_instance(
            build_head_work(),
            capabilities=build_capabilities(),
            recipe=build_recipe(card=head),
        )
        is None
    )


def test_a_top_k_request_whose_k_differs_from_the_instances_cap_is_refused():
    work = build_work(outputs=(build_scores(retention="top_k", min_score=0.005, top_k=3),))

    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(work, capabilities=CAPPED_AT_FIVE, recipe=RECIPE)

    assert exc.value.code == errors.TOP_K_DISAGREES
    assert "3" in exc.value.detail and "5" in exc.value.detail


def test_a_top_k_request_matching_the_instances_cap_is_accepted():
    work = build_work(outputs=(build_scores(retention="top_k", min_score=0.005, top_k=5),))

    assert (
        refuse_instance(work, capabilities=CAPPED_AT_FIVE, recipe=RECIPE)
        is None
    )


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(build_scores(), id="full"),
        pytest.param(build_scores(retention="thresholded", min_score=0.005), id="thresholded"),
    ],
)
def test_an_unreduced_request_against_a_capped_instance_is_refused_by_retention(request_):
    with pytest.raises(errors.EngineError) as exc:
        refuse_instance(
            build_work(outputs=(request_,)),
            capabilities=CAPPED_AT_FIVE,
            recipe=RECIPE,
        )

    assert exc.value.code == errors.RETENTION_UNSUPPORTED


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_an_embeddings_request_naming_no_width_takes_the_recipes(dtype):
    assert (
        refuse_instance(
            build_work(card=EMBEDDING_CARD, outputs=(build_embeddings(),)),
            capabilities=build_capabilities(
                emits_embeddings=True, embedding_dtype="float32", embedding_dim=1024
            ),
            recipe=build_recipe(card=EMBEDDING_CARD, dtype=dtype),
        )
        is None
    )


REFUSING_CALLS = {
    "scores_not_emitted": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(
            emits_scores=False, score_domain=None, supported_retention=frozenset()
        ),
        recipe=RECIPE,
    ),
    "retention_unsupported": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(
            supported_retention=frozenset({"top_k"}), native_top_k=5
        ),
        recipe=RECIPE,
    ),
    "full_retention_reduced": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(native_score_floor=0.05),
        recipe=RECIPE,
    ),
    "score_floor_disagrees": lambda: refuse_instance(
        build_work(outputs=(build_scores(retention="thresholded", min_score=0.5),)),
        capabilities=build_capabilities(native_score_floor=0.005),
        recipe=RECIPE,
    ),
    "min_score_out_of_domain": lambda: refuse_instance(
        build_work(outputs=(build_scores(retention="thresholded", min_score=1.5),)),
        capabilities=build_capabilities(native_score_floor=1.5),
        recipe=RECIPE,
    ),
    "min_score_out_of_domain_from_a_top_k_policy": lambda: refuse_instance(
        build_work(outputs=(build_scores(), build_detections(TopKPolicy(k=5, min_score=2.0)))),
        capabilities=build_capabilities(),
        recipe=RECIPE,
    ),
    "embeddings_not_emitted_by_the_instance": lambda: refuse_instance(
        build_work(card=build_card(can_emit_embeddings=True, embedding_dim=8), outputs=(build_embeddings(),)),
        capabilities=build_capabilities(emits_embeddings=False),
        recipe=build_recipe(card=build_card(can_emit_embeddings=True, embedding_dim=8)),
    ),
    "embeddings_not_emitted_by_the_card": lambda: refuse_request(
        build_work(card=build_card(can_emit_embeddings=False), outputs=(build_embeddings(),)),
        registry=build_registry(),
    ),
    "embedding_emission_disagrees": lambda: refuse_instance(
        build_work(card=build_card(can_emit_embeddings=False)),
        capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=512),
        recipe=build_recipe(card=build_card(can_emit_embeddings=False)),
    ),
    "embedding_dim_disagrees": lambda: refuse_instance(
        build_work(card=EMBEDDING_CARD, outputs=(build_embeddings(),)),
        capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=512),
        recipe=build_recipe(card=EMBEDDING_CARD),
    ),
    "embedding_source_dtype_invalid": lambda: refuse_instance(
        build_work(card=EMBEDDING_CARD),
        capabilities=build_capabilities(
            emits_embeddings=True, embedding_dim=1024, embedding_dtype=None
        ),
        recipe=build_recipe(card=EMBEDDING_CARD),
    ),
    "embedding_dtype_disagrees": lambda: refuse_instance(
        build_work(card=EMBEDDING_CARD, outputs=(build_embeddings(storage_dtype="float16"),)),
        capabilities=build_capabilities(emits_embeddings=True, embedding_dtype="float32", embedding_dim=1024),
        recipe=build_recipe(card=EMBEDDING_CARD, dtype="float32"),
    ),
    "registry_required": lambda: refuse_request(
        build_work(model=build_pinned_model(**UNLABELLED)), registry=None
    ),
    "recipe_model_disagrees": lambda: refuse_instance(
        build_work(),
        capabilities=build_capabilities(),
        recipe=build_recipe(card=build_card(model_version="2")),
    ),
    "registry_fingerprint_mismatch": lambda: refuse_request(
        build_work(),
        registry=build_registry(fingerprint=OTHER_FINGERPRINT),
    ),
    "top_k_disagrees": lambda: refuse_instance(
        build_work(outputs=(build_scores(retention="top_k", min_score=0.005, top_k=3),)),
        capabilities=CAPPED_AT_FIVE,
        recipe=RECIPE,
    ),
    "head_class_not_in_registry": lambda: refuse_request(
        build_head_work(card=build_head(classes=("owl", "barred-owl"))),
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
