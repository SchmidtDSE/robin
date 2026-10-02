"""What the engine declines to attempt, decided entirely from in-memory values."""

import pytest

from robin_contracts.cards import (
    AudioGeometry,
    HeadCard,
    InferenceParam,
    ModelCard,
    ModelRef,
    RunnerResampled,
)
from robin_contracts.embedding_transforms import L2Norm
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
    ThresholdPolicy,
    TopKPolicy,
)
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)
from robin_inference_engine import errors
from robin_inference_engine.validate_request import refuse_request

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
        "window_duration": 3.0,
        "window_overlap": 0.0,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "taxa_registry_digest": REGISTRY_FINGERPRINT,
        "audio": AudioGeometry(
            downmix="mean",
            resampler=RunnerResampled(algorithm="soxr_hq"),
            pad="centre_crop_end_pad",
        ),
        "backend": "tensorflow",
        "embedding_transform": L2Norm(),
        "dtype": "float32",
    }
    return ModelCard(**(fields | overrides))


def build_embedding_card(**overrides) -> ModelCard:
    fields = {"can_emit_embeddings": True, "embedding_dim": 1024, "embedding_dtype": "float32"}
    return build_card(**(fields | overrides))


EMBEDDING_CARD = build_embedding_card()

# A model whose library returns only the scores at or above 0.005.
THRESHOLDED_CARD = build_card(min_detection_threshold=0.005)


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
    }
    return PinnedModel(**(fields | overrides))


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


UNLABELLED = {"files": {"weights": WEIGHTS}}
REGISTRY = build_registry()


def refused(
    work: InferenceWork, *, registry: TaxonRegistry | None = REGISTRY
) -> errors.EngineError:
    with pytest.raises(errors.EngineError) as exc:
        refuse_request(work, registry=registry)
    return exc.value


def accepted(work: InferenceWork, *, registry: TaxonRegistry | None = REGISTRY) -> bool:
    return refuse_request(work, registry=registry) is None


# ---------------------------------------------------------------------------
# Scores: what the card says the model emits, against what is requested.
# ---------------------------------------------------------------------------


def test_scores_from_a_card_that_emits_none_are_refused():
    error = refused(build_work(card=build_card(score_domain=None, taxa_registry_digest=None)))

    assert error.code == errors.SCORES_NOT_EMITTED
    assert "score_domain" in error.detail


def test_full_retention_from_a_model_with_a_floor_of_its_own_is_refused():
    # The scores below the model's floor do not exist, so no stream it emits is full.
    error = refused(build_work(card=THRESHOLDED_CARD))

    assert error.code == errors.FULL_RETENTION_REDUCED
    assert "0.005" in error.detail


@pytest.mark.parametrize("floor", [0.0, -0.5])
def test_full_retention_from_a_model_whose_floor_excludes_nothing_is_accepted(floor):
    assert accepted(build_work(card=build_card(min_detection_threshold=floor)))


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(build_scores(retention="thresholded", min_score=0.001), id="thresholded"),
        pytest.param(build_scores(retention="top_k", min_score=0.001, top_k=5), id="top_k"),
    ],
)
def test_a_floor_below_the_models_own_is_refused(request_):
    # The scores between the two floors were never emitted, so the stream would be
    # published under a floor it was not produced at.
    error = refused(build_work(card=THRESHOLDED_CARD, outputs=(request_,)))

    assert error.code == errors.SCORE_FLOOR_BELOW_MODEL_FLOOR
    assert "0.001" in error.detail and "0.005" in error.detail


@pytest.mark.parametrize("floor", [0.005, 0.5])
@pytest.mark.parametrize("retention", ["thresholded", "top_k"])
def test_a_floor_at_or_above_the_models_own_is_accepted(retention, floor):
    top_k = 5 if retention == "top_k" else None
    request_ = build_scores(retention=retention, min_score=floor, top_k=top_k)

    assert accepted(build_work(card=THRESHOLDED_CARD, outputs=(request_,)))


@pytest.mark.parametrize("k", [1, 3, 51, 10_000])
def test_any_k_is_accepted(k):
    request_ = build_scores(retention="top_k", min_score=0.0, top_k=k)

    assert accepted(build_work(outputs=(request_,)))


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(build_scores(retention="thresholded", min_score=0.0), id="thresholded"),
        pytest.param(build_scores(retention="top_k", min_score=0.2, top_k=5), id="top_k"),
    ],
)
def test_a_reduced_request_from_a_model_that_emits_every_score_is_accepted(request_):
    assert accepted(build_work(outputs=(request_,)))


@pytest.mark.parametrize("floor", [1.5, -0.1])
def test_a_min_score_outside_the_score_domain_is_refused(floor):
    from_request = build_work(outputs=(build_scores(retention="thresholded", min_score=floor),))
    assert refused(from_request).code == errors.MIN_SCORE_OUT_OF_DOMAIN

    from_policy = build_work(
        outputs=(build_scores(), build_detections(ThresholdPolicy(min_score=floor)))
    )
    assert refused(from_policy).code == errors.MIN_SCORE_OUT_OF_DOMAIN


# ---------------------------------------------------------------------------
# Embeddings: the card allows them, and the storage width is the recipe's.
# ---------------------------------------------------------------------------


def test_embeddings_forbidden_by_the_card_are_refused():
    work = build_work(card=build_card(can_emit_embeddings=False), outputs=(build_embeddings(),))

    assert refused(work).code == errors.EMBEDDINGS_NOT_EMITTED


def test_embeddings_the_card_allows_are_accepted():
    assert accepted(build_work(card=EMBEDDING_CARD, outputs=(build_embeddings(),)))


def test_an_embeddings_only_work_needs_no_registry():
    card = build_embedding_card(embedding_dim=1280)
    work = build_work(
        card=card,
        model=build_pinned_model(card, **UNLABELLED),
        outputs=(build_embeddings(),),
    )

    assert accepted(work, registry=None)


def test_an_embeddings_request_at_the_recipes_width_is_accepted():
    card = build_embedding_card(dtype="float16")

    assert accepted(
        build_work(card=card, outputs=(build_embeddings(storage_dtype="float16"),))
    )


@pytest.mark.parametrize(
    ("requested", "declared"), [("float16", "float32"), ("float32", "float16")]
)
def test_an_embeddings_request_at_another_width_is_refused(requested, declared):
    # The recipe's dtype is inside the recipe fingerprint, so honouring the request
    # would let two works sharing one fingerprint produce different bytes.
    card = build_embedding_card(dtype=declared)

    error = refused(build_work(card=card, outputs=(build_embeddings(storage_dtype=requested),)))

    assert error.code == errors.EMBEDDING_DTYPE_DISAGREES
    assert requested in error.detail and declared in error.detail


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_an_embeddings_request_naming_no_width_takes_the_recipes(dtype):
    card = build_embedding_card(dtype=dtype)

    assert accepted(build_work(card=card, outputs=(build_embeddings(),)))


@pytest.mark.parametrize(("source", "storage"), [("float32", "float16"), ("float16", "float32")])
def test_the_emitted_precision_need_not_be_the_stored_one(source, storage):
    card = build_embedding_card(embedding_dtype=source, dtype=storage)

    assert accepted(build_work(card=card, outputs=(build_embeddings(),)))


# ---------------------------------------------------------------------------
# Settings: the card declares every name a work may set, and its type.
# ---------------------------------------------------------------------------

TUNABLE_CARD = build_card(
    inference_params=(
        InferenceParam(name="overlap", type="float"),
        InferenceParam(name="top_n", type="int"),
    )
)


@pytest.mark.parametrize(
    ("card", "settings"),
    [
        pytest.param(build_card(), {"top_k": 5}, id="a_card_declaring_none"),
        pytest.param(TUNABLE_CARD, {"overlap": 0.5, "overlpa": 0.5}, id="a_misspelt_name"),
    ],
)
def test_a_setting_the_card_does_not_declare_is_refused(card, settings):
    error = refused(build_work(card=card, settings=settings))

    assert error.code == errors.SETTING_UNDECLARED
    undeclared = sorted(set(settings) - {"overlap"})[0]
    assert undeclared in error.detail


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("top_n", True, id="a_bool_for_an_int"),
        pytest.param("top_n", 1.5, id="a_float_for_an_int"),
        pytest.param("top_n", "8", id="text_for_an_int"),
        pytest.param("top_n", None, id="null_for_an_int"),
        pytest.param("overlap", 1, id="an_int_for_a_float"),
        pytest.param("overlap", False, id="a_bool_for_a_float"),
        pytest.param("overlap", "0.5", id="text_for_a_float"),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused(name, value):
    error = refused(build_work(card=TUNABLE_CARD, settings={name: value}))

    assert error.code == errors.SETTING_TYPE_MISMATCH
    assert name in error.detail and repr(value) in error.detail


def test_declared_settings_of_the_right_type_are_accepted():
    assert accepted(build_work(card=TUNABLE_CARD, settings={"overlap": 0.5, "top_n": 8}))


def test_no_settings_are_accepted_by_any_card():
    assert accepted(build_work(card=build_card(), settings={}))


# ---------------------------------------------------------------------------
# The registry, and heads.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("card", [build_card(), build_head()], ids=["backbone", "head"])
def test_scores_from_a_work_pinning_no_registry_file_are_refused(card):
    work = build_work(card=card, model=build_pinned_model(card, **UNLABELLED))

    error = refused(work, registry=None)

    assert error.code == errors.REGISTRY_REQUIRED
    assert REGISTRY_ROLE in error.detail


def test_a_loaded_registry_differing_from_its_pinned_file_is_refused():
    error = refused(build_work(), registry=build_registry(fingerprint=OTHER_FINGERPRINT))
    assert error.code == errors.REGISTRY_FINGERPRINT_MISMATCH
    assert REGISTRY_FILE.digest in error.detail
    assert OTHER_FINGERPRINT in error.detail

    assert refused(build_work(), registry=None).code == errors.REGISTRY_FINGERPRINT_MISMATCH


def a_work_pinning_another_registry_than_its_cards() -> errors.EngineError:
    other = PinnedFile(uri="s3://b/other.csv", digest=OTHER_FINGERPRINT, size_bytes=8)
    work = build_work(model=build_pinned_model(files={"weights": WEIGHTS, REGISTRY_ROLE: other}))
    return refused(work, registry=build_registry(fingerprint=OTHER_FINGERPRINT))


def test_a_pinned_registry_the_card_does_not_name_is_refused():
    error = a_work_pinning_another_registry_than_its_cards()

    assert (error.code, error.stage) == (
        errors.REGISTRY_DISAGREES_WITH_CARD,
        errors.VALIDATE_REQUEST,
    )
    assert OTHER_FINGERPRINT in error.detail
    assert REGISTRY_FINGERPRINT in error.detail
    assert "owl/1" in error.detail


def test_an_embeddings_only_work_on_a_card_naming_no_registry_may_pin_one():
    card = build_embedding_card(score_domain=None, taxa_registry_digest=None)

    assert accepted(build_work(card=card, outputs=(build_embeddings(),)))


def test_a_head_class_outside_the_registry_is_refused():
    error = refused(
        build_head_work(card=build_head(classes=("owl", "barred-owl"))),
        registry=build_registry(labels=("owl",)),
    )

    assert error.code == errors.HEAD_CLASS_NOT_IN_REGISTRY


def test_a_head_work_is_refused_because_no_head_runs_yet():
    error = refused(build_head_work())

    assert error.code == errors.HEAD_NOT_SUPPORTED
    assert "amy-head/1" in error.detail


def test_a_satisfiable_work_is_not_refused():
    assert accepted(build_work())


REFUSING_CALLS = {
    "scores_not_emitted": lambda: refused(build_work(card=build_card(score_domain=None, taxa_registry_digest=None))),
    "full_retention_reduced": lambda: refused(build_work(card=THRESHOLDED_CARD)),
    "score_floor_below_model_floor": lambda: refused(
        build_work(
            card=THRESHOLDED_CARD,
            outputs=(build_scores(retention="thresholded", min_score=0.001),),
        )
    ),
    "min_score_out_of_domain": lambda: refused(
        build_work(outputs=(build_scores(retention="thresholded", min_score=1.5),))
    ),
    "min_score_out_of_domain_from_a_top_k_policy": lambda: refused(
        build_work(outputs=(build_scores(), build_detections(TopKPolicy(k=5, min_score=2.0))))
    ),
    "embeddings_not_emitted": lambda: refused(
        build_work(card=build_card(can_emit_embeddings=False), outputs=(build_embeddings(),))
    ),
    "embedding_dtype_disagrees": lambda: refused(
        build_work(card=EMBEDDING_CARD, outputs=(build_embeddings(storage_dtype="float16"),))
    ),
    "setting_undeclared": lambda: refused(build_work(settings={"top_k": 5})),
    "setting_type_mismatch": lambda: refused(
        build_work(card=TUNABLE_CARD, settings={"top_n": True})
    ),
    "registry_required": lambda: refused(
        build_work(model=build_pinned_model(**UNLABELLED)), registry=None
    ),
    "registry_fingerprint_mismatch": lambda: refused(
        build_work(), registry=build_registry(fingerprint=OTHER_FINGERPRINT)
    ),
    "registry_disagrees_with_card": a_work_pinning_another_registry_than_its_cards,
    "head_class_not_in_registry": lambda: refused(
        build_head_work(card=build_head(classes=("owl", "barred-owl"))),
        registry=build_registry(labels=("owl",)),
    ),
    "head_not_supported": lambda: refused(build_head_work()),
}


@pytest.mark.parametrize("name", sorted(REFUSING_CALLS))
def test_every_refusal_raises_engine_error_with_the_validate_request_stage(name):
    error = REFUSING_CALLS[name]()

    assert error.code in errors.VALIDATE_REQUEST_FAILURES
    assert error.stage == errors.VALIDATE_REQUEST


def test_every_refusal_code_is_reachable():
    raised = {refuse().code for refuse in REFUSING_CALLS.values()}

    # run_work refuses this one itself, and its own tests reach it.
    raised_by_the_engine = {errors.EMBEDDING_INPUT_NOT_AVAILABLE}
    assert raised == set(errors.VALIDATE_REQUEST_FAILURES) - raised_by_the_engine
