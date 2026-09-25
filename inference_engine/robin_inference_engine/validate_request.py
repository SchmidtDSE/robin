"""What the engine refuses to attempt, checked before any audio is touched.

Neither entry point loads, reads, opens or fetches anything: every argument is an
already-verified in-memory value. An engine that cannot do what was asked says so
rather than degrading to something adjacent and publishing it, so every message names
both sides of the disagreement.
"""

from collections.abc import Iterator
from typing import get_args

from robin_contracts.cards import HeadCard, ModelCard, model_ref
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    ScoresRequest,
)
from robin_contracts.protocols import EmbeddingDtype, ModelCapabilities
from robin_contracts.registry import TaxonRegistry
from robin_contracts.specs import Recipe
from robin_contracts.work import REGISTRY_ROLE, InferenceWork
from robin_inference_engine import errors
from robin_inference_engine.accept_window import PROBABILITY_RANGE

EMBEDDING_DTYPES: tuple[str, ...] = get_args(EmbeddingDtype)


def refuse_request(work: InferenceWork, *, registry: TaxonRegistry | None) -> None:
    """Everything knowable before the model instance exists."""
    card = work.model.card
    _refuse_embeddings_the_card_forbids(work, card)
    _refuse_unlabelled_scores(work)
    _refuse_a_substituted_registry(work, registry)
    _refuse_head_classes_the_registry_does_not_declare(card, registry)


def refuse_instance(
    work: InferenceWork,
    *,
    capabilities: ModelCapabilities,
    recipe: Recipe,
) -> None:
    """What only a configured instance can answer, once the factory has returned it."""
    card = work.model.card
    _refuse_a_recipe_for_another_model(card, recipe)
    scores = _scores(work)
    if scores is not None:
        _refuse_scores_the_instance_does_not_emit(capabilities)
        _refuse_an_unsupported_retention(scores, capabilities)
        _refuse_a_reduced_full_stream(scores, capabilities)
        _refuse_a_floor_the_instance_does_not_impose(scores, capabilities)
        _refuse_a_k_the_instance_does_not_apply(scores, capabilities)
    _refuse_a_floor_outside_the_score_domain(work, capabilities)
    _refuse_an_embedding_emission_its_card_forbids(capabilities, card)
    _refuse_an_embedding_width_its_card_contradicts(capabilities, card)
    _refuse_an_embedding_precision_the_instance_will_not_emit(capabilities)
    _refuse_embeddings_the_instance_does_not_emit(work, capabilities)
    _refuse_a_storage_width_the_recipe_does_not_declare(work, recipe)


def _scores(work: InferenceWork) -> ScoresRequest | None:
    return next((one for one in work.outputs if isinstance(one, ScoresRequest)), None)


def _embeddings(work: InferenceWork) -> EmbeddingsRequest | None:
    return next((one for one in work.outputs if isinstance(one, EmbeddingsRequest)), None)


def _detections(work: InferenceWork) -> DetectionsRequest | None:
    return next((one for one in work.outputs if isinstance(one, DetectionsRequest)), None)


def _refused(code: str, detail: str) -> errors.EngineError:
    return errors.EngineError(code, errors.VALIDATE_REQUEST, detail)


def _card_id(card: ModelCard | HeadCard) -> str:
    return f"{card.model_name}/{card.model_version}"


def _refuse_a_recipe_for_another_model(card: ModelCard | HeadCard, recipe: Recipe) -> None:
    # Every artifact header records both the recipe and the work's model, so a recipe
    # naming another model would publish a header that contradicts itself. The digest
    # is compared too: an adapter built from a stale copy of the card has the right name.
    expected = model_ref(card)
    if recipe.model != expected:
        raise _refused(
            errors.RECIPE_MODEL_DISAGREES,
            f"the instance's recipe names model {recipe.model.id} ({recipe.model.digest}) "
            f"but the work runs {expected.id} ({expected.digest})",
        )


def _refuse_embeddings_the_card_forbids(
    work: InferenceWork, card: ModelCard | HeadCard
) -> None:
    if _embeddings(work) is None or not isinstance(card, ModelCard):
        return
    if not card.can_emit_embeddings:
        raise _refused(
            errors.EMBEDDINGS_NOT_EMITTED,
            f"embeddings were requested but card {_card_id(card)} declares "
            f"can_emit_embeddings false",
        )


def _refuse_unlabelled_scores(work: InferenceWork) -> None:
    if _scores(work) is None or REGISTRY_ROLE in work.model.files:
        return
    raise _refused(
        errors.REGISTRY_REQUIRED,
        f"scores were requested but the work pins no {REGISTRY_ROLE!r} file, so "
        f"nothing gives the model's output positions a meaning",
    )


def _refuse_a_substituted_registry(
    work: InferenceWork, registry: TaxonRegistry | None
) -> None:
    # Compared with the bytes that were loaded, not only the file that was verified,
    # so a file replaced between the two reads is still refused.
    pinned = work.model.files.get(REGISTRY_ROLE)
    if pinned is None:
        return
    if registry is None:
        raise _refused(
            errors.REGISTRY_FINGERPRINT_MISMATCH,
            f"the work pins {REGISTRY_ROLE} file digest {pinned.digest} but no registry "
            f"was loaded",
        )
    if registry.fingerprint != pinned.digest:
        raise _refused(
            errors.REGISTRY_FINGERPRINT_MISMATCH,
            f"the work pins {REGISTRY_ROLE} file digest {pinned.digest} but the loaded "
            f"registry's fingerprint is {registry.fingerprint}",
        )


def _refuse_head_classes_the_registry_does_not_declare(
    card: ModelCard | HeadCard, registry: TaxonRegistry | None
) -> None:
    if not isinstance(card, HeadCard):
        return
    if registry is None:
        raise _refused(
            errors.HEAD_CLASS_NOT_IN_REGISTRY,
            f"head {_card_id(card)} declares {len(card.classes)} classes but no "
            f"registry was supplied to declare them",
        )
    missing = [label for label in card.classes if label not in registry.labels]
    if missing:
        raise _refused(
            errors.HEAD_CLASS_NOT_IN_REGISTRY,
            f"head {_card_id(card)} declares class {missing[0]!r}, which registry "
            f"{registry.fingerprint} does not",
        )


def _refuse_scores_the_instance_does_not_emit(capabilities: ModelCapabilities) -> None:
    if not capabilities.emits_scores:
        raise _refused(
            errors.SCORES_NOT_EMITTED,
            "scores were requested but this instance declares emits_scores false",
        )


def _refuse_an_unsupported_retention(
    scores: ScoresRequest, capabilities: ModelCapabilities
) -> None:
    if scores.retention not in capabilities.supported_retention:
        raise _refused(
            errors.RETENTION_UNSUPPORTED,
            f"retention {scores.retention!r} is not in this instance's "
            f"supported_retention {sorted(capabilities.supported_retention)}",
        )


def _refuse_a_reduced_full_stream(
    scores: ScoresRequest, capabilities: ModelCapabilities
) -> None:
    # A floor at or below the domain minimum excludes nothing, so it does not by
    # itself disqualify full. A configured top-k is declared by native_top_k, which
    # ModelCapabilities only allows when supported_retention is exactly top_k, so the
    # retention check has already refused full against it.
    minimum = PROBABILITY_RANGE[0]
    floor = capabilities.native_score_floor
    if scores.retention == "full" and floor is not None and floor > minimum:
        raise _refused(
            errors.FULL_RETENTION_REDUCED,
            f"full retention was requested but this instance declares "
            f"native_score_floor {floor}, above the domain minimum {minimum}",
        )


def _refuse_a_floor_the_instance_does_not_impose(
    scores: ScoresRequest, capabilities: ModelCapabilities
) -> None:
    """A reduced stream is published under the request's floor, so they must be one floor."""
    # Exact, because both sides are declared constants rather than measurements: a
    # tolerance would let a genuinely different floor through in either direction.
    if scores.retention == "full":
        return
    if scores.min_score != capabilities.native_score_floor:
        raise _refused(
            errors.SCORE_FLOOR_DISAGREES,
            f"{scores.retention} retention would be published under the requested "
            f"floor {scores.min_score}, but this instance declares native_score_floor "
            f"{capabilities.native_score_floor}",
        )


def _refuse_a_k_the_instance_does_not_apply(
    scores: ScoresRequest, capabilities: ModelCapabilities
) -> None:
    """A top-k stream is published under the request's k, so they must be one k."""
    if scores.retention != "top_k":
        return
    if scores.top_k != capabilities.native_top_k:
        raise _refused(
            errors.TOP_K_DISAGREES,
            f"top_k retention would be published under the requested top_k "
            f"{scores.top_k}, but this instance declares native_top_k "
            f"{capabilities.native_top_k}",
        )


def _declared_floors(work: InferenceWork) -> Iterator[tuple[str, float | None]]:
    scores = _scores(work)
    if scores is not None:
        yield "the scores request", scores.min_score
    detections = _detections(work)
    if detections is not None:
        yield "the detection policy", detections.policy.min_score


def _refuse_a_floor_outside_the_score_domain(
    work: InferenceWork, capabilities: ModelCapabilities
) -> None:
    low, high = PROBABILITY_RANGE
    for source, floor in _declared_floors(work):
        if floor is None:
            continue
        if capabilities.score_domain != "probability":
            raise _refused(
                errors.MIN_SCORE_OUT_OF_DOMAIN,
                f"{source} declares min_score {floor} but this instance declares "
                f"score_domain {capabilities.score_domain!r}",
            )
        if not low <= floor <= high:
            raise _refused(
                errors.MIN_SCORE_OUT_OF_DOMAIN,
                f"{source} declares min_score {floor}, outside the probability domain "
                f"[{low}, {high}]",
            )


def _refuse_an_embedding_emission_its_card_forbids(
    capabilities: ModelCapabilities, card: ModelCard | HeadCard
) -> None:
    if not capabilities.emits_embeddings or not isinstance(card, ModelCard):
        return
    if not card.can_emit_embeddings:
        raise _refused(
            errors.EMBEDDING_EMISSION_DISAGREES,
            f"this instance emits embeddings while card {_card_id(card)} declares "
            f"can_emit_embeddings false",
        )


def _refuse_an_embedding_width_its_card_contradicts(
    capabilities: ModelCapabilities, card: ModelCard | HeadCard
) -> None:
    # An instance that emits embeddings must declare a width its card agrees with,
    # whether or not this work asks for embeddings: the contradiction is the defect,
    # and a work that never requests them would otherwise carry it silently.
    if not capabilities.emits_embeddings:
        return
    if capabilities.embedding_dim is None:
        raise _refused(
            errors.EMBEDDING_DIM_DISAGREES,
            "this instance emits embeddings but declares no embedding_dim",
        )
    declared = card.embedding_dim if isinstance(card, ModelCard) else None
    if declared is not None and declared != capabilities.embedding_dim:
        raise _refused(
            errors.EMBEDDING_DIM_DISAGREES,
            f"this instance declares embedding_dim {capabilities.embedding_dim} while "
            f"card {_card_id(card)} declares {declared}",
        )


def _refuse_an_embedding_precision_the_instance_will_not_emit(
    capabilities: ModelCapabilities,
) -> None:
    # Checked whether or not this work asks for embeddings, for the same reason the
    # width above it is: the artifact header records the declared precision, so an
    # instance declaring none, or declaring one it will not emit, contradicts itself.
    declared = capabilities.embedding_dtype
    accepted = ", ".join(EMBEDDING_DTYPES)
    if not capabilities.emits_embeddings:
        if declared is not None:
            raise _refused(
                errors.EMBEDDING_SOURCE_DTYPE_INVALID,
                f"this instance declares embedding_dtype {declared!r} while declaring "
                f"emits_embeddings false",
            )
        return
    if declared is None:
        raise _refused(
            errors.EMBEDDING_SOURCE_DTYPE_INVALID,
            f"this instance declares emits_embeddings true but no embedding_dtype; "
            f"it must declare one of {accepted}",
        )
    if declared not in EMBEDDING_DTYPES:
        raise _refused(
            errors.EMBEDDING_SOURCE_DTYPE_INVALID,
            f"this instance declares embedding_dtype {declared!r}, which is not one "
            f"of {accepted}",
        )


def _refuse_embeddings_the_instance_does_not_emit(
    work: InferenceWork, capabilities: ModelCapabilities
) -> None:
    if _embeddings(work) is not None and not capabilities.emits_embeddings:
        raise _refused(
            errors.EMBEDDINGS_NOT_EMITTED,
            "embeddings were requested but this instance declares "
            "emits_embeddings false",
        )


def _refuse_a_storage_width_the_recipe_does_not_declare(
    work: InferenceWork, recipe: Recipe
) -> None:
    # The recipe's dtype is inside the recipe fingerprint, so honoring the request
    # instead would let two works sharing one fingerprint produce different bytes.
    embeddings = _embeddings(work)
    if embeddings is None or embeddings.storage_dtype in (None, recipe.dtype):
        return
    raise _refused(
        errors.EMBEDDING_DTYPE_DISAGREES,
        f"embeddings were requested at storage_dtype {embeddings.storage_dtype!r}, "
        f"but recipe {recipe.id} declares dtype {recipe.dtype!r}",
    )
