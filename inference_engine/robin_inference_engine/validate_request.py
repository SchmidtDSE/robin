"""What the engine refuses to attempt, checked before any recording's input is fetched.

Nothing here loads, reads, opens or fetches anything: every argument is an
already-verified in-memory value. An engine that cannot do what was asked says so
rather than degrading to something adjacent and publishing it, so every message names
both sides of the disagreement.
"""

from collections.abc import Iterator

from robin_contracts.cards import HeadCard, ModelCard, model_ref
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.protocols import JsonScalar
from robin_contracts.registry import TaxonRegistry
from robin_contracts.work import REGISTRY_ROLE, InferenceWork
from robin_inference_engine import errors
from robin_inference_engine.accept_window import SCORE_RANGE
from robin_inference_engine.requested_outputs import (
    detections_request,
    embeddings_request,
    scores_request,
)

# bool is an int subclass, so types are matched exactly: True is not an int setting.
_SETTING_TYPES: dict[str, type] = {"int": int, "float": float}


def refuse_request(work: InferenceWork, *, registry: TaxonRegistry | None) -> None:
    """Everything the engine refuses before it builds the model, all read from the card."""
    card = work.model.card
    # First: whether the card is a head decides which of its fields the rest read.
    _refuse_an_input_the_card_does_not_read(work, card)
    if isinstance(card, HeadCard):
        _refuse_a_backbone_the_head_does_not_read(work.input.backbone, card)
        _refuse_embeddings_from_a_head(work, card)
    _refuse_embeddings_the_card_forbids(work, card)
    _refuse_unlabelled_scores(work)
    _refuse_a_substituted_registry(work, registry)
    _refuse_a_registry_the_card_does_not_pin(work, card)
    _refuse_settings_the_card_does_not_declare(work, card)
    scores = scores_request(work)
    if scores is not None:
        _refuse_scores_the_card_does_not_emit(card)
        _refuse_a_full_request_from_a_thresholded_stream(scores, card)
    _refuse_a_floor_outside_the_score_domain(work)
    if scores is not None:
        _refuse_a_floor_below_the_models_own(scores, card)
    _refuse_a_storage_width_the_card_does_not_declare(work, card)


def _refused(code: str, detail: str) -> errors.EngineError:
    return errors.EngineError(code, errors.VALIDATE_REQUEST, detail)


def _card_id(card: ModelCard | HeadCard) -> str:
    return f"{card.model_name}/{card.model_version}"


def _refuse_an_input_the_card_does_not_read(
    work: InferenceWork, card: ModelCard | HeadCard
) -> None:
    # A head reads the embeddings its backbone saved; a backbone reads audio.
    expected = "embedding_artifact" if isinstance(card, HeadCard) else "audio"
    if work.input.kind != expected:
        raise _refused(
            errors.INPUT_KIND_DISAGREES_WITH_CARD,
            f"the work's input is {work.input.kind!r}, but card {_card_id(card)} reads "
            f"{expected!r} input",
        )


def _refuse_a_backbone_the_head_does_not_read(backbone: ModelCard, head: HeadCard) -> None:
    given = model_ref(backbone)
    if given != head.backbone:
        raise _refused(
            errors.HEAD_BACKBONE_DISAGREES,
            f"the work's input comes from backbone {given.id} at {given.digest}, but head "
            f"{_card_id(head)} reads backbone {head.backbone.id} at {head.backbone.digest}",
        )
    if not backbone.can_emit_embeddings:
        raise _refused(
            errors.HEAD_BACKBONE_DISAGREES,
            f"backbone {given.id} declares can_emit_embeddings false, but head "
            f"{_card_id(head)} reads its embeddings",
        )
    if backbone.embedding_dim != head.embedding_dim:
        raise _refused(
            errors.HEAD_BACKBONE_DISAGREES,
            f"backbone {given.id} declares embedding_dim {backbone.embedding_dim}, but head "
            f"{_card_id(head)} takes {head.embedding_dim} values",
        )


def _refuse_embeddings_from_a_head(work: InferenceWork, head: HeadCard) -> None:
    if embeddings_request(work) is not None:
        raise _refused(
            errors.HEAD_EMITS_NO_EMBEDDINGS,
            f"embeddings were requested, but head {_card_id(head)} writes none: it reads "
            f"its backbone's",
        )


def _refuse_embeddings_the_card_forbids(
    work: InferenceWork, card: ModelCard | HeadCard
) -> None:
    if embeddings_request(work) is None or not isinstance(card, ModelCard):
        return
    if not card.can_emit_embeddings:
        raise _refused(
            errors.EMBEDDINGS_NOT_EMITTED,
            f"embeddings were requested but card {_card_id(card)} declares "
            f"can_emit_embeddings false",
        )


def _refuse_unlabelled_scores(work: InferenceWork) -> None:
    if scores_request(work) is None or REGISTRY_ROLE in work.model.files:
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


def _refuse_a_registry_the_card_does_not_pin(
    work: InferenceWork, card: ModelCard | HeadCard
) -> None:
    pinned = work.model.files.get(REGISTRY_ROLE)
    if pinned is None or card.taxa_registry_digest is None:
        return
    if pinned.digest != card.taxa_registry_digest:
        raise _refused(
            errors.REGISTRY_DISAGREES_WITH_CARD,
            f"the work pins {REGISTRY_ROLE} file digest {pinned.digest} but card "
            f"{_card_id(card)} declares taxa_registry_digest {card.taxa_registry_digest}",
        )


def _refuse_settings_the_card_does_not_declare(
    work: InferenceWork, card: ModelCard | HeadCard
) -> None:
    # A head card declares no settings.
    params = card.inference_params if isinstance(card, ModelCard) else ()
    declared = {param.name: param.type for param in params}
    for name, value in work.settings.items():
        if name not in declared:
            raise _refused(
                errors.SETTING_UNDECLARED,
                f"setting {name!r} is not declared by card {_card_id(card)}, which "
                f"declares {sorted(declared)}",
            )
        _refuse_a_setting_of_another_type(name, value, declared[name], card)


def _refuse_a_setting_of_another_type(
    name: str, value: JsonScalar, declared: str, card: ModelCard
) -> None:
    if type(value) is not _SETTING_TYPES[declared]:
        raise _refused(
            errors.SETTING_TYPE_MISMATCH,
            f"setting {name!r} is {value!r}, but card {_card_id(card)} declares it "
            f"{declared}",
        )


def _refuse_scores_the_card_does_not_emit(card: ModelCard | HeadCard) -> None:
    if card.score_domain is None:
        raise _refused(
            errors.SCORES_NOT_EMITTED,
            f"scores were requested but card {_card_id(card)} declares score_domain "
            f"null, so the model emits none",
        )


def _refuse_a_full_request_from_a_thresholded_stream(
    scores: ScoresRequest, card: ModelCard | HeadCard
) -> None:
    # A floor at or below the domain minimum excludes nothing, so such a model emits
    # every label. Above it, the scores under the floor were never produced.
    minimum = SCORE_RANGE[0]
    floor = card.min_detection_threshold
    if scores.retention == "full" and floor > minimum:
        raise _refused(
            errors.FULL_RETENTION_REDUCED,
            f"full retention was requested but card {_card_id(card)} declares "
            f"min_detection_threshold {floor}, above the domain minimum {minimum}, so "
            f"the model does not emit every score",
        )


def _refuse_a_floor_below_the_models_own(
    scores: ScoresRequest, card: ModelCard | HeadCard
) -> None:
    """The scores between the two floors were never produced, so no floor can be below it."""
    floor = card.min_detection_threshold
    if scores.retention != "full" and scores.min_score < floor:
        raise _refused(
            errors.SCORE_FLOOR_BELOW_MODEL_FLOOR,
            f"{scores.retention} retention was requested at floor {scores.min_score}, "
            f"below the min_detection_threshold {floor} card {_card_id(card)} declares",
        )


def _declared_floors(work: InferenceWork) -> Iterator[tuple[str, float | None]]:
    scores = scores_request(work)
    if scores is not None:
        yield "the scores request", scores.min_score
    detections = detections_request(work)
    if detections is not None:
        yield "the detection policy", detections.policy.min_score


def _refuse_a_floor_outside_the_score_domain(work: InferenceWork) -> None:
    # A floor exists only beside a scores request, whose card was checked to emit
    # scores, so the score range bounds it.
    low, high = SCORE_RANGE
    for source, floor in _declared_floors(work):
        if floor is not None and not low <= floor <= high:
            raise _refused(
                errors.MIN_SCORE_OUT_OF_DOMAIN,
                f"{source} declares min_score {floor}, outside the score range "
                f"[{low}, {high}]",
            )


def _refuse_a_storage_width_the_card_does_not_declare(
    work: InferenceWork, card: ModelCard | HeadCard
) -> None:
    # A head work asking for embeddings was refused above, so a request here is a
    # backbone's. The card's dtype is inside the recipe fingerprint, so honoring the
    # request instead would let two works sharing one fingerprint produce different bytes.
    embeddings = embeddings_request(work)
    if embeddings is None or embeddings.storage_dtype in (None, card.dtype):
        return
    raise _refused(
        errors.EMBEDDING_DTYPE_DISAGREES,
        f"embeddings were requested at storage_dtype {embeddings.storage_dtype!r}, "
        f"but card {_card_id(card)} declares dtype {card.dtype!r}",
    )
