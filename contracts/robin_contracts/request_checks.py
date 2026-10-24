"""The checks on a request that need only in-memory values.

The engine runs them before it attempts a work, and the control plane runs them before
it stores a configuration, so a bad request is refused with the same code in both places.
Nothing here loads, reads, opens or fetches anything. An engine that cannot do what
was asked says so rather than degrading to something adjacent and publishing it, so
every message names both sides of the disagreement.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from robin_contracts.cards import HeadCard, ModelCard, backbone_mismatch
from robin_contracts.output_contracts import (
    DetectionsRequest,
    EmbeddingsRequest,
    OutputRequest,
    ScoresRequest,
)
from robin_contracts.protocols import JsonScalar
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    EmbeddingArtifactInput,
    PinnedModel,
)

# The range every score a model emits must be in.
SCORE_RANGE = (0.0, 1.0)

SCORES_NOT_EMITTED = "scores_not_emitted"
EMBEDDINGS_NOT_EMITTED = "embeddings_not_emitted"
FULL_RETENTION_REDUCED = "full_retention_reduced"
SCORE_FLOOR_BELOW_MODEL_FLOOR = "score_floor_below_model_floor"
MIN_SCORE_OUT_OF_DOMAIN = "min_score_out_of_domain"
EMBEDDING_DTYPE_DISAGREES = "embedding_dtype_disagrees"
REGISTRY_REQUIRED = "registry_required"
REGISTRY_DISAGREES_WITH_CARD = "registry_disagrees_with_card"
SETTING_UNDECLARED = "setting_undeclared"
SETTING_TYPE_MISMATCH = "setting_type_mismatch"
WINDOW_OVERLAP_INVALID = "window_overlap_invalid"
INPUT_KIND_DISAGREES_WITH_CARD = "input_kind_disagrees_with_card"
HEAD_BACKBONE_DISAGREES = "head_backbone_disagrees"
HEAD_EMITS_NO_EMBEDDINGS = "head_emits_no_embeddings"

REQUEST_PROBLEM_CODES = (
    SCORES_NOT_EMITTED,
    EMBEDDINGS_NOT_EMITTED,
    FULL_RETENTION_REDUCED,
    SCORE_FLOOR_BELOW_MODEL_FLOOR,
    MIN_SCORE_OUT_OF_DOMAIN,
    EMBEDDING_DTYPE_DISAGREES,
    REGISTRY_REQUIRED,
    REGISTRY_DISAGREES_WITH_CARD,
    SETTING_UNDECLARED,
    SETTING_TYPE_MISMATCH,
    WINDOW_OVERLAP_INVALID,
    INPUT_KIND_DISAGREES_WITH_CARD,
    HEAD_BACKBONE_DISAGREES,
    HEAD_EMITS_NO_EMBEDDINGS,
)

# bool is an int subclass, so types are matched exactly: True is not an int setting.
_SETTING_TYPES: dict[str, type] = {"int": int, "float": float}

_Request = TypeVar("_Request", ScoresRequest, EmbeddingsRequest, DetectionsRequest)


@dataclass(frozen=True)
class RequestProblem:
    """Why a request is refused: one of `REQUEST_PROBLEM_CODES`, and a message."""

    code: str
    detail: str


def check_request_for_problem(
    model: PinnedModel,
    input: AudioInput | EmbeddingArtifactInput,
    settings: Mapping[str, JsonScalar],
    outputs: Sequence[OutputRequest],
) -> RequestProblem | None:
    """The first reason the engine would refuse this request, or None."""
    found = _problems(model, input, settings, outputs)
    return next((problem for problem in found if problem is not None), None)


def _problems(
    model: PinnedModel,
    input: AudioInput | EmbeddingArtifactInput,
    settings: Mapping[str, JsonScalar],
    outputs: Sequence[OutputRequest],
) -> Iterator[RequestProblem | None]:
    # Each check runs only when the checks before it found nothing, so a check may rely
    # on what an earlier one established.
    card = model.card

    yield _find_input_mismatch(input, card)
    if isinstance(card, HeadCard):
        assert isinstance(input, EmbeddingArtifactInput)
        yield _find_backbone_mismatch(input.backbone, card)
        yield _find_undeclared_settings(input.backbone_settings, input.backbone)
        yield _find_stalled_overlap(
            input.backbone_settings, input.backbone
        )
        yield _find_embeddings_from_head(outputs, card)
    yield _find_forbidden_embeddings(outputs, card)
    yield _find_unlabelled_scores(model, outputs)
    yield _find_registry_mismatch(model, card)
    yield _find_undeclared_settings(settings, card)
    yield _find_stalled_overlap(settings, card)
    scores = _requested(outputs, ScoresRequest)
    if scores is not None:
        yield _find_unavailable_scores(card)
        yield _find_full_scores_from_thresholded(scores, card)
    yield _find_floor_outside_domain(outputs)
    if scores is not None:
        yield _find_floor_below_model(scores, card)
    yield _find_undeclared_storage_width(outputs, card)


def _requested(outputs: Sequence[OutputRequest], kind: type[_Request]) -> _Request | None:
    return next((one for one in outputs if isinstance(one, kind)), None)


def _card_id(card: ModelCard | HeadCard) -> str:
    return f"{card.model_name}/{card.model_version}"


def _find_input_mismatch(
    input: AudioInput | EmbeddingArtifactInput, card: ModelCard | HeadCard
) -> RequestProblem | None:
    # A head reads the embeddings its backbone saved; a backbone reads audio.
    expected = "embedding_artifact" if isinstance(card, HeadCard) else "audio"
    if input.kind == expected:
        return None
    return RequestProblem(
        INPUT_KIND_DISAGREES_WITH_CARD,
        f"the work's input is {input.kind!r}, but card {_card_id(card)} reads "
        f"{expected!r} input",
    )


def _find_backbone_mismatch(
    backbone: ModelCard, head: HeadCard
) -> RequestProblem | None:
    mismatch = backbone_mismatch(head, backbone)
    if mismatch is None:
        return None
    return RequestProblem(HEAD_BACKBONE_DISAGREES, mismatch)


def _find_embeddings_from_head(
    outputs: Sequence[OutputRequest], head: HeadCard
) -> RequestProblem | None:
    if _requested(outputs, EmbeddingsRequest) is None:
        return None
    return RequestProblem(
        HEAD_EMITS_NO_EMBEDDINGS,
        f"embeddings were requested, but head {_card_id(head)} writes none: it reads "
        f"its backbone's",
    )


def _find_forbidden_embeddings(
    outputs: Sequence[OutputRequest], card: ModelCard | HeadCard
) -> RequestProblem | None:
    if _requested(outputs, EmbeddingsRequest) is None or not isinstance(card, ModelCard):
        return None
    if card.can_emit_embeddings:
        return None
    return RequestProblem(
        EMBEDDINGS_NOT_EMITTED,
        f"embeddings were requested but card {_card_id(card)} declares "
        f"can_emit_embeddings false",
    )


def _find_unlabelled_scores(
    model: PinnedModel, outputs: Sequence[OutputRequest]
) -> RequestProblem | None:
    if _requested(outputs, ScoresRequest) is None or REGISTRY_ROLE in model.files:
        return None
    return RequestProblem(
        REGISTRY_REQUIRED,
        f"scores were requested but the work pins no {REGISTRY_ROLE!r} file, so "
        f"nothing gives the model's output positions a meaning",
    )


def _find_registry_mismatch(
    model: PinnedModel, card: ModelCard | HeadCard
) -> RequestProblem | None:
    pinned = model.files.get(REGISTRY_ROLE)
    if pinned is None or card.taxa_registry_digest is None:
        return None
    if pinned.digest == card.taxa_registry_digest:
        return None
    return RequestProblem(
        REGISTRY_DISAGREES_WITH_CARD,
        f"the work pins {REGISTRY_ROLE} file digest {pinned.digest} but card "
        f"{_card_id(card)} declares taxa_registry_digest {card.taxa_registry_digest}",
    )


def _find_undeclared_settings(
    settings: Mapping[str, JsonScalar], card: ModelCard | HeadCard
) -> RequestProblem | None:
    # A head card declares no settings.
    params = card.inference_params if isinstance(card, ModelCard) else ()
    declared = {param.name: param.type for param in params}
    for name, value in settings.items():
        if name not in declared:
            return RequestProblem(
                SETTING_UNDECLARED,
                f"setting {name!r} is not declared by card {_card_id(card)}, which "
                f"declares {sorted(declared)}",
            )
        if type(value) is not _SETTING_TYPES[declared[name]]:
            return RequestProblem(
                SETTING_TYPE_MISMATCH,
                f"setting {name!r} is {value!r}, but card {_card_id(card)} declares it "
                f"{declared[name]}",
            )
    return None


def _find_stalled_overlap(
    settings: Mapping[str, JsonScalar], card: ModelCard | HeadCard
) -> RequestProblem | None:
    # The checks above leave a float here only when a backbone card declares the setting.
    overlap = settings.get("window_overlap")
    if type(overlap) is not float or 0.0 <= overlap < card.window_duration:
        return None
    return RequestProblem(
        WINDOW_OVERLAP_INVALID,
        f"setting 'window_overlap' is {overlap!r}, but card {_card_id(card)} needs it "
        f"at least 0.0 and less than its window_duration {card.window_duration}",
    )


def _find_unavailable_scores(card: ModelCard | HeadCard) -> RequestProblem | None:
    if card.score_domain is not None:
        return None
    return RequestProblem(
        SCORES_NOT_EMITTED,
        f"scores were requested but card {_card_id(card)} declares score_domain "
        f"null, so the model emits none",
    )


def _find_full_scores_from_thresholded(
    scores: ScoresRequest, card: ModelCard | HeadCard
) -> RequestProblem | None:
    # A floor at or below the domain minimum excludes nothing, so such a model emits
    # every label. Above it, the scores under the floor were never produced.
    minimum = SCORE_RANGE[0]
    floor = card.min_detection_threshold
    if scores.retention != "full" or floor <= minimum:
        return None
    return RequestProblem(
        FULL_RETENTION_REDUCED,
        f"full retention was requested but card {_card_id(card)} declares "
        f"min_detection_threshold {floor}, above the domain minimum {minimum}, so "
        f"the model does not emit every score",
    )


def _find_floor_below_model(
    scores: ScoresRequest, card: ModelCard | HeadCard
) -> RequestProblem | None:
    """The scores between the two floors were never produced, so no floor can be below it."""
    floor = card.min_detection_threshold
    if scores.retention == "full" or scores.min_score >= floor:
        return None
    return RequestProblem(
        SCORE_FLOOR_BELOW_MODEL_FLOOR,
        f"{scores.retention} retention was requested at floor {scores.min_score}, "
        f"below the min_detection_threshold {floor} card {_card_id(card)} declares",
    )


def _declared_floors(outputs: Sequence[OutputRequest]) -> Iterator[tuple[str, float | None]]:
    scores = _requested(outputs, ScoresRequest)
    if scores is not None:
        yield "the scores request", scores.min_score
    detections = _requested(outputs, DetectionsRequest)
    if detections is not None:
        yield "the detection policy", detections.policy.min_score


def _find_floor_outside_domain(
    outputs: Sequence[OutputRequest],
) -> RequestProblem | None:
    # A floor exists only beside a scores request, whose card was checked to emit
    # scores, so the score range bounds it.
    low, high = SCORE_RANGE
    for source, floor in _declared_floors(outputs):
        if floor is not None and not low <= floor <= high:
            return RequestProblem(
                MIN_SCORE_OUT_OF_DOMAIN,
                f"{source} declares min_score {floor}, outside the score range "
                f"[{low}, {high}]",
            )
    return None


def _find_undeclared_storage_width(
    outputs: Sequence[OutputRequest], card: ModelCard | HeadCard
) -> RequestProblem | None:
    # A head request for embeddings was refused above, so a request here is a
    # backbone's. The card's dtype is inside the recipe fingerprint, so honoring the
    # request instead would let two works sharing one fingerprint produce different bytes.
    embeddings = _requested(outputs, EmbeddingsRequest)
    if embeddings is None or embeddings.storage_dtype in (None, card.dtype):
        return None
    return RequestProblem(
        EMBEDDING_DTYPE_DISAGREES,
        f"embeddings were requested at storage_dtype {embeddings.storage_dtype!r}, "
        f"but card {_card_id(card)} declares dtype {card.dtype!r}",
    )
