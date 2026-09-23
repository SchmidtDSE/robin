"""The provenance metadata shared by every artifact the engine writes.

It is stored in each artifact's header rather than in a database, so a reader holding
only the file can tell what produced it.
"""

from collections.abc import Iterable, Mapping

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.results import ArtifactContractId
from robin_contracts.specs import Recipe
from robin_contracts.work import InferenceWork, work_digest
from robin_inference_engine import errors

# Named because other modules use these keys on their own.
CONTRACT_KEY = "robin.contract"
REGISTRY_URI_KEY = "robin.registry_uri"
REGISTRY_FINGERPRINT_KEY = "robin.registry_fingerprint"
SCORE_DOMAIN_KEY = "robin.score_domain"
SCORE_RETENTION_KEY = "robin.score_retention"
SCORE_FLOOR_KEY = "robin.score_floor"
SCORE_TOP_K_KEY = "robin.score_top_k"
EMBEDDING_DIM_KEY = "robin.embedding_dim"
EMBEDDING_SOURCE_DTYPE_KEY = "robin.embedding_source_dtype"
EMBEDDING_STORAGE_DTYPE_KEY = "robin.embedding_storage_dtype"
BACKBONE_REF_KEY = "robin.backbone_ref"
BACKBONE_CARD_DIGEST_KEY = "robin.backbone_card_digest"

# Every artifact carries these keys, whatever its contract.
REQUIRED_KEYS: tuple[str, ...] = (
    CONTRACT_KEY,
    "robin.work_digest",
    "robin.recipe_fingerprint",
    "robin.recipe",
    "robin.model_ref",
    "robin.model_card_digest",
    "robin.model_file_digests",
    "robin.recording_map_uri",
    "robin.recording_map_checksum",
)

# Written together, and only when a registry supplied the label vocabulary. An
# artifact with no labels, such as embeddings, carries neither.
REGISTRY_KEYS: tuple[str, ...] = (REGISTRY_URI_KEY, REGISTRY_FINGERPRINT_KEY)

SCORE_KEYS: tuple[str, ...] = (SCORE_DOMAIN_KEY, SCORE_RETENTION_KEY)

EMBEDDING_KEYS: tuple[str, ...] = (
    EMBEDDING_DIM_KEY,
    EMBEDDING_SOURCE_DTYPE_KEY,
    EMBEDDING_STORAGE_DTYPE_KEY,
    BACKBONE_REF_KEY,
    BACKBONE_CARD_DIGEST_KEY,
)


def required_metadata(
    *,
    contract_id: ArtifactContractId,
    work: InferenceWork,
    recipe: Recipe,
    registry_uri: str | None,
    registry_fingerprint: str | None,
    recording_map_uri: str,
    recording_map_checksum: str,
) -> dict[bytes, bytes]:
    """The provenance keys every window artifact carries, as Arrow metadata."""
    _require_a_whole_registry_binding(registry_uri, registry_fingerprint)
    values: dict[str, str] = {
        CONTRACT_KEY: contract_id,
        "robin.work_digest": work_digest(work),
        # Not a duplicate: one key holds the recipe, the other its digest.
        "robin.recipe_fingerprint": recipe.id,
        "robin.recipe": canonical_json_bytes(recipe).decode("utf-8"),
        "robin.model_ref": work.model.ref.id,
        "robin.model_card_digest": work.model.ref.digest,
        "robin.model_file_digests": _model_file_digests(work),
        "robin.recording_map_uri": recording_map_uri,
        "robin.recording_map_checksum": recording_map_checksum,
    }
    if registry_uri is not None:
        values[REGISTRY_URI_KEY] = registry_uri
    if registry_fingerprint is not None:
        values[REGISTRY_FINGERPRINT_KEY] = registry_fingerprint
    return _encode(values)


def embedding_metadata(
    work: InferenceWork, *, dim: int, source_dtype: str, storage_dtype: str
) -> dict[bytes, bytes]:
    """The keys that make an opaque run of floats readable."""
    # Written even when they repeat the model keys: a head's model is not its backbone.
    backbone = work.model.backbone
    return _encode(
        {
            EMBEDDING_DIM_KEY: str(dim),
            EMBEDDING_SOURCE_DTYPE_KEY: source_dtype,
            EMBEDDING_STORAGE_DTYPE_KEY: storage_dtype,
            BACKBONE_REF_KEY: work.model.ref.id if backbone is None else backbone.id,
            BACKBONE_CARD_DIGEST_KEY: (
                work.model.ref.digest if backbone is None else backbone.digest
            ),
        }
    )


def score_metadata(request: ScoresRequest, *, score_domain: str) -> dict[bytes, bytes]:
    """The keys recording how much of the label vocabulary a score stream kept.

    The floor and k are taken from the request. A model whose own floor differs is
    refused before inference, so they describe the rows written, not only the request.
    """
    values: dict[str, str] = {
        SCORE_DOMAIN_KEY: score_domain,
        SCORE_RETENTION_KEY: request.retention,
    }
    if request.retention != "full":
        values[SCORE_FLOOR_KEY] = repr(request.min_score)
    if request.retention == "top_k":
        values[SCORE_TOP_K_KEY] = repr(request.top_k)
    return _encode(values)


def decode_metadata(raw: Mapping[bytes, bytes] | None) -> dict[str, str]:
    """Arrow's byte keys and values, decoded to text.

    Arrow reports a missing header as `None`, which becomes an empty mapping for the
    caller's key check to refuse. Bytes that are not valid UTF-8 raise a typed failure.
    """
    if raw is None:
        return {}
    decoded: dict[str, str] = {}
    for key, value in raw.items():
        name = _decode_text(key, what=f"the metadata key {key!r}")
        decoded[name] = _decode_text(value, what=f"the value of {name}")
    return decoded


def require_metadata_keys(
    metadata: Mapping[str, str], keys: Iterable[str], *, contract_id: str
) -> None:
    """Refuse an artifact missing any required key, naming every absent key."""
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise errors.EngineError(
            errors.ARTIFACT_METADATA_INCOMPLETE,
            errors.READ_INPUT_ARTIFACT,
            f"{contract_id} requires metadata this artifact does not carry: "
            f"{', '.join(missing)}",
        )


def _require_a_whole_registry_binding(
    registry_uri: str | None, registry_fingerprint: str | None
) -> None:
    """Refuse a registry uri without its fingerprint, or a fingerprint without one."""
    # Half a pair is the engine contradicting itself, not an absent binding.
    if (registry_uri is None) == (registry_fingerprint is None):
        return
    raise RuntimeError(
        f"{REGISTRY_URI_KEY} and {REGISTRY_FINGERPRINT_KEY} are written together or "
        f"not at all, got {registry_uri!r} and {registry_fingerprint!r}"
    )


def _decode_text(raw: bytes, *, what: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise errors.EngineError(
            errors.ARTIFACT_MALFORMED,
            errors.READ_INPUT_ARTIFACT,
            f"{what} is not valid UTF-8: {exc}",
        ) from exc


def _model_file_digests(work: InferenceWork) -> str:
    # Every pinned file, without its uri: a uri is a location, not an identity. The
    # work's file order survives, because canonical JSON sorts keys, never array items.
    return canonical_json_bytes(
        [one.model_dump(mode="json", exclude={"uri"}) for one in work.model.files]
    ).decode("utf-8")


def _encode(values: Mapping[str, str]) -> dict[bytes, bytes]:
    return {key.encode("utf-8"): value.encode("utf-8") for key, value in values.items()}
