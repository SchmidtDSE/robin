"""Registered models: a card and its pinned files, fixed after registration."""

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import psycopg
from psycopg.types.json import Json
from pydantic import TypeAdapter, ValidationError

from robin_contracts.cards import HeadCard, ModelCard, card_digest, model_ref
from robin_contracts.local_files import get_local_path
from robin_contracts.tasks import output_version_id
from robin_contracts.work import REGISTRY_ROLE, PinnedFile, PinnedModel
from robin_models import registration
from robin_run_manager.database import Database, transaction
from robin_run_manager.errors import (
    INVALID_REQUEST,
    MODEL_CONFLICT,
    NOT_FOUND,
    UNKNOWN_MODEL,
    Refused,
)
from robin_run_manager.operator_files import field_problems, read_yaml_or_refuse
from robin_run_manager.pages import DEFAULT_LIMIT, Page, start_page


_COLUMNS = "model_id, card, card_digest, files, registered_by, registered_at"
_CARD_ADAPTER = TypeAdapter(ModelCard | HeadCard)


@dataclass(frozen=True)
class Registered:
    """The result of `register_model`. `created` is False if the model was already registered."""

    model_id: str
    card_digest: str
    created: bool


def get_built_in_card(model_id: str) -> ModelCard:
    """The card that robin-models ships for `model_id`, such as `owl/v4`.

    Raises `Refused` with `invalid_request` for a model it does not ship.
    """
    try:
        return registration.get_bundled_card(model_id)
    except KeyError:
        known = list(registration.get_built_in_model_ids())
        raise Refused(
            INVALID_REQUEST,
            f"{model_id} is not a built-in model. The built-in models are {', '.join(known)}. "
            "Give another model's card with --card.",
            details={"model_id": model_id, "built_in_models": known},
        ) from None


def get_roles_by_model() -> dict[str, frozenset[str]]:
    """The file roles that each built-in model and each head runtime always reads.

    Every model also takes the taxa registry, which it needs only for scores.
    """
    roles = {
        model_id: registration.get_file_roles(registration.get_bundled_card(model_id))
        for model_id in registration.get_built_in_model_ids()
    }
    return roles | {
        f"{runtime} heads": runtime_roles
        for runtime, runtime_roles in registration.get_head_runtime_roles().items()
    }


def read_card_file(path: Path) -> ModelCard | HeadCard:
    """Read a card from a YAML file whose fields are the card's fields.

    Raises `Refused` with `invalid_request` for a file that is not a valid card.
    """
    loaded = read_yaml_or_refuse(path, "model card")
    try:
        return _CARD_ADAPTER.validate_python(loaded)
    except ValidationError as error:
        # The details give each problem, so the Pydantic error adds nothing.
        raise Refused(
            INVALID_REQUEST,
            f"{path} is not a valid model card.",
            details={"file": str(path), "problems": field_problems(error, "file")},
        ) from None


def prepare_model(card: ModelCard | HeadCard, locations: Mapping[str, str]) -> PinnedModel:
    """Pin each file by its bytes, and check the files against the card, before registration.

    `locations` maps each role to an absolute local path or a `file:///` URI. Each file is
    read once for its sha256 and size. The registry file must be the one the card names.
    Does no database work. Raises `Refused` with `invalid_request`.
    """
    model_id = model_ref(card).id
    roles = _get_file_roles_or_refuse(card, model_id)
    _refuse_wrong_roles(model_id, roles, set(locations))
    pinned = {role: _pin(role, location) for role, location in locations.items()}
    registry = pinned.get(REGISTRY_ROLE)
    if registry is not None and card.taxa_registry_digest not in (None, registry.digest):
        raise Refused(
            INVALID_REQUEST,
            f"The {REGISTRY_ROLE} file is not the registry that {model_id}'s card names: "
            f"it hashes to {registry.digest}, and the card names {card.taxa_registry_digest}.",
            details={"role": REGISTRY_ROLE, "location": locations[REGISTRY_ROLE]},
        )
    return PinnedModel(card=card, files=pinned)


def prepare_built_in_model(model_id: str, locations: Mapping[str, str]) -> PinnedModel:
    """Prepare a built-in model, such as `owl/v4`, with the card that robin-models ships.

    As `prepare_model`, except for the registry. If `locations` has no `taxa_registry`, this
    uses the registry file that robin-models ships. The stored location is a `robin-models:`
    URI, which the worker reads from its own installed robin-models.
    Raises `Refused` with `invalid_request`.
    """
    model = prepare_model(get_built_in_card(model_id), locations)
    if REGISTRY_ROLE in model.files:
        return model
    location = registration.get_bundled_registry_location(model_id)
    path = registration.get_bundled_file(location)
    registry = _pin_path(REGISTRY_ROLE, location, location, path)
    return PinnedModel(card=model.card, files=model.files | {REGISTRY_ROLE: registry})


def register_model(database: Database, model: PinnedModel, *, actor: str) -> Registered:
    """Register `model` under its name and version.

    If the same card and files are already registered, nothing changes and `created` is
    False. The row keeps its first `registered_by`. Another card or other files under the
    same name and version are refused with `model_conflict`: a changed model needs a new
    version.
    """
    model_id = model_ref(model.card).id
    digest = card_digest(model.card)
    files = {role: pinned.model_dump(mode="json") for role, pinned in model.files.items()}
    with transaction(database) as connection:
        inserted = connection.execute(
            "INSERT INTO models (model_id, card, card_digest, files, registered_by) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (model_id) DO NOTHING RETURNING model_id",
            (model_id, Json(model.card.model_dump(mode="json")), digest, Json(files), actor),
        ).fetchone()
        # A new statement sees a row that a concurrent registration committed.
        stored = connection.execute(
            "SELECT card_digest, files FROM models WHERE model_id = %s", (model_id,)
        ).fetchone()
    differs = []
    if stored["card_digest"] != digest:
        differs.append("card")
    if stored["files"] != files:
        differs.append("files")
    if differs:
        raise Refused(
            MODEL_CONFLICT,
            f"Model {model_id} is already registered with another {' and '.join(differs)}. "
            "A changed model needs a new version.",
            details={"model_id": model_id, "differs": differs},
        )
    return Registered(model_id=model_id, card_digest=digest, created=inserted is not None)


def get_stored_model(connection: psycopg.Connection, model_id: str) -> PinnedModel:
    """The registered model `model_id`, read in the caller's transaction.

    Raises `Refused` with `unknown_model` if it is not registered.
    """
    row = connection.execute(
        "SELECT card, files FROM models WHERE model_id = %s", (model_id,)
    ).fetchone()
    if row is None:
        raise Refused(
            UNKNOWN_MODEL,
            f"Model {model_id} is not registered. Register it with `robin models register`.",
            details={"model_id": model_id},
        )
    return PinnedModel.model_validate({"card": row["card"], "files": row["files"]})


def list_models(
    database: Database, *, limit: int = DEFAULT_LIMIT, cursor: str | None = None
) -> Page:
    """A page of registered models, ordered by `model_id`."""
    paging = start_page("models", cursor, limit, {})
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            f"SELECT {_COLUMNS} FROM models "
            "WHERE %(after)s::text IS NULL OR model_id > %(after)s "
            "ORDER BY model_id LIMIT %(limit)s",
            {"after": paging.after, "limit": paging.limit + 1},
        ).fetchall()
    return paging.make_page(rows, key=lambda row: row["model_id"], item=_get_summary)


def show_model(database: Database, model_id: str) -> dict:
    """One model: its summary, its card, and its files with their locations.

    Raises `Refused` with `not_found`.
    """
    with transaction(database, read_only=True) as connection:
        row = connection.execute(
            f"SELECT {_COLUMNS} FROM models WHERE model_id = %s", (model_id,)
        ).fetchone()
    if row is None:
        raise Refused(
            NOT_FOUND, f"Model {model_id} is not registered.", details={"model_id": model_id}
        )
    return _get_summary(row) | {"card": row["card"], "files": row["files"]}


def _get_summary(row: dict) -> dict:
    card = _CARD_ADAPTER.validate_python(row["card"])
    return {
        "model_id": row["model_id"],
        "kind": "head" if isinstance(card, HeadCard) else "model",
        "card_digest": row["card_digest"],
        "output_version_id": output_version_id(card),
        "registered_by": row["registered_by"],
        "registered_at": row["registered_at"].isoformat(),
    }


def _get_file_roles_or_refuse(card: ModelCard | HeadCard, model_id: str) -> frozenset[str]:
    roles = registration.get_file_roles(card)
    if roles is None:
        raise Refused(
            INVALID_REQUEST,
            f"robin-models has no adapter for {model_id}, so Robin cannot run it.",
            details={"model_id": model_id},
        )
    return roles


def _refuse_wrong_roles(model_id: str, roles: frozenset[str], given: set[str]) -> None:
    missing = sorted(roles - given)
    unknown = sorted(given - roles - {REGISTRY_ROLE})
    if not missing and not unknown:
        return
    sentences = [
        f"{model_id} reads the roles {', '.join(sorted(roles))}, and {REGISTRY_ROLE} for scores."
    ]
    if missing:
        sentences.append(f"Missing: {', '.join(missing)}.")
    if unknown:
        sentences.append(f"Not read: {', '.join(unknown)}.")
    raise Refused(
        INVALID_REQUEST,
        " ".join(sentences),
        details={"model_id": model_id, "missing_roles": missing, "unknown_roles": unknown},
    )


def _pin(role: str, location: str) -> PinnedFile:
    uri, path = _get_local_file(role, location)
    return _pin_path(role, location, uri, path)


def _pin_path(role: str, location: str, uri: str, path: Path) -> PinnedFile:
    """Pin the file at `path` by its sha256 and size, with `uri` as its stored location.

    `role` and `location` are the names that the operator gave. Errors use them.
    """
    sha256 = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as file:
            while chunk := file.read(1 << 20):
                sha256.update(chunk)
                size += len(chunk)
    except (OSError, ValueError) as error:
        # ValueError: a path that the system cannot open, such as one with a null byte.
        raise Refused(
            INVALID_REQUEST,
            f"The {role} file {location} cannot be read: {getattr(error, 'strerror', None) or error}.",
            details={"role": role, "location": location},
        ) from None
    return PinnedFile(uri=uri, digest=f"sha256:{sha256.hexdigest()}", size_bytes=size)


def _get_local_file(role: str, location: str) -> tuple[str, Path]:
    """The URI Robin stores for `location`, and the path to read.

    Registration reads local files only. A bare path is stored as its `file:///` URI, and
    a `file:` URI is stored as written.
    """
    try:
        path = get_local_path(location)
    except ValueError:
        raise Refused(
            INVALID_REQUEST,
            f"The {role} location {location} is not a local file. Give an absolute path or a "
            "file:/// URI. Registering files on S3 is not supported.",
            details={"role": role, "location": location},
        ) from None
    if os.path.isabs(location):
        return PurePosixPath(location).as_uri(), path
    return location, path
