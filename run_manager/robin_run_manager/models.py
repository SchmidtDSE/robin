"""Registered models: a card and its pinned files, fixed after registration."""

from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.types.json import Json
from pydantic import TypeAdapter, ValidationError

from robin_contracts.cards import HeadCard, ModelCard, card_digest, model_ref
from robin_contracts.tasks import output_version_id
from robin_contracts.work import PinnedModel
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
_CARD = TypeAdapter(ModelCard | HeadCard)


@dataclass(frozen=True)
class Registered:
    """The result of `register_model`. `created` is False if the model was already registered."""

    model_id: str
    card_digest: str
    created: bool


def read_registration(path: Path) -> PinnedModel:
    """Read a registration file: `card`, with the card's fields, and `files` by role.

    Raises `Refused` with `invalid_request` for a file that is not a valid registration.
    """
    loaded = read_yaml_or_refuse(path, "model registration")
    try:
        return PinnedModel.model_validate(loaded)
    except ValidationError as error:
        problems = field_problems(error, "file")
    raise Refused(
        INVALID_REQUEST,
        f"{path} is not a valid model registration.",
        details={"file": str(path), "problems": problems},
    )


def register_model(database: Database, model: PinnedModel, *, actor: str) -> Registered:
    """Register `model` under its name and version.

    The same card and files again return the existing row, with its first actor. Other
    content under the same name and version is refused with `model_conflict`: a changed
    model needs a new version.
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


def stored_model(connection: psycopg.Connection, model_id: str) -> PinnedModel:
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
    return paging.make_page(rows, key=lambda row: row["model_id"], item=_summary)


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
    return _summary(row) | {"card": row["card"], "files": row["files"]}


def _summary(row: dict) -> dict:
    card = _CARD.validate_python(row["card"])
    return {
        "model_id": row["model_id"],
        "kind": "head" if isinstance(card, HeadCard) else "model",
        "card_digest": row["card_digest"],
        "output_version_id": output_version_id(card),
        "registered_by": row["registered_by"],
        "registered_at": row["registered_at"].isoformat(),
    }
