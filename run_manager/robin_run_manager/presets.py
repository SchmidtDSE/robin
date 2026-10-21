"""Presets: a name for a configuration. A preset never changes after it is created."""

from dataclasses import dataclass

import psycopg
from psycopg.types.json import Json

from robin_run_manager.configuration import (
    Backbone,
    BackboneInput,
    Configuration,
    UnresolvedConfiguration,
    configuration_digest,
    resolve,
)
from robin_run_manager.database import Database, transaction
from robin_run_manager.errors import (
    INVALID_REQUEST,
    NOT_FOUND,
    PRESET_CONFLICT,
    UNKNOWN_CONFIGURATION,
    Refused,
)
from robin_run_manager.models import stored_model
from robin_run_manager.pages import DEFAULT_LIMIT, Page, start_page


_SELECT = (
    "SELECT p.name, p.configuration_digest, p.created_by, p.created_at, c.configuration "
    "FROM presets p JOIN configurations c ON c.digest = p.configuration_digest"
)


@dataclass(frozen=True)
class Created:
    """The result of `create_preset`. `created` is False if the preset already existed."""

    name: str
    configuration_digest: str
    created: bool


def create_preset(
    database: Database, name: str, unresolved: UnresolvedConfiguration, *, actor: str
) -> Created:
    """Resolve `unresolved` and give the configuration the name `name`.

    The same name and configuration again return the existing preset, with its first
    actor. The same name with another configuration is refused with `preset_conflict`,
    and nothing changes. An empty name is refused with `invalid_request`.
    """
    if not name.strip():
        raise Refused(INVALID_REQUEST, "A preset name must not be empty.", details={"name": name})
    with transaction(database) as connection:
        model = stored_model(connection, unresolved.model)
        backbone = None
        if isinstance(unresolved.input, BackboneInput):
            backbone = _stored_configuration(connection, unresolved.input.backbone_configuration)
        configuration = resolve(unresolved, model, backbone)
        digest = configuration_digest(configuration)
        connection.execute(
            "INSERT INTO configurations (digest, model_id, configuration) VALUES (%s, %s, %s) "
            "ON CONFLICT (digest) DO NOTHING",
            (digest, unresolved.model, Json(configuration.model_dump(mode="json"))),
        )
        inserted = connection.execute(
            "INSERT INTO presets (name, configuration_digest, created_by) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO NOTHING RETURNING name",
            (name, digest, actor),
        ).fetchone()
        # A new statement sees a preset that a concurrent create committed.
        stored = connection.execute(
            "SELECT configuration_digest FROM presets WHERE name = %s", (name,)
        ).fetchone()["configuration_digest"]
        if stored != digest:
            # Raised in the transaction, so the configuration above is not stored.
            raise Refused(
                PRESET_CONFLICT,
                f"Preset {name} already names another configuration, {stored}. "
                "Choose another name.",
                details={"name": name, "configuration_digest": stored},
            )
    return Created(name=name, configuration_digest=digest, created=inserted is not None)


def _stored_configuration(connection: psycopg.Connection, digest: str) -> Backbone:
    row = connection.execute(
        "SELECT model_id, configuration FROM configurations WHERE digest = %s", (digest,)
    ).fetchone()
    if row is None:
        raise Refused(
            UNKNOWN_CONFIGURATION,
            f"No stored configuration has the digest {digest}. Check backbone_configuration "
            "in the file.",
            details={"configuration_digest": digest},
        )
    return Backbone(
        configuration=Configuration.model_validate(row["configuration"]),
        model=stored_model(connection, row["model_id"]),
    )


def list_presets(
    database: Database, *, limit: int = DEFAULT_LIMIT, cursor: str | None = None
) -> Page:
    """A page of presets, ordered by name."""
    paging = start_page("presets", cursor, limit, {})
    with transaction(database, read_only=True) as connection:
        rows = connection.execute(
            f"{_SELECT} WHERE %(after)s::text IS NULL OR p.name > %(after)s "
            "ORDER BY p.name LIMIT %(limit)s",
            {"after": paging.after, "limit": paging.limit + 1},
        ).fetchall()
    return paging.make_page(rows, key=lambda row: row["name"], item=_summary)


def show_preset(database: Database, name: str) -> dict:
    """One preset: its summary and its resolved configuration, which holds no file locations.

    Raises `Refused` with `not_found`.
    """
    with transaction(database, read_only=True) as connection:
        row = connection.execute(f"{_SELECT} WHERE p.name = %s", (name,)).fetchone()
    if row is None:
        raise Refused(NOT_FOUND, f"There is no preset {name}.", details={"name": name})
    return _summary(row) | {"configuration": row["configuration"]}


def _summary(row: dict) -> dict:
    configuration = Configuration.model_validate(row["configuration"])
    return {
        "name": row["name"],
        "configuration_digest": row["configuration_digest"],
        "model_name": configuration.model.name,
        "model_version": configuration.model.version,
        "output_kinds": [output.kind for output in configuration.outputs],
        "created_by": row["created_by"],
        "created_at": row["created_at"].isoformat(),
    }
