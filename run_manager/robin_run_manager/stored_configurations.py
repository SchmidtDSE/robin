"""Stored configurations: resolve one and store it, or read one back by its digest."""

from typing import NamedTuple

import psycopg
from psycopg.types.json import Json

from robin_contracts.work import AudioInput, EmbeddingArtifactInput, PinnedModel
from robin_run_manager.configuration import (
    BackboneInput,
    Configuration,
    ConfigurationInput,
    UnresolvedConfiguration,
    get_configuration_digest,
    get_embedding_input,
    resolve_configuration,
)
from robin_run_manager.errors import UNKNOWN_CONFIGURATION, Refused
from robin_run_manager.models import get_stored_model


class StoredConfiguration(NamedTuple):
    """A stored configuration and its model."""

    configuration: Configuration
    model: PinnedModel


def add_configuration(
    connection: psycopg.Connection, unresolved: UnresolvedConfiguration
) -> tuple[Configuration, str]:
    """Resolve `unresolved`, store it if it is new, and return it with its digest.

    For a head, the backbone configuration must be stored already.
    """
    model = get_stored_model(connection, unresolved.model)
    work_input = get_work_input(connection, unresolved.input)
    configuration = resolve_configuration(unresolved, model, work_input)
    digest = get_configuration_digest(configuration)
    connection.execute(
        "INSERT INTO configurations (digest, model_id, configuration) VALUES (%s, %s, %s) "
        "ON CONFLICT (digest) DO NOTHING",
        (digest, unresolved.model, Json(configuration.model_dump(mode="json"))),
    )
    return configuration, digest


def get_stored_configuration(connection: psycopg.Connection, digest: str) -> StoredConfiguration:
    """The stored configuration `digest`, and its model.

    Raises `Refused` with `unknown_configuration` if no configuration has the digest.
    """
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
    return StoredConfiguration(
        configuration=Configuration.model_validate(row["configuration"]),
        model=get_stored_model(connection, row["model_id"]),
    )


def get_work_input(
    connection: psycopg.Connection, configuration_input: ConfigurationInput
) -> AudioInput | EmbeddingArtifactInput:
    """The input that the engine reads for `configuration_input`.

    The audio input is used as written. A head reads the embeddings of its backbone
    configuration, which must be stored already.
    """
    if not isinstance(configuration_input, BackboneInput):
        return configuration_input
    backbone = get_stored_configuration(connection, configuration_input.backbone_configuration)
    return get_embedding_input(backbone.configuration, backbone.model)
