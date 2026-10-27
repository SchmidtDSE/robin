"""Submit a run: store it in `preparing` with everything that preparation needs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Json, Jsonb

from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.configuration import BackboneInput, UnresolvedConfiguration
from robin_run_manager.database import Database, transaction
from robin_run_manager.errors import (
    INVALID_REQUEST,
    UNKNOWN_PRESET,
    UNSUPPORTED_LOCATION,
    Refused,
)
from robin_run_manager.location_forms import get_unsupported_locations
from robin_run_manager.recording_inputs import MAX_PROBLEMS, RecordingManifest
from robin_run_manager.stored_configurations import (
    StoredConfiguration,
    add_configuration,
    get_stored_configuration,
)
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS, get_frozen_output_versions


@dataclass(frozen=True)
class Submitted:
    """The result of `submit_run`."""

    run_id: UUID
    configuration_digest: str
    recording_count: int


def submit_run(
    database: Database,
    *,
    preset: str | None,
    configuration: UnresolvedConfiguration | None,
    recordings: RecordingManifest,
    execution_settings: Mapping[str, Any],
    actor: str,
    release_versions: Sequence[ModelOutputVersion] = RELEASE_OUTPUT_VERSIONS,
) -> Submitted:
    """Store a new run of `recordings` in `preparing`. Give a preset or a configuration.

    The run freezes its configuration, output versions, recording list, and
    `execution_settings`, which `get_execution_settings` makes from the profile. It
    claims nothing: preparation does that later. Each recording and model file must have
    a location form that the settings' compute kind can read.
    """
    if (preset is None) == (configuration is None):
        raise Refused(INVALID_REQUEST, "Give a preset or a configuration, and not both.")
    with transaction(database) as connection:
        digest = _get_configuration_digest(connection, preset, configuration)
        stored = get_stored_configuration(connection, digest)
        versions = get_frozen_output_versions(_get_cards(connection, stored), release_versions)
        _check_locations(execution_settings["compute"]["kind"], recordings, stored)
        run_id = _insert_run(connection, actor, execution_settings, digest, versions, recordings)
    return Submitted(
        run_id=run_id, configuration_digest=digest, recording_count=len(recordings.recordings)
    )


def _get_configuration_digest(
    connection: psycopg.Connection,
    preset: str | None,
    configuration: UnresolvedConfiguration | None,
) -> str:
    """The digest of the preset's configuration, or of `configuration`, stored if it is new."""
    if preset is not None:
        return _get_preset_digest(connection, preset)
    _, digest = add_configuration(connection, configuration)
    return digest


def _get_cards(
    connection: psycopg.Connection, stored: StoredConfiguration
) -> list[ModelCard | HeadCard]:
    """The card of the configuration's model, and for a head, also its backbone's card."""
    cards = [stored.model.card]
    if isinstance(stored.configuration.input, BackboneInput):
        backbone = get_stored_configuration(
            connection, stored.configuration.input.backbone_configuration
        )
        cards.append(backbone.model.card)
    return cards


def _check_locations(
    compute_kind: str, recordings: RecordingManifest, stored: StoredConfiguration
) -> None:
    """Refuse the run if `compute_kind` cannot read a recording or a model file."""
    unsupported = get_unsupported_locations(compute_kind, [
        *(("recording", recording.audio_uri) for recording in recordings.recordings),
        *(("model_file", pinned.uri) for pinned in stored.model.files.values()),
    ])
    if not unsupported:
        return
    first = unsupported[0]
    role = first["role"].replace("_", " ")
    forms = " or ".join(first["accepted_forms"])
    more = (
        f" In all, {len(unsupported)} locations cannot be read." if len(unsupported) > 1 else ""
    )
    raise Refused(
        UNSUPPORTED_LOCATION,
        f"Compute kind {compute_kind} cannot read the {role} {first['location']}. "
        f"Give {forms}.{more}",
        details={
            "compute_kind": compute_kind,
            "problems": unsupported[:MAX_PROBLEMS],
            "truncated": len(unsupported) > MAX_PROBLEMS,
        },
    )


def _insert_run(
    connection: psycopg.Connection,
    actor: str,
    execution_settings: Mapping[str, Any],
    digest: str,
    versions: dict[str, int],
    recordings: RecordingManifest,
) -> UUID:
    """Store the run, its frozen configuration and versions, and its recording list."""
    run_id = connection.execute(
        "INSERT INTO runs (kind, submitted_by, execution_settings) "
        "VALUES ('inference', %s, %s) RETURNING run_id",
        (actor, Json(execution_settings)),
    ).fetchone()["run_id"]
    provenance = recordings.provenance | {"recording_count": len(recordings.recordings)}
    connection.execute(
        "INSERT INTO inference_runs (run_id, configuration_digest, output_versions, "
        "provenance) VALUES (%s, %s, %s, %s)",
        (run_id, digest, Jsonb(versions), Json(provenance)),
    )
    with connection.cursor().copy(
        "COPY run_recordings (run_id, namespace, value, audio_uri, duration_seconds) "
        "FROM STDIN"
    ) as copy:
        for recording in recordings.recordings:
            copy.write_row((
                run_id,
                recording.namespace,
                recording.value,
                recording.audio_uri,
                recording.duration_seconds,
            ))
    return run_id


def _get_preset_digest(connection: psycopg.Connection, name: str) -> str:
    row = connection.execute(
        "SELECT configuration_digest FROM presets WHERE name = %s", (name,)
    ).fetchone()
    if row is None:
        raise Refused(
            UNKNOWN_PRESET,
            f"There is no preset {name}. See the presets with `robin presets list`.",
            details={"name": name},
        )
    return row["configuration_digest"]
