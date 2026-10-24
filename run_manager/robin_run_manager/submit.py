"""Submit a run: store it in `preparing` with everything that preparation needs."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

import psycopg
from psycopg.types.json import Json, Jsonb

from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager.configuration import BackboneInput, UnresolvedConfiguration
from robin_run_manager.database import Database, transaction
from robin_run_manager.errors import INVALID_REQUEST, UNKNOWN_PRESET, Refused
from robin_run_manager.recording_inputs import RecordingFile
from robin_run_manager.stored_configurations import add_configuration, get_stored_configuration
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
    recordings: RecordingFile,
    actor: str,
    release_versions: Sequence[ModelOutputVersion] = RELEASE_OUTPUT_VERSIONS,
) -> Submitted:
    """Store a new run of `recordings` in `preparing`. Give a preset or a configuration.

    The run freezes its configuration, output versions, and recording list. It claims
    nothing: preparation does that later.
    """
    if (preset is None) == (configuration is None):
        raise Refused(INVALID_REQUEST, "Give a preset or a configuration, and not both.")
    with transaction(database) as connection:
        if preset is not None:
            digest = _get_preset_digest(connection, preset)
        else:
            _, digest = add_configuration(connection, configuration)
        stored = get_stored_configuration(connection, digest)
        cards = [stored.model.card]
        if isinstance(stored.configuration.input, BackboneInput):
            backbone = get_stored_configuration(
                connection, stored.configuration.input.backbone_configuration
            )
            cards.append(backbone.model.card)
        versions = get_frozen_output_versions(cards, release_versions)
        run_id = connection.execute(
            "INSERT INTO runs (kind, submitted_by) VALUES ('inference', %s) RETURNING run_id",
            (actor,),
        ).fetchone()["run_id"]
        observed = recordings.observed | {"recording_count": len(recordings.recordings)}
        connection.execute(
            "INSERT INTO inference_runs (run_id, configuration_digest, output_versions, "
            "observed_provenance) VALUES (%s, %s, %s, %s)",
            (run_id, digest, Jsonb(versions), Json(observed)),
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
    return Submitted(
        run_id=run_id, configuration_digest=digest, recording_count=len(recordings.recordings)
    )


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
