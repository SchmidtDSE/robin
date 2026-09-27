"""What a caller supplies to run a work: a way to fetch files, and a place to publish."""

from pathlib import Path
from typing import Protocol

from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord


class FileAcquisition(Protocol):
    """Makes the bytes at a uri available as a local file, and releases it after.

    The engine checks each model file `fetch` returns against its pinned digest. Audio
    is used as fetched.
    """

    def fetch(self, uri: str) -> Path: ...
    def release(self, path: Path) -> None: ...


class ArtifactWriter(Protocol):
    """Publishes one recording's finished local file and returns the record that names it.

    The file is published at the location its kind and recording give under the
    writer's root. The writer is create-only there: publishing the same bytes again
    succeeds, and different bytes already at that location are refused. `create` reads
    `source` before returning and must not depend on it afterwards: the engine deletes it.
    """

    def create(
        self,
        *,
        kind: ArtifactKind,
        contract_id: ArtifactContractId,
        namespace: str,
        value: str,
        source: Path,
        checksum: str,
        rows: int,
    ) -> ArtifactRecord: ...
