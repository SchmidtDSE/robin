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
    """Publishes a finished local file and returns the record that names it.

    `create` reads `source` before returning and must not depend on it afterwards: the
    engine deletes it. It is create-only and content-addressed, so publishing the same
    bytes again succeeds and different bytes under the same key are refused.
    """

    def create(
        self,
        *,
        kind: ArtifactKind,
        contract_id: ArtifactContractId,
        source: Path,
        checksum: str,
        rows: int,
    ) -> ArtifactRecord: ...
