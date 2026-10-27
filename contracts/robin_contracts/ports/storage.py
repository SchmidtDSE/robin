"""The storage port: how the engine gets input files and publishes artifacts, and how the
run manager writes and reads task files."""

from pathlib import Path
from typing import Protocol

from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord


class FileProvider(Protocol):
    """Makes the bytes at a uri available as a local file, and releases it after.

    The engine checks each model file `fetch` returns against its pinned digest, and
    each embeddings file against the checksum its recording names. Audio is used as
    fetched.
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


class TaskStorage(Protocol):
    """Writes and reads a task's files for the run manager, such as `work.json`.

    `create` writes `data` once at `uri` and makes missing folders. It raises
    `FileExistsError` if something is already there. A reader never sees part of a file.

    `read` returns the bytes at `uri`, or None only when the file is confirmed absent.
    It raises for any other failure.
    """

    def create(self, uri: str, data: bytes) -> None: ...
    def read(self, uri: str) -> bytes | None: ...
