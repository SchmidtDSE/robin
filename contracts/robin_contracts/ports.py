"""The ports: the services a caller passes to `run_work` for the engine to use.

A port is a protocol. The engine calls its methods, and an implementation satisfies it
by having the same methods, without importing it. The engine and robin's adapters may
not import each other, so the protocols are defined here, in the package both import.

Each port is named for what it does and what it handles: a `FileProvider` provides
files, and an `ArtifactWriter` writes artifacts. Robin's own implementations live in
`robin_adapters`, one subpackage per port and one module per backend, such as
`robin_adapters.file_provider.local`. A new port follows the same pattern.
"""

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
