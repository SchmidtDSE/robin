"""The `ArtifactWriter` port over the local file system."""

import hashlib
import os
import uuid
from pathlib import Path
from typing import BinaryIO

from robin_contracts.canonical import checksum_file
from robin_contracts.layout import artifact_path
from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord

_CHUNK_BYTES = 1024 * 1024


class LocalArtifactWriter:
    """Publishes each file under `root`, at the path its kind and recording give.

    The file is copied to a hidden temporary name, checked, then hard-linked to its
    final name. A hard link fails if the name exists, so no reader sees a partial file
    and no file is replaced. Different bytes already there raise `FileExistsError`.
    The root must be on a file system that supports hard links.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

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
    ) -> ArtifactRecord:
        final = self._root / artifact_path(kind, namespace, value)
        final.parent.mkdir(parents=True, exist_ok=True)
        # The leading dot hides the file from dataset readers, and no reader's file
        # pattern matches the `.tmp` ending.
        temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(source, "rb") as original, open(temporary, "xb") as staged:
                copied = _copy_hashing(original, staged)
            if copied != checksum:
                raise ValueError(f"{source} hashes to {copied}, not the {checksum} supplied")
            _link_or_replay(temporary, final, checksum)
        finally:
            temporary.unlink(missing_ok=True)
        return ArtifactRecord(
            kind=kind,
            contract_id=contract_id,
            namespace=namespace,
            value=value,
            uri=final.as_uri(),
            checksum=checksum,
            size_bytes=final.stat().st_size,
            rows=rows,
        )


def _link_or_replay(temporary: Path, final: Path, checksum: str) -> None:
    try:
        os.link(temporary, final)
    except FileExistsError:
        existing = checksum_file(final)
        if existing != checksum:
            raise FileExistsError(
                f"{final} already holds {existing}; refusing to publish {checksum} there"
            ) from None


def _copy_hashing(source: BinaryIO, target: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(_CHUNK_BYTES):
        digest.update(chunk)
        target.write(chunk)
    return "sha256:" + digest.hexdigest()
