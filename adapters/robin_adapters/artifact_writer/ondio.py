"""The `ArtifactWriter` port over an `s3://` root, through ondio."""

import tempfile
from pathlib import Path
from typing import Any

import ondio
from robin_contracts.canonical import checksum_file
from robin_contracts.layout import artifact_path
from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord
from robin_contracts.tasks import output_root_problem

from robin_adapters.location import LOCAL_STORE, split_location


class OndioArtifactWriter:
    """Publishes each file under an `s3://` root, uploading it from disk.

    The file goes at the path its kind and recording give. The same bytes already there
    succeed without an upload, and different bytes raise `FileExistsError`. Two writers
    storing one file at once are not refused: the later upload wins. `backend_options`
    are passed to every ondio call, which passes them to its backend, such as
    `profile_name` for S3.
    """

    def __init__(self, root: str, **backend_options: Any) -> None:
        self._root = _check_output_root(root)
        self._backend_options = backend_options

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
        hashed = checksum_file(source)
        if hashed != checksum:
            raise ValueError(f"{source} hashes to {hashed}, not the {checksum} supplied")
        uri = f"{self._root}/{artifact_path(kind, namespace, value)}"
        if ondio.exists(uri, **self._backend_options):
            self._check_existing(uri, checksum)
        else:
            ondio.upload(uri, source, **self._backend_options)
        return ArtifactRecord(
            kind=kind,
            contract_id=contract_id,
            namespace=namespace,
            value=value,
            uri=uri,
            checksum=checksum,
            size_bytes=source.stat().st_size,
            rows=rows,
        )

    def _check_existing(self, uri: str, checksum: str) -> None:
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "existing"
            ondio.download(uri, copy, **self._backend_options)
            existing = checksum_file(copy)
        if existing != checksum:
            raise FileExistsError(
                f"{uri} already holds {existing}; refusing to publish {checksum} there"
            )


def _check_output_root(root: str) -> str:
    """`root` as ondio reads it, with no trailing `/`, or `ValueError` if it is not usable.

    A usable root is a remote root that passes the result check's own root rule.
    """
    place = split_location(root)
    if place is None or place[0] == LOCAL_STORE or output_root_problem(
        root, split_location=split_location
    ):
        raise ValueError(f"{root!r} is not a usable s3:// root with a bucket")
    store, path = place
    path = path.removesuffix("/")
    return f"{store}/{path}" if path else store
