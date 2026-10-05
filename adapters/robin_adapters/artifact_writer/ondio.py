"""The `ArtifactWriter` port over an `s3://` or `gs://` root, through ondio."""

import tempfile
from pathlib import Path
from typing import Any

import ondio
from robin_contracts.canonical import checksum_file
from robin_contracts.layout import artifact_path
from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord

_SCHEMES = ("s3", "gs")


class OndioArtifactWriter:
    """Publishes each file under an `s3://` or `gs://` root, uploading it from disk.

    The file goes at the path its kind and recording give. The same bytes already there
    succeed without an upload, and different bytes raise `FileExistsError`. Two writers
    storing one file at once are not refused: the later upload wins. `backend_options`
    are passed to every ondio call, which passes them to its backend, such as
    `profile_name` for S3.
    """

    def __init__(self, root: str, **backend_options: Any) -> None:
        self._root = _checked_root(root)
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


def _checked_root(root: str) -> str:
    """`root` without its trailing `/`, or `ValueError` if it is no `s3://` or `gs://` root.

    ondio cuts a uri at `?` or `#`, so neither may appear. Percent-encoding is kept.
    """
    stripped = root.rstrip("/")
    scheme, separator, rest = stripped.partition("://")
    if (
        not separator
        or scheme not in _SCHEMES
        or "?" in root
        or "#" in root
        or "" in rest.split("/")
    ):
        raise ValueError(f"{root!r} is not an s3:// or gs:// root with a bucket")
    return stripped
