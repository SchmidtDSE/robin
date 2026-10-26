"""Storage adapters for S3, through ondio: `OndioFileProvider` and `OndioArtifactWriter`."""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import ondio
from robin_contracts.canonical import checksum_file
from robin_contracts.layout import artifact_path
from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord
from robin_contracts.tasks import output_root_problem

from robin_adapters.storage.location import LOCAL_STORE, split_location


class OndioFileProvider:
    """Downloads the object at a URI into a new folder under `directory`.

    `release` deletes the file and its folder. Every ondio call receives
    `backend_options`, such as `profile_name` for S3.
    """

    def __init__(self, directory: Path, **backend_options: Any) -> None:
        self._directory = Path(directory).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._backend_options = backend_options
        self._fetched: set[Path] = set()

    def fetch(self, uri: str) -> Path:
        name = uri.rsplit("/", 1)[-1]
        # `.` and `..` would name the folder or its parent rather than a file in it.
        if name in ("", ".", ".."):
            raise ValueError(f"{uri!r} has no file name after its last '/'")
        folder = self._directory / uuid.uuid4().hex
        folder.mkdir()
        path = folder / name
        try:
            ondio.download(uri, path, **self._backend_options)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self._fetched.add(path)
        return path

    def release(self, path: Path) -> None:
        if path not in self._fetched:
            raise ValueError(f"{path} was not fetched by this provider, or is already released")
        self._fetched.remove(path)
        shutil.rmtree(path.parent)


class OndioArtifactWriter:
    """Publishes each file under an `s3://` root, at the path that `artifact_path` gives.

    Two writers that store one file at the same time are not refused: the upload that
    finishes last replaces the other. Every ondio call receives `backend_options`.
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
    """Returns `root` with no trailing `/`.

    Raises `ValueError` if `root` is not a usable `s3://` root.
    """
    place = split_location(root)
    if place is None or place[0] == LOCAL_STORE or output_root_problem(
        root, split_location=split_location
    ):
        raise ValueError(f"{root!r} is not a usable s3:// root with a bucket")
    store, path = place
    path = path.removesuffix("/")
    return f"{store}/{path}" if path else store
