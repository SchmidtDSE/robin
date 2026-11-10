"""Storage adapters for an object store, such as S3: `ObjectStoreFileProvider`,
`ObjectStoreArtifactWriter`, and `ObjectStoreTaskStorage`. They work through ondio."""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import ondio
from ondio.backends import StorageBackend
from robin_contracts.canonical import checksum_file
from robin_contracts.layout import artifact_path
from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord
from robin_contracts.tasks import output_root_problem

from robin_adapters.storage.location import LOCAL_STORE, split_location


class ObjectStoreFileProvider:
    """Downloads the object at a URI into a new folder under `directory`.

    `release` deletes the file and its folder. It builds one ondio backend for each
    store, such as one S3 bucket, with `backend_options`, on the first fetch from that
    store. It uses that backend for every fetch from the store after that. It builds no
    backend in `__init__`, so a command that fetches nothing needs no credentials.
    """

    def __init__(self, directory: Path, **backend_options: Any) -> None:
        self._directory = Path(directory).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._backend_options = backend_options
        self._backends: dict[str, StorageBackend] = {}
        self._fetched: set[Path] = set()

    def fetch(self, uri: str) -> Path:
        name = uri.rsplit("/", 1)[-1]
        # `.` and `..` would name the folder or its parent rather than a file in it.
        if name in ("", ".", ".."):
            raise ValueError(f"{uri!r} has no file name after its last '/'")
        backend = self._get_backend(uri)
        folder = self._directory / uuid.uuid4().hex
        folder.mkdir()
        path = folder / name
        try:
            ondio.download(uri, path, backend=backend)
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

    def _get_backend(self, uri: str) -> StorageBackend:
        store = ondio.locate(uri).store
        if store not in self._backends:
            self._backends[store] = ondio.get_backend(store, **self._backend_options)
        return self._backends[store]


class ObjectStoreArtifactWriter:
    """Publishes each file under an `s3://` root, at the path that `artifact_path` gives.

    Two writers that store one file at the same time are not refused: the upload that
    finishes last replaces the other. It builds one ondio backend for `root`, with
    `backend_options`, on the first `create`, and uses it for every call after that.
    """

    def __init__(self, root: str, **backend_options: Any) -> None:
        self._root = _check_output_root(root)
        self._backend_options = backend_options
        self._backend: StorageBackend | None = None

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
        backend = self._get_backend()
        if ondio.exists(uri, backend=backend):
            self._check_existing(uri, checksum, backend)
        else:
            ondio.upload(uri, source, backend=backend)
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

    def _check_existing(self, uri: str, checksum: str, backend: StorageBackend) -> None:
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "existing"
            ondio.download(uri, copy, backend=backend)
            existing = checksum_file(copy)
        if existing != checksum:
            raise FileExistsError(
                f"{uri} already holds {existing}; refusing to publish {checksum} there"
            )

    def _get_backend(self) -> StorageBackend:
        if self._backend is None:
            self._backend = ondio.get_backend(self._root, **self._backend_options)
        return self._backend


class ObjectStoreTaskStorage:
    """Writes and reads task files under `root`, such as `work.json`, through ondio.

    It builds one ondio backend for `root`, with `backend_options`, on the first
    `create` or `read`, and uses it for every call after that. It does not build the
    backend in `__init__`, so a command that never reads or writes a task file needs no
    storage credentials.
    """

    def __init__(self, root: str, **backend_options: Any) -> None:
        self._root = root
        self._backend_options = backend_options
        self._backend: StorageBackend | None = None

    def __repr__(self) -> str:
        # The options can hold an endpoint or a credential, so they are not shown.
        return f"ObjectStoreTaskStorage({self._root!r})"

    def create(self, uri: str, data: bytes) -> None:
        try:
            ondio.create(uri, data, backend=self._get_backend())
        except ondio.ObjectExistsError as exc:
            raise FileExistsError(f"{uri} already exists") from exc

    def read(self, uri: str) -> bytes | None:
        try:
            return ondio.read(uri, backend=self._get_backend())
        except ondio.ObjectNotFoundError:
            return None

    def get_worker_env_vars(self) -> dict[str, str]:
        """Returns the environment variables that make a worker's ondio calls use this
        storage's options, such as its endpoint.

        It builds no backend, so it needs no storage credentials.
        """
        return ondio.get_environment(self._root, **self._backend_options)

    def _get_backend(self) -> StorageBackend:
        if self._backend is None:
            self._backend = ondio.get_backend(self._root, **self._backend_options)
        return self._backend


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
