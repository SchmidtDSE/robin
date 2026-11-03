"""The storage port: how the engine gets input files and publishes artifacts, and how the
run manager writes and reads task files.

Each storage backend also has a settings class, a `StorageSettings`, which checks the
profile's storage section.
"""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord
from robin_contracts.tasks import get_unsafe_segment


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


def is_s3_uri(location: str) -> bool:
    """Whether `location` is an `s3://` URI that names a bucket."""
    bucket, _, _ = location.removeprefix("s3://").partition("/")
    return location.startswith("s3://") and bucket != ""


class StorageSettings(BaseModel):
    """The profile's storage section: where Robin writes a run's work files, result
    files, and outputs. Each storage kind has a subclass, which names its own fields.

    `root` is the folder that holds every run's files. A field that the kind does not
    have is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    root: str = Field(min_length=1)

    @classmethod
    def get_path_in_store(cls, root: str) -> str:
        """Returns the path of `root` inside its store, such as `robin/runs` for
        `s3://lab/robin/runs`. Raises `ValueError` if `root` does not have the form of
        the kind. The message names the form, never the value."""
        raise NotImplementedError

    @field_validator("root")
    @classmethod
    def _check_root(cls, root: str) -> str:
        # Storage may clean an empty, `.`, or `..` part, but a worker refuses a root that
        # keeps one, and a run keeps its root. So the profile refuses them all.
        if get_unsafe_segment(cls.get_path_in_store(root)) is not None:
            raise ValueError("give a root with no empty, '.', or '..' part")
        return root

    def get_backend_options(self) -> dict[str, str]:
        """Returns the options that the storage adapter needs, other than the root, such
        as an S3 endpoint."""
        return {}

    def __repr_args__(self):
        # Any field other than `kind` can hold a value that must not be shown, such as a
        # bucket name, so only the field names are shown.
        yield "kind", self.kind
        yield "fields", sorted(set(type(self).model_fields) - {"kind"})
