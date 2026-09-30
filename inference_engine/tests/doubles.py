"""Hermetic stand-ins for a model, the ports, and an installed model distribution.

Every double appends to one shared call log, so a single list shows the order in which
the engine called separate components. The doubles do not check their arguments; the
tests do.
"""

import hashlib
import importlib
import shutil
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.inputs import Input
from robin_contracts.layout import artifact_path
from robin_contracts.protocols import ModelContext
from robin_contracts.records import WindowOutput
from robin_contracts.results import ArtifactContractId, ArtifactKind, ArtifactRecord
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)

CallLog = list[tuple[object, ...]]

ENTRY_POINT_GROUP = "robin.models"

# A registry the loader accepts, declaring two labels.
REGISTRY_CSV = b"class_index,label,label_kind\n0,owl,non_taxonomic\n1,rain,non_taxonomic\n"

# Factories an installed test distribution hands out, keyed by the module that
# exposes them. Filled and emptied by `installed_factory`.
REGISTERED_FACTORIES: dict[str, Callable[[ModelContext], object]] = {}


class ScriptedModel:
    """A model that yields a fixed list of windows for each recording it is run on.

    `script[n]` is what the nth call to `run` yields. A script entry that is an
    exception is raised at that point, so a model can fail after yielding some windows.
    With `reuse_buffer`, every embedding is written into one array the model keeps and
    yields again, which a real adapter must never do.
    """

    def __init__(
        self,
        *,
        script: Sequence[Sequence[WindowOutput | Exception]],
        calls: CallLog,
        reuse_buffer: bool = False,
        fail_after_recording: Exception | None = None,
        fail_clean_up: Exception | None = None,
    ) -> None:
        self._script = script
        self._calls = calls
        self._runs = 0
        self._reuse_buffer = reuse_buffer
        self._buffer = None
        self._fail_after_recording = fail_after_recording
        self._fail_clean_up = fail_clean_up

    def run(self, input: Input) -> Iterator[WindowOutput]:
        self._calls.append(("run", input.path))
        items = self._script[self._runs] if self._runs < len(self._script) else ()
        self._runs += 1
        return self._yield(items)

    def after_recording(self) -> None:
        self._calls.append(("after_recording",))
        if self._fail_after_recording is not None:
            raise self._fail_after_recording

    def clean_up(self) -> None:
        self._calls.append(("clean_up",))
        if self._fail_clean_up is not None:
            raise self._fail_clean_up

    def _yield(self, items: Sequence[WindowOutput | Exception]) -> Iterator[WindowOutput]:
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield self._handed_over(item)

    def _handed_over(self, window: WindowOutput) -> WindowOutput:
        if window.embedding is None:
            return window
        if not self._reuse_buffer:
            # A fresh array per yield, so running one script twice shares nothing.
            return replace(window, embedding=window.embedding.copy())
        if self._buffer is None:
            self._buffer = window.embedding.copy()
        else:
            self._buffer[...] = window.embedding
        return replace(window, embedding=self._buffer)


class LocalFiles:
    """Returns the local file a test wrote for each uri.

    A uri in `missing` returns a path with no file behind it.
    """

    def __init__(
        self,
        name: str,
        paths: Mapping[str, Path],
        calls: CallLog,
        *,
        raise_on: Mapping[str, Exception] | None = None,
        missing: Sequence[str] = (),
        fail_release: Exception | None = None,
    ) -> None:
        self._name = name
        self._paths = paths
        self._calls = calls
        self._raise_on = raise_on or {}
        self._missing = missing
        self._fail_release = fail_release
        self.returned: list[Path] = []

    def fetch(self, uri: str) -> Path:
        self._calls.append((self._name, "fetch", uri))
        if uri in self._raise_on:
            raise self._raise_on[uri]
        path = self._paths[uri]
        if uri in self._missing:
            path = path.with_name(path.name + ".absent")
        self.returned.append(path)
        return path

    def release(self, path: Path) -> None:
        self._calls.append((self._name, "release", path))
        if self._fail_release is not None:
            raise self._fail_release


class CopyingWriter:
    """Copies each staged file to its recording's place under `destination`.

    It is create-only there: a file already holding the same bytes is left as it is
    and its uri recorded in `replayed`, and one holding other bytes is refused.
    `fail_on` counts creates from 1.
    """

    def __init__(
        self, destination: Path, calls: CallLog, *, fail_on: int | None = None
    ) -> None:
        self._destination = destination
        self._calls = calls
        self._fail_on = fail_on
        self._creates = 0
        self.replayed: list[str] = []

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
        self._calls.append(("create", kind, namespace, value))
        self._creates += 1
        if self._creates == self._fail_on:
            raise OSError(f"create number {self._creates} failed")
        published = self._destination / artifact_path(kind, namespace, value)
        if published.exists():
            if published.read_bytes() != source.read_bytes():
                raise FileExistsError(f"{published} already holds different bytes")
            self.replayed.append(published.as_uri())
        else:
            published.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, published)
        return ArtifactRecord(
            kind=kind,
            contract_id=contract_id,
            namespace=namespace,
            value=value,
            uri=published.as_uri(),
            checksum=checksum,
            size_bytes=published.stat().st_size,
            rows=rows,
        )


@contextmanager
def installed_distribution(
    directory: Path,
    *,
    name: str,
    entry_points: Mapping[str, str],
    modules: Mapping[str, str],
) -> Iterator[None]:
    """Install a distribution called `name` registering `entry_points` as models.

    It is put on `sys.path`, so the real `importlib.metadata` lookup finds it.
    """
    site = directory / f"site-{uuid.uuid4().hex}"
    dist_info = site / f"{name.replace('-', '_')}-0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0\n")
    lines = [f"[{ENTRY_POINT_GROUP}]"]
    lines += [f"{entry} = {value}" for entry, value in entry_points.items()]
    (dist_info / "entry_points.txt").write_text("\n".join(lines) + "\n")
    for module, source in modules.items():
        (site / f"{module}.py").write_text(source)

    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(site))
        for module in modules:
            sys.modules.pop(module, None)
        shutil.rmtree(site)
        importlib.invalidate_caches()


@contextmanager
def installed_factory(
    directory: Path,
    ref_id: str,
    factory: Callable[[ModelContext], object],
    *,
    distribution: str = "robin-test-model",
) -> Iterator[str]:
    """Install a distribution whose `ref_id` entry point is `factory`.

    Yields the entry point's module name; the module is imported only when the entry
    point is loaded.
    """
    module = f"robin_test_factory_{uuid.uuid4().hex}"
    REGISTERED_FACTORIES[module] = factory
    source = f"import doubles\n\nbuild = doubles.REGISTERED_FACTORIES[{module!r}]\n"
    try:
        with installed_distribution(
            directory,
            name=distribution,
            entry_points={ref_id: f"{module}:build"},
            modules={module: source},
        ):
            yield module
    finally:
        del REGISTERED_FACTORIES[module]


def pinned_file(path: Path, uri: str) -> PinnedFile:
    """Pin `path` under `uri` at its true size and digest."""
    data = path.read_bytes()
    return PinnedFile(
        uri=uri,
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


class WorkBuilder:
    """Writes a work's model files and audio under `directory`, pinned at their true digests."""

    def __init__(
        self,
        directory: Path,
        card: ModelCard | HeadCard,
        *,
        registry_csv: bytes | None = REGISTRY_CSV,
    ) -> None:
        self.card = card
        self.directory = directory
        self.model_paths: dict[str, Path] = {}
        self.audio_paths: dict[str, Path] = {}
        self.files: dict[str, PinnedFile] = {}
        self._write_model_file("weights", "weights.bin", b"not really weights")
        if registry_csv is not None:
            self._write_model_file(REGISTRY_ROLE, "registry.csv", registry_csv)

    def model_path(self, role: str) -> Path:
        return self.model_paths[self.files[role].uri]

    def recording(
        self, value: str, *, namespace: str = "test", audio: bytes | None = None, **overrides
    ) -> RecordingRef:
        """A recording whose audio is a small file written for it."""
        path = self.directory / "audio" / namespace / f"{value}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio if audio is not None else f"audio {namespace} {value}".encode())
        uri = f"file:///recordings/{namespace}/{value}.wav"
        self.audio_paths[uri] = path
        fields = {"namespace": namespace, "value": value, "audio_uri": uri}
        return RecordingRef(**(fields | overrides))

    def work(self, recordings: Sequence[RecordingRef], **overrides) -> InferenceWork:
        fields = {
            "schema_version": "robin.inference-work/1",
            "recordings": tuple(recordings),
            "model": PinnedModel(card=self.card, files=dict(self.files)),
            "input": AudioInput(),
            "settings": {"gain": 1.0},
            "resources": {},
            "outputs": (),
        }
        return InferenceWork(**(fields | overrides))

    def _write_model_file(self, role: str, name: str, data: bytes) -> None:
        path = self.directory / "model" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        uri = f"file:///model/{name}"
        self.model_paths[uri] = path
        self.files[role] = pinned_file(path, uri)
