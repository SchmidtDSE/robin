"""A test model, a builder for work files, and helpers to run the worker.

The test environment has no ML runtime, so the worker's tests run a small model
written in plain Python. It is installed as a distribution with a `robin.models`
entry point, so the engine finds it by its normal lookup.
"""

import hashlib
import importlib
import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.tasks import InferenceTask, ModelOutputVersion
from robin_contracts.work import (
    REGISTRY_ROLE,
    AudioInput,
    InferenceWork,
    PinnedFile,
    PinnedModel,
    RecordingRef,
)

TEST_MODEL_ID = "model:test-model/1"
TEST_VERSIONS = {TEST_MODEL_ID: 1}
TEST_MODEL_MODULE = "robin_worker_test_model"
STARTED_VARIABLE = "ROBIN_TEST_MODEL_STARTED"

REGISTRY_CSV = b"class_index,label,label_kind\n0,owl,non_taxonomic\n1,rain,non_taxonomic\n"

# Yields one window for each recording. When STARTED_VARIABLE names a file, it writes
# that file and then waits, so that a test can stop the worker while the model runs.
TEST_MODEL_SOURCE = f'''
import os
import time

from robin_contracts.records import ClassScore, WindowOutput


class TestModel:
    def run(self, input):
        marker = os.environ.get({STARTED_VARIABLE!r})
        if marker:
            open(marker, "w").close()
            while True:
                time.sleep(0.05)
        yield WindowOutput(
            start=0.0, end=3.0, scores=(ClassScore("owl", 0.9), ClassScore("rain", 0.1))
        )

    def after_recording(self):
        pass

    def clean_up(self):
        pass


def build(context):
    return TestModel()
'''


def build_card() -> ModelCard:
    return ModelCard(
        model_name="test-model",
        model_version="1",
        runtime="none",
        window_duration=3.0,
        sample_rate=16000,
        min_detection_threshold=0.0,
        score_domain="sigmoid",
        taxa_registry_digest="sha256:" + hashlib.sha256(REGISTRY_CSV).hexdigest(),
        audio=AudioGeometry(
            downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
        ),
        backend="none",
        dtype="float32",
    )


def installed_test_model(directory: Path) -> Path:
    """Write a folder that installs the test model when it is on the import path."""
    site = directory / "site"
    dist_info = site / "robin_test_model-0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: robin-test-model\nVersion: 0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[robin.models]\ntest-model/1 = {TEST_MODEL_MODULE}:build\n"
    )
    (site / f"{TEST_MODEL_MODULE}.py").write_text(TEST_MODEL_SOURCE)
    return site


@contextmanager
def on_sys_path(site: Path) -> Iterator[None]:
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(site))
        sys.modules.pop(TEST_MODEL_MODULE, None)
        importlib.invalidate_caches()


class TaskBuilder:
    """Writes a task's model files, audio, and work file under `directory`.

    `location` gives the URI the work names for each local file. By default it is
    the file's absolute path. A test on S3 uploads the files and names them there.
    """

    def __init__(self, directory: Path, *, location: Callable[[Path], str] = str) -> None:
        self.directory = directory
        self.location = location
        self.task_id = uuid.uuid4()
        self.task_dir = directory / "runs" / "r1" / "tasks" / str(self.task_id)
        self.task_dir.mkdir(parents=True)
        self.work_path = self.task_dir / "work.json"
        self.files = {
            "weights": self._model_file("weights.bin", b"test weights"),
            REGISTRY_ROLE: self._model_file("registry.csv", REGISTRY_CSV),
        }

    def recording(self, value: str, *, audio_uri: str | None = None) -> RecordingRef:
        path = self.directory / "audio" / f"{value}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"audio {value}".encode())
        uri = self.location(path) if audio_uri is None else audio_uri
        return RecordingRef(namespace="test", value=value, audio_uri=uri)

    def work(self, recordings: Sequence[RecordingRef] | None = None, **overrides) -> InferenceWork:
        if recordings is None:
            recordings = (self.recording("r1"), self.recording("r2"))
        fields = {
            "schema_version": "robin.inference-work/1",
            "recordings": tuple(recordings),
            "model": PinnedModel(card=build_card(), files=dict(self.files)),
            "input": AudioInput(),
            "settings": {},
            "resources": {},
            "outputs": (ScoresRequest(contract_id="robin.scores.parquet/1", retention="full"),),
        }
        return InferenceWork(**(fields | overrides))

    def task(self, **overrides) -> InferenceTask:
        fields = {
            "schema_version": "robin.inference-task/1",
            "task_id": self.task_id,
            "output_root": self.location(self.task_dir / "outputs"),
            "output_version": ModelOutputVersion(model=TEST_MODEL_ID, version=1),
        }
        if "work" not in overrides:
            fields["work"] = self.work()
        return InferenceTask(**(fields | overrides))

    def write(self, **overrides) -> tuple[Path, InferenceTask]:
        """Write the work file for a task. Keyword arguments replace envelope fields."""
        task = self.task(**overrides)
        self.work_path.write_text(task.model_dump_json())
        return self.work_path, task

    def write_bytes(self, data: bytes) -> Path:
        self.work_path.write_bytes(data)
        return self.work_path

    @property
    def result_path(self) -> Path:
        return self.task_dir / "result.json"

    @property
    def outputs(self) -> Path:
        return self.task_dir / "outputs"

    def _model_file(self, name: str, data: bytes) -> PinnedFile:
        path = self.directory / "model" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return PinnedFile(
            uri=self.location(path),
            digest="sha256:" + hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )


# Starts the worker with the real version table and the test model's version.
WORKER_WITH_TEST_VERSIONS = f"""
import sys

from robin_contracts.tasks import ModelOutputVersion
from robin_models.output_versions import OUTPUT_VERSIONS
from robin_worker.cli import main

test_model = ModelOutputVersion(model={TEST_MODEL_ID!r}, version=1)
sys.exit(main(sys.argv[1:], versions=(*OUTPUT_VERSIONS, test_model)))
"""

CONSOLE_SCRIPT = Path(sys.executable).parent / "robin-worker"


def worker_command(directory: Path, args: Sequence[str], *, test_versions: bool) -> list[str]:
    """The command that runs the worker.

    With `test_versions`, it is a script that adds the test model's version to the
    real table. Otherwise it is the console script.
    """
    if not test_versions:
        return [str(CONSOLE_SCRIPT), *args]
    script = directory / "run_worker.py"
    script.write_text(WORKER_WITH_TEST_VERSIONS)
    return [sys.executable, str(script), *args]


def worker_env(*, site: Path | None, env: dict[str, str | None] | None) -> dict[str, str]:
    """The worker's environment. A value of None in `env` removes that variable."""
    environment = dict(os.environ) | {"ROBIN_RELEASE": "test-release"}
    if site is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(site), environment.get("PYTHONPATH")))
        )
    for name, value in (env or {}).items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def run_worker(
    directory: Path,
    args: Sequence[str],
    *,
    site: Path | None = None,
    env: dict[str, str | None] | None = None,
    test_versions: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        worker_command(directory, args, test_versions=test_versions),
        env=worker_env(site=site, env=env),
        capture_output=True,
        text=True,
        timeout=120,
    )


def popen_worker(
    directory: Path,
    args: Sequence[str],
    *,
    site: Path | None = None,
    env: dict[str, str | None] | None = None,
) -> subprocess.Popen:
    return subprocess.Popen(
        worker_command(directory, args, test_versions=True),
        env=worker_env(site=site, env=env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
