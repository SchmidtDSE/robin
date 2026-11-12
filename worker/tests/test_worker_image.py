"""The worker image installs Robin's own packages only from the wheels it was built with,
and every model and head runtime that robin-models has.

Robin's packages are not on PyPI, and each is version 0.0.0. If pip may choose a Robin
package by name, a package of the same name published on PyPI at a higher version
would replace Robin's own in the image, and run with the job role's access.

The GPU image also turns on TensorFlow's deterministic GPU operations, so that repeated
runs of the same work give identical values.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "worker.Dockerfile"
GPU_DOCKERFILE = ROOT / "docker" / "worker-gpu.Dockerfile"
MODELS_PYPROJECT = ROOT / "models" / "pyproject.toml"
ROBIN_PACKAGES = ("contracts", "inference_engine", "models", "adapters", "worker")


def get_pip_install_text() -> str:
    text = DOCKERFILE.read_text()
    # The RUN step that installs, with its line continuations joined.
    step = re.search(r"^RUN pip install .*?(?<!\\)$", text, re.MULTILINE | re.DOTALL)
    assert step is not None, "the Dockerfile has no RUN pip install step"
    return step.group(0)


def test_each_robin_package_is_installed_from_its_wheel_file():
    install = get_pip_install_text()

    for package in ROBIN_PACKAGES:
        assert f"/tmp/robin-wheels/robin_{package}-*.whl" in install, package


def test_no_robin_package_is_installed_by_name():
    install = get_pip_install_text()

    assert "--find-links" not in install
    named = re.findall(r"(?<![\w/])robin-[a-z-]+", install)
    assert named == [], named


def test_the_image_installs_every_extra_of_robin_models():
    install = get_pip_install_text()
    extras = tomllib.loads(MODELS_PYPROJECT.read_text())["project"]["optional-dependencies"]

    installed = re.search(r"robin_models-\*\.whl\)\[([^\]]*)\]", install)
    assert installed is not None, "the Dockerfile installs robin-models with no extras"
    assert set(installed.group(1).split(",")) == set(extras)


def test_the_gpu_image_turns_on_deterministic_gpu_operations():
    text = GPU_DOCKERFILE.read_text()

    assert re.search(r"^ENV TF_DETERMINISTIC_OPS=1$", text, re.MULTILINE), (
        "the GPU image does not set TF_DETERMINISTIC_OPS=1"
    )
