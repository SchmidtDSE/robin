"""The worker image installs Robin's own packages only from the wheels it was built with.

Robin's packages are not on PyPI, and each is version 0.0.0. If pip may choose a Robin
package by name, a package of the same name published on PyPI at a higher version
would replace Robin's own in the image, and run with the job role's access.
"""

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "worker.Dockerfile"
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
