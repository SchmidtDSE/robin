"""Fixtures for the install-shape smoke tests.

The tests install built wheels into per-test virtual environments and
assert that the resulting install answers the required/forbidden import
matrix for each SP02-supported shape.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WHEELS_DIR = REPO_ROOT / ".wheelcheck" / "dist"

# The checker lives alongside its tests; make it importable by name.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _has_wheels(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.whl"))


@pytest.fixture(scope="session")
def built_wheels_dir() -> Path:
    override = os.environ.get("ROBIN_WHEELS_DIR")
    directory = Path(override).resolve() if override else DEFAULT_WHEELS_DIR
    if _has_wheels(directory):
        return directory
    if override:
        pytest.fail(f"ROBIN_WHEELS_DIR={directory} contains no *.whl files")
    subprocess.run(
        ["pixi", "run", "-e", "test", "wheels"],
        check=True,
        cwd=REPO_ROOT,
    )
    if not _has_wheels(directory):
        pytest.fail(f"pixi wheels task did not produce wheels in {directory}")
    return directory
