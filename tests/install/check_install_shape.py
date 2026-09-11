"""Install-shape checker for ROBIN wheels.

For a given shape name, install the corresponding wheel(s) into an
isolated virtual environment and assert two things:

- Every *required* import for the shape succeeds.
- Every *forbidden* import raises ``ModuleNotFoundError``.

Every shape additionally asserts that seven heavy runtime dependencies
that no SP02 distribution declares (``tensorflow``, ``torch``,
``birdnet``, ``psycopg``, ``boto3``, ``duckdb``, ``pyarrow``) remain
absent from the install.

Invoked from CI as::

    python tools/check_install_shape.py <shape> --wheels-dir <dir> --venv <path>

Also imported by ``tests/install/test_shape_*.py`` for the same
assertions under pytest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class InstallShapeError(RuntimeError):
    """A required assertion for an install shape did not hold."""


@dataclass(frozen=True)
class Shape:
    name: str
    install: tuple[str, ...]
    required: tuple[str, ...]
    forbidden_robin: tuple[str, ...]


_ROBIN_COMPONENTS = (
    "robin_contracts",
    "robin_inference_engine",
    "robin_run_manager",
    "robin_worker",
    "robin_adapters",
)

_FORBIDDEN_RUNTIMES = (
    "tensorflow",
    "torch",
    "birdnet",
    "psycopg",
    "boto3",
    "duckdb",
    "pyarrow",
)


SHAPES: dict[str, Shape] = {}


def _register(name: str, install: tuple[str, ...], required: tuple[str, ...]) -> None:
    forbidden = tuple(m for m in _ROBIN_COMPONENTS if m not in required)
    SHAPES[name] = Shape(name=name, install=install, required=required, forbidden_robin=forbidden)


_register(
    "contracts",
    install=("robin-contracts",),
    required=("robin_contracts",),
)
_register(
    "inference-engine",
    install=("robin-inference-engine",),
    required=("robin_inference_engine", "robin_contracts"),
)
_register(
    "run-manager",
    install=("robin-run-manager",),
    required=("robin_run_manager", "robin_contracts"),
)
_register(
    "worker",
    install=("robin-worker",),
    required=_ROBIN_COMPONENTS,
)
_register(
    "adapters",
    install=("robin-adapters",),
    required=("robin_adapters", "robin_contracts"),
)
_register(
    "meta-all",
    install=("robin-bioacoustics[all]",),
    required=_ROBIN_COMPONENTS,
)


def check_shape(shape_name: str, wheels_dir: Path, venv: Path) -> None:
    """Install ``shape_name`` into ``venv`` from ``wheels_dir`` and check imports.

    Raises ``InstallShapeError`` if any required import fails or any
    forbidden import succeeds.
    """
    if shape_name not in SHAPES:
        raise InstallShapeError(f"unknown shape {shape_name!r}; known: {sorted(SHAPES)}")
    shape = SHAPES[shape_name]
    wheels_dir = wheels_dir.resolve()
    if not wheels_dir.is_dir():
        raise InstallShapeError(f"wheels dir {wheels_dir} does not exist")

    _create_venv(venv)
    python = _venv_python(venv)

    _run([str(python), "-m", "pip", "install", "--upgrade", "--quiet", "pip"])
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels_dir),
            *shape.install,
        ]
    )

    for module in shape.required:
        _assert_importable(python, module)
    for module in shape.forbidden_robin:
        _assert_not_importable(python, module, reason=f"forbidden for shape {shape.name!r}")
    for module in _FORBIDDEN_RUNTIMES:
        _assert_not_importable(python, module, reason="never a runtime dependency in SP02")


def _create_venv(venv: Path) -> None:
    venv.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "venv", str(venv)])


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _assert_importable(python: Path, module: str) -> None:
    result = subprocess.run(
        [str(python), "-I", "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InstallShapeError(
            f"required import {module!r} failed:\n{result.stderr.strip()}"
        )


def _assert_not_importable(python: Path, module: str, reason: str) -> None:
    result = subprocess.run(
        [str(python), "-I", "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise InstallShapeError(
            f"{module!r} unexpectedly importable ({reason})"
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shape", choices=sorted(SHAPES))
    parser.add_argument("--wheels-dir", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        check_shape(args.shape, args.wheels_dir, args.venv)
    except InstallShapeError as exc:
        print(f"install-shape check FAILED: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"subprocess failed: {exc.cmd}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}", file=sys.stderr)
        return 2
    print(f"install-shape check OK: {args.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
