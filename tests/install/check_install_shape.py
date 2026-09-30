"""Install-shape checker for ROBIN wheels.

For a given shape name, install the corresponding wheel(s) into an
isolated virtual environment and assert two things:

- Every *required* import for the shape succeeds.
- Every *forbidden* import raises ``ModuleNotFoundError``.

Heavy runtime dependencies are checked per shape. Every entry in
``_HEAVY_RUNTIMES`` a shape does not declare must be absent from the
install, and every one it does declare must be importable: asserting
presence is what ties the list to the dependency, where a permitted-only
list would let a declared dependency disappear with this check still green.

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
    runtimes: tuple[str, ...]


_ROBIN_COMPONENTS = (
    "robin_contracts",
    "robin_inference_engine",
    "robin_run_manager",
    "robin_worker",
    "robin_adapters",
    "robin_models",
)

_HEAVY_RUNTIMES = (
    "tensorflow",
    "tf_keras",
    "sox_tensorflow",
    "soundfile",
    "PIL",
    "torch",
    "birdnet",
    "psycopg",
    "boto3",
    "duckdb",
    "pyarrow",
)


SHAPES: dict[str, Shape] = {}


def _register(
    name: str,
    install: tuple[str, ...],
    required: tuple[str, ...],
    runtimes: tuple[str, ...] = (),
) -> None:
    unknown = [module for module in runtimes if module not in _HEAVY_RUNTIMES]
    if unknown:
        raise InstallShapeError(f"shape {name!r} declares unknown runtimes {unknown}")
    forbidden = tuple(m for m in _ROBIN_COMPONENTS if m not in required)
    SHAPES[name] = Shape(
        name=name,
        install=install,
        required=required,
        forbidden_robin=forbidden,
        runtimes=runtimes,
    )


_register(
    "contracts",
    install=("robin-contracts",),
    required=("robin_contracts",),
)
_register(
    "inference-engine",
    install=("robin-inference-engine",),
    required=("robin_inference_engine", "robin_contracts"),
    runtimes=("pyarrow", "duckdb"),
)
_register(
    "run-manager",
    install=("robin-run-manager",),
    required=("robin_run_manager", "robin_contracts"),
)
_register(
    "worker",
    install=("robin-worker",),
    required=(
        "robin_contracts",
        "robin_inference_engine",
        "robin_run_manager",
        "robin_worker",
        "robin_adapters",
    ),
    runtimes=("pyarrow", "duckdb"),
)
_register(
    "adapters",
    install=("robin-adapters",),
    required=(
        "robin_adapters",
        "robin_adapters.file_provider.local",
        "robin_adapters.artifact_writer.local",
        "robin_contracts",
    ),
)
_register(
    "meta-all",
    install=("robin-bioacoustics[all]",),
    required=(
        "robin_contracts",
        "robin_inference_engine",
        "robin_run_manager",
        "robin_worker",
        "robin_adapters",
    ),
    runtimes=("pyarrow", "duckdb"),
)
_register(
    "models",
    install=("robin-models",),
    required=("robin_models", "robin_models.owl", "robin_contracts"),
)
_register(
    "meta-owl",
    install=("robin-bioacoustics[owl]",),
    required=(
        "robin_inference_engine",
        "robin_contracts",
        "robin_models",
        "robin_models.owl",
        "robin_models.owl.adapter",
    ),
    runtimes=(
        "tensorflow",
        "tf_keras",
        "sox_tensorflow",
        "soundfile",
        "PIL",
        "pyarrow",
        "duckdb",
    ),
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
    for module in shape.runtimes:
        _assert_importable(python, module)
    for module in _HEAVY_RUNTIMES:
        if module in shape.runtimes:
            continue
        _assert_not_importable(
            python, module, reason=f"not declared by shape {shape.name!r}"
        )


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
