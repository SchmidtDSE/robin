"""Import-boundary tests for the ROBIN package skeleton.

The checker is a private helper defined in this module (it moves to a
shared location when a second consumer needs it). It parses a module's
``import`` / ``from ... import`` statements with ``ast`` and classifies
each imported name against the allowed-edges table in ``allowed_edges``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from .allowed_edges import (
    ADAPTERS_DEFAULT_ALLOWED,
    ADAPTERS_PERSISTENCE_ALLOWED,
    ADAPTERS_PERSISTENCE_PREFIX,
    ADAPTERS_PREFIX,
    FIXED_ALLOWED_ROBIN_IMPORTS,
    FORBIDDEN_PREFIXES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each entry is the directory directly containing a distributable
# package's flat import-package directory. Discovered by walking
# */robin_*/**/*.py at the repository root rather than a hardcoded
# list, so the check tracks whichever project directories exist.
REAL_PACKAGE_ROOTS = sorted(
    {path.parent for path in REPO_ROOT.glob("*/robin_*") if path.is_dir()}
)


def _prefix_matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _module_name_for(file_path: Path, package_root: Path) -> str:
    """Dotted module name of file_path relative to package_root.

    package_root is the directory directly containing the top-level
    flat ``robin_<name>`` import-package directory.
    """
    rel = file_path.relative_to(package_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_module_names(file_path: Path) -> set[str]:
    tree = ast.parse(file_path.read_text(), filename=str(file_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                names.add(node.module)
    return names


def _allowed_robin_prefixes_for(module_name: str) -> set[str] | None:
    """robin_* prefixes module_name may import, or None if unconstrained."""
    if _prefix_matches(module_name, ADAPTERS_PERSISTENCE_PREFIX):
        return ADAPTERS_PERSISTENCE_ALLOWED
    if _prefix_matches(module_name, ADAPTERS_PREFIX):
        return ADAPTERS_DEFAULT_ALLOWED
    for owner, allowed in FIXED_ALLOWED_ROBIN_IMPORTS.items():
        if _prefix_matches(module_name, owner):
            return allowed
    return None


def check_module(module_name: str, imported_names: set[str]) -> list[str]:
    """Return violation strings for one module's imports. [] means compliant."""
    violations: list[str] = []

    for imported in imported_names:
        for forbidden in FORBIDDEN_PREFIXES:
            if imported.startswith(forbidden):
                violations.append(
                    f"{module_name} imports {imported} (forbidden prefix {forbidden!r})"
                )

    allowed = _allowed_robin_prefixes_for(module_name)
    if allowed is None:
        return violations

    for imported in imported_names:
        if not imported.startswith("robin_"):
            continue
        if any(_prefix_matches(imported, edge) for edge in allowed):
            continue
        violations.append(
            f"{module_name} imports {imported}, outside its allowed edges {sorted(allowed)}"
        )
    return violations


def check_file(file_path: Path, package_root: Path) -> list[str]:
    module_name = _module_name_for(file_path, package_root)
    imported = _imported_module_names(file_path)
    return check_module(module_name, imported)


def check_tree(package_root: Path) -> list[str]:
    violations: list[str] = []
    for file_path in sorted(package_root.rglob("*.py")):
        violations.extend(check_file(file_path, package_root))
    return violations


def _write_module(root: Path, dotted_name: str, import_lines: list[str]) -> Path:
    parts = dotted_name.split(".")
    dir_path = root.joinpath(*parts[:-1])
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{parts[-1]}.py"
    file_path.write_text("\n".join(import_lines) + "\n" if import_lines else "")
    return file_path


# ---------------------------------------------------------------------------
# Fixture-based checker verification (test-first).
# ---------------------------------------------------------------------------


def test_adapters_storage_importing_run_manager_is_rejected(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_adapters.storage.s3",
        ["from robin_run_manager.ports import persistence"],
    )
    violations = check_file(module, tmp_path)
    assert violations, "robin_adapters.storage.s3 must not import robin_run_manager"


def test_adapters_persistence_importing_run_manager_is_accepted(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_adapters.persistence.postgres",
        ["from robin_run_manager.ports import persistence"],
    )
    violations = check_file(module, tmp_path)
    assert not violations, (
        "robin_adapters.persistence.postgres may import robin_run_manager: "
        f"{violations}"
    )


def test_inference_engine_importing_run_manager_is_rejected(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_inference_engine.pipeline",
        ["from robin_run_manager import runs"],
    )
    violations = check_file(module, tmp_path)
    assert violations, "robin_inference_engine must not import robin_run_manager"


def test_contracts_importing_inference_engine_is_rejected(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_contracts.identity",
        ["import robin_inference_engine"],
    )
    violations = check_file(module, tmp_path)
    assert violations, "robin_contracts must not import any other robin_* package"


def test_worker_importing_all_siblings_is_accepted(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_worker.entry",
        [
            "import robin_contracts",
            "import robin_inference_engine",
            "import robin_run_manager",
            "import robin_adapters",
        ],
    )
    violations = check_file(module, tmp_path)
    assert not violations, f"robin_worker may import all four siblings: {violations}"


def test_any_module_importing_soundhub_is_rejected(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_inference_engine.pipeline",
        ["import soundhub_core"],
    )
    violations = check_file(module, tmp_path)
    assert violations, "no robin_* module may import soundhub_*"


def test_inference_engine_importing_contracts_is_accepted(tmp_path):
    module = _write_module(
        tmp_path,
        "robin_inference_engine.foo",
        ["from robin_contracts import bar"],
    )
    violations = check_file(module, tmp_path)
    assert not violations, (
        f"robin_inference_engine.foo importing robin_contracts.bar is allowed: {violations}"
    )


# ---------------------------------------------------------------------------
# Real-tree scan.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package_root", REAL_PACKAGE_ROOTS, ids=lambda p: p.name)
def test_real_package_tree_has_no_boundary_violations(package_root):
    violations = check_tree(package_root)
    assert not violations, violations
