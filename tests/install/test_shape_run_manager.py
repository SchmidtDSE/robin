from __future__ import annotations

from pathlib import Path

from check_install_shape import check_shape


def test_run_manager_install_shape(built_wheels_dir: Path, tmp_path: Path) -> None:
    check_shape("run-manager", wheels_dir=built_wheels_dir, venv=tmp_path / "venv")
