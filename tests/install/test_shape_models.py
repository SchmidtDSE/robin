from __future__ import annotations

from pathlib import Path

from check_install_shape import check_shape


def test_models_install_shape(built_wheels_dir: Path, tmp_path: Path) -> None:
    check_shape("models", wheels_dir=built_wheels_dir, venv=tmp_path / "venv")


def test_meta_owl_install_shape(built_wheels_dir: Path, tmp_path: Path) -> None:
    check_shape("meta-owl", wheels_dir=built_wheels_dir, venv=tmp_path / "venv")


def test_meta_birdnet_install_shape(built_wheels_dir: Path, tmp_path: Path) -> None:
    check_shape("meta-birdnet", wheels_dir=built_wheels_dir, venv=tmp_path / "venv")


def test_meta_perch_install_shape(built_wheels_dir: Path, tmp_path: Path) -> None:
    check_shape("meta-perch", wheels_dir=built_wheels_dir, venv=tmp_path / "venv")


def test_meta_onnx_head_install_shape(built_wheels_dir: Path, tmp_path: Path) -> None:
    check_shape("meta-onnx-head", wheels_dir=built_wheels_dir, venv=tmp_path / "venv")
