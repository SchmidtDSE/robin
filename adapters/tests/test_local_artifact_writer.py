import hashlib
import inspect
import os
from pathlib import Path

import pytest
from robin_contracts.layout import artifact_path
from robin_contracts.ports import ArtifactWriter

from robin_adapters.artifact_writer import local
from robin_adapters.artifact_writer.local import LocalArtifactWriter


def test_it_has_the_signature_its_port_declares():
    declared = inspect.signature(ArtifactWriter.create)
    supplied = inspect.signature(LocalArtifactWriter.create)
    assert list(supplied.parameters) == list(declared.parameters)
    for name, parameter in declared.parameters.items():
        assert supplied.parameters[name].kind == parameter.kind, name


# ---------------------------------------------------------------------------
# Publishing.
# ---------------------------------------------------------------------------

DATA = b"a finished artifact"


def _checksum(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _staged(directory: Path, data: bytes = DATA) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"staged-{len(list(directory.iterdir()))}"
    path.write_bytes(data)
    return path


def _create(writer, source, *, checksum=None, namespace="soundhub", value="42"):
    return writer.create(
        kind="scores",
        contract_id="robin.scores.parquet/1",
        namespace=namespace,
        value=value,
        source=source,
        checksum=_checksum(source.read_bytes()) if checksum is None else checksum,
        rows=7,
    )


def _location(root: Path, namespace="soundhub", value="42") -> Path:
    return root / artifact_path("scores", namespace, value)


def test_a_file_is_published_at_its_recordings_location(tmp_path):
    root = tmp_path / "root"
    source = _staged(tmp_path / "staging")

    record = _create(LocalArtifactWriter(root), source)

    published = _location(root)
    assert published.read_bytes() == DATA
    assert record.kind == "scores"
    assert record.contract_id == "robin.scores.parquet/1"
    assert record.namespace == "soundhub"
    assert record.value == "42"
    assert record.uri == published.as_uri()
    assert record.checksum == _checksum(DATA)
    assert record.size_bytes == published.stat().st_size == len(DATA)
    assert record.rows == 7


def test_the_whole_file_is_there_when_its_name_appears(tmp_path, monkeypatch):
    root = tmp_path / "root"
    source = _staged(tmp_path / "staging", b"x")
    seen = []
    real_link = os.link

    def link_then_read(src, dst, *args, **kwargs):
        real_link(src, dst, *args, **kwargs)
        seen.append(Path(dst).read_bytes())

    monkeypatch.setattr(os, "link", link_then_read)
    _create(LocalArtifactWriter(root), source)

    assert seen == [b"x"]


def test_the_published_file_does_not_depend_on_the_source(tmp_path):
    root = tmp_path / "root"
    source = _staged(tmp_path / "staging")

    _create(LocalArtifactWriter(root), source)
    source.unlink()

    assert _location(root).read_bytes() == DATA


def test_only_the_published_file_is_left_in_its_folder(tmp_path):
    root = tmp_path / "root"

    _create(LocalArtifactWriter(root), _staged(tmp_path / "staging"))

    assert list(_location(root).parent.iterdir()) == [_location(root)]


def test_a_relative_root_publishes_under_its_absolute_form(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    writer = LocalArtifactWriter(Path("relative-root"))
    # The root is fixed when the writer is built, not when it publishes.
    monkeypatch.chdir(elsewhere)

    record = _create(writer, _staged(tmp_path / "staging"))

    published = _location(tmp_path / "relative-root")
    assert published.read_bytes() == DATA
    assert record.uri == published.as_uri()


def test_a_root_that_does_not_exist_yet_is_created(tmp_path):
    root = tmp_path / "not" / "yet"

    _create(LocalArtifactWriter(root), _staged(tmp_path / "staging"))

    assert _location(root).is_file()


# ---------------------------------------------------------------------------
# Refusing and replaying.
# ---------------------------------------------------------------------------


def test_a_wrong_checksum_is_refused_and_publishes_nothing(tmp_path):
    root = tmp_path / "root"
    source = _staged(tmp_path / "staging")
    wrong = _checksum(b"other bytes")

    with pytest.raises(ValueError) as refused:
        _create(LocalArtifactWriter(root), source, checksum=wrong)

    assert str(source) in str(refused.value)
    assert wrong in str(refused.value)
    assert _checksum(DATA) in str(refused.value)
    assert not _location(root).exists()
    assert list(_location(root).parent.iterdir()) == []


def test_the_same_bytes_again_are_a_replay(tmp_path):
    root = tmp_path / "root"
    writer = LocalArtifactWriter(root)
    first = _create(writer, _staged(tmp_path / "staging"))
    before = _location(root).stat()

    second = _create(writer, _staged(tmp_path / "staging"))

    after = _location(root).stat()
    assert second == first
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert list(_location(root).parent.iterdir()) == [_location(root)]


def test_different_bytes_at_the_location_are_refused(tmp_path):
    root = tmp_path / "root"
    planted = _location(root)
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"damaged")

    with pytest.raises(FileExistsError) as refused:
        _create(LocalArtifactWriter(root), _staged(tmp_path / "staging"))

    assert str(planted) in str(refused.value)
    assert _checksum(b"damaged") in str(refused.value)
    assert _checksum(DATA) in str(refused.value)
    assert planted.read_bytes() == b"damaged"
    assert list(planted.parent.iterdir()) == [planted]


def test_a_failure_mid_copy_publishes_nothing_and_leaves_nothing(tmp_path, monkeypatch):
    root = tmp_path / "root"

    def copy_part_then_fail(source, target):
        target.write(b"part of it")
        raise OSError("disk full")

    monkeypatch.setattr(local, "_copy_hashing", copy_part_then_fail)

    with pytest.raises(OSError, match="disk full"):
        _create(LocalArtifactWriter(root), _staged(tmp_path / "staging"))

    assert not _location(root).exists()
    assert list(_location(root).parent.iterdir()) == []
