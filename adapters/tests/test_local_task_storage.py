import os

import pytest

from robin_adapters.storage.local import LocalTaskStorage


def test_create_makes_the_folders_and_writes_the_bytes(tmp_path):
    path = tmp_path / "runs" / "r" / "tasks" / "t" / "work.json"

    LocalTaskStorage().create(path.as_uri(), b"{}")

    assert path.read_bytes() == b"{}"


def test_create_refuses_a_file_that_is_there(tmp_path):
    path = tmp_path / "work.json"
    storage = LocalTaskStorage()
    storage.create(str(path), b"first")

    with pytest.raises(FileExistsError):
        storage.create(str(path), b"second")

    assert path.read_bytes() == b"first"


def test_read_gives_none_only_for_an_absent_file(tmp_path):
    storage = LocalTaskStorage()
    path = tmp_path / "result.json"

    assert storage.read(path.as_uri()) is None
    assert storage.read(str(tmp_path / "absent" / "result.json")) is None
    path.write_bytes(b"{}")
    assert storage.read(path.as_uri()) == b"{}"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file with mode 000")
def test_read_raises_for_a_folder_or_an_unreadable_file(tmp_path):
    storage = LocalTaskStorage()
    folder = tmp_path / "result.json"
    folder.mkdir()
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_bytes(b"{}")
    unreadable.chmod(0o000)

    try:
        with pytest.raises(OSError):
            storage.read(folder.as_uri())
        with pytest.raises(OSError):
            storage.read(unreadable.as_uri())
    finally:
        unreadable.chmod(0o600)


def test_create_leaves_no_temporary_file(tmp_path):
    storage = LocalTaskStorage()
    path = tmp_path / "work.json"
    storage.create(str(path), b"first")
    with pytest.raises(FileExistsError):
        storage.create(str(path), b"second")

    assert [entry.name for entry in tmp_path.iterdir()] == ["work.json"]
