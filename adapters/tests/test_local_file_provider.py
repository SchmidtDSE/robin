import inspect
from pathlib import Path
from urllib.parse import quote

import pytest
from robin_contracts.ports.storage import FileProvider

from robin_adapters.storage.local import LocalFileProvider


def test_it_has_the_signature_its_port_declares():
    for member in ("fetch", "release"):
        declared = inspect.signature(getattr(FileProvider, member))
        supplied = inspect.signature(getattr(LocalFileProvider, member))
        assert list(supplied.parameters) == list(declared.parameters), member
        for name, parameter in declared.parameters.items():
            assert supplied.parameters[name].kind == parameter.kind, (member, name)


def _file(directory: Path, name: str, data: bytes = b"audio") -> Path:
    path = directory / name
    path.write_bytes(data)
    return path


def _file_uri(path: Path, host: str = "") -> str:
    return f"file://{host}{quote(str(path))}"


def test_a_file_uri_gives_the_files_path(tmp_path):
    path = _file(tmp_path, "a.wav")

    assert LocalFileProvider().fetch(_file_uri(path)) == path


def test_a_location_that_is_not_local_is_refused():
    with pytest.raises(ValueError):
        LocalFileProvider().fetch("s3://bucket/key")


def test_a_missing_file_is_not_found(tmp_path):
    uri = _file_uri(tmp_path / "absent.wav")

    with pytest.raises(FileNotFoundError) as missing:
        LocalFileProvider().fetch(uri)
    assert uri in str(missing.value)


def test_a_folder_is_not_found(tmp_path):
    folder = tmp_path / "folder.wav"
    folder.mkdir()

    with pytest.raises(FileNotFoundError) as missing:
        LocalFileProvider().fetch(str(folder))
    assert str(folder) in str(missing.value)


def test_release_leaves_the_file_as_it_was(tmp_path):
    path = _file(tmp_path, "a.wav", b"the recording")
    files = LocalFileProvider()

    fetched = files.fetch(str(path))
    files.release(fetched)
    files.release(fetched)

    assert path.read_bytes() == b"the recording"
