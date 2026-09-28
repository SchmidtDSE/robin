import inspect
from pathlib import Path
from urllib.parse import quote

import pytest
from robin_contracts.ports import FileProvider

from robin_adapters.file_provider.local import LocalFileProvider


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


def test_an_absolute_path_gives_itself(tmp_path):
    path = _file(tmp_path, "a.wav")

    assert LocalFileProvider().fetch(str(path)) == path


def test_a_localhost_file_uri_gives_the_same_path(tmp_path):
    path = _file(tmp_path, "a.wav")

    assert LocalFileProvider().fetch(_file_uri(path, host="localhost")) == path


def test_a_percent_encoded_uri_gives_the_decoded_path(tmp_path):
    path = _file(tmp_path, "a b.wav")
    assert LocalFileProvider().fetch(_file_uri(path)) == path


@pytest.mark.parametrize("name", ["a%20b.wav", "a?b.wav", "a#b.wav"])
def test_an_absolute_path_is_used_literally(tmp_path, name):
    path = _file(tmp_path, name)
    _file(tmp_path, "a b.wav", b"a decoy the decoded name would find")

    assert LocalFileProvider().fetch(str(path)) == path


@pytest.mark.parametrize("name", ["a%20b.wav", "a?b.wav", "a#b.wav"])
def test_an_encoded_uri_gives_the_same_literal_path(tmp_path, name):
    path = _file(tmp_path, name)

    assert LocalFileProvider().fetch(_file_uri(path)) == path


def test_a_relative_file_uri_is_refused_even_when_the_file_exists(tmp_path, monkeypatch):
    _file(tmp_path, "relative.wav")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="file:relative.wav"):
        LocalFileProvider().fetch("file:relative.wav")


@pytest.mark.parametrize("suffix", ["?x=1", "#part", "?", "#"])
def test_a_file_uri_with_a_query_or_fragment_is_refused(tmp_path, suffix):
    path = _file(tmp_path, "a.wav")
    uri = _file_uri(path) + suffix

    with pytest.raises(ValueError) as refused:
        LocalFileProvider().fetch(uri)
    assert uri in str(refused.value)


@pytest.mark.parametrize(
    "uri",
    [
        "a.wav",
        "file://otherhost/data/a.wav",
        "s3://bucket/key",
        "https://example.org/a.wav",
    ],
)
def test_anything_but_a_local_absolute_location_is_refused(tmp_path, monkeypatch, uri):
    _file(tmp_path, "a.wav")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as refused:
        LocalFileProvider().fetch(uri)
    assert uri in str(refused.value)


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
