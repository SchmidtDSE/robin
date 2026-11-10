import inspect
from pathlib import Path

import boto3
import ondio
import pytest
from robin_contracts.ports.storage import FileProvider

import robin_adapters.storage.object_store.adapter as provider_module
from robin_adapters.storage.object_store.adapter import ObjectStoreFileProvider


def _put(bucket: str, key: str, data: bytes) -> str:
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


def _writing_spy(calls: list[dict]):
    def download(uri, out_path, **kwargs):
        calls.append(kwargs)
        Path(out_path).write_bytes(b"downloaded")

    return download


def test_it_has_the_signature_its_port_declares():
    for member in ("fetch", "release"):
        declared = inspect.signature(getattr(FileProvider, member))
        supplied = inspect.signature(getattr(ObjectStoreFileProvider, member))
        assert list(supplied.parameters) == list(declared.parameters), member
        for name, parameter in declared.parameters.items():
            assert supplied.parameters[name].kind == parameter.kind, (member, name)


def test_a_fetch_gives_an_absolute_path_under_the_directory_holding_the_object(
    tmp_path, s3_bucket
):
    directory = tmp_path / "scratch"
    uri = _put(s3_bucket, "models/birdnet/model.bin", b"the model")

    path = ObjectStoreFileProvider(directory).fetch(uri)

    assert path.is_absolute()
    assert path.is_relative_to(directory)
    assert path.name == "model.bin"
    assert path.read_bytes() == b"the model"


def test_an_encoded_key_is_fetched_as_written(tmp_path, s3_bucket):
    uri = _put(s3_bucket, "a%2Fb/c%25 d=e.bin", b"as written")
    _put(s3_bucket, "a/b/c% d=e.bin", b"decoded")

    path = ObjectStoreFileProvider(tmp_path / "scratch").fetch(uri)

    assert path.name == "c%25 d=e.bin"
    assert path.read_bytes() == b"as written"


def test_two_fetches_of_one_uri_give_two_paths(tmp_path, s3_bucket):
    uri = _put(s3_bucket, "a.wav", b"audio")
    files = ObjectStoreFileProvider(tmp_path / "scratch")

    first = files.fetch(uri)
    second = files.fetch(uri)

    assert first != second
    assert first.read_bytes() == second.read_bytes() == b"audio"


def test_a_missing_object_is_not_found_and_leaves_nothing(tmp_path, s3_bucket):
    directory = tmp_path / "scratch"
    files = ObjectStoreFileProvider(directory)

    with pytest.raises(ondio.ObjectNotFoundError):
        files.fetch(f"s3://{s3_bucket}/absent.wav")

    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("last", ["", ".", ".."])
def test_a_uri_with_no_file_name_is_refused_before_downloading(tmp_path, monkeypatch, last):
    calls: list[dict] = []
    monkeypatch.setattr(provider_module.ondio, "download", _writing_spy(calls))
    uri = f"s3://robin-test/models/{last}"

    with pytest.raises(ValueError) as refused:
        ObjectStoreFileProvider(tmp_path / "scratch").fetch(uri)

    assert uri in str(refused.value)
    assert calls == []


def test_release_deletes_the_file_and_its_folder(tmp_path, s3_bucket):
    directory = tmp_path / "scratch"
    files = ObjectStoreFileProvider(directory)
    path = files.fetch(_put(s3_bucket, "a.wav", b"audio"))

    files.release(path)

    assert list(directory.iterdir()) == []


def test_releasing_a_path_it_never_returned_is_refused(tmp_path):
    other = tmp_path / "other.wav"
    other.write_bytes(b"audio")

    with pytest.raises(ValueError) as refused:
        ObjectStoreFileProvider(tmp_path / "scratch").release(other)

    assert str(other) in str(refused.value)
    assert other.read_bytes() == b"audio"


def test_releasing_a_path_twice_is_refused(tmp_path, s3_bucket):
    files = ObjectStoreFileProvider(tmp_path / "scratch")
    path = files.fetch(_put(s3_bucket, "a.wav", b"audio"))
    files.release(path)

    with pytest.raises(ValueError) as refused:
        files.release(path)

    assert str(path) in str(refused.value)


def _backend_spies(monkeypatch) -> tuple[list[tuple], list]:
    """Records each backend build, and the backend that each download receives."""
    builds = []
    received = []
    real_get_backend = provider_module.ondio.get_backend
    real_download = provider_module.ondio.download

    def get_backend(uri, **options):
        backend = real_get_backend(uri, **options)
        builds.append((uri, options, backend))
        return backend

    def download(uri, out_path, **kwargs):
        received.append(kwargs.get("backend"))
        return real_download(uri, out_path, **kwargs)

    monkeypatch.setattr(provider_module.ondio, "get_backend", get_backend)
    monkeypatch.setattr(provider_module.ondio, "download", download)
    return builds, received


def test_one_backend_serves_every_fetch_from_a_store(tmp_path, s3_bucket, monkeypatch):
    uris = [_put(s3_bucket, f"{name}.wav", name.encode()) for name in ("a", "b", "c")]
    builds, received = _backend_spies(monkeypatch)
    files = ObjectStoreFileProvider(tmp_path / "scratch", region_name="us-east-1")
    assert builds == []

    paths = [files.fetch(uri) for uri in uris]

    [(store, options, backend)] = builds
    assert (store, options) == (f"s3://{s3_bucket}", {"region_name": "us-east-1"})
    assert all(given is backend for given in received) and len(received) == 3
    assert [path.read_bytes() for path in paths] == [b"a", b"b", b"c"]


def test_each_store_has_its_own_backend(tmp_path, s3_bucket, monkeypatch):
    boto3.client("s3").create_bucket(Bucket="robin-other")
    first = _put(s3_bucket, "a.wav", b"first bucket")
    second = _put("robin-other", "a.wav", b"other bucket")
    builds, _ = _backend_spies(monkeypatch)
    files = ObjectStoreFileProvider(tmp_path / "scratch")

    paths = [files.fetch(first), files.fetch(second), files.fetch(first)]

    assert [store for store, _, _ in builds] == [f"s3://{s3_bucket}", "s3://robin-other"]
    assert [path.read_bytes() for path in paths] == [
        b"first bucket",
        b"other bucket",
        b"first bucket",
    ]


def test_a_relative_directory_gives_absolute_paths(tmp_path, monkeypatch, s3_bucket):
    work = tmp_path / "work"
    elsewhere = tmp_path / "elsewhere"
    work.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(work)
    files = ObjectStoreFileProvider(Path("scratch"))

    path = files.fetch(_put(s3_bucket, "a.wav", b"audio"))
    monkeypatch.chdir(elsewhere)

    assert path.is_absolute()
    assert path.read_bytes() == b"audio"
    files.release(path)
    assert list((work / "scratch").iterdir()) == []


def test_a_partial_download_leaves_nothing(tmp_path, monkeypatch):
    def download(uri, out_path, **kwargs):
        Path(out_path).write_bytes(b"part of")
        raise OSError("connection reset")

    monkeypatch.setattr(provider_module.ondio, "download", download)
    directory = tmp_path / "scratch"

    with pytest.raises(OSError, match="connection reset"):
        ObjectStoreFileProvider(directory).fetch("s3://robin-test/a.wav")

    assert list(directory.iterdir()) == []


def test_the_directory_is_created(tmp_path):
    directory = tmp_path / "a" / "b"

    ObjectStoreFileProvider(directory)

    assert directory.is_dir()
