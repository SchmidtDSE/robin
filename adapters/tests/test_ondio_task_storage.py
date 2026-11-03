import pytest

from robin_adapters.storage import ondio as storage_module
from robin_adapters.storage.ondio import OndioTaskStorage

ROOT = "s3://robin-test/robin"
WORK = f"{ROOT}/runs/r/tasks/t/work.json"


def _get_backend_spy(monkeypatch) -> list[tuple]:
    calls = []
    real = storage_module.ondio.get_backend

    def spy(uri, **options):
        calls.append((uri, options))
        return real(uri, **options)

    monkeypatch.setattr(storage_module.ondio, "get_backend", spy)
    return calls


def test_create_writes_once_and_a_second_create_is_file_exists(s3_bucket):
    storage = OndioTaskStorage(ROOT)
    storage.create(WORK, b"first")

    with pytest.raises(FileExistsError):
        storage.create(WORK, b"second")

    assert storage.read(WORK) == b"first"


def test_read_of_a_missing_object_gives_none(s3_bucket):
    assert OndioTaskStorage(ROOT).read(WORK) is None


def test_one_backend_serves_every_call(s3_bucket, monkeypatch):
    calls = _get_backend_spy(monkeypatch)
    storage = OndioTaskStorage(ROOT, region_name="us-east-1")

    storage.create(WORK, b"{}")
    storage.read(WORK)
    storage.read(f"{ROOT}/runs/r/tasks/t/result.json")

    assert calls == [(ROOT, {"region_name": "us-east-1"})]


def test_a_storage_that_is_not_used_builds_no_backend(monkeypatch):
    calls = _get_backend_spy(monkeypatch)

    repr(OndioTaskStorage(ROOT, endpoint_url="http://127.0.0.1:9000"))

    assert calls == []


def test_repr_shows_no_endpoint():
    shown = repr(OndioTaskStorage(ROOT, endpoint_url="http://127.0.0.1:9000"))

    assert ROOT in shown
    assert "9000" not in shown


def test_the_worker_environment_carries_the_options_and_builds_no_backend(monkeypatch):
    calls = _get_backend_spy(monkeypatch)
    storage = OndioTaskStorage(ROOT, endpoint_url="http://127.0.0.1:9000")

    environment = storage.get_worker_env_vars()

    assert environment == storage_module.ondio.get_environment(
        ROOT, endpoint_url="http://127.0.0.1:9000"
    )
    assert environment != {}
    assert calls == []
