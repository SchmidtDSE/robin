"""`robin files export`."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from robin_adapters.storage.local.adapter import LocalTaskStorage
from robin_cli import main
from robin_cli.main import EXIT_REFUSED, EXIT_UNAVAILABLE


class CreateFails(LocalTaskStorage):
    def create(self, uri, data):
        raise RuntimeError("token=secret")


def test_files_export_writes_the_index_where_it_says(robin, environ):
    assert robin("db", "migrate").code == 0

    ran = robin("--json", "files", "export")

    assert ran.code == 0, ran.stderr
    data = json.loads(ran.stdout)
    root = Path(environ["ROBIN_PROFILE"]) / "storage"
    assert data["uri"] == f"{root}/index/current_files.parquet"
    assert data["dated_uri"].startswith(f"{root}/index/history/")
    index = Path(data["uri"]).read_bytes()
    assert pq.read_table(data["uri"]).num_rows == data["rows"]
    assert data["checksum"] == "sha256:" + hashlib.sha256(index).hexdigest()
    assert Path(data["dated_uri"]).read_bytes() == index


def test_files_export_without_storage_is_refused(robin, environ):
    folder = Path(environ["ROBIN_PROFILE"])
    config = yaml.safe_load((folder / "config.yaml").read_text())
    (folder / "config.yaml").write_text(yaml.safe_dump({"database": config["database"]}))
    assert robin("db", "migrate").code == 0

    ran = robin("--json", "files", "export")

    assert ran.code == EXIT_REFUSED
    assert ran.error()["reason"] == "invalid_profile"
    assert not (folder / "storage" / "index").exists()


def test_files_export_reports_a_storage_failure_without_its_text(robin, monkeypatch):
    assert robin("db", "migrate").code == 0
    real = main.compose
    monkeypatch.setattr(main, "compose", lambda profile: dataclasses.replace(
        real(profile), task_storage={"local": CreateFails()}
    ))

    ran = robin("--json", "files", "export")

    assert ran.code == EXIT_UNAVAILABLE
    assert ran.error()["reason"] == "storage_unavailable"
    for text in (ran.stdout, ran.stderr):
        assert "secret" not in text
        assert "Traceback" not in text
