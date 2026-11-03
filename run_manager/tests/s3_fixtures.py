"""Fixtures for the tests that need a real S3 server: RustFS in Docker.

`conftest.py` imports this module. The server starts only for a test that uses
`s3_server` or `bucket`, and one server serves every such test in the session. Without Docker
these tests fail with a message; they are never skipped.
"""

import shutil
import subprocess
import time
import uuid

import boto3
import botocore.exceptions
import pytest

__all__ = ["bucket", "s3_server"]

RUSTFS_IMAGE = "rustfs/rustfs:1.0.4"
# The user that the image runs as. Its data and log folders must be writable by it.
_RUSTFS_USER = "10001"
_ACCESS_KEY = "robin-test-key"
_SECRET_KEY = "robin-test-secret"
_READY_TIMEOUT_S = 60


@pytest.fixture(scope="session")
def s3_server():
    """The endpoint and the keys of an empty S3 server in Docker."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("The S3 tests need Docker, and `docker` is not on PATH.")
    owner = f"uid={_RUSTFS_USER},gid={_RUSTFS_USER}"
    started = subprocess.run(
        [
            docker, "run", "--detach", "--rm",
            "--env", f"RUSTFS_ACCESS_KEY={_ACCESS_KEY}",
            "--env", f"RUSTFS_SECRET_KEY={_SECRET_KEY}",
            "--publish", "127.0.0.1::9000",
            # The image keeps its data in a volume, which would stay after the container
            # is removed, so the data and the logs are kept in memory.
            "--tmpfs", f"/data:{owner}",
            "--tmpfs", f"/logs:{owner}",
            RUSTFS_IMAGE,
        ],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        pytest.fail(
            "The S3 tests need Docker, and Docker could not start RustFS. "
            f"Is the Docker daemon running?\n{started.stderr}"
        )
    container = started.stdout.strip()
    try:
        server = {
            "endpoint": f"http://127.0.0.1:{_get_published_port(docker, container)}",
            "access_key": _ACCESS_KEY,
            "secret_key": _SECRET_KEY,
        }
        _wait_until_ready(docker, container, server)
        yield server
    finally:
        subprocess.run([docker, "rm", "--force", "--volumes", container], capture_output=True)


def _get_published_port(docker: str, container: str) -> int:
    shown = subprocess.run(
        [docker, "port", container, "9000/tcp"], capture_output=True, text=True, check=True
    ).stdout
    return int(shown.splitlines()[0].rsplit(":", 1)[1])


def _wait_until_ready(docker: str, container: str, server: dict) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=server["endpoint"],
        aws_access_key_id=server["access_key"],
        aws_secret_access_key=server["secret_key"],
        region_name="us-east-1",
    )
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            client.list_buckets()
            return
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError):
            if time.monotonic() > deadline:
                logs = subprocess.run(
                    [docker, "logs", container], capture_output=True, text=True
                )
                pytest.fail(f"RustFS did not start.\n{logs.stdout}{logs.stderr}")
            time.sleep(0.5)


@pytest.fixture
def bucket(s3_server, monkeypatch, tmp_path):
    """A new, empty bucket on the S3 server.

    The run manager and the worker get the server's keys from the environment, and no
    AWS file or endpoint of the developer's own. The endpoint is not in the
    environment: it must reach the worker through the profile.
    """
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", s3_server["access_key"])
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", s3_server["secret_key"])
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-aws-credentials"))
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
        monkeypatch.delenv(name, raising=False)
    name = f"robin-test-{uuid.uuid4().hex[:12]}"
    boto3.client("s3", endpoint_url=s3_server["endpoint"]).create_bucket(Bucket=name)
    return name
