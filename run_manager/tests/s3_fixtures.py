"""A fixture for the tests that need a real S3 server: RustFS in Docker.

`conftest.py` imports this module. The server starts only for a test that uses
`s3_server`, and one server serves every such test in the session. Without Docker
these tests fail with a message; they are never skipped.
"""

import shutil
import subprocess
import time

import boto3
import botocore.exceptions
import pytest

__all__ = ["s3_server"]

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
