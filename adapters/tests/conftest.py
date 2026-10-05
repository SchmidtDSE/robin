import boto3
import moto
import pytest


@pytest.fixture
def s3_bucket(tmp_path, monkeypatch):
    """The name of an empty bucket in moto's in-process S3, reached with fake credentials.

    The AWS config and credentials files point at paths that don't exist, so no test
    reads the developer's own.
    """
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-aws-credentials"))
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with moto.mock_aws():
        boto3.client("s3").create_bucket(Bucket="robin-test")
        yield "robin-test"
