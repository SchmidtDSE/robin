import boto3
import moto
import pytest


@pytest.fixture
def fake_aws_login(tmp_path, monkeypatch):
    """Fake AWS credentials, and no AWS files or endpoint of the developer's own.

    The AWS config and credentials files point at paths that don't exist, and the
    endpoint overrides are unset.
    """
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-aws-credentials"))
    endpoints = ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL_BATCH")
    for name in ("AWS_PROFILE", *endpoints):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def s3_bucket(fake_aws_login):
    """The name of an empty bucket in moto's in-process S3, reached with fake credentials."""
    with moto.mock_aws():
        boto3.client("s3").create_bucket(Bucket="robin-test")
        yield "robin-test"
