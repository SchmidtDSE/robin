import boto3
import moto
import pytest


@pytest.fixture
def s3_bucket(tmp_path, monkeypatch):
    """The name of an empty bucket in moto's in-process S3, reached with fake credentials.

    The AWS config and credentials files point at paths that don't exist, and the
    endpoint overrides are unset, so no test reads the developer's own files or endpoint.
    """
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-aws-credentials"))
    for name in ("AWS_PROFILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
        monkeypatch.delenv(name, raising=False)
    with moto.mock_aws():
        boto3.client("s3").create_bucket(Bucket="robin-test")
        yield "robin-test"
