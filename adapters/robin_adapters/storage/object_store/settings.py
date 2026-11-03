"""The settings class of the object store backend, for S3 and services that work like S3."""

from typing import Annotated, Literal

from pydantic import AfterValidator, AnyHttpUrl
from robin_contracts.ports.storage import StorageSettings, is_s3_uri


def _refuse_url_with_login(url: AnyHttpUrl) -> str:
    # A run keeps the URL, and Robin keeps no credential. It gives the URL as text,
    # because the field keeps text.
    if url.username is not None or url.password is not None:
        raise ValueError(
            "give a URL without a user name or password; Robin keeps no credential"
        )
    return str(url)


# An http or https URL with no user name or password, as text in its normal form, such
# as `http://host:9000/`.
HttpUrlWithoutLogin = Annotated[AnyHttpUrl, AfterValidator(_refuse_url_with_login)]


class S3StorageSettings(StorageSettings):
    """Robin writes its work files, result files, and outputs under an S3 `root`."""

    kind: Literal["s3"]
    # The S3 service's URL, for a service that is not AWS S3.
    endpoint: HttpUrlWithoutLogin | None = None

    @classmethod
    def get_path_in_store(cls, root: str) -> str:
        if not is_s3_uri(root):
            raise ValueError("give an s3:// URI with a bucket")
        return root.removeprefix("s3://").partition("/")[2]

    def get_backend_options(self) -> dict[str, str]:
        return {} if self.endpoint is None else {"endpoint_url": self.endpoint}
