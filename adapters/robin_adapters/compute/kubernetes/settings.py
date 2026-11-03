"""The setup class of the `kubernetes` compute kind."""

from typing import Literal

from pydantic import Field

from robin_contracts.ports.compute import ComputeSetupSettings


class KubernetesSetup(ComputeSetupSettings):
    """Tasks run as Kubernetes jobs, which read and write S3."""

    TARGET_FIELDS = ("kind", "namespace")
    # A bundled location names only a model's file, so only `model_file` accepts it.
    ACCEPTED_FORMS = {
        "recording": ("s3",),
        "model_file": ("s3", "bundled"),
        "embeddings": ("s3",),
        "storage_root": ("s3",),
    }

    kind: Literal["kubernetes"]
    namespace: str = Field(min_length=1)
    # The name of the Kubernetes Secret that holds the S3 credentials.
    s3_secret_name: str = Field(min_length=1)
