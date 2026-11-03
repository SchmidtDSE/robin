"""The setup class of the `local` compute kind."""

from typing import Literal

from robin_contracts.ports.compute import ComputeSetupSettings


class LocalSetup(ComputeSetupSettings):
    """Tasks run on this computer. The worker is not held to the compute request."""

    TARGET_FIELDS = ("kind",)
    # A bundled location names only a model's file, so only `model_file` accepts it.
    ACCEPTED_FORMS = {
        "recording": ("path", "file", "s3"),
        "model_file": ("path", "file", "s3", "bundled"),
        "embeddings": ("path", "file", "s3"),
        "storage_root": ("path", "file", "s3"),
    }

    kind: Literal["local"]
