"""The setup class of each compute kind, which the profile's `compute.setups` uses.

Each class is in the `settings` module of its kind in `robin_adapters.compute`. A
`settings` module imports no adapter, so a profile loads without the packages that an
adapter needs, such as boto3.
"""

from collections.abc import Mapping
from typing import Annotated

from pydantic import Field

from robin_adapters.compute.aws_batch.settings import AwsBatchSetup
from robin_adapters.compute.kubernetes.settings import KubernetesSetup
from robin_adapters.compute.local.settings import LocalSetup
from robin_contracts.ports.compute import ComputeSetupSettings

# A compute setup of any kind, chosen by its `kind`.
ComputeSetup = Annotated[
    LocalSetup | KubernetesSetup | AwsBatchSetup, Field(discriminator="kind")
]

# The setup class of each compute kind, by its `kind`.
SETUP_CLASSES: Mapping[str, type[ComputeSetupSettings]] = {
    "local": LocalSetup,
    "kubernetes": KubernetesSetup,
    "aws-batch": AwsBatchSetup,
}
