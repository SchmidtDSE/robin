"""The output version of each model's code in this release.

Importing this module loads no ML runtime.
"""

from robin_contracts.tasks import ModelOutputVersion

# Increase a version in the change that changes that model's outputs, whatever the
# cause: the adapter, the engine, preprocessing, or a locked dependency. A change that
# leaves the outputs the same keeps the version. Every head on one runtime shares that
# runtime's version.
OUTPUT_VERSIONS: tuple[ModelOutputVersion, ...] = (
    ModelOutputVersion(model="model:owl/v4", version=1),
    ModelOutputVersion(model="model:birdnet/v2p4", version=1),
    ModelOutputVersion(model="model:perch/v8", version=1),
    ModelOutputVersion(model="head_runtime:onnx", version=1),
)
