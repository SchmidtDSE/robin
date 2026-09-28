import inspect

import pytest

from robin_contracts.ports import ArtifactWriter, FileProvider
from robin_contracts.results import ArtifactContractId, ArtifactKind


@pytest.mark.parametrize(
    ("port", "members"),
    [(FileProvider, {"fetch", "release"}), (ArtifactWriter, {"create"})],
    ids=["file-provider", "artifact-writer"],
)
def test_each_port_declares_exactly_its_members(port, members):
    declared = {name for name in vars(port) if not name.startswith("_")}
    assert declared == members


def test_the_writer_is_asked_for_a_declared_kind_and_contract():
    parameters = inspect.signature(ArtifactWriter.create).parameters

    assert parameters["kind"].annotation is ArtifactKind
    assert parameters["contract_id"].annotation is ArtifactContractId
