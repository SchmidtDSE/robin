import inspect

from robin_contracts.ports.storage import ArtifactWriter
from robin_contracts.results import ArtifactContractId, ArtifactKind


def test_the_writer_is_asked_for_a_declared_kind_and_contract():
    parameters = inspect.signature(ArtifactWriter.create).parameters

    assert parameters["kind"].annotation is ArtifactKind
    assert parameters["contract_id"].annotation is ArtifactContractId
