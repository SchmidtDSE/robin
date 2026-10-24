"""Output versions: what a run freezes when it is submitted.

Only this module reads the installed release's table of output versions.
"""

from collections.abc import Mapping, Sequence

from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.tasks import ModelOutputVersion, output_version_id
from robin_models.output_versions import OUTPUT_VERSIONS
from robin_run_manager.errors import UNKNOWN_OUTPUT_VERSION, Refused

# The output version of each model's code in the installed release.
RELEASE_OUTPUT_VERSIONS: tuple[ModelOutputVersion, ...] = OUTPUT_VERSIONS


def get_frozen_output_versions(
    cards: Sequence[ModelCard | HeadCard], release_versions: Sequence[ModelOutputVersion]
) -> dict[str, int]:
    """The versions that a run freezes for `cards`, by output-version ID.

    `cards` is the card of the run's model and, for a head, the card of its backbone. A head
    run freezes its backbone's version because it reads the backbone's embeddings.
    """
    versions = {entry.model: entry.version for entry in release_versions}
    frozen = {}
    for card in cards:
        identifier = output_version_id(card)
        if identifier not in versions:
            raise Refused(
                UNKNOWN_OUTPUT_VERSION,
                f"The installed Robin release has no output version for {identifier}, so "
                "it cannot run it. Use a release that has this model.",
                details={"output_version_id": identifier},
            )
        frozen[identifier] = versions[identifier]
    return frozen


def get_outdated_versions(
    run_versions: Mapping[str, int], release_versions: Sequence[ModelOutputVersion]
) -> dict[str, dict]:
    """Each of the run's versions that `release_versions` changed or does not have."""
    versions = {entry.model: entry.version for entry in release_versions}
    return {
        identifier: {"run": version, "release": versions.get(identifier)}
        for identifier, version in run_versions.items()
        if versions.get(identifier) != version
    }
