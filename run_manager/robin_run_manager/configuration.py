"""The inference configuration: what a run computes, without its recordings.

An operator writes an unresolved configuration, which names its model. Robin resolves it
against the registered model, checks it as the engine would, and identifies the resolved
configuration by its digest.
This module does no database work: the callers look up the models.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, NoReturn

from pydantic import BaseModel, Field, ValidationError

from robin_contracts.canonical import sha256_v1
from robin_contracts.cards import ModelCard, card_digest
from robin_contracts.output_contracts import EmbeddingsRequest
from robin_contracts.request_checks import check_request_for_problem
from robin_contracts.work import (
    AudioInput,
    BytesDigest,
    CanonicalDigest,
    EmbeddingArtifactInput,
    FiniteScalarMap,
    NonEmptyText,
    OutputRequests,
    PinnedModel,
)
from robin_run_manager.errors import INVALID_CONFIGURATION, Refused
from robin_run_manager.operator_files import field_problems, read_yaml_or_refuse

CONFIGURATION_FORMAT = "robin.configuration/1"

# A head configuration names a backbone configuration that requests no embeddings.
BACKBONE_REQUESTS_NO_EMBEDDINGS = "backbone_requests_no_embeddings"


class BackboneInput(BaseModel, frozen=True, extra="forbid"):
    """A head reads the embeddings that one backbone configuration saved."""

    kind: Literal["embedding_artifact"] = "embedding_artifact"
    backbone_configuration: CanonicalDigest


ConfigurationInput = Annotated[AudioInput | BackboneInput, Field(discriminator="kind")]


class UnresolvedConfiguration(BaseModel, frozen=True, extra="forbid"):
    """What an operator writes: the model's `name/version`, the input, settings, and outputs."""

    model: NonEmptyText
    input: ConfigurationInput
    settings: FiniteScalarMap = {}
    outputs: OutputRequests


class ResolvedModel(BaseModel, frozen=True, extra="forbid"):
    """A model as a configuration names it: its card, and the digest of each file by role.

    File locations are left out, so a model whose files move keeps its configurations.
    """

    name: str
    version: str
    card_digest: CanonicalDigest
    files: Mapping[str, BytesDigest]


class Configuration(BaseModel, frozen=True, extra="forbid"):
    """What a run computes, pinned to the model's digests. Its digest is its identity."""

    format: Literal["robin.configuration/1"] = CONFIGURATION_FORMAT
    model: ResolvedModel
    input: ConfigurationInput
    settings: FiniteScalarMap = {}
    outputs: OutputRequests


def read_configuration(path: Path) -> UnresolvedConfiguration:
    """Read a configuration file.

    Raises `Refused`: `invalid_request` if the file cannot be read as YAML, and
    `invalid_configuration` for any problem in what it holds.
    """
    loaded = read_yaml_or_refuse(path, "configuration")
    try:
        return UnresolvedConfiguration.model_validate(loaded)
    except ValidationError as error:
        # The details give each problem, so the Pydantic error adds nothing.
        raise Refused(
            INVALID_CONFIGURATION,
            f"{path} is not a valid configuration.",
            details={"file": str(path), "problems": field_problems(error, "file")},
        ) from None


def resolve_configuration(
    unresolved: UnresolvedConfiguration,
    model: PinnedModel,
    work_input: AudioInput | EmbeddingArtifactInput,
) -> Configuration:
    """Check `unresolved` against its model and resolve it.

    `model` is the registered model that `unresolved` names. `work_input` is what the
    engine reads: the audio input as written, or for a head, the input that
    `get_embedding_input` gives for its backbone.
    Raises `Refused` with `invalid_configuration`.
    """
    problem = check_request_for_problem(model, work_input, unresolved.settings, unresolved.outputs)
    if problem is not None:
        _refuse(problem.code, problem.detail)
    return Configuration(
        model=ResolvedModel(
            name=model.card.model_name,
            version=model.card.model_version,
            card_digest=card_digest(model.card),
            files={role: pinned.digest for role, pinned in model.files.items()},
        ),
        input=unresolved.input,
        settings=unresolved.settings,
        # One request of each kind, in a fixed order: the order as written is not part
        # of what the configuration computes.
        outputs=tuple(sorted(unresolved.outputs, key=lambda output: output.kind)),
    )


def get_embedding_input(
    backbone: Configuration, backbone_model: PinnedModel
) -> EmbeddingArtifactInput:
    """The input of a head: the embeddings that its backbone configuration saves.

    Raises `Refused` with `invalid_configuration` if the backbone saves no embeddings.
    """
    embeddings = next((o for o in backbone.outputs if isinstance(o, EmbeddingsRequest)), None)
    if embeddings is None:
        _refuse(
            BACKBONE_REQUESTS_NO_EMBEDDINGS,
            f"backbone configuration {get_configuration_digest(backbone)} requests no "
            "embeddings, so a head cannot read its output",
        )
    # A stored head configuration cannot request embeddings, so this backbone is a model.
    assert isinstance(backbone_model.card, ModelCard)
    return EmbeddingArtifactInput(
        contract_id=embeddings.contract_id,
        backbone=backbone_model.card,
        backbone_settings=backbone.settings,
    )


def get_configuration_digest(configuration: Configuration) -> str:
    """The identity of a configuration."""
    return sha256_v1(configuration.model_dump(mode="json"))


def _refuse(code: str, detail: str) -> NoReturn:
    raise Refused(
        INVALID_CONFIGURATION,
        f"The configuration is not valid: {detail}.",
        details={"code": code},
    )
