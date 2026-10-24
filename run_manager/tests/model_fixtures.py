"""Registered models for tests, read from the fixture files without reading their pinned files."""

from pathlib import Path

import yaml

from robin_contracts.work import PinnedModel

FIXTURES = Path(__file__).parent / "fixtures"


def read_pinned_model(name: str) -> PinnedModel:
    """The model in `fixtures/<name>.yaml`: a card and its pinned files, given by hand."""
    return PinnedModel.model_validate(yaml.safe_load((FIXTURES / f"{name}.yaml").read_text()))
