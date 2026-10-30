"""Helpers for the CLI tests: the result of one run, and a profile folder."""

import json
from dataclasses import dataclass

import yaml

PASSWORD_VARIABLE = "ROBIN_TEST_DB_PASSWORD"


@dataclass(frozen=True)
class Ran:
    code: int
    stdout: str
    stderr: str

    def error(self) -> dict:
        """The JSON error body: the last line of stderr."""
        return json.loads(self.stderr.strip().splitlines()[-1])


def write_profile(folder, database_settings, **changes):
    """A profile folder for environment `test`. A change to None removes that field."""
    folder.mkdir()
    database = database_settings.model_dump(mode="json")
    database["password"] = "${" + PASSWORD_VARIABLE + "}"
    database.update(changes)
    database = {name: value for name, value in database.items() if value is not None}
    sections = {
        "database": database,
        "compute": {"setups": {"local": {"kind": "local"}}, "models": {"owl/v4": {}}},
        "storage": {"kind": "local", "root": str(folder / "storage")},
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(sections))
    (folder / "test.yaml").write_text("{}\n")
    return folder


def write_owl_files(folder, weights=b"bytes that stand for the weights"):
    """Write OWL's weights and its bundled registry, and return `models register`'s arguments."""
    from importlib.resources import files

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "weights.h5").write_bytes(weights)
    registry = files("robin_models.owl") / "resources" / "taxa_registry.csv"
    (folder / "taxa_registry.csv").write_bytes(registry.read_bytes())
    return [
        "--file", f"weights={folder / 'weights.h5'}",
        "--file", f"taxa_registry={folder / 'taxa_registry.csv'}",
    ]
