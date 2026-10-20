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
    sections = {"database": database, "catalogs": {"soundhub": {"kind": "soundhub"}}}
    (folder / "config.yaml").write_text(yaml.safe_dump(sections))
    (folder / "test.yaml").write_text("{}\n")
    return folder
