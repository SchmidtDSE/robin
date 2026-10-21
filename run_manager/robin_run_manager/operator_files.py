"""Reading the YAML files that an operator gives to a command."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from robin_contracts.cards import read_yaml_mapping
from robin_run_manager.errors import INVALID_REQUEST, Refused


def read_yaml_or_refuse(path: Path, what: str) -> dict[str, Any]:
    """The mapping in the YAML file at `path`. `what` names the file in messages.

    Raises `Refused` with `invalid_request` if the file cannot be read, is not YAML,
    has a repeated key, or does not hold a mapping.
    """
    try:
        return read_yaml_mapping(path, what)
    except (OSError, ValueError) as error:
        problem = str(error)
    raise Refused(
        INVALID_REQUEST,
        f"{path} cannot be read as a {what}: {problem}",
        details={"file": str(path)},
    )


def field_problems(error: ValidationError, whole: str) -> list[dict[str, str]]:
    """Each field error as `{"field", "message"}`. `whole` names a problem with the whole file.

    The values are left out, so that a secret from a profile is never shown.
    """
    return [
        {
            "field": ".".join(str(part) for part in problem["loc"]) or whole,
            "message": problem["msg"],
        }
        for problem in error.errors(include_url=False, include_context=False, include_input=False)
    ]
