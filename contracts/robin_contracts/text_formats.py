"""How Robin reads the YAML and JSON that a person writes.

Every command reads them with the same rules, so one text never means two things:

- A repeated key is refused. The parsers would keep the last value and say nothing.
- In YAML, a date stays text. A JSON value cannot be a date.
- In JSON, a number that is not finite, such as `NaN` or `1e400`, is refused.

An `InvalidText` message never quotes the text, because the text can hold a
secret. Its cause is the parser's error, which can quote it. A caller that reads a
secret raises its own error outside the handler.
"""

import json
import math
from typing import Any

import yaml


class InvalidText(ValueError):
    """The text is not valid. `problem` says why, after a subject such as "The file"."""

    def __init__(self, problem: str, *, line: int | None = None, column: int | None = None):
        self.problem = problem
        self.line = line
        self.column = column
        where = "" if line is None else f" at line {line}, column {column}"
        super().__init__(f"{problem}{where}")


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            # A key that is a list or a mapping is refused by SafeLoader itself.
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise InvalidText(
                    f"has the key {key!r} more than once",
                    line=key_node.start_mark.line + 1,
                    column=key_node.start_mark.column + 1,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_Loader.yaml_implicit_resolvers = {
    first: [(tag, rule) for tag, rule in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load_yaml(text: str) -> Any:
    """Returns the value in the YAML `text`. Raises `InvalidText` if it is not valid."""
    try:
        return yaml.load(text, Loader=_Loader)
    except InvalidText:
        raise
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        if mark is None:
            raise InvalidText("is not valid YAML") from error
        raise InvalidText(
            "is not valid YAML", line=mark.line + 1, column=mark.column + 1
        ) from error
    except (ValueError, TypeError, AttributeError, RecursionError) as error:
        # PyYAML raises these for a tagged value that it cannot build, such as
        # `!!timestamp 2026-99-99`. Their messages can quote the value.
        raise InvalidText("holds a value that YAML cannot build") from error


def load_json(text: str) -> Any:
    """Returns the value in the JSON `text`. Raises `InvalidText` if it is not valid."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_refuse_repeated_keys,
            parse_constant=_refuse_constant,
            parse_float=_parse_float,
        )
    except InvalidText:
        raise
    except json.JSONDecodeError as error:
        # `msg` is the problem only. The error's own text can quote the input.
        raise InvalidText(
            f"is not valid JSON: {error.msg}", line=error.lineno, column=error.colno
        ) from error
    except RecursionError as error:
        raise InvalidText("is nested too deeply") from error
    except ValueError as error:
        # Python raises it for an integer with more digits than it reads.
        raise InvalidText("holds a number with too many digits") from error


def _refuse_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise InvalidText(f"has the key {key!r} more than once")
        mapping[key] = value
    return mapping


def _refuse_constant(name: str) -> Any:
    raise InvalidText(f"holds {name}, which is not a JSON number")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise InvalidText("holds a number that is too large")
    return value
