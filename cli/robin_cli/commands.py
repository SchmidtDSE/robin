"""What each `robin` command runs, and how its result is printed.

Each handler takes a `Context` and the parsed arguments, and returns an `Output`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robin_run_manager import migrations
from robin_run_manager.compose import Services


@dataclass(frozen=True)
class Output:
    """A command's result. With --json, `data` is printed; otherwise `text`.

    `notice` is an extra message for people. It goes to stderr, and only without --json.
    """

    data: Any
    text: str
    notice: str | None = None


@dataclass(frozen=True)
class Context:
    """What each command gets: the services, the actor, and the selected profile.

    `actor` is the database login, which PostgreSQL has checked.
    """

    services: Services
    actor: str
    profile: Path
    environment: str


def migrate(context: Context, arguments) -> Output:
    names = [migration.name for migration in migrations.migrate(context.services.database)]
    text = "\n".join(f"Applied {name}." for name in names) or "The schema is current."
    return Output(data={"applied": names}, text=text)


def whoami(context: Context, arguments) -> Output:
    database = context.services.profile.database
    data = {
        "login": context.actor,
        "host": database.host,
        "database": database.dbname,
        "profile": str(context.profile),
        "environment": context.environment,
    }
    return Output(data=data, text="\n".join(f"{name}: {value}" for name, value in data.items()))
