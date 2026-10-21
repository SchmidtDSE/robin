"""What each `robin` command runs, and how its result is printed.

Each handler takes a `Context` and the parsed arguments, and returns an `Output`.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robin_run_manager import migrations, models, presets
from robin_run_manager.compose import Services
from robin_run_manager.configuration import read_configuration
from robin_run_manager.pages import Page


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


def register_model(context: Context, arguments) -> Output:
    model = models.read_registration(Path(arguments.file))
    registered = models.register_model(context.services.database, model, actor=context.actor)
    data = {
        "model_id": registered.model_id,
        "card_digest": registered.card_digest,
        "created": registered.created,
    }
    if registered.created:
        return Output(data=data, text=f"Registered {registered.model_id}.")
    return Output(data=data, text=f"{registered.model_id} was already registered.")


def list_models(context: Context, arguments) -> Output:
    return _list(
        models.list_models,
        context,
        arguments,
        lambda item: f"{item['model_id']}  {item['kind']}  {item['card_digest']}",
    )


def show_model(context: Context, arguments) -> Output:
    return _show(models.show_model(context.services.database, arguments.id))


def create_preset(context: Context, arguments) -> Output:
    unresolved = read_configuration(Path(arguments.configuration))
    created = presets.create_preset(
        context.services.database, arguments.name, unresolved, actor=context.actor
    )
    data = {
        "name": created.name,
        "configuration_digest": created.configuration_digest,
        "created": created.created,
    }
    if created.created:
        text = f"Created preset {created.name}: {created.configuration_digest}."
    else:
        text = f"Preset {created.name} already exists: {created.configuration_digest}."
    return Output(data=data, text=text)


def list_presets(context: Context, arguments) -> Output:
    return _list(
        presets.list_presets,
        context,
        arguments,
        lambda item: (
            f"{item['name']}  {item['model_name']}/{item['model_version']}  "
            f"{item['configuration_digest']}"
        ),
    )


def show_preset(context: Context, arguments) -> Output:
    return _show(presets.show_preset(context.services.database, arguments.name))


def _list(
    read: Callable[..., Page], context: Context, arguments, line: Callable[[dict], str]
) -> Output:
    database = context.services.database
    if arguments.all:
        # The filters stay the same on every page. If a page fails, nothing is printed.
        items, cursor = [], None
        while True:
            page = read(database, limit=arguments.limit, cursor=cursor)
            items.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        page = Page(items=tuple(items), next_cursor=None)
    else:
        page = read(database, limit=arguments.limit, cursor=arguments.cursor)
    notice = None
    if page.next_cursor is not None:
        notice = f"There are more. Show them with --cursor {page.next_cursor}"
    return Output(
        data=page.body(), text="\n".join(line(item) for item in page.items), notice=notice
    )


def _show(data: dict) -> Output:
    lines = []
    for name, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        lines.append(f"{name}: {value}")
    return Output(data=data, text="\n".join(lines))
