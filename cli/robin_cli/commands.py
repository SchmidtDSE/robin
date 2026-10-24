"""What each `robin` command runs, and how its result is printed.

Each handler takes a `Context` and the parsed arguments, and returns an `Output`.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from robin_run_manager import migrations, models, presets, runs, submit
from robin_run_manager.compose import Services
from robin_run_manager.configuration import read_configuration
from robin_run_manager.pages import Page
from robin_run_manager.recording_inputs import read_recordings_file, read_uris_file


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
    if arguments.card is not None:
        card = models.read_card_file(arguments.card)
        model = models.prepare_model(card, dict(arguments.files))
    else:
        model = models.prepare_built_in_model(arguments.model, dict(arguments.files))

    registered_model = models.register_model(context.services.database, model, actor=context.actor)
    data = {
        "model_id": registered_model.model_id,
        "card_digest": registered_model.card_digest,
        "created": registered_model.created,
    }
    if registered_model.created:
        return Output(data=data, text=f"Registered {registered_model.model_id}.")
    return Output(data=data, text=f"{registered_model.model_id} was already registered.")


def list_models(context: Context, arguments) -> Output:
    return _list(
        partial(models.list_models, context.services.database),
        arguments,
        lambda item: f"{item['model_id']}  {item['kind']}  {item['card_digest']}",
    )


def show_model(context: Context, arguments) -> Output:
    return _show(models.show_model(context.services.database, arguments.id))


def create_preset(context: Context, arguments) -> Output:
    unresolved = read_configuration(arguments.configuration)
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
        partial(presets.list_presets, context.services.database),
        arguments,
        lambda item: (
            f"{item['name']}  {item['model_name']}/{item['model_version']}  "
            f"{item['configuration_digest']}"
        ),
    )


def show_preset(context: Context, arguments) -> Output:
    return _show(presets.show_preset(context.services.database, arguments.name))


def submit_run(context: Context, arguments) -> Output:
    if arguments.uris is not None:
        recordings = read_uris_file(arguments.uris, arguments.namespace)
    else:
        recordings = read_recordings_file(arguments.recordings)

    configuration = (
        read_configuration(arguments.configuration)
        if arguments.configuration is not None
        else None
    )

    submitted_run = submit.submit_run(
        context.services.database,
        preset=arguments.preset,
        configuration=configuration,
        recordings=recordings,
        actor=context.actor,
    )
    data = {
        "run_id": str(submitted_run.run_id),
        "status": "preparing",
        "configuration_digest": submitted_run.configuration_digest,
        "recording_count": submitted_run.recording_count,
    }
    text = (
        f"Submitted run {submitted_run.run_id} with {submitted_run.recording_count} recordings. "
        "It is preparing."
    )
    return Output(data=data, text=text)


def list_runs(context: Context, arguments) -> Output:
    read = partial(
        runs.list_runs,
        context.services.database,
        submitted_by=arguments.submitted_by,
        created_after=arguments.created_after,
    )
    return _list(
        read,
        arguments,
        lambda item: (
            f"{item['run_id']}  {item['status']}  {item['submitted_by']}  "
            f"{item['created_at']}  {item['recording_count']}"
        ),
    )


def show_run(context: Context, arguments) -> Output:
    return _show(runs.show_run(context.services.database, arguments.run_id))


def list_run_recordings(context: Context, arguments) -> Output:
    read = partial(
        runs.list_run_recordings,
        context.services.database,
        arguments.run_id,
        state=arguments.state,
        skip_reason=arguments.skip_reason,
    )
    return _list(
        read,
        arguments,
        lambda item: (
            f"{item['namespace']}  {item['value']}  {item['state'] or '-'}  "
            f"{item['skip_reason'] or '-'}"
        ),
    )


def list_run_tasks(context: Context, arguments) -> Output:
    return _list(
        partial(runs.list_run_tasks, context.services.database, arguments.run_id),
        arguments,
        lambda item: f"{item['task_id']}  {item['state']}  {item['recording_count']}",
    )


def _list(read: Callable[..., Page], arguments, line: Callable[[dict], str]) -> Output:
    if arguments.all:
        page = _read_all_pages(read, arguments.limit)
    else:
        page = read(limit=arguments.limit, cursor=arguments.cursor)
    notice = None
    if page.next_cursor is not None:
        notice = f"There are more. Show them with --cursor {page.next_cursor}"
    return Output(
        data=page.body(), text="\n".join(line(item) for item in page.items), notice=notice
    )


def _read_all_pages(read: Callable[..., Page], limit: int) -> Page:
    """All pages as one page. The filters stay the same on every page.

    If a page fails, nothing is printed.
    """
    items, cursor = [], None
    while True:
        page = read(limit=limit, cursor=cursor)
        items.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return Page(items=tuple(items), next_cursor=None)


def _show(data: dict) -> Output:
    lines = []
    for name, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        lines.append(f"{name}: {value}")
    return Output(data=data, text="\n".join(lines))
