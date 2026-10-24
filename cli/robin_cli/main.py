"""The `robin` operator command.

`robin [--profile DIR] [--env ENV] [--json] <command> ...` loads the profile, logs in
to the profile's database, runs one shared operation, and prints its result.
Success output goes to stdout. Messages and errors go to stderr. With --json, every
error is one JSON body on stderr.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from robin_run_manager import models
from robin_run_manager.compose import Services, compose
from robin_run_manager.database import current_login
from robin_run_manager.errors import NotConfigured, OperationError, OutcomeUncertain, Unavailable
from robin_run_manager.migrations import require_current_schema
from robin_run_manager.pages import DEFAULT_LIMIT, MAX_LIMIT
from robin_run_manager.profile import load_profile
from robin_run_manager.runs import RECORDING_STATES, SKIP_REASONS

from robin_cli import commands
from robin_cli.commands import Context, Output

PROFILE_VARIABLE = "ROBIN_PROFILE"
ENVIRONMENT_VARIABLE = "ROBIN_ENV"

EXIT_SUCCESS = 0
EXIT_REFUSED = 1
EXIT_LOCAL_PROBLEM = 2
EXIT_OUTCOME_UNCERTAIN = 3
EXIT_UNAVAILABLE = 4
EXIT_INTERNAL_ERROR = 5

# The pool logs each failed background connection. With no handler, Python prints these
# records on stderr, which would break the rule of one JSON body on stderr. A failure
# that matters reaches the command as an error, and the CLI reports that.
logging.getLogger("psycopg.pool").addHandler(logging.NullHandler())


class _LocalProblem(Exception):
    """Wrong syntax, or no profile or environment is selected. Exit code 2.

    `text` is argparse's own output for a syntax error. It is shown instead of the
    message when the output is not JSON.
    """

    def __init__(self, reason: str, message: str, *, text: str = "") -> None:
        super().__init__(message)
        self.body = {"reason": reason, "message": message, "details": {}}
        self.text = text


# ---------------------------------------------------------------------------
# Run one command.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    """Run the command in `argv` and return the exit code.

    --help and --version print on stdout and exit through argparse.
    """
    environ = os.environ if environ is None else environ
    argv = sys.argv[1:] if argv is None else list(argv)
    # Found before parsing, so that a syntax error is also reported as JSON.
    json_mode = "--json" in argv
    try:
        arguments = _parse(_build_parser(), argv)
        directory, environment = _get_selection(arguments, environ)
        profile = load_profile(directory, environment, environ)
        if not json_mode:
            print(f"robin: profile {directory}, environment {environment}", file=sys.stderr)
        services = compose(profile)
        with contextlib.closing(services.database):
            output = _run(arguments, services, directory, environment)
    except _LocalProblem as problem:
        if problem.text and not json_mode:
            sys.stderr.write(problem.text)
        else:
            _print_error(problem.body, json_mode=json_mode)
        return EXIT_LOCAL_PROBLEM
    except OperationError as error:
        # For example: a profile file is missing or not valid, the database refused
        # the login or cannot be reached, or the operation refused.
        return _report(error, json_mode=json_mode)
    except Exception:
        return _report_defect(json_mode=json_mode)
    _print(output, json_mode=json_mode)
    return EXIT_SUCCESS


def _run(
    arguments: argparse.Namespace, services: Services, directory: Path, environment: str
) -> Output:
    actor = current_login(services.database)
    if arguments.needs_schema:
        require_current_schema(services.database)
    return arguments.run(Context(services, actor, directory, environment), arguments)


def _parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Parse `argv`. A syntax error becomes a `_LocalProblem` that holds argparse's text.

    A command's `check` tests the rules that argparse cannot. It also raises a `_LocalProblem`.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            arguments = parser.parse_args(argv)
    except SystemExit as exit:
        if exit.code == 0:
            raise  # --help or --version; their text is on stdout
        text = captured.getvalue()
        lines = text.strip().splitlines()
        message = lines[-1] if lines else "The command line is not valid."
        raise _LocalProblem("invalid_arguments", message, text=text) from None
    if arguments.check is not None:
        arguments.check(arguments)
    return arguments


def _get_selection(
    arguments: argparse.Namespace, environ: Mapping[str, str]
) -> tuple[Path, str]:
    directory = arguments.profile or environ.get(PROFILE_VARIABLE)
    environment = arguments.env or environ.get(ENVIRONMENT_VARIABLE)
    if not directory:
        raise _LocalProblem(
            "not_configured", f"No profile. Use --profile, or set {PROFILE_VARIABLE}."
        )
    if not environment:
        raise _LocalProblem(
            "not_configured",
            f"No environment. Use --env, or set {ENVIRONMENT_VARIABLE}. "
            "Robin has no default environment.",
        )
    return Path(directory), environment


# ---------------------------------------------------------------------------
# The parser, and the commands that are not in a group.
# ---------------------------------------------------------------------------


OVERVIEW = """\
Every command except --help and --version needs a profile folder (--profile or
ROBIN_PROFILE) and one environment in it (--env or ROBIN_ENV). There is no default
environment.

The usual order of work:

  robin db migrate
  robin models register owl/v4 --file weights=.model_assets/owl/v4/weights.h5
  robin presets create dawn-owl --configuration owl.yaml
  robin runs submit --uris uris.txt --namespace survey --preset dawn-owl
  robin runs show RUN

This version stores submitted runs but does not prepare or run them yet. A submitted
run stays in "preparing".

Use --json in scripts. Exit codes: 0 success, 1 refused, 2 command or profile
problem, 3 outcome uncertain, 4 unavailable, 5 internal error.

Give robin COMMAND ACTION --help for the details of each command.
"""


MIGRATE_EPILOG = """\
Safe to run again: each migration is applied once. Commands that read or write
Robin's tables refuse with schema_not_current until the schema is current.

Example:

  robin db migrate
"""

WHOAMI_EPILOG = """\
Robin knows you by your database login. This works before robin db migrate.

Example:

  robin --profile ~/robin-profile --env local whoami
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robin",
        description="Register models, name configurations, and submit them to run on "
        "lists of recordings.",
        epilog=OVERVIEW,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile", metavar="DIR", help=f"the profile folder; replaces {PROFILE_VARIABLE}"
    )
    parser.add_argument(
        "--env", metavar="ENV", help=f"the environment in the profile; replaces {ENVIRONMENT_VARIABLE}"
    )
    parser.add_argument("--json", action="store_true", help="print results and errors as JSON")
    parser.add_argument("--version", action="version", version=f"robin {version('robin-cli')}")
    nouns = parser.add_subparsers(dest="noun", required=True, metavar="COMMAND")

    db = _add_group(nouns, "db", "the database schema")
    _add_command(
        db, "migrate", commands.migrate, "apply the migrations that the database does not have",
        needs_schema=False,
        epilog=MIGRATE_EPILOG,
    )
    _add_command(
        nouns, "whoami", commands.whoami,
        "show your database login and the selected profile", needs_schema=False,
        epilog=WHOAMI_EPILOG,
    )

    _add_model_commands(nouns)
    _add_preset_commands(nouns)
    _add_run_commands(nouns)
    return parser


# ---------------------------------------------------------------------------
# robin models
# ---------------------------------------------------------------------------


REGISTRATION_FORMAT = """\
Name a built-in model, or give the card of another model with --card. Give each
file the model reads with --file ROLE=LOCATION. Robin reads every file to pin it
by its sha256 digest and size, and checks it before anything is stored.

The files each model always reads, by role:

{roles}

Every model can also take a taxa_registry file. It is needed only for scores:
without it, a configuration of the model cannot request scores. It must be the
registry that the card names. A built-in model takes the registry that robin-models
ships, stored as robin-models:MODEL/taxa_registry.csv, unless --file
taxa_registry=LOCATION gives another copy. A model given with --card takes its
registry only with --file.

A head is given with --card. A LOCATION is a path or a file:/// URI. A relative
path is taken from the current folder, and Robin stores the absolute path. Files on
S3 cannot be registered yet.

Example:

  robin models register owl/v4 --file weights=.model_assets/owl/v4/weights.h5
"""


MODEL_SHOW_EPILOG = """\
Prints the model's card, and each file's role, location, sha256 digest, and size.
The card's inference_params are the settings that a configuration can give.

Example:

  robin models show owl/v4
"""


def _add_model_commands(nouns) -> None:
    model_commands = _add_group(nouns, "models", "registered models")
    register = _add_command(
        model_commands, "register", commands.register_model,
        "register a model and the files it reads", check=_check_register,
        epilog=REGISTRATION_FORMAT.format(
            roles=_get_role_text(models.get_roles_by_model())
        ),
    )
    register.add_argument(
        "model", metavar="MODEL", nargs="?", help="a built-in model, for example owl/v4"
    )
    register.add_argument(
        "--card", type=Path, metavar="FILE", help="the card of a model that is not built in"
    )
    register.add_argument(
        "--file", metavar="ROLE=LOCATION", dest="files", action="append", required=True,
        type=_parse_role_and_location, help="one file the model reads; repeat for each role",
    )
    _add_list_command(
        model_commands, "list", commands.list_models, "list models",
        columns="MODEL  KIND  CARD_DIGEST", note="KIND is model or head.",
        example="robin models list --all",
    )
    _add_command(
        model_commands, "show", commands.show_model, "show one model",
        epilog=MODEL_SHOW_EPILOG,
    ).add_argument("id", metavar="ID", help="the model's name/version, for example owl/v4")


def _get_role_text(roles_by_model) -> str:
    return "\n".join(f"  {name:<14}{', '.join(sorted(roles))}" for name, roles in roles_by_model.items())


_URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def _parse_role_and_location(text: str) -> tuple[str, str]:
    """Parse ROLE=LOCATION. A relative path is made absolute from the current folder."""
    role, equals, location = text.partition("=")
    if not equals or not role or not location:
        raise argparse.ArgumentTypeError(f"{text!r} is not ROLE=LOCATION")
    if not location.startswith("/") and _URI_SCHEME.match(location) is None:
        # abspath, not resolve: a symlink stays the stored location, so the digest check
        # still catches a link that is pointed at a changed file.
        location = os.path.abspath(location)
    return role, location


# ---------------------------------------------------------------------------
# robin presets
# ---------------------------------------------------------------------------


CONFIGURATION_FORMAT = """\
A configuration file is YAML with these fields:

  model     the registered model, as name/version
  input     kind: audio, for a model that reads audio; or kind: embedding_artifact
            and backbone_configuration: DIGEST, for a head. DIGEST is the digest
            of the backbone configuration whose embeddings the head reads; that
            configuration must request embeddings.
  settings  optional; only the names that the model card lists under
            inference_params (see robin models show MODEL)
  outputs   a list with at most one of each kind:
              kind: scores, contract_id: robin.scores.parquet/1, and a retention:
                full (no other fields), thresholded (with min_score), or top_k
                (with top_k and min_score)
              kind: embeddings, contract_id: robin.embeddings.parquet/1, and an
                optional storage_dtype: float32 or float16
              kind: detections, contract_id: robin.detections.parquet/1, and a
                policy: {kind: threshold, min_score: X} or {kind: top_k, k: N},
                where top_k can also take min_score

Example configuration:

  model: owl/v4
  input:
    kind: audio
  outputs:
    - kind: scores
      contract_id: robin.scores.parquet/1
      retention: full
"""


PRESET_CREATE_EPILOG = f"""\
Robin checks the configuration against the registered model and stores it by its
digest. Creating the same name with the same configuration again is safe. The same
name with another configuration is refused: a preset never changes.

{CONFIGURATION_FORMAT}
Example:

  robin presets create dawn-owl --configuration owl.yaml
"""

PRESET_SHOW_EPILOG = """\
Prints the preset's digest, who created it and when, and its resolved configuration.

Example:

  robin presets show dawn-owl
"""


def _add_preset_commands(nouns) -> None:
    preset_commands = _add_group(nouns, "presets", "named configurations")
    create = _add_command(
        preset_commands, "create", commands.create_preset, "name a configuration",
        epilog=PRESET_CREATE_EPILOG,
    )
    create.add_argument("name", metavar="NAME", help="the new preset's name")
    create.add_argument(
        "--configuration", type=Path, metavar="FILE", required=True, help="the configuration to name"
    )
    _add_list_command(
        preset_commands, "list", commands.list_presets, "list presets",
        columns="NAME  MODEL  CONFIGURATION_DIGEST", example="robin presets list --limit 20",
    )
    _add_command(
        preset_commands, "show", commands.show_preset, "show one preset",
        epilog=PRESET_SHOW_EPILOG,
    ).add_argument("name", metavar="NAME", help="the preset's name")


# ---------------------------------------------------------------------------
# robin runs
# ---------------------------------------------------------------------------


RECORDINGS_FORMAT = """\
Give the recordings in one of two forms.

--recordings FILE is JSON Lines: one object on each line, with namespace, value,
audio_uri, and an optional duration_seconds. namespace and value identify the
recording; audio_uri is where its audio is.

  {"namespace": "survey", "value": "site-1/0500", "audio_uri": "s3://audio/site-1/0500.wav"}

--uris FILE has one absolute path or URI on each line, such as s3://audio/a.wav.
Give --namespace with it. An absolute path is stored as its file:/// URI, and a
file: line must already be in that form. Each URI is also the recording's value.

Both files are UTF-8. Empty lines are skipped. Each recording is given once.
"""

RUN_STATUSES = """\
A run's status is one of:

  preparing               stored; Robin has not prepared it yet
  preparation_failed      preparation stopped; failures.preparation says why
  queued                  tasks wait to start
  running                 at least one task has started and not ended
  finished                every task ended, or there were none, and no
                          recording failed
  finished_with_failures  every task ended, and at least one recording failed
  cancelled               the run was cancelled
  withdrawn               the run was withdrawn

"finished" means that the run's tasks ended, not that each recording has a result.
A skipped recording, for example one already covered, has no result from this run.
"""


RUN_HELP = "the run's ID, from robin runs submit or robin runs list"


RUN_SUBMIT_EPILOG = f"""\
Robin stores the run, its configuration, and its recordings, with the status
preparing. This version does not prepare or run it yet. Give a preset, or a
configuration file in the format that robin presets create --help describes.

{RECORDINGS_FORMAT}
Examples:

  robin runs submit --uris uris.txt --namespace survey --preset dawn-owl
  robin runs submit --recordings recordings.jsonl --configuration owl.yaml
"""

RUN_SHOW_EPILOG = f"""\
Prints the run's status, who submitted it and when, its configuration digest and
output versions, its task and recording counts by state, its skip counts by reason,
and any failure.

{RUN_STATUSES}
Example:

  robin runs show 0b9c1f9e-5d1c-4f43-9f43-3a6f0e8d2c11
"""


def _add_run_commands(nouns) -> None:
    run_commands = _add_group(nouns, "runs", "inference runs")
    submit = _add_command(
        run_commands, "submit", commands.submit_run,
        "submit a run of a list of recordings", check=_check_submit,
        epilog=RUN_SUBMIT_EPILOG,
    )
    inputs = submit.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--recordings", type=Path, metavar="FILE", help="a JSON Lines file with one recording on each line"
    )
    inputs.add_argument(
        "--uris", type=Path, metavar="FILE", help="a file with one absolute path or URI on each line"
    )
    submit.add_argument(
        "--namespace", metavar="NS",
        help="with --uris: the collection that the recordings belong to",
    )
    computes = submit.add_mutually_exclusive_group(required=True)
    computes.add_argument("--preset", metavar="P", help="the preset to run")
    computes.add_argument(
        "--configuration", type=Path, metavar="FILE", help="a configuration file to run"
    )

    listed = _add_list_command(
        run_commands, "list", commands.list_runs, "list runs, newest first",
        columns="RUN  STATUS  SUBMITTED_BY  CREATED_AT  RECORDINGS",
        note="robin runs show --help lists the statuses.",
        example="robin runs list --submitted-by ana --created-after 2026-10-01T00:00:00Z",
    )
    listed.add_argument(
        "--submitted-by", metavar="ACTOR", help="only runs that this database login submitted"
    )
    listed.add_argument(
        "--created-after", metavar="T",
        help="only runs created after T, an ISO 8601 time with a time zone",
    )
    _add_command(
        run_commands, "show", commands.show_run, "show one run",
        epilog=RUN_SHOW_EPILOG,
    ).add_argument("run_id", metavar="RUN", help=RUN_HELP)
    recordings = _add_list_command(
        run_commands, "recordings", commands.list_run_recordings, "list a run's recordings",
        columns="NAMESPACE  VALUE  STATE  SKIP_REASON",
        note="STATE is - until preparation gives the recording one. SKIP_REASON is - "
        "unless the\nrecording was skipped.",
        example="robin runs recordings RUN --state skipped --skip-reason already_covered",
    )
    recordings.add_argument("run_id", metavar="RUN", help=RUN_HELP)
    recordings.add_argument(
        "--state", metavar="S", help=f"only recordings in this state: {', '.join(RECORDING_STATES)}"
    )
    recordings.add_argument(
        "--skip-reason", metavar="R",
        help=f"only recordings skipped for this reason: {', '.join(SKIP_REASONS)}",
    )
    _add_list_command(
        run_commands, "tasks", commands.list_run_tasks, "list a run's tasks",
        columns="TASK  STATE  RECORDINGS", example="robin runs tasks RUN",
    ).add_argument("run_id", metavar="RUN", help=RUN_HELP)


# ---------------------------------------------------------------------------
# Add groups, commands, and their shared options.
# ---------------------------------------------------------------------------


def _add_list_command(
    group, name: str, run, help: str, *, columns: str, example: str, note: str | None = None
):
    """Add a command that prints a page of items, one on each line, with the page options.

    The epilog gives the columns, the optional note, and the example.
    """
    epilog = f"Each line is: {columns}\n"
    if note:
        epilog += f"{note}\n"
    epilog += f"\nExample:\n\n  {example}\n"
    command = _add_command(group, name, run, help, epilog=epilog)
    _add_page_options(command)
    return command


def _add_page_options(command) -> None:
    # In their own help section, so that a command's filters show first.
    section = command.add_argument_group("page options")
    section.add_argument(
        "--limit", metavar="N", type=int, default=DEFAULT_LIMIT,
        help=f"items on each page, 1 to {MAX_LIMIT} (default {DEFAULT_LIMIT})",
    )
    where = section.add_mutually_exclusive_group()
    where.add_argument("--cursor", metavar="C", help="start after the page that gave this cursor")
    where.add_argument("--all", action="store_true", help="follow every page")


def _add_group(nouns, name: str, help: str):
    return nouns.add_parser(name, help=help, description=help).add_subparsers(
        dest="verb", required=True, metavar="ACTION"
    )


def _add_command(
    group,
    name: str,
    run,
    help: str,
    *,
    needs_schema: bool = True,
    check: Callable[[argparse.Namespace], None] | None = None,
    epilog: str | None = None,
):
    # The raw formatter keeps the epilog's line breaks and indented examples.
    command = group.add_parser(
        name, help=help, description=help, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Also accepted after the command. SUPPRESS keeps a --json given before it.
    command.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="print results and errors as JSON",
    )
    command.set_defaults(run=run, needs_schema=needs_schema, check=check)
    return command


# ---------------------------------------------------------------------------
# Check the rules that argparse cannot.
# ---------------------------------------------------------------------------


def _check_submit(arguments: argparse.Namespace) -> None:
    """Check the submit options that argparse cannot: --namespace goes with --uris only."""
    if arguments.uris is not None and arguments.namespace is None:
        raise _LocalProblem("invalid_arguments", "--uris needs --namespace.")
    if arguments.recordings is not None and arguments.namespace is not None:
        raise _LocalProblem(
            "invalid_arguments",
            "--namespace is used only with --uris. Each line of --recordings has its own.",
        )


def _check_register(arguments: argparse.Namespace) -> None:
    """Check the register options that argparse cannot: one model source, and each role once."""
    if (arguments.model is None) == (arguments.card is None):
        raise _LocalProblem(
            "invalid_arguments", "Name a built-in model, or give --card, but not both."
        )
    roles = [role for role, _ in arguments.files]
    repeated = sorted({role for role in roles if roles.count(role) > 1})
    if repeated:
        raise _LocalProblem(
            "invalid_arguments", f"Give each role once. Repeated: {', '.join(repeated)}."
        )


# ---------------------------------------------------------------------------
# Print results and errors.
# ---------------------------------------------------------------------------


def _print(output: Output, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(output.data))
        return
    if output.text:
        print(output.text)
    if output.notice:
        print(f"robin: {output.notice}", file=sys.stderr)


def _report(error: OperationError, *, json_mode: bool) -> int:
    _print_error(error.body(), json_mode=json_mode)
    if isinstance(error, NotConfigured):
        return EXIT_LOCAL_PROBLEM
    if isinstance(error, OutcomeUncertain):
        return EXIT_OUTCOME_UNCERTAIN
    if isinstance(error, Unavailable):
        return EXIT_UNAVAILABLE
    return EXIT_REFUSED


def _report_defect(*, json_mode: bool) -> int:
    request_id = uuid.uuid4().hex
    print(f"robin: internal error, request {request_id}:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    body = {
        "reason": "internal_error",
        "message": "Robin failed because of a defect. The lines above give the details.",
        "details": {"request_id": request_id},
    }
    _print_error(body, json_mode=json_mode)
    return EXIT_INTERNAL_ERROR


def _print_error(body: dict, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(body), file=sys.stderr)
    else:
        print(f"robin: {body['message']} ({body['reason']})", file=sys.stderr)
