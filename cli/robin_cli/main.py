"""The `robin` operator command.

`robin [--profile DIR] [--env ENV] [--json] <command> ...` loads the profile, logs in
to the profile's database, runs one shared operation, and prints its result.
Success output goes to stdout. Messages and errors go to stderr. With --json, every
error is one JSON body on stderr.
"""

import argparse
import contextlib
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from robin_contracts.local_files import has_uri_scheme
from robin_run_manager import models
from robin_run_manager.compose import Services, compose
from robin_run_manager.database import NoticeListener, check_login, current_login, listen
from robin_run_manager.errors import (
    CatalogFailed,
    ComputeUnavailable,
    InternalError,
    NotConfigured,
    OperationError,
    OutcomeUncertain,
    SourceUnavailable,
    StorageUnavailable,
    Unavailable,
)
from robin_run_manager.migrations import require_current_schema
from robin_run_manager.pages import DEFAULT_LIMIT, MAX_LIMIT
from robin_run_manager.profile import load_profile
from robin_run_manager.reconcile import ReconcileReport, StepFailure
from robin_run_manager.runs import RECORDING_STATES, SKIP_REASONS

from robin_cli import commands
from robin_cli.color import BOLD, DIM, RED_BOLD, STATUS_COLORS, YELLOW, add_color
from robin_cli.commands import Context, Output

PROFILE_VARIABLE = "ROBIN_PROFILE"
ENVIRONMENT_VARIABLE = "ROBIN_ENV"

EXIT_SUCCESS = 0
EXIT_REFUSED = 1
EXIT_LOCAL_PROBLEM = 2
EXIT_OUTCOME_UNCERTAIN = 3
EXIT_UNAVAILABLE = 4
EXIT_INTERNAL_ERROR = 5

# When a command reports more than one failure, the first code in this order wins.
EXIT_CODES_MOST_SERIOUS_FIRST = (
    EXIT_OUTCOME_UNCERTAIN,
    EXIT_INTERNAL_ERROR,
    EXIT_UNAVAILABLE,
    EXIT_LOCAL_PROBLEM,
    EXIT_REFUSED,
)

DEFAULT_INTERVAL_SECONDS = 30
# The longest time the loop waits before it sees a stop signal.
_STOP_CHECK_SECONDS = 0.5

# The pool logs each failed background connection. With no handler, Python prints these
# records on stderr, which would break the rule of one JSON body on stderr. A failure
# that matters reaches the command as an error, and the CLI reports that.
logging.getLogger("psycopg.pool").addHandler(logging.NullHandler())


class _LocalProblem(Exception):
    """Wrong syntax, or no profile or environment is selected. Exit code 2.

    For a syntax error, `usage` is argparse's usage text and the message is argparse's
    error line, such as `robin runs submit: error: ...`. Without JSON, both are shown
    as argparse shows them. `command` is the command whose help the hint names, such as
    `robin runs submit`.
    """

    def __init__(
        self, reason: str, message: str, *, usage: str = "", command: str = "robin"
    ) -> None:
        super().__init__(message)
        self.body = {"reason": reason, "message": message, "details": {}}
        self.usage = usage
        self.command = command


class _Parser(argparse.ArgumentParser):
    """An argparse parser that raises a `_LocalProblem` instead of printing and exiting.

    Each subparser is also a `_Parser`, because argparse makes subparsers of the
    parent's class. So `prog` names the command that the user typed.
    """

    def error(self, message: str):
        raise _LocalProblem(
            "invalid_arguments",
            f"{self.prog}: error: {message}",
            usage=self.format_usage(),
            command=self.prog,
        )


def is_terminal(stream) -> bool:
    """True if `stream` is a terminal. The tests replace this function."""
    return stream.isatty()


def is_color_on(stream, *, json_mode: bool, environ: Mapping[str, str]) -> bool:
    """True if output on `stream` gets color: not --json, a terminal, and not turned off.

    A `NO_COLOR` that is not empty, or `TERM=dumb`, turns color off.
    """
    if json_mode or environ.get("NO_COLOR") or environ.get("TERM") == "dumb":
        return False
    return is_terminal(stream)


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
    table = not json_mode and is_terminal(sys.stdout)
    color = is_color_on(sys.stdout, json_mode=json_mode, environ=environ)
    error_color = is_color_on(sys.stderr, json_mode=json_mode, environ=environ)
    try:
        arguments = _parse(_build_parser(), argv)
        directory, environment = _get_selection(arguments, environ)
        profile = load_profile(directory, environment, environ)
        if not json_mode:
            line = f"robin: profile {directory}, environment {environment}"
            print(add_color(line, DIM, enabled=error_color), file=sys.stderr)
        services = compose(profile)
        with contextlib.closing(services.database):
            context = _get_context(arguments, services, directory, environment, environ)
            if getattr(arguments, "loop", False):
                return _reconcile_in_loop(context, arguments, json_mode=json_mode)
            output = arguments.run(context, arguments)
    except _LocalProblem as problem:
        _print_local_problem(problem, json_mode=json_mode, color=error_color)
        return EXIT_LOCAL_PROBLEM
    except OperationError as error:
        # For example: a profile file is missing or not valid, the database refused
        # the login or cannot be reached, or the operation refused.
        return _report(error, json_mode=json_mode, color=error_color)
    except Exception:
        return _report_defect(json_mode=json_mode, color=error_color)
    _print(output, json_mode=json_mode, table=table, color=color, notice_color=error_color)
    return _get_most_serious_exit_code(output.errors)


def _get_context(
    arguments: argparse.Namespace,
    services: Services,
    directory: Path,
    environment: str,
    environ: Mapping[str, str],
) -> Context:
    actor = current_login(services.database)
    if arguments.needs_schema:
        require_current_schema(services.database)
    return Context(services, actor, directory, environment, environ)


def _parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Parse `argv`. A syntax error raises a `_LocalProblem` from `_Parser.error`.

    A command's `check` tests the rules that argparse cannot. If it returns a message,
    that message becomes a `_LocalProblem`.
    """
    arguments = parser.parse_args(argv)
    if arguments.check is not None:
        message = arguments.check(arguments)
        if message is not None:
            raise _LocalProblem("invalid_arguments", message, command=arguments.prog)
    return arguments


def _get_selection(
    arguments: argparse.Namespace, environ: Mapping[str, str]
) -> tuple[Path, str]:
    directory = arguments.profile or environ.get(PROFILE_VARIABLE)
    environment = arguments.env or environ.get(ENVIRONMENT_VARIABLE)
    if not directory and not environment:
        raise _LocalProblem(
            "not_configured",
            "No profile and no environment. Use --profile and --env, or set "
            f"{PROFILE_VARIABLE} and {ENVIRONMENT_VARIABLE}.",
        )
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
  robin reconcile --loop
  robin runs show RUN

robin reconcile --loop prepares each submitted run, starts its tasks with the
profile's compute, and records their results. Stop it with Ctrl-C.

Human output has color only on a terminal. Set NO_COLOR, or TERM=dumb, to turn
color off.

Use --json in scripts. Exit codes: 0 success, 1 refused, 2 command or profile
problem, 3 outcome uncertain, 4 the database, a catalog source, the compute, or
the task storage is unavailable, 5 internal error or a catalog adapter failed.

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
    parser = _Parser(
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
    _add_reconcile_command(nouns)
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
        columns=commands.MODEL_COLUMNS, note="KIND is model or head.",
        example="robin models list --all",
    )
    _add_command(
        model_commands, "show", commands.show_model, "show one model",
        epilog=MODEL_SHOW_EPILOG,
    ).add_argument("id", metavar="ID", help="the model's name/version, for example owl/v4")


def _get_role_text(roles_by_model) -> str:
    return "\n".join(f"  {name:<14}{', '.join(sorted(roles))}" for name, roles in roles_by_model.items())


def _parse_role_and_location(text: str) -> tuple[str, str]:
    """Parse ROLE=LOCATION. A relative path is made absolute from the current folder."""
    role, equals, location = text.partition("=")
    if not equals or not role or not location:
        raise argparse.ArgumentTypeError(f"{text!r} is not ROLE=LOCATION")
    if not location.startswith("/") and not has_uri_scheme(location):
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
        columns=commands.PRESET_COLUMNS, example="robin presets list --limit 20",
    )
    _add_command(
        preset_commands, "show", commands.show_preset, "show one preset",
        epilog=PRESET_SHOW_EPILOG,
    ).add_argument("name", metavar="NAME", help="the preset's name")


# ---------------------------------------------------------------------------
# robin runs
# ---------------------------------------------------------------------------


RECORDINGS_FORMAT = """\
Give the recordings in one of three forms. Each recording is given once.

--recordings FILE is JSON Lines, one recording on each line. namespace and value
identify it, and duration_seconds is optional:

  {"namespace": "survey", "value": "site-1/0500", "audio_uri": "s3://audio/site-1/0500.wav"}

--uris FILE has one absolute path, file:/// URI, or s3:// URI on each line, and
each is also the recording's value. Give --namespace with it. A path is stored as
its file:/// URI, and a file: line must already be in that form.

Both files are UTF-8, and empty lines are skipped.

--selection JSON or --selection @FILE (a YAML file) asks the catalog adapter that
the profile's catalog section names: kind for a built-in one, or code
(module:function) for an installed one. Each ${NAME} in its settings is filled in
only for --selection. The built-in kinds are:

  folder-layout           files under root that match pattern; give each field of
                          the pattern a value, a list, or {"from": ..., "to": ...}
  csv-list                the rows of a CSV file; give {"file": PATH}

If the catalog step fails, nothing is stored. The exit code is 1 for a bad
selection, bad catalog settings, or a refusal from the source; 2 if the profile has
no catalog section or the adapter is not installed; 4 if the source is unavailable
(try again); and 5 if the adapter failed. --json gives the reason.

Each audio location, and each file of the configuration's model, must have a form
that the profile's compute kind can read:

  local                   an absolute path, a file:/// URI, or an s3:// URI
  kubernetes              an s3:// URI

A model file can also be a robin-models: location, for a built-in model's taxa
registry. Robin refuses any other form, such as http://, with unsupported_location,
and stores nothing.
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
preparing. robin reconcile prepares and runs it. Give a preset, or a configuration
file in the format that robin presets create --help describes.

The profile must have a compute section and a storage section. The run keeps
both, so a later change to the profile does not change the run. Without a compute
section, the command stops with not_configured.

{RECORDINGS_FORMAT}
Examples:

  robin runs submit --uris uris.txt --namespace survey --preset dawn-owl
  robin runs submit --recordings recordings.jsonl --configuration owl.yaml
  robin runs submit --selection '{{"site": "04"}}' --preset dawn-owl
  robin runs submit --selection @selection.yaml --preset dawn-owl
"""

RUN_SHOW_EPILOG = f"""\
Prints the run's status, who submitted it and when, its configuration digest and
output versions, its provenance (where its recording list came from), the compute
and storage settings that it keeps from submit, its task and recording counts by
state, its skip counts by reason, and its failures: why preparation failed, and
the failed recordings counted by the scope and code of their failure, most first
(at most 20; "truncated" says if there are more).

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
        "--uris", type=Path, metavar="FILE", help="a file with one path or URI on each line"
    )
    inputs.add_argument(
        "--selection", metavar="JSON|@FILE",
        help="what to ask the profile's catalog adapter for, as JSON or a YAML file",
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
        columns=commands.RUN_COLUMNS,
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
        columns=commands.RECORDING_COLUMNS,
        note="STATE is - until preparation gives the recording one. SKIP_REASON is - "
        "unless the\nrecording was skipped.",
        example="robin runs recordings RUN --state skipped --skip-reason already_covered",
    )
    recordings.add_argument("run_id", metavar="RUN", help=RUN_HELP)
    recordings.add_argument(
        "--state", metavar="S", choices=RECORDING_STATES,
        help=f"only recordings in this state: {', '.join(RECORDING_STATES)}",
    )
    recordings.add_argument(
        "--skip-reason", metavar="R", choices=SKIP_REASONS,
        help=f"only recordings skipped for this reason: {', '.join(SKIP_REASONS)}",
    )
    _add_list_command(
        run_commands, "tasks", commands.list_run_tasks, "list a run's tasks",
        columns=commands.TASK_COLUMNS, example="robin runs tasks RUN",
        note="REASON is why a failed task failed, or - for any other task. --json also\n"
        "gives each task's execution times, its release, and the failure's detail.",
    ).add_argument("run_id", metavar="RUN", help=RUN_HELP)


# ---------------------------------------------------------------------------
# robin reconcile
# ---------------------------------------------------------------------------


RECONCILE_EPILOG = f"""\
One pass does these steps in this order. A failure in one task or run is reported,
and the pass goes on.

  1. ask the compute about each task whose submit may not have been made, and
     submit it again only if it has no result and no execution
  2. ask the compute about each submitted or running task
  3. record the result of each task whose execution ended
  4. release each ended task in the compute
  5. prepare at most 10 submitted runs, of any user, oldest first
  6. start queued tasks, oldest first, while the compute has free slots
     (compute.max_in_flight in the profile)

Steps 1 to 4 and 6 act only on your runs, and only on runs that were submitted
with the same compute and storage settings as the profile. The pass counts your
other tasks as tasks_queued_elsewhere and tasks_active_elsewhere. They are not
failures.

The profile must have a compute section with a kind that this version can run:
local. Otherwise the command stops with not_configured (exit code 2).

The report gives the counts prepared, launched, outcomes_recorded, released,
stops_requested, tasks_queued_elsewhere, and tasks_active_elsewhere, and one entry
in failures for each failure: its step, run_id, task_id, reason, message, and
details. The exit code is 0 if nothing failed. Otherwise it is the most serious
failure's code, in the order 3 (outcome uncertain), 5 (internal error), 4 (the
database, the compute, or the task storage is unavailable), and 1. An internal
error writes its stack and a request ID on stderr, without the error's text.

With --loop, Robin does a pass, then waits until a run is submitted or --interval
seconds have passed, and does the next pass. With --json it prints each report on
one line; without --json it prints a short line on stderr for each pass. Before each
pass it logs in to the database again. If the database cannot be reached, the
report of the next pass has the failure, with the step login or listen, and the loop
tries again later. A pass that cannot log in or listen does no steps. If the database refuses the login, it stops with exit
code 1; if your AWS credentials ended, it stops with exit code 2: sign in again.
SIGINT (Ctrl-C) or SIGTERM stops it after the current operation on a task or run,
with exit code 0.

Examples:

  robin reconcile
  robin --json reconcile --loop --interval 60
"""


def _add_reconcile_command(nouns) -> None:
    command = _add_command(
        nouns, "reconcile", commands.reconcile_once,
        "prepare runs, start tasks, and record their results", check=_check_reconcile,
        epilog=RECONCILE_EPILOG,
    )
    command.add_argument(
        "--loop", action="store_true", help="do passes until SIGINT or SIGTERM"
    )
    command.add_argument(
        "--interval", metavar="SECONDS", type=_parse_interval,
        help=f"with --loop: the longest wait between passes (default {DEFAULT_INTERVAL_SECONDS})",
    )


def _parse_interval(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds greater than 0")
    return seconds


def _reconcile_in_loop(
    context: Context, arguments: argparse.Namespace, *, json_mode: bool
) -> int:
    """Do passes until SIGINT or SIGTERM, and give exit code 0.

    Before each pass, a new login checks that the database still accepts the profile's
    login. An `Unavailable` from the login check or the listener is a failure in the
    report of that pass, which then does no steps, and the loop goes on. Any other
    error stops the loop, and the caller reports it.
    """
    database = context.services.database
    interval = arguments.interval or DEFAULT_INTERVAL_SECONDS
    stopping = threading.Event()
    listener: NoticeListener | None = None
    # The failures of the last wait, which go in the report of the next pass.
    failures: list[StepFailure] = []
    with _stop_on_signals(stopping):
        try:
            while not stopping.is_set():
                step = "login"
                try:
                    check_login(database)
                    step = "listen"
                    # Listen before the pass, so that a submit during the pass is seen.
                    listener = listener or listen(database)
                except Unavailable as error:
                    failures.append(StepFailure(step, None, None, error))
                    output = commands.get_pass_output(ReconcileReport(failures=failures))
                else:
                    output = commands.reconcile_once(
                        context, arguments, should_stop=stopping.is_set, failures=failures
                    )
                _print_pass(output, json_mode=json_mode)
                failures = []
                listener = _wait_for_next_pass(listener, interval, stopping, failures)
        finally:
            if listener is not None:
                listener.close()
    return EXIT_SUCCESS


@contextlib.contextmanager
def _stop_on_signals(stopping: threading.Event):
    """Set `stopping` on SIGINT or SIGTERM in the block, in place of stopping Robin."""
    numbers = (signal.SIGINT, signal.SIGTERM)
    previous = {number: signal.signal(number, lambda *_: stopping.set()) for number in numbers}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _wait_for_next_pass(
    listener: NoticeListener | None,
    interval: float,
    stopping: threading.Event,
    failures: list[StepFailure],
) -> NoticeListener | None:
    """Wait for a notice, `interval` seconds, or a stop signal, whichever comes first.

    Gives the listener, or None if it failed or there was none. A failed listener is
    added to `failures`, and the next pass opens a new one.
    """
    deadline = time.monotonic() + interval
    while not stopping.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Wait in short parts, because a signal does not end a wait.
        part = min(remaining, _STOP_CHECK_SECONDS)
        if listener is None:
            time.sleep(part)
            continue
        try:
            if listener.wait(part):
                break
        except Unavailable as error:
            failures.append(StepFailure("listen", None, None, error))
            listener.close()
            listener = None
    return listener


def _print_pass(output: Output, *, json_mode: bool) -> None:
    """Print the report of one pass of the loop: a JSON line on stdout with --json, else
    its lines on stderr."""
    if json_mode:
        print(json.dumps(output.data), flush=True)
    else:
        print(f"robin: {time.strftime('%H:%M:%S')} {output.text}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Add groups, commands, and their shared options.
# ---------------------------------------------------------------------------


LIST_LAYOUT = """\
On a terminal, the first line is a header and the columns are aligned. In a pipe
or a file, there is no header, and two spaces separate the values on each line.
Use --json for output that scripts can depend on.
"""


def _add_list_command(
    group,
    name: str,
    run,
    help: str,
    *,
    columns: tuple[str, ...],
    example: str,
    note: str | None = None,
):
    """Add a command that prints a page of items, one on each line, with the page options.

    The epilog gives the columns, the optional note, the layout, and the example.
    """
    epilog = f"Columns: {'  '.join(columns)}\n"
    if note:
        epilog += f"{note}\n"
    epilog += f"{LIST_LAYOUT}\nExample:\n\n  {example}\n"
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
    check: Callable[[argparse.Namespace], str | None] | None = None,
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
    command.set_defaults(run=run, needs_schema=needs_schema, check=check, prog=command.prog)
    return command


# ---------------------------------------------------------------------------
# Check the rules that argparse cannot.
# ---------------------------------------------------------------------------


def _check_submit(arguments: argparse.Namespace) -> str | None:
    """Check the submit options that argparse cannot: --namespace goes with --uris only."""
    if arguments.uris is not None and arguments.namespace is None:
        return "--uris needs --namespace."
    if arguments.uris is None and arguments.namespace is not None:
        return (
            "--namespace is used only with --uris. --recordings and --selection give "
            "each recording its namespace."
        )
    return None


def _check_reconcile(arguments: argparse.Namespace) -> str | None:
    if arguments.interval is not None and not arguments.loop:
        return "--interval is used only with --loop."
    return None


def _check_register(arguments: argparse.Namespace) -> str | None:
    """Check the register options that argparse cannot: one model source, and each role once."""
    if arguments.model is None and arguments.card is None:
        return "Name a built-in model, or give --card."
    if arguments.model is not None and arguments.card is not None:
        return "Name a built-in model or give --card, not both."
    roles = [role for role, _ in arguments.files]
    repeated = sorted({role for role in roles if roles.count(role) > 1})
    if repeated:
        return f"Give each role once. Repeated: {', '.join(repeated)}."
    return None


# ---------------------------------------------------------------------------
# Print results and errors.
# ---------------------------------------------------------------------------


def _print(
    output: Output, *, json_mode: bool, table: bool, color: bool, notice_color: bool
) -> None:
    """Print the result. With `table`, a list has a header and aligned columns.

    Without `table`, two spaces separate the values of a list row. `color` is for
    stdout and `notice_color` for the notice on stderr.
    """
    if json_mode:
        print(json.dumps(output.data))
        return
    if output.columns and table:
        lines = _get_table_lines(output.columns, output.rows, color=color)
    elif output.columns:
        lines = ["  ".join(values) for values in output.rows]
    elif output.fields:
        lines = _get_field_lines(output.fields, color=color)
    else:
        lines = [output.text] if output.text else []
    if lines:
        print("\n".join(lines))
    if output.notice:
        print(add_color(f"robin: {output.notice}", DIM, enabled=notice_color), file=sys.stderr)


def _get_table_lines(
    columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...], *, color: bool
) -> list[str]:
    """The header and the rows. Each column but the last is padded to its widest value.

    Color is added after the padding, so that the codes do not change the widths.
    """
    widths = [
        max(len(values[index]) for values in (columns, *rows)) for index in range(len(columns))
    ]
    # Only the run list has a STATUS column. Its values are the keys of STATUS_COLORS.
    status = columns.index("STATUS") if "STATUS" in columns else None
    lines = [add_color("  ".join(_pad(columns, widths)), BOLD, enabled=color)]
    for values in rows:
        cells = _pad(values, widths)
        if status is not None:
            cells[status] = add_color(
                cells[status], STATUS_COLORS.get(values[status]), enabled=color
            )
        lines.append("  ".join(cells))
    return lines


def _pad(values: tuple[str, ...], widths: list[int]) -> list[str]:
    """Each value padded to its column's width, except the last value."""
    padded = [value.ljust(width) for value, width in zip(values[:-1], widths)]
    return [*padded, values[-1]]


def _get_field_lines(fields: tuple[tuple[str, str], ...], *, color: bool) -> list[str]:
    """`name: value` lines, with the value of `status` in its color."""
    lines = []
    for name, value in fields:
        # Only a run has a `status` field. Its values are the keys of STATUS_COLORS.
        if name == "status":
            value = add_color(value, STATUS_COLORS.get(value), enabled=color)
        lines.append(f"{name}: {value}")
    return lines


def _report(error: OperationError, *, json_mode: bool, color: bool) -> int:
    _print_error(error.body(), json_mode=json_mode, color=color)
    return _get_exit_code(error)


def _get_exit_code(error: OperationError) -> int:
    if isinstance(error, NotConfigured):
        return EXIT_LOCAL_PROBLEM
    if isinstance(error, OutcomeUncertain):
        return EXIT_OUTCOME_UNCERTAIN
    if isinstance(error, (Unavailable, SourceUnavailable, ComputeUnavailable, StorageUnavailable)):
        return EXIT_UNAVAILABLE
    if isinstance(error, (CatalogFailed, InternalError)):
        return EXIT_INTERNAL_ERROR
    return EXIT_REFUSED


def _get_most_serious_exit_code(errors: Sequence[OperationError]) -> int:
    codes = {_get_exit_code(error) for error in errors}
    return next((code for code in EXIT_CODES_MOST_SERIOUS_FIRST if code in codes), EXIT_SUCCESS)


def _report_defect(*, json_mode: bool, color: bool) -> int:
    request_id = uuid.uuid4().hex
    prefix = add_color("robin:", RED_BOLD, enabled=color)
    print(f"{prefix} internal error, request {request_id}:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    body = {
        "reason": "internal_error",
        "message": "Robin failed because of a defect. The lines above give the details.",
        "details": {"request_id": request_id},
    }
    _print_error(body, json_mode=json_mode, color=color)
    return EXIT_INTERNAL_ERROR


def _print_local_problem(problem: _LocalProblem, *, json_mode: bool, color: bool) -> None:
    """Print the problem. Without --json, a command-line error ends with a hint line."""
    if problem.usage and not json_mode:
        sys.stderr.write(problem.usage)
        prefix = f"{problem.command}:"
        rest = problem.body["message"].removeprefix(prefix)
        print(f"{add_color(prefix, RED_BOLD, enabled=color)}{rest}", file=sys.stderr)
    else:
        _print_error(problem.body, json_mode=json_mode, color=color)
    if not json_mode and problem.body["reason"] == "invalid_arguments":
        print(add_color(f"See: {problem.command} --help", YELLOW, enabled=color), file=sys.stderr)


def _print_error(body: dict, *, json_mode: bool, color: bool) -> None:
    if json_mode:
        print(json.dumps(body), file=sys.stderr)
    else:
        prefix = add_color("robin:", RED_BOLD, enabled=color)
        print(f"{prefix} {body['message']} ({body['reason']})", file=sys.stderr)
