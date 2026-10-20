import pytest

from robin_cli.main import main
from robin_run_manager import migrations

from cli_doubles import PASSWORD_VARIABLE, Ran, write_profile
from postgres_fixtures import *  # noqa: F403


@pytest.fixture
def environ(tmp_path, database_settings):
    """Variables that select a profile, environment `test`, for the test's database."""
    folder = write_profile(tmp_path / "profile", database_settings)
    return {
        "ROBIN_PROFILE": str(folder),
        "ROBIN_ENV": "test",
        PASSWORD_VARIABLE: database_settings.password.get_secret_value(),
    }


@pytest.fixture
def robin(capsys, environ):
    """Run `robin` in this process. Returns the exit code, stdout, and stderr."""

    def run(*argv, env=None):
        try:
            code = main(list(argv), environ=environ if env is None else env)
        except SystemExit as exit:
            code = exit.code
        captured = capsys.readouterr()
        return Ran(code=code, stdout=captured.out, stderr=captured.err)

    return run


@pytest.fixture
def packaged_migration(monkeypatch, tmp_path):
    """Use one test migration, `0001_first`, as this release's migrations."""
    folder = tmp_path / "packaged"
    folder.mkdir()
    (folder / "0001_first.sql").write_text("CREATE TABLE first (x integer);")
    packaged = migrations.load_migrations(folder)
    monkeypatch.setattr(migrations, "packaged_migrations", lambda: packaged)
