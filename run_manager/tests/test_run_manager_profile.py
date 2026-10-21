"""Loading a profile: `config.yaml`, `{environment}.yaml`, and `${VAR}` references."""

import pytest
import yaml

from robin_run_manager.errors import Refused
from robin_run_manager.profile import ProfileNotFound, load_profile

SECRET = "s3cr3t-value-for-tests"
ADAPTER_SECRET = "adapter-token-for-tests"
DATABASE = {
    "host": "localhost",
    "dbname": "robin",
    "user": "robin",
    "password": "${ROBIN_DB_PASSWORD}",
}
ENVIRON = {"ROBIN_DB_PASSWORD": SECRET}


def write(folder, name, value):
    folder.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else yaml.safe_dump(value)
    (folder / name).write_text(text)


def invalid(tmp_path, environ=ENVIRON) -> Refused:
    """Load environment `dev` and return the `invalid_profile` refusal."""
    with pytest.raises(Refused) as raised:
        load_profile(tmp_path, "dev", environ)
    assert raised.value.reason == "invalid_profile"
    return raised.value


def shown(refusal: Refused) -> str:
    """Everything that a user can see of a refusal."""
    return f"{refusal.message} {refusal.details} {refusal!r}"


def test_the_environment_file_replaces_whole_sections(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE, "storage": {"kind": "local", "root": "/data"}})
    write(tmp_path, "dev.yaml", {"storage": {"kind": "s3", "bucket": "robin"}})

    profile = load_profile(tmp_path, "dev", ENVIRON)

    assert profile.storage.kind == "s3"
    assert profile.storage.model_extra == {"bucket": "robin"}
    assert profile.database.password.get_secret_value() == SECRET


def test_config_yaml_is_optional(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    assert load_profile(tmp_path, "dev", ENVIRON).database.host == "localhost"


def test_a_missing_variable_is_refused_and_named(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    refusal = invalid(tmp_path, environ={})

    assert "ROBIN_DB_PASSWORD" in refusal.message
    assert refusal.details == {"field": "database.password", "variable": "ROBIN_DB_PASSWORD"}


def test_a_variable_only_in_a_replaced_section_is_not_needed(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE, "storage": {"kind": "s3", "bucket": "${OLD_BUCKET}"}})
    write(tmp_path, "dev.yaml", {"storage": {"kind": "local", "root": "/data"}})

    assert load_profile(tmp_path, "dev", ENVIRON).storage.kind == "local"


def test_an_empty_environment_file_keeps_config_yaml(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE})
    write(tmp_path, "dev.yaml", "")

    assert load_profile(tmp_path, "dev", ENVIRON).database.dbname == "robin"


@pytest.mark.parametrize("environment", ["", "../prod", "Prod", "config", "a/b", "dev.yaml"])
def test_an_environment_that_is_not_a_plain_name_is_not_found(tmp_path, environment):
    write(tmp_path / "inner", "config.yaml", {"database": DATABASE})
    write(tmp_path, "prod.yaml", {"database": DATABASE})

    with pytest.raises(ProfileNotFound, match="environment"):
        load_profile(tmp_path / "inner", environment, ENVIRON)


def test_a_missing_environment_file_is_not_found(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE})

    with pytest.raises(ProfileNotFound, match="dev.yaml"):
        load_profile(tmp_path, "dev", ENVIRON)


def test_a_missing_profile_folder_is_not_found(tmp_path):
    with pytest.raises(ProfileNotFound, match="folder"):
        load_profile(tmp_path / "absent", "dev", ENVIRON)


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", "- a\n- b\n")

    assert invalid(tmp_path).details == {"file": "dev.yaml"}


def test_invalid_yaml_is_refused_with_its_file_and_position(tmp_path):
    write(tmp_path, "dev.yaml", "database: [\n")

    refusal = invalid(tmp_path)

    assert refusal.details["file"] == "dev.yaml"
    assert refusal.details["line"] >= 1
    assert "dev.yaml" in refusal.message


def test_an_invalid_yaml_error_does_not_show_the_source_text(tmp_path):
    # PyYAML's own message quotes the line, so this password would appear in it.
    write(tmp_path, "dev.yaml", f'database:\n  password: "{SECRET}\n')

    refusal = invalid(tmp_path)

    assert SECRET not in shown(refusal)
    assert refusal.__cause__ is None
    assert refusal.__context__ is None or SECRET not in str(refusal.__context__)


def test_an_unknown_section_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "databse": {}})

    refusal = invalid(tmp_path)

    assert [problem["field"] for problem in refusal.details["problems"]] == ["databse"]


@pytest.mark.parametrize("storage", [None, {"kind": "local", "root": "/data"}])
def test_a_sagemaker_launcher_needs_s3_storage(tmp_path, storage):
    sections = {"database": DATABASE, "launcher": {"kind": "sagemaker"}}
    if storage is not None:
        sections["storage"] = storage
    write(tmp_path, "dev.yaml", sections)

    refusal = invalid(tmp_path)
    assert "s3" in refusal.message
    # A problem with the whole profile names the profile, not a field.
    assert [problem["field"] for problem in refusal.details["problems"]] == ["profile"]


def test_a_pool_minimum_above_its_maximum_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", {"database": {**DATABASE, "pool_min_size": 5, "pool_max_size": 2}})

    assert "pool_min_size" in invalid(tmp_path).message


def test_a_sagemaker_launcher_with_s3_storage_is_accepted(tmp_path):
    write(tmp_path, "dev.yaml", {
        "database": DATABASE,
        "launcher": {"kind": "sagemaker", "max_in_flight": 10},
        "storage": {"kind": "s3", "bucket": "robin"},
    })

    assert load_profile(tmp_path, "dev", ENVIRON).launcher.kind == "sagemaker"


def test_a_validation_error_does_not_show_a_secret(tmp_path):
    # The secret reaches a field that refuses it, so validation fails on it.
    write(tmp_path, "dev.yaml", {"database": {**DATABASE, "port": "${ROBIN_DB_PASSWORD}"}})

    refusal = invalid(tmp_path)

    assert [problem["field"] for problem in refusal.details["problems"]] == ["database.port"]
    assert SECRET not in shown(refusal)
    assert refusal.__cause__ is None


def test_the_profile_shows_no_password_or_adapter_value(tmp_path):
    write(tmp_path, "dev.yaml", {
        "database": DATABASE,
        "storage": {"kind": "s3", "secret_access_key": "${ADAPTER_SECRET}"},
        "catalogs": {"soundhub": {"kind": "soundhub", "token": "${ADAPTER_SECRET}"}},
    })

    profile = load_profile(tmp_path, "dev", {**ENVIRON, "ADAPTER_SECRET": ADAPTER_SECRET})

    for text in (repr(profile), str(profile)):
        assert SECRET not in text
        assert ADAPTER_SECRET not in text
        assert "token" in text
    assert profile.catalogs["soundhub"].model_extra == {"token": ADAPTER_SECRET}


def test_a_file_that_is_not_utf8_is_refused_with_its_name(tmp_path):
    tmp_path.joinpath("dev.yaml").write_bytes(b"database:\n  password: \xff\xfe\n")

    refusal = invalid(tmp_path)

    assert refusal.details["file"] == "dev.yaml"
    assert refusal.__cause__ is None


def test_a_file_that_cannot_be_read_is_refused_with_its_name(tmp_path, monkeypatch):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    def unreadable(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(type(tmp_path), "read_text", unreadable)

    refusal = invalid(tmp_path)

    assert refusal.details == {"file": "dev.yaml"}
    assert refusal.__cause__ is None


def test_a_value_that_yaml_cannot_build_is_refused_without_the_value(tmp_path):
    # PyYAML raises ValueError, not YAMLError, for a date that does not exist.
    write(tmp_path, "dev.yaml", "database:\n  password: 2026-99-99\n")

    refusal = invalid(tmp_path)

    assert refusal.details == {"file": "dev.yaml"}
    assert "2026-99-99" not in shown(refusal)
    assert refusal.__cause__ is None
    assert refusal.__context__ is None


def test_a_value_that_contains_itself_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", "database: &database\n  itself: *database\n")

    refusal = invalid(tmp_path)

    assert refusal.details == {"field": "database.itself"}


def test_an_empty_catalog_name_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "catalogs": {"": {"kind": "soundhub"}}})

    refusal = invalid(tmp_path)

    assert [problem["field"].split(".")[0] for problem in refusal.details["problems"]] == ["catalogs"]


RDS = {
    "host": "robin.example.us-west-2.rds.amazonaws.com",
    "dbname": "robin",
    "user": "alice",
    "auth": "rds_iam",
    "region": "us-west-2",
}


def ca_bundle(tmp_path) -> str:
    path = tmp_path / "rds-ca.pem"
    path.write_text("Robin does not read this file. libpq reads it when it connects.\n")
    return str(path)


def test_password_is_the_default_login(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    assert load_profile(tmp_path, "dev", ENVIRON).database.auth == "password"


def test_an_rds_iam_login_needs_no_password(tmp_path):
    write(tmp_path / "p", "dev.yaml", {"database": {**RDS, "sslrootcert": ca_bundle(tmp_path)}})

    database = load_profile(tmp_path / "p", "dev", {}).database

    assert database.auth == "rds_iam"
    assert database.password is None
    assert str(database.sslrootcert) == ca_bundle(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"auth": "password"},
        {"password": "p"},
        {"region": None},
        {"sslrootcert": None},
        {"auth": "password", "password": "p"},
        {"auth": "kerberos"},
    ],
    ids=[
        "password-without-a-password",
        "rds-iam-with-a-password",
        "rds-iam-without-a-region",
        "rds-iam-without-a-ca-bundle",
        "password-with-rds-iam-fields",
        "unknown-auth",
    ],
)
def test_login_fields_that_do_not_agree_are_refused(tmp_path, change):
    database = {**RDS, "sslrootcert": ca_bundle(tmp_path), **change}
    database = {name: value for name, value in database.items() if value is not None}
    write(tmp_path / "p", "dev.yaml", {"database": database})

    with pytest.raises(Refused) as raised:
        load_profile(tmp_path / "p", "dev", {})

    assert raised.value.reason == "invalid_profile"
    assert raised.value.details["problems"][0]["field"].startswith("database")


def test_a_ca_bundle_that_does_not_exist_is_refused(tmp_path):
    database = {**RDS, "sslrootcert": str(tmp_path / "absent.pem")}
    write(tmp_path / "p", "dev.yaml", {"database": database})

    with pytest.raises(Refused) as raised:
        load_profile(tmp_path / "p", "dev", {})

    assert raised.value.details["problems"][0]["field"] == "database.sslrootcert"


def test_a_relative_ca_bundle_path_is_refused(tmp_path):
    database = {**RDS, "sslrootcert": "certs/rds-ca.pem"}
    write(tmp_path / "p", "dev.yaml", {"database": database})

    with pytest.raises(Refused) as raised:
        load_profile(tmp_path / "p", "dev", {})

    problem = raised.value.details["problems"][0]
    assert problem["field"] == "database.sslrootcert"
    assert "absolute" in problem["message"]
