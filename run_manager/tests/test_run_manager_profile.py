"""Loading a profile: `config.yaml`, `{environment}.yaml`, and `${VAR}` references."""

import pytest
import yaml

from robin_contracts.ports.compute import ComputeRequest
from robin_run_manager.errors import NotConfigured, Refused
from robin_run_manager.profile import (
    get_execution_settings,
    get_launch_settings,
    get_resolved_catalog_settings,
    load_profile,
)

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
    write(tmp_path, "dev.yaml", {"storage": {"kind": "s3", "root": "s3://robin/", "region": "x"}})

    profile = load_profile(tmp_path, "dev", ENVIRON)

    assert profile.storage.root == "s3://robin/"
    assert profile.storage.model_extra == {"region": "x"}
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

    with pytest.raises(NotConfigured, match="environment"):
        load_profile(tmp_path / "inner", environment, ENVIRON)


def test_a_missing_environment_file_is_not_found(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE})

    with pytest.raises(NotConfigured, match="dev.yaml"):
        load_profile(tmp_path, "dev", ENVIRON)


def test_a_missing_environment_lists_the_profile_environments(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE})
    write(tmp_path, "test.yaml", {})
    write(tmp_path, "local.yaml", {})

    with pytest.raises(NotConfigured) as raised:
        load_profile(tmp_path, "prod", ENVIRON)

    assert raised.value.details["environments"] == ["local", "test"]


def test_a_missing_profile_folder_is_not_found(tmp_path):
    with pytest.raises(NotConfigured, match="folder"):
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


def test_a_pool_minimum_above_its_maximum_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", {"database": {**DATABASE, "pool_min_size": 5, "pool_max_size": 2}})

    assert "pool_min_size" in invalid(tmp_path).message


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
        "compute": {"kind": "local", "api_key": "${ADAPTER_SECRET}"},
        "storage": {"kind": "s3", "root": "s3://robin/", "secret_access_key": "${ADAPTER_SECRET}"},
    })

    profile = load_profile(tmp_path, "dev", {**ENVIRON, "ADAPTER_SECRET": ADAPTER_SECRET})

    for text in (repr(profile), str(profile)):
        assert SECRET not in text
        assert ADAPTER_SECRET not in text
        assert "secret_access_key" in text
        assert "api_key" in text
    assert profile.storage.model_extra == {"secret_access_key": ADAPTER_SECRET}


@pytest.mark.parametrize("named", [{"kind": "csv-list"}, {"code": "lab.catalog:build_adapter"}])
def test_catalog_repr_shows_no_setting_value(tmp_path, named):
    write(tmp_path, "dev.yaml", {
        "database": DATABASE,
        "catalog": {**named, "token": ADAPTER_SECRET},
    })

    profile = load_profile(tmp_path, "dev", ENVIRON)

    for text in (repr(profile), str(profile), repr(profile.catalog)):
        assert ADAPTER_SECRET not in text
        assert "token" in text
        assert next(iter(named.values())) in text


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
    write(tmp_path, "dev.yaml", "database:\n  password: !!timestamp 2026-99-99\n")

    refusal = invalid(tmp_path)

    assert refusal.details == {"file": "dev.yaml"}
    assert "2026-99-99" not in shown(refusal)
    assert refusal.__cause__ is None
    assert refusal.__context__ is None


def test_a_value_that_contains_itself_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", "database: &database\n  itself: *database\n")

    refusal = invalid(tmp_path)

    assert refusal.details == {"field": "database.itself"}


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


def test_recordings_per_task_defaults_to_100(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    assert load_profile(tmp_path, "dev", ENVIRON).preparation.recordings_per_task == 100


@pytest.mark.parametrize(
    "preparation", [{"recordings_per_task": 0}, {"recordings_per_task": 5, "x": 1}]
)
def test_recordings_per_task_must_be_at_least_1(tmp_path, preparation):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "preparation": preparation})

    refusal = invalid(tmp_path)

    assert refusal.details["problems"][0]["field"].startswith("preparation.")


LOCAL_COMPUTE = {"kind": "local"}
LOCAL_STORAGE = {"kind": "local", "root": "/data/robin"}
KUBERNETES = {
    "kind": "kubernetes",
    "namespace": "robin",
    "s3_secret_name": "robin-s3",
    "s3_endpoint": "https://s3.example",
}
S3_STORAGE = {"kind": "s3", "root": "s3://lab/robin/"}
CATALOG = {"kind": "csv-list", "namespace": "lab", "session_token": "${SESSION_TOKEN}"}


def test_compute_and_storage_reprs_show_no_setting_value(tmp_path):
    endpoint = f"https://robin:{ADAPTER_SECRET}@s3.example"
    write(tmp_path, "dev.yaml", {
        "database": DATABASE,
        "compute": {**KUBERNETES, "s3_endpoint": endpoint},
        "storage": {"kind": "s3", "root": f"s3://{ADAPTER_SECRET}/"},
    })

    profile = load_profile(tmp_path, "dev", ENVIRON)

    for text in (repr(profile), str(profile)):
        assert ADAPTER_SECRET not in text
        assert "kubernetes" in text
        assert "s3_endpoint" in text
        assert "root" in text


def get_problem_fields(refusal: Refused) -> list[str]:
    return [problem["field"] for problem in refusal.details["problems"]]


def test_a_catalog_variable_that_is_not_set_does_not_stop_loading(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "catalog": CATALOG})

    profile = load_profile(tmp_path, "dev", ENVIRON)

    assert profile.catalog.kind == "csv-list"


def test_get_resolved_catalog_settings_fills_in_variables(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "catalog": CATALOG})
    profile = load_profile(tmp_path, "dev", ENVIRON)

    settings = get_resolved_catalog_settings(profile, {"SESSION_TOKEN": ADAPTER_SECRET})

    assert settings == {"namespace": "lab", "session_token": ADAPTER_SECRET}


def test_get_resolved_catalog_settings_refuses_a_missing_variable(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "catalog": CATALOG})
    profile = load_profile(tmp_path, "dev", ENVIRON)

    with pytest.raises(Refused) as raised:
        get_resolved_catalog_settings(profile, {})

    assert raised.value.reason == "invalid_profile"
    assert raised.value.details == {"field": "catalog.session_token", "variable": "SESSION_TOKEN"}


@pytest.mark.parametrize(
    "catalog",
    [{"namespace": "lab"}, {"kind": "csv-list", "code": "lab.catalog:build_adapter"}],
    ids=["neither", "both"],
)
def test_catalog_needs_exactly_one_of_kind_and_code(tmp_path, catalog):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "catalog": catalog})

    assert get_problem_fields(invalid(tmp_path)) == ["catalog"]


@pytest.mark.parametrize(
    "code", ["lab.catalog", "lab.catalog:", ":build_adapter", "lab catalog:build", "1lab:build"]
)
def test_catalog_code_must_name_a_module_and_function(tmp_path, code):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "catalog": {"code": code}})

    assert get_problem_fields(invalid(tmp_path)) == ["catalog.code"]


def test_an_unknown_compute_kind_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", {
        "database": DATABASE, "compute": {"kind": "sagemaker"}, "storage": S3_STORAGE,
    })

    assert get_problem_fields(invalid(tmp_path)) == ["compute"]


@pytest.mark.parametrize("missing", ["namespace", "s3_secret_name", "s3_endpoint"])
def test_kubernetes_needs_its_three_fields(tmp_path, missing):
    compute = {name: value for name, value in KUBERNETES.items() if name != missing}
    write(tmp_path, "dev.yaml", {"database": DATABASE, "compute": compute, "storage": S3_STORAGE})

    refusal = invalid(tmp_path)

    assert get_problem_fields(refusal) == [f"compute.kubernetes.{missing}"]


@pytest.mark.parametrize("storage", [LOCAL_STORAGE, {"kind": "local", "root": "file:///data"}])
def test_kubernetes_refuses_a_storage_root_that_is_not_s3(tmp_path, storage):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "compute": KUBERNETES, "storage": storage})

    refusal = invalid(tmp_path)

    assert get_problem_fields(refusal) == ["storage"]
    assert "storage.root" in refusal.message
    assert "s3://" in refusal.message


@pytest.mark.parametrize("compute", [LOCAL_COMPUTE, KUBERNETES], ids=["local", "kubernetes"])
def test_compute_without_storage_is_refused(tmp_path, compute):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "compute": compute})

    assert get_problem_fields(invalid(tmp_path)) == ["storage"]


def test_a_profile_without_compute_or_storage_loads(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    profile = load_profile(tmp_path, "dev", ENVIRON)

    assert profile.compute is None
    assert profile.storage is None


def test_an_unknown_storage_kind_is_refused(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "storage": {"kind": "gcs", "root": "/data"}})

    assert get_problem_fields(invalid(tmp_path)) == ["storage"]


def test_storage_needs_a_root(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "storage": {"kind": "local"}})

    assert get_problem_fields(invalid(tmp_path)) == ["storage.local.root"]


@pytest.mark.parametrize(
    "storage",
    [
        {"kind": "local", "root": "s3://lab/robin/"},
        {"kind": "s3", "root": "/data/robin"},
        {"kind": "local", "root": "data/robin"},
        {"kind": "s3", "root": "s3://"},
        {"kind": "s3", "root": "s3:///robin/"},
    ],
    ids=["local-with-s3", "s3-with-a-path", "relative-path", "s3-with-nothing", "s3-no-bucket"],
)
def test_storage_root_must_have_the_form_of_its_kind(tmp_path, storage):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "storage": storage})

    assert get_problem_fields(invalid(tmp_path)) == [f"storage.{storage['kind']}.root"]


@pytest.mark.parametrize(
    "storage",
    [
        {"kind": "local", "root": "/data/../robin"},
        {"kind": "local", "root": "/data//robin"},
        {"kind": "local", "root": "/data/./robin"},
        {"kind": "local", "root": "file:///data/%2E%2E/robin"},
        {"kind": "s3", "root": "s3://lab/a/../b"},
        {"kind": "s3", "root": "s3://lab//robin"},
    ],
    ids=["parent", "empty", "dot", "encoded-parent", "s3-parent", "s3-empty"],
)
def test_storage_root_has_no_empty_dot_or_parent_part(tmp_path, storage):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "storage": storage})

    assert get_problem_fields(invalid(tmp_path)) == [f"storage.{storage['kind']}.root"]


@pytest.mark.parametrize(
    "storage",
    [
        {"kind": "local", "root": "/"},
        {"kind": "local", "root": "/data/robin/"},
        {"kind": "local", "root": "file:///data/robin"},
        {"kind": "s3", "root": "s3://lab"},
        {"kind": "s3", "root": "s3://lab/robin/"},
    ],
)
def test_a_storage_root_with_safe_parts_loads(tmp_path, storage):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "storage": storage})

    assert load_profile(tmp_path, "dev", ENVIRON).storage.root == storage["root"]


def test_get_execution_settings_freezes_compute_and_storage(tmp_path):
    write(tmp_path, "dev.yaml", {
        "database": DATABASE,
        "compute": {**KUBERNETES, "image": "robin:1"},
        "storage": {**S3_STORAGE, "root": "s3://lab/${LAB}/", "region": "x"},
    })
    profile = load_profile(tmp_path, "dev", {**ENVIRON, "LAB": "owls"})

    assert get_execution_settings(profile) == {
        "compute": {
            "kind": "kubernetes",
            "namespace": "robin",
            "s3_secret_name": "robin-s3",
            "s3_endpoint": "https://s3.example",
        },
        "storage": {"kind": "s3", "root": "s3://lab/owls/"},
    }


def test_get_execution_settings_needs_compute(tmp_path):
    write(tmp_path, "dev.yaml", {"database": DATABASE})

    with pytest.raises(NotConfigured):
        get_execution_settings(load_profile(tmp_path, "dev", ENVIRON))


LAUNCH_FIELDS = {
    "max_in_flight": 4,
    "request": {
        "requests": {"cpu": "2", "memory": "8Gi"},
        "limits": {"memory": "8Gi"},
        "active_deadline_seconds": 21600,
    },
    "resources": {"batch_size": 32, "device": "cpu"},
    "models": {
        "perch/v8": {"request": {"requests": {"cpu": "4"}}, "resources": {"batch_size": 16}},
        "birdnet/v2": {"resources": {"threads": 2}},
    },
}


def load_with_compute(tmp_path, compute):
    write(tmp_path, "dev.yaml", {"database": DATABASE, "compute": compute, "storage": LOCAL_STORAGE})
    return load_profile(tmp_path, "dev", ENVIRON)


def test_a_model_entry_replaces_the_request_and_the_resources_as_a_whole(tmp_path):
    profile = load_with_compute(tmp_path, {**LOCAL_COMPUTE, **LAUNCH_FIELDS})

    perch = get_launch_settings(profile, "perch/v8")
    birdnet = get_launch_settings(profile, "birdnet/v2")

    assert perch.request == ComputeRequest(requests={"cpu": "4"})
    assert perch.resources == {"batch_size": 16}
    assert birdnet.request == ComputeRequest.model_validate(LAUNCH_FIELDS["request"])
    assert birdnet.resources == {"threads": 2}


@pytest.mark.parametrize("launch", [LAUNCH_FIELDS, {}], ids=["with-defaults", "without"])
def test_a_model_without_an_entry_gets_the_defaults(tmp_path, launch):
    profile = load_with_compute(tmp_path, {**LOCAL_COMPUTE, **launch})

    settings = get_launch_settings(profile, "other/v1")

    assert settings.request == ComputeRequest.model_validate(launch.get("request", {}))
    assert settings.resources == launch.get("resources", {})


def test_max_in_flight_defaults_to_10_and_must_be_at_least_one(tmp_path):
    assert load_with_compute(tmp_path, LOCAL_COMPUTE).compute.max_in_flight == 10
    write(tmp_path, "dev.yaml", {
        "database": DATABASE, "compute": {**KUBERNETES, "max_in_flight": 0}, "storage": S3_STORAGE,
    })

    assert get_problem_fields(invalid(tmp_path)) == ["compute.kubernetes.max_in_flight"]


@pytest.mark.parametrize("value", [float("inf"), float("nan")], ids=["inf", "nan"])
@pytest.mark.parametrize("where", ["default", "model"])
def test_resources_refuse_a_value_that_is_not_finite(tmp_path, value, where):
    if where == "default":
        compute = {**LOCAL_COMPUTE, "resources": {"scale": value}}
    else:
        compute = {**LOCAL_COMPUTE, "models": {"perch/v8": {"resources": {"scale": value}}}}
    write(tmp_path, "dev.yaml", {"database": DATABASE, "compute": compute, "storage": LOCAL_STORAGE})

    [field] = get_problem_fields(invalid(tmp_path))

    assert field.startswith("compute.local.")
    assert field.endswith("resources")


@pytest.mark.parametrize("compute", [LOCAL_COMPUTE, KUBERNETES], ids=["local", "kubernetes"])
def test_execution_settings_leave_out_the_launch_fields(tmp_path, compute):
    storage = LOCAL_STORAGE if compute is LOCAL_COMPUTE else S3_STORAGE
    write(tmp_path, "dev.yaml", {"database": DATABASE, "compute": compute, "storage": storage})
    without = get_execution_settings(load_profile(tmp_path, "dev", ENVIRON))
    write(tmp_path, "dev.yaml", {
        "database": DATABASE, "compute": {**compute, **LAUNCH_FIELDS}, "storage": storage,
    })

    assert get_execution_settings(load_profile(tmp_path, "dev", ENVIRON)) == without


def test_a_repeated_key_is_refused_with_its_position(tmp_path):
    write(tmp_path, "dev.yaml", "database:\n  host: a\n  host: b\n")

    refusal = invalid(tmp_path)

    assert refusal.details == {"file": "dev.yaml", "line": 3, "column": 3}
    assert "'host' more than once" in refusal.message


def test_an_unquoted_date_is_text(tmp_path):
    write(tmp_path, "config.yaml", {"database": DATABASE})
    write(tmp_path, "dev.yaml", "catalog:\n  kind: csv-list\n  since: 2026-05-01\n")

    profile = load_profile(tmp_path, "dev", ENVIRON)

    assert profile.catalog.model_extra == {"since": "2026-05-01"}
