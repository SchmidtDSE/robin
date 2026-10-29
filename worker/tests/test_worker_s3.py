"""Tasks whose files are on S3.

moto's S3 runs in the test process, so these tests call the worker in that process.
"""

import uuid

import boto3
import pytest
from robin_adapters.storage.location import split_location
from robin_contracts.canonical import checksum_bytes, checksum_file
from robin_contracts.tasks import InferenceTaskResult, task_result_problem

from robin_worker.cli import main
from robin_worker.run import run_task
from worker_doubles import TEST_VERSIONS, TaskBuilder


@pytest.fixture
def s3(s3_bucket, test_model_site):
    return boto3.client("s3")


def s3_location(base):
    return lambda path: f"s3://robin-test/{path.relative_to(base).as_posix()}"


def key(uri):
    return uri.removeprefix("s3://robin-test/")


def upload(s3, path, uri):
    s3.put_object(Bucket="robin-test", Key=key(uri), Body=path.read_bytes())


def upload_tree(s3, base):
    location = s3_location(base)
    for path in base.rglob("*"):
        if path.is_file():
            upload(s3, path, location(path))


def read_object(s3, uri) -> bytes:
    return s3.get_object(Bucket="robin-test", Key=key(uri))["Body"].read()


def run(work_uri, lines=None):
    log = [] if lines is None else lines
    return run_task(work_uri, versions=TEST_VERSIONS, release="test-release", log=log.append)


@pytest.mark.parametrize("suffix", ["?x/work.json", "#x/work.json", "?", "#"])
def test_an_s3_work_uri_with_a_query_or_fragment_is_refused_before_inference(
    tmp_path, s3, capsys, suffix
):
    builder = TaskBuilder(tmp_path / "task")
    work_path, task = builder.write()
    work_uri = "s3://robin-test/tasks/t/work.json"
    upload(s3, work_path, work_uri)

    with pytest.raises(SystemExit) as stopped:
        main(["run", work_uri + suffix], versions=(task.output_version,))

    assert stopped.value.code == 2
    assert not builder.outputs.exists()
    assert read_object(s3, work_uri) == work_path.read_bytes()
    objects = s3.list_objects_v2(Bucket="robin-test")["Contents"]
    assert [one["Key"] for one in objects] == [key(work_uri)]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "must be an absolute local path" in captured.err
    assert "no result was written" in captured.err


def test_a_task_entirely_on_s3_completes(tmp_path, s3):
    base = tmp_path / "task"
    builder = TaskBuilder(base, location=s3_location(base))
    work_path, task = builder.write()
    upload_tree(s3, base)
    work_uri = s3_location(base)(work_path)

    code = main(["run", work_uri], versions=(task.output_version,))

    assert code == 0
    result = InferenceTaskResult.model_validate_json(
        read_object(s3, work_uri.removesuffix("work.json") + "result.json")
    )
    assert result.result.outcome == "completed"
    assert task.output_root == f"s3://robin-test/runs/r1/tasks/{task.task_id}/outputs"
    for artifact in result.result.artifacts:
        assert artifact.uri.startswith(task.output_root + "/")
    problem = task_result_problem(
        result,
        task=task,
        work_file_digest=checksum_file(work_path),
        split_location=split_location,
    )
    assert problem is None


def test_local_model_files_with_s3_audio_and_outputs_complete(tmp_path, s3):
    builder = TaskBuilder(tmp_path / "task")
    audio = s3_location(tmp_path / "task")
    recordings = (
        builder.recording("r1", audio_uri=audio(tmp_path / "task" / "audio" / "r1.wav")),
        builder.recording("r2", audio_uri=audio(tmp_path / "task" / "audio" / "r2.wav")),
    )
    for one in recordings:
        upload(s3, tmp_path / "task" / "audio" / f"{one.value}.wav", one.audio_uri)
    root = "s3://robin-test/runs/r1/outputs"
    work_path, _ = builder.write(work=builder.work(recordings), output_root=root)

    code = run(str(work_path))

    assert code == 0
    result = InferenceTaskResult.model_validate_json(builder.result_path.read_bytes())
    assert result.result.outcome == "completed"
    assert [one.uri.startswith(root + "/") for one in result.result.artifacts] == [True, True]


def test_an_existing_result_on_s3_is_not_changed(tmp_path, s3):
    base = tmp_path / "task"
    builder = TaskBuilder(base, location=s3_location(base))
    work_path, _ = builder.write()
    upload_tree(s3, base)
    work_uri = s3_location(base)(work_path)
    result_uri = work_uri.removesuffix("work.json") + "result.json"
    s3.put_object(Bucket="robin-test", Key=key(result_uri), Body=b"earlier")
    lines = []

    code = run(work_uri, lines)

    assert code == 3
    assert read_object(s3, result_uri) == b"earlier"
    assert any("already exists" in line for line in lines)


def test_a_second_execution_on_s3_writes_its_own_folder(tmp_path, s3):
    base = tmp_path / "task"
    builder = TaskBuilder(base, location=s3_location(base))
    work_path, task = builder.write()
    upload_tree(s3, base)
    work_uri = s3_location(base)(work_path)
    result_uri = work_uri.removesuffix("work.json") + "result.json"

    first = main(["run", work_uri], versions=(task.output_version,))
    committed = read_object(s3, result_uri)
    second = main(["run", work_uri], versions=(task.output_version,))

    assert (first, second) == (0, 3)
    assert read_object(s3, result_uri) == committed
    prefix = key(task.output_root) + "/"
    objects = s3.list_objects_v2(Bucket="robin-test", Prefix=prefix)["Contents"]
    folders = {one["Key"].removeprefix(prefix).split("/")[0] for one in objects}
    assert len(folders) == 2
    assert all(uuid.UUID(one) for one in folders)
    result = InferenceTaskResult.model_validate_json(committed)
    for artifact in result.result.artifacts:
        assert checksum_bytes(read_object(s3, artifact.uri)) == artifact.checksum


def test_an_s3_root_with_a_query_is_refused_as_invalid(tmp_path, s3):
    builder = TaskBuilder(tmp_path / "task")
    builder.write(output_root="s3://robin-test/a?b")

    code = run(str(builder.work_path))

    assert code == 1
    result = InferenceTaskResult.model_validate_json(builder.result_path.read_bytes())
    assert (result.result.outcome, result.result.code) == ("refused", "work_file_invalid")
