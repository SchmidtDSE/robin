import ondio
import pytest
from robin_adapters.artifact_writer.local import LocalArtifactWriter
from robin_adapters.artifact_writer.ondio import OndioArtifactWriter

from robin_worker import storage
from robin_worker.storage import TaskFiles

# ---------------------------------------------------------------------------
# Reading and creating whole files.
# ---------------------------------------------------------------------------


def test_a_local_path_and_a_file_uri_read_the_same_bytes(tmp_path):
    path = tmp_path / "work.json"
    path.write_bytes(b"{}")

    assert storage.read_bytes(str(path)) == b"{}"
    assert storage.read_bytes(path.as_uri()) == b"{}"


def test_reading_a_missing_file_raises_os_error(tmp_path):
    with pytest.raises(OSError):
        storage.read_bytes(str(tmp_path / "missing.json"))


def test_reading_an_unsupported_scheme_raises_value_error():
    with pytest.raises(ValueError, match="gs://b/x"):
        storage.read_bytes("gs://b/x")


def test_create_writes_the_bytes(tmp_path):
    path = tmp_path / "result.json"

    storage.create(str(path), b"first")

    assert path.read_bytes() == b"first"


def test_a_second_create_is_refused_and_keeps_the_first_bytes(tmp_path):
    path = tmp_path / "result.json"
    storage.create(str(path), b"first")

    with pytest.raises(FileExistsError):
        storage.create(str(path), b"second")

    assert path.read_bytes() == b"first"


def test_a_second_create_on_s3_is_refused_and_keeps_the_first_bytes(s3_bucket):
    uri = f"s3://{s3_bucket}/tasks/t/result.json"
    storage.create(uri, b"first")

    with pytest.raises(FileExistsError):
        storage.create(uri, b"second")

    assert ondio.read(uri) == b"first"


def test_create_in_a_missing_local_folder_is_refused_and_makes_no_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.create(str(tmp_path / "missing" / "result.json"), b"first")

    assert not (tmp_path / "missing").exists()


def test_create_keeps_question_marks_and_hashes_in_a_local_name(tmp_path):
    folder = tmp_path / "a?b#c"
    folder.mkdir()

    storage.create(str(folder / "result.json"), b"first")

    assert (folder / "result.json").read_bytes() == b"first"


# ---------------------------------------------------------------------------
# The artifact writer for an output root.
# ---------------------------------------------------------------------------


def test_a_local_root_gets_the_local_writer(tmp_path):
    assert isinstance(storage.artifact_writer(str(tmp_path / "outputs")), LocalArtifactWriter)


def test_an_s3_root_gets_the_ondio_writer():
    assert isinstance(storage.artifact_writer("s3://bucket/outputs"), OndioArtifactWriter)


@pytest.mark.parametrize("root", ["gs://bucket/x", "s3://bucket/a?b"])
def test_an_unsupported_root_raises_value_error(root):
    with pytest.raises(ValueError):
        storage.artifact_writer(root)


# ---------------------------------------------------------------------------
# Model files and inputs.
# ---------------------------------------------------------------------------


def test_a_local_file_is_used_where_it_is(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    files = TaskFiles(tmp_path / "scratch")

    fetched = files.fetch(str(path))
    files.release(fetched)

    assert fetched == path
    assert path.read_bytes() == b"weights"
    assert not (tmp_path / "scratch").exists()


def test_one_local_file_fetched_twice_is_released_twice(tmp_path):
    # Two model roles can name one file, and the engine holds both until the end.
    path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    files = TaskFiles(tmp_path / "scratch")

    first, second = files.fetch(str(path)), files.fetch(str(path))
    files.release(first)
    files.release(second)

    with pytest.raises(ValueError):
        files.release(path)


def test_fetching_an_unsupported_scheme_names_the_uri(tmp_path):
    with pytest.raises(ValueError, match="gs://b/x"):
        TaskFiles(tmp_path / "scratch").fetch("gs://b/x")


def test_releasing_a_path_it_did_not_return_is_refused(tmp_path):
    with pytest.raises(ValueError):
        TaskFiles(tmp_path / "scratch").release(tmp_path / "other.bin")
