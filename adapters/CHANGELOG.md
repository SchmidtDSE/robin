# Changelog

## Unreleased

### Added

- One folder for each compute kind in `robin_adapters.compute`: `local`, `kubernetes`,
  and `aws_batch`. Each has a `settings` module with the kind's setup class, on the base
  class `ComputeSetupSettings` of `robin_contracts.ports.compute`. A setup class checks
  its section of the profile, names its target fields and the location forms it
  accepts, and checks a new run's request. A `settings` module imports only pydantic and
  `robin_contracts`, so a profile loads without an adapter's packages.
- One folder for each storage backend in `robin_adapters.storage`: `local` and
  `object_store`. Each has a `settings` module with the backend's settings class
  (`LocalStorageSettings`, `S3StorageSettings`), on the base class `StorageSettings` of
  `robin_contracts.ports.storage`. A settings class checks the profile's storage
  section, refuses a field that its kind does not have, and gives the options that the
  adapter needs. `is_s3_uri` is in `robin_contracts.ports.storage`.
- `robin_adapters.compute.aws_batch.adapter`, the `aws-batch` compute adapter, with the
  `aws-batch` extra (boto3). `AwsBatchCompute` runs each task as one AWS Batch job named
  for the task, with one attempt, the task's request as resource overrides, its
  deadline as the job timeout, and the tags `robin-run` and `robin-setup`. Before it
  submits, it looks for a job of the task's name, and returns that job if there is one.
  `describe` counts every job of the task's name, and `stop` ends each of them that has
  not ended. `release` does nothing. A refusal by Batch (`ClientException`) is
  `SubmissionRejected`.
- `robin_adapters.compute.local.adapter`, the `local` compute adapter. `LocalCompute` runs
  `robin-worker run <work URI>` as a detached process in a new session, with its output
  in `worker.log` in the task folder. It starts at most one worker for each task, and
  finds a running worker again in the process table, also from a later process. A task
  runs while any process stays in the worker's process group. It does not enforce the
  compute request.
- `LocalTaskStorage` in `robin_adapters.storage.local.adapter`. It writes a task file once at a
  local path or a `file://` URI, and refuses to write over a file that is there. It
  reads a task file, and gives None only when the file is absent.
- `robin_adapters.catalog.folder_layout`, the `folder-layout` catalog adapter. It finds
  recordings by the paths of the files under a local folder, with a pattern such as
  `site{site}/{date}/{recorder}_{date}_{time}.wav`, and makes each identity from the
  pattern's fields. A selection gives a text value, a list of text values, or a text
  range for each field. A folder that cannot be read stops the selection, so that no
  run gets a list with recordings missing.
- `robin_adapters.catalog.csv_list`, the `csv-list` catalog adapter. It reads one
  recording from each row of a UTF-8 CSV file that the selection names, with the
  identity from one column, the audio from another, and an optional duration column. An
  audio cell can be a URI, an absolute path, or a path relative to the CSV file's
  folder. An empty identity or audio cell, a NUL character, or a bad duration stops the
  selection with the row number.

### Changed

- The storage adapters move to `robin_adapters.storage`, with one folder for each
  backend. `robin_adapters.storage.local.adapter` holds `LocalFileProvider` and
  `LocalArtifactWriter`. `robin_adapters.storage.object_store.adapter` holds
  `ObjectStoreFileProvider`, `ObjectStoreArtifactWriter`, and `ObjectStoreTaskStorage`,
  which work through ondio. They were the `Ondio*` classes of
  `robin_adapters.storage.ondio`, which is removed. The extra is still `ondio`.
  `robin_adapters.location` is now
  `robin_adapters.storage.location`. The old paths `robin_adapters.file_provider`,
  `robin_adapters.artifact_writer`, and `robin_adapters.location` are removed.
- The protocols move from `robin_contracts.ports` to `robin_contracts.ports.storage`.
