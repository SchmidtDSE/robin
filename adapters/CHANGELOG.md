# Changelog

## Unreleased

### Added

- `robin_adapters.compute.local`, the `local` compute adapter. `LocalCompute` runs
  `robin-worker run <work URI>` as a detached process in a new session, with its output
  in `worker.log` in the task folder. It starts at most one worker for each task, and
  finds a running worker again in the process table, also from a later process. A task
  runs while any process stays in the worker's process group. It does not enforce the
  compute request.
- `LocalTaskStorage` in `robin_adapters.storage.local`. It writes a task file once at a
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

- The storage adapters move to `robin_adapters.storage`, with one module for each
  backend. `robin_adapters.storage.local` holds `LocalFileProvider` and
  `LocalArtifactWriter`. `robin_adapters.storage.ondio` holds `OndioFileProvider` and
  `OndioArtifactWriter`. `robin_adapters.location` is now
  `robin_adapters.storage.location`. The old paths `robin_adapters.file_provider`,
  `robin_adapters.artifact_writer`, and `robin_adapters.location` are removed.
- The protocols move from `robin_contracts.ports` to `robin_contracts.ports.storage`.
