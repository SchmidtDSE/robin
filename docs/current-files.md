# The current files

The view `current_files` in the database gives one row for each output file of each
recording's current result. It is a published interface: its name and columns change
only with a breaking release.

| Column | What it holds |
|---|---|
| `namespace`, `value` | The recording. |
| `configuration_digest` | The configuration that made the result. |
| `output_versions` | The model output versions of the result, as a JSON object. |
| `up_to_date` | True if no result has a higher version of any of these output versions. |
| `kind` | `scores`, `embeddings`, or `detections`. |
| `contract_id` | The file's format, such as `robin.scores.parquet/1`. |
| `uri` | Where the file is. |
| `checksum` | The file's `sha256:` checksum. |
| `run_id` | The run that made the result. |

The current result of a recording and a configuration is its result in a run that is
not withdrawn, with the newest output versions.

## The index file

`robin files export` copies `current_files` to one Parquet file in the profile's
storage, so that a reader that cannot reach the database can find the current files.
The file is `<storage root>/index/current_files.parquet`. Each export also keeps a
dated copy at `<storage root>/index/history/<time>/current_files.parquet`, where
`<time>` is the UTC time of the export, such as `20261117T120000Z`.

The file's contract is `robin.current-files.parquet/1`. It is in the Parquet metadata
under `robin.contract`. A change to the columns is a new contract and a new file name.

| Column | What it holds |
|---|---|
| `namespace`, `value` | The recording. |
| `model_id` | The model of the configuration, such as `owl/v4`. |
| `presets` | The names of the presets of the configuration, sorted. It is empty if the configuration has no preset. |
| `configuration_digest` | The configuration that made the result. |
| `kind` | `scores`, `embeddings`, or `detections`. |
| `contract_id` | The file's format, such as `robin.scores.parquet/1`. |
| `uri` | Where the file is. |
| `checksum` | The file's `sha256:` checksum. |
| `size_bytes` | The file's size in bytes. |
| `rows` | The file's row count. |
| `output_versions` | The model output versions of the result, as JSON text with sorted keys. |
| `up_to_date` | True if no result has a higher version of any of these output versions. |
| `run_id` | The run that made the result. |

`size_bytes` and `rows` are empty for files that Robin recorded before this release.

The index changes only when `robin files export` runs, so run it after each run ends.
Only one export runs at a time. A second export that starts while one runs is
refused with `export_in_progress`, and writes nothing.
