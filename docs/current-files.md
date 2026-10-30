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
