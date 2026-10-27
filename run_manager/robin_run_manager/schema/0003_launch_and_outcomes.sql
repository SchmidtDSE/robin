-- How each task is launched and how it ends, the failure of each recording, and the
-- current_files view.

ALTER TABLE tasks
    ADD COLUMN compute_ref text,          -- the compute adapter's name for the execution
    ADD COLUMN work_uri text,
    ADD COLUMN work_file_digest text,     -- the sha256: checksum of the work file's bytes
    ADD COLUMN compute_request json,      -- what the task asks of the compute service
    ADD COLUMN work_resources json,       -- what the work file gives the worker's code
    ADD COLUMN reserved_at timestamptz,
    ADD COLUMN submitted_at timestamptz,
    ADD COLUMN started_at timestamptz,
    ADD COLUMN ended_at timestamptz,
    ADD COLUMN outcome_recorded_at timestamptz,
    ADD COLUMN released_at timestamptz,
    ADD COLUMN robin_release text,        -- the Robin release that the result names
    ADD COLUMN failure json,              -- {reason, message, detail}
    -- A failed or cancelled task can end before it is reserved.
    ADD CHECK (
        state NOT IN ('submitting', 'submitted', 'running', 'finalizing', 'succeeded')
        OR (reserved_at IS NOT NULL AND work_file_digest IS NOT NULL)
    ),
    ADD CHECK ((state = 'failed') = (failure IS NOT NULL)),
    ADD CHECK (
        released_at IS NULL
        OR (state IN ('succeeded', 'failed', 'cancelled') AND reserved_at IS NOT NULL)
    );

ALTER TABLE recording_outcomes
    -- {scope, code, stage, message}. The scope is "recording" for the recording's own
    -- failure, and "task" when its whole task failed; then there is no stage.
    ADD COLUMN failure json,
    ADD CHECK ((state = 'failed') = (failure IS NOT NULL));

-- The current output files: one row for each file of each current result.
--
-- The current result of a recording and configuration is its covered outcome in a run
-- that is not withdrawn, with the greatest output_versions. up_to_date is true when each
-- of the result's versions is the highest version of that ID in any covered or withdrawn
-- outcome, of any configuration.
--
-- This view and its columns are a published interface: a change to them is a breaking
-- change.
CREATE VIEW current_files AS
WITH current_results AS (
    SELECT DISTINCT ON (o.namespace, o.value, o.configuration_digest)
        o.run_id, o.namespace, o.value, o.configuration_digest, o.output_versions
    FROM recording_outcomes o
    JOIN runs r ON r.run_id = o.run_id
    WHERE o.state = 'covered' AND r.withdrawn_at IS NULL
    -- Two jsonb objects with the same keys are ordered by their values in key order.
    -- Each release's versions are equal or higher than the last release's, so the
    -- greatest object holds the newest versions.
    ORDER BY o.namespace, o.value, o.configuration_digest, o.output_versions DESC
),
highest_versions AS (
    SELECT version.key AS output_version_id, max(version.value::numeric) AS highest
    FROM recording_outcomes o, jsonb_each(o.output_versions) AS version
    WHERE o.state IN ('covered', 'withdrawn')
    GROUP BY version.key
)
SELECT
    c.namespace,
    c.value,
    c.configuration_digest,
    c.output_versions,
    NOT EXISTS (
        SELECT 1
        FROM jsonb_each(c.output_versions) AS version
        JOIN highest_versions h ON h.output_version_id = version.key
        WHERE version.value::numeric < h.highest
    ) AS up_to_date,
    f.kind,
    f.contract_id,
    f.uri,
    f.checksum,
    c.run_id
FROM current_results c
JOIN output_files f
    ON f.run_id = c.run_id AND f.namespace = c.namespace AND f.value = c.value;

COMMENT ON VIEW current_files IS
    'Published interface. One row for each output file of the current result of each '
    'recording and configuration: its covered outcome in a run that is not withdrawn, '
    'with the greatest output_versions. up_to_date is true when no covered or withdrawn '
    'outcome has a higher version of any of its output version IDs.';
