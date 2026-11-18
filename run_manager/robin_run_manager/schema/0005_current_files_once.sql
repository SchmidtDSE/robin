-- The view current_files, with the same columns and rows as in 0003. Only the
-- highest_versions step changes: it is now MATERIALIZED. Postgres computes it once for
-- each query, and the up_to_date check of each row reads the result. Before, Postgres
-- could compute it again for each row, which read every recording outcome once for
-- each row of the view.

CREATE OR REPLACE VIEW current_files AS
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
highest_versions AS MATERIALIZED (
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
