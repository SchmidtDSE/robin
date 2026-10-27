-- Runs, their recordings and tasks, their output files, and the outcome of each recording.
-- output_versions is jsonb, not json, because a unique index and a foreign key compare it.

CREATE TABLE runs (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind text NOT NULL CHECK (kind IN ('inference')),
    submitted_by text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    state text NOT NULL DEFAULT 'preparing'
        CHECK (state IN ('preparing', 'preparation_failed', 'ready')),
    preparation_failure json,             -- {reason, message, details}
    -- The compute and storage settings that the run keeps from the profile at submit.
    execution_settings json NOT NULL CHECK (json_typeof(execution_settings) = 'object'),
    cancelled_at timestamptz,
    cancelled_by text,
    withdrawn_at timestamptz,
    withdrawn_by text,
    CHECK ((state = 'preparation_failed') = (preparation_failure IS NOT NULL)),
    CHECK ((cancelled_at IS NULL) = (cancelled_by IS NULL)),
    CHECK ((withdrawn_at IS NULL) = (withdrawn_by IS NULL))
);

CREATE INDEX runs_by_creation ON runs (created_at, run_id);

CREATE TABLE inference_runs (
    run_id uuid PRIMARY KEY REFERENCES runs,
    configuration_digest text NOT NULL REFERENCES configurations,
    output_versions jsonb NOT NULL CHECK (jsonb_typeof(output_versions) = 'object'),
    provenance json NOT NULL,
    -- The target of the outcomes' foreign key, so that each outcome copies its run's
    -- configuration and versions.
    UNIQUE (run_id, configuration_digest, output_versions)
);

-- The submitted list. No row is updated or deleted.
CREATE TABLE run_recordings (
    run_id uuid NOT NULL REFERENCES inference_runs,
    namespace text NOT NULL,
    value text NOT NULL,
    audio_uri text NOT NULL,
    duration_seconds double precision CHECK (duration_seconds > 0),
    PRIMARY KEY (run_id, namespace, value)
);

CREATE TABLE tasks (
    task_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES runs,
    state text NOT NULL CHECK (state IN (
        'queued', 'submitting', 'submitted', 'running', 'finalizing',
        'succeeded', 'failed', 'cancelled'
    )),
    retry_of uuid REFERENCES tasks,
    created_by text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX tasks_by_run ON tasks (run_id, created_at, task_id);

CREATE TABLE output_files (
    file_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES runs,
    task_id uuid NOT NULL REFERENCES tasks,
    namespace text NOT NULL,
    value text NOT NULL,
    kind text NOT NULL,
    contract_id text NOT NULL,
    uri text NOT NULL,
    checksum text NOT NULL,
    UNIQUE (run_id, namespace, value, kind)
);

CREATE TABLE recording_outcomes (
    run_id uuid NOT NULL,
    namespace text NOT NULL,
    value text NOT NULL,
    configuration_digest text NOT NULL,
    output_versions jsonb NOT NULL,
    state text NOT NULL CHECK (state IN (
        'skipped', 'pending', 'covered', 'failed', 'released', 'withdrawn'
    )),
    skip_reason text CHECK (skip_reason IN (
        'already_covered', 'in_progress', 'input_not_available', 'input_has_no_windows',
        'input_location_unsupported'
    )),
    task_id uuid REFERENCES tasks,        -- the task that holds a pending claim
    input_file_id bigint REFERENCES output_files,  -- a head's pinned embeddings file
    PRIMARY KEY (run_id, namespace, value),
    FOREIGN KEY (run_id, namespace, value) REFERENCES run_recordings,
    FOREIGN KEY (run_id, configuration_digest, output_versions)
        REFERENCES inference_runs (run_id, configuration_digest, output_versions),
    CHECK ((state = 'skipped') = (skip_reason IS NOT NULL))
);

CREATE INDEX recording_outcomes_by_task ON recording_outcomes (task_id);

-- A recording has at most one claim for each configuration and set of output versions:
-- unfinished work or a valid result.
CREATE UNIQUE INDEX recording_claims
    ON recording_outcomes (namespace, value, configuration_digest, output_versions)
    WHERE state IN ('pending', 'covered');
