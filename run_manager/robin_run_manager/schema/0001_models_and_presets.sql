-- Registered models, the configurations that use them, and preset names.
-- No row is updated or deleted.
-- Documents are json, not jsonb: json keeps the exact text, so a document reads back
-- with the same values and digest. jsonb would turn a float such as 1e16 into an integer.

CREATE TABLE models (
    model_id text PRIMARY KEY,            -- name/version, for example owl/v4
    card json NOT NULL,
    card_digest text NOT NULL,
    files json NOT NULL,                 -- role -> {uri, digest, size_bytes}
    registered_by text NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE configurations (
    digest text PRIMARY KEY,
    model_id text NOT NULL REFERENCES models,
    configuration json NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE presets (
    name text PRIMARY KEY,
    configuration_digest text NOT NULL REFERENCES configurations,
    created_by text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
