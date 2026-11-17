-- The size and row count of each output file, and the windows and audio checksum of
-- each covered recording. The values come from the task's result.json. Rows recorded
-- before this migration have null in these columns: their values are only in result.json.

ALTER TABLE output_files
    ADD COLUMN size_bytes bigint CHECK (size_bytes >= 0),
    ADD COLUMN rows bigint CHECK (rows >= 0);

ALTER TABLE recording_outcomes
    ADD COLUMN windows_completed integer CHECK (windows_completed >= 0),
    ADD COLUMN first_window_start_s double precision,
    ADD COLUMN last_window_end_s double precision,
    -- The sha256: checksum of the audio file that the engine read. It is null when the
    -- input is saved embeddings.
    ADD COLUMN input_checksum text;
