# Output versions

An output version names one version of a model's outputs: the code that makes them, from
the adapter and the engine to the preprocessing and the locked dependencies. The table in
`robin_models/output_versions.py` gives each model's current version. A version goes up
in the change that changes that model's outputs, for any reason. A change that leaves the
outputs the same keeps the version. A run freezes the versions it uses, so outputs made
under an older version show as out of date.

Each bump adds an entry here, in the same change. An entry gives the date, the commit,
the cause, how the outputs changed and by how much, and what the change means for
outputs made under the version before. Entries are newest first for each model, and the
models are in the table's order.

## model:owl/v4 version 3

- **Date:** 2026-11-19
- **Commit:** this change (the hash goes here after the commit)
- **Cause:** a dependency. OWL makes its spectrograms with sox_numpy 0.1 in place of
  sox_tensorflow 0.1.3. sox_numpy keeps 0.1.3's algorithm, with NumPy's rfft in place of
  TensorFlow's. Its pixels differ from 0.1.3's in about 0.002% of its test pixels, always
  by one level. Its agreement with the `sox` binary is as good as 0.1.3's or better (see
  sox_numpy's README). The adapter also sets `OMP_NUM_THREADS=1` unless it is set, which
  does not change the pixels.
- **Outputs:** on the CPU, in the 14 cases of the check (worker image, linux/amd64), the
  windows and labels are the same. Most scores change, by at most 8.4e-4 on the 3 real
  recordings (3,969 windows) and at most 4.7e-6 on the 11 made files. No window's top
  label changes. Silence gives the same scores.
- **Old outputs:** version 2 scores differ from version 3 scores by the amounts above.

## model:owl/v4 version 2

- **Date:** 2026-11-18
- **Commit:** 0ace788
- **Cause:** the adapter. OWL moved to the shared window loop, and makes its
  spectrograms on the CPU on every worker. Before, it made them on the GPU of a GPU
  worker.
- **Outputs:** on the CPU, the scores files had equal bytes in all 14 cases of the check
  (3 real recordings and 11 made files, at 8 to 96 kHz). On a GPU worker, nobody proved
  that the scores stayed the same, so the version went up.
- **Old outputs:** version 1 outputs made on the CPU are the same as version 2 outputs.
  Version 1 outputs made on a GPU worker can differ.

## model:owl/v4 version 1

- **Date:** 2026-10-09
- **Commit:** 6378ee4
- **Cause:** the first version, when the table was added.
- **Outputs:** none before.
- **Old outputs:** none before.

## model:birdnet/v2p4 version 1

- **Date:** 2026-10-09
- **Commit:** 6378ee4
- **Cause:** the first version, when the table was added.
- **Outputs:** none before.
- **Old outputs:** none before.

## model:perch/v1p8 version 1

- **Date:** 2026-10-09
- **Commit:** 6378ee4
- **Cause:** the first version, when the table was added. The entry was named
  `model:perch/v8` until adf2b62 (2026-11-13), which renamed the model to say that it is
  Perch v1.8. The rename did not change the scores.
- **Outputs:** none before.
- **Old outputs:** none before. Outputs made before adf2b62 name `model:perch/v8`.

## head_runtime:onnx version 1

- **Date:** 2026-10-09
- **Commit:** 6378ee4
- **Cause:** the first version, when the table was added.
- **Outputs:** none before.
- **Old outputs:** none before.
