# 🐦🎙️ robin

ROBIN (**R**eusable **O**pen **B**ioacoustic **In**frastructure) is an open ecosystem of tools for bioacoustic research—from model validation and fine-tuning to inference over field recordings and serving results.

Its modular design supports both local research workflows and deployed services. Use the inference engine on its own, compare and adapt models with evaluation and training tools, or add a control plane and API to manage durable runs and published outputs.

The ecosystem includes:

- **Inference Engine**: run different models (e.g. BirdNET, Perch) in scalable, containerized form, and produce consistent scores, embeddings, and detections.
- **Evaluation and training module**: validate predictions, compare models, and fine-tune models for new datasets.
- **Control plane and API**: coordinate execution, track runs, and publish results.
- **Bidirectional provenance tracking**: determine what a given recording is being used for,
    and what data products were used to produce a set of detections or trained model.
- **Shared contracts**: keep model interfaces, data formats, and artifact identities consistent across tools.

Platform adapters connect ROBIN to existing catalogs, storage, and access policies. SoundHub is the first integration; the core tools are designed to work independently.

**Status:** ROBIN is being designed and extracted from the SoundHub model runner branch `mdc/refactor` by @gottacatchenall. Package boundaries and public interfaces are still evolving.

## Checking OWL against the retired runner's scores

`models/tests/test_owl_end_to_end.py` runs robin's OWL v4 on a fixed field recording and
compares every score with the retired SoundHub model runner's OWL on the same audio. Every
score must be within 10⁻³ of the runner's. The two environments render spectrograms that
differ by one grey level in a few pixels, which moves scores by up to about 2.3 × 10⁻⁴, so a
tighter tolerance would fail on the rendering alone.

The weights and the recording are not in the repository. The test skips when either is
missing and fails when either is not the expected file.

**The weights.** `PNW-Cnet_v4_TF.h5`, 11,144,960 bytes, sha256
`b35e445fa294e90c55330a4efbc230fe6ae2ed3f1c7a3d804db6c86fba28eaa5`. Set
`ROBIN_OWL_V4_WEIGHTS` to its path.

**The recording.** A 48 kHz, mono, 16-bit FLAC of 898 seconds (43,104,000 frames), sha256
`94d3acd8e3562ffaade43e3c0df4bc7a7241ff8ceadcb7d42409af14e8f651bd`. Set
`ROBIN_OWL_BASELINE_AUDIO` to its path. It is frames 314,016,000 to 357,120,000 (6,542 s to
7,440 s) of a SoundHub recording that is 48 kHz, mono, 16-bit, 1,555,200,000 frames long,
with sha256 `422ac83435d94993df96eb008d431db1a5bc03af8e5e1b76ec4ac0a207bc5a65`. This cuts it,
reading only the part it needs, and with libsndfile 1.2.2 it reproduces the hash above:

```python
import soundfile as sf

with sf.SoundFile("source.flac") as source:
    source.seek(6542 * source.samplerate)
    audio = source.read(898 * source.samplerate, dtype="int16")
sf.write("owl_baseline.flac", audio, 48000, subtype="PCM_16")
```

**Running it.**

```sh
ROBIN_OWL_V4_WEIGHTS=/path/to/PNW-Cnet_v4_TF.h5 \
ROBIN_OWL_BASELINE_AUDIO=/path/to/owl_baseline.flac \
pixi run -e owl test-owl
```

**The expected scores.** `models/tests/owl_v4_expected_scores.arrow` was made once by the
retired runner's OWL, on the CPU with batch size 64 and every label kept. Its schema
metadata records the audio and weights hashes, the runner's commit and the runtime versions.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
