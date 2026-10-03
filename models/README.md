# robin-models

`robin-models` holds ROBIN's model adapters, each with its bundled card and taxa registry, and
each model's runtime behind an extra.

## BirdNET v2.4

### Where the files come from

BirdNET's files are in the archive `BirdNET_v2.4_protobuf.zip`, in Zenodo record 15050749
(DOI `10.5281/zenodo.15050749`, <https://zenodo.org/records/15050749>). Download it from
<https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip>.

- Size: 124,522,908 bytes
- sha256: `a2b99e7bb621da755fc6db785e51784bd12b22cf4518a0a0c9d09bec966149e4`
- MD5: `3863ddaa19ad6acf73622cb1c6086c40` (Zenodo's published checksum)

### What a work pins

| Role | Path in the archive | sha256 | Bytes |
| --- | --- | --- | --- |
| `saved_model` | `audio-model/saved_model.pb` | `63a03e31f3bc4bbd3ac03aeb261b5bf529c173c652d77df4d9b0c0b39f09122d` | 6,929,070 |
| `variables_data` | `audio-model/variables/variables.data-00000-of-00001` | `b05a11f1d7501351c2d48ef9aa96803a54ac76ba23e3370d7bab138d91609418` | 51,445,883 |
| `variables_index` | `audio-model/variables/variables.index` | `5ac18d1811c430fecc5286ea1792a3363cb5906b390105743c122f1e2812106e` | 15,092 |
| `labels` | `labels/en_us.txt` | `b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16` | 259,740 |
| `taxa_registry` | bundled: `robin_models/birdnet/resources/taxa_registry.csv` | `b758a58bee475d45b1fb04a8f42cbaa13c794d2dc782f0e86968e6c1515d4c2d` | 838,069 |

`labels` and `taxa_registry` are needed only for scores.

### How to use them

robin never rehosts these files. Download the archive from Zenodo, unpack it, and pin each file
by its path, digest and size. The engine checks every digest and size before the model is built.

A work that wants only embeddings should pin neither `taxa_registry` nor `labels`. A pinned
registry makes the adapter compute scores, at the cost of one matrix multiply per batch, and the
engine discards them when they weren't asked for.

### The registry

The bundled registry is copied byte for byte from taxa_bureau commit `1213658`,
`taxa_registries/birdnet_taxa_registry.csv`. Its labels equal `labels/en_us.txt` line for line,
and the adapter checks that when it is built.

### Running it

Install `robin-models[birdnet]`. BirdNET runs on the CPU only. Each batch runs the SavedModel's
`embeddings` signature once, and the scores are its output multiplied by the model's own
classifier layer, which is what the `basic` signature computes.

### Credit

BirdNET is developed by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab
of Ornithology and Chemnitz University of Technology (<https://birdnet.cornell.edu>).

### Licence

The model is for non-commercial use. Upstream states its licence as CC BY-NC 4.0 (the Zenodo
record) and as CC BY-NC-SA 4.0 (BirdNET-Analyzer's README). It is not covered by robin's
BSD-3-Clause licence.
