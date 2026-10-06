# robin-models

`robin-models` holds ROBIN's model adapters and its head runtime, each with its runtime behind an
extra. The backbone adapters bundle their card and taxa registry; the head runtime bundles
neither.

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

`full` score retention writes one row per label per window: about 7.8 million rows per hour of
audio (1,200 windows × 6,522 labels). Use `thresholded` or `top_k` unless you need every score.

### Credit

BirdNET is developed by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab
of Ornithology and Chemnitz University of Technology (<https://birdnet.cornell.edu>).

### Licence

The model is for non-commercial use. Upstream states its licence as CC BY-NC 4.0 (the Zenodo
record) and as CC BY-NC-SA 4.0 (BirdNET-Analyzer's README). It is not covered by robin's
BSD-3-Clause licence.

## Perch v8

### Where the files come from

Perch v8's files are Google's bird vocalization classifier for TensorFlow 2, version 8, on Kaggle:
<https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8>.
Download its archive,
`bird-vocalization-classifier-tensorflow2-bird-vocalization-classifier-v8.tar.gz`.

- Size: 88,258,975 bytes
- sha256: `e58f0d9e73c34c4762f4c73d60716ebd0f26b61f35d9335a1141f7868f41d574`

Kaggle's page now calls this "the Perch v1 model" and points to Perch 2. Perch 2 is a different
model; this adapter runs version 8 only.

### What a work pins

Paths are inside the unpacked archive.

| Role | Path in the archive | sha256 | Bytes |
| --- | --- | --- | --- |
| `saved_model` | `saved_model.pb` | `8d603a05e7712c070447815e8835490d4a9b951310ecc60e4ec9ab1b50730a72` | 3,881,694 |
| `variables_data` | `variables/variables.data-00000-of-00001` | `b1c81c99b09d8ac37c55ac09ff14e3dd4c98f9f8de341d7d99161ebc9fe41aad` | 95,806,615 |
| `variables_index` | `variables/variables.index` | `93fe310af9edf5f2662c3a32c75b9a96c89fe2b9905f45e5d1213adba4737a67` | 8,406 |
| `labels` | `assets/label.csv` | `0c85cbc1d8391b67641a307d5f2219f96366db8ed229fc13d281a2e681ce83cc` | 97,298 |
| `taxa_registry` | bundled: `robin_models/perch/resources/taxa_registry.csv` | `ca28f370cb5924af9966fee9fcd0a12d57632bba85a01a3f5908f9011ef99fd9` | 1,074,422 |

`labels` and `taxa_registry` are needed only for scores.

### How to use them

robin never rehosts these files. Download the archive from Kaggle, unpack it, and pin each file
by its path, digest and size. The engine checks every digest and size before the model is built.

A work that wants only embeddings should pin neither `taxa_registry` nor `labels`. Unlike
BirdNET, a pinned registry costs no extra computation: one call returns both outputs.

### The registry

The bundled registry is copied byte for byte from taxa_bureau commit `1213658`,
`taxa_registries/perch-v8_taxa_registry.csv`. Its labels equal the lines of `assets/label.csv`
after its `ebird2021` header, and the adapter checks that when it is built. 15 of its 10,932 eBird
codes have no GBIF taxon, and are published as `unresolved`.

### Running it

Install `robin-models[perch]`. Perch runs on the CPU only. Each batch is one call to the
SavedModel's `serving_default` signature, which returns the embedding and the classifier's
logits; the scores are the logits' sigmoid, between 0 and 1. Audio not at 32 kHz is resampled
with SciPy's polyphase resampler, as Perch's training data was, so a window equals the same part
of the whole recording resampled at once. Embeddings are stored as the model returns them, not
normalised.

`full` score retention writes one row per label per window: about 7.9 million rows per hour of
audio (720 windows × 10,932 labels). Use `thresholded` or `top_k` unless you need every score.

### Credit

Perch is developed by Google Research and Google DeepMind. Cite Ghani et al., "Global birdsong
embeddings enable superior transfer learning for bioacoustic classification", *Scientific
Reports* 13, 22876 (2023).

### Licence

Kaggle states the licence of version 8 as Apache 2.0. It is not covered by robin's BSD-3-Clause
licence.

## Running more than one model

Run one TensorFlow model per process. OWL needs legacy Keras, and TensorFlow chooses its Keras
version once, when it is first imported. So a process that first imports BirdNET's or Perch's
adapter can't then load OWL.

## Heads

A head is a small model that scores one backbone's saved embeddings. It is three files the work
pins: a head card, an ONNX graph (role `graph`) and a taxa registry (role `taxa_registry`). robin
ships none of them. A head runs from what the work pins, and the engine checks every file's digest
and size before the head is built.

### The card

```yaml
model_name: example-head
model_version: "1"
runtime: onnx
backbone: {name: perch, version: v8, digest: 'sha256:v1:<the backbone card digest>'}
embedding_dim: 1280
min_detection_threshold: 0.0
score_domain: sigmoid
taxa_registry_digest: sha256:<the registry file's sha256>
```

- `model_name` and `model_version` name the head.
- `runtime` names the code that runs the head. `onnx` is the one runtime robin has.
- `backbone` is the backbone whose embeddings the head reads: its `name`, `version` and card
  `digest`.
- `embedding_dim` is the width the graph takes. It must equal the backbone's.
- `min_detection_threshold` means what it means on a model card.
- `score_domain` means what it means on a model card, and is `sigmoid`.
- `taxa_registry_digest` is the sha256 of the registry file that labels the head's scores.

### The input

Each recording's input is its embeddings file, as its backbone wrote it, named on the recording
by uri and checksum. The engine checks that the file came from the card's backbone, at the card's
width, for that recording, before the head sees a vector.

### The graph

The graph has one input, float32, of shape `[batch, embedding_dim]`, and one output, float32, of
shape `[batch, number of registry rows]`. The batch dimension must not be fixed. The input is the
backbone's raw vectors as stored, widened to float32. Each output column is the score of the
registry row at that position. The graph applies its own activation; robin applies none, and no
clipping or threshold. A non-finite output fails the recording.

### Normalisation

robin applies no normalisation to a head's input. Any normalisation the head was trained with,
such as `x / (‖x‖₂ + 1e-8)`, must be part of the graph. A graph that leaves it out still runs and
gives plausible wrong scores, and robin can't detect that.

### Running it

Install `robin-models[onnx-head]`. A head runs on the CPU only, with onnxruntime's CPU provider,
in batches of `batch_size` rows (default 256).

### Where outputs go

A head's scores and detections use the same paths as its backbone's, so a head work needs a
writer root other than its backbone's. A writer refuses to replace a file that differs.
