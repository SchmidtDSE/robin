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

## License

BSD-3-Clause. See [LICENSE](LICENSE).
