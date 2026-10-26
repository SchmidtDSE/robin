# Changelog

## Unreleased

### Changed

- The storage adapters move to `robin_adapters.storage`, with one module for each
  backend. `robin_adapters.storage.local` holds `LocalFileProvider` and
  `LocalArtifactWriter`. `robin_adapters.storage.ondio` holds `OndioFileProvider` and
  `OndioArtifactWriter`. `robin_adapters.location` is now
  `robin_adapters.storage.location`. The old paths `robin_adapters.file_provider`,
  `robin_adapters.artifact_writer`, and `robin_adapters.location` are removed.
- The protocols move from `robin_contracts.ports` to `robin_contracts.ports.storage`.
