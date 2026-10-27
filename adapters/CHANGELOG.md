# Changelog

## Unreleased

### Added

- `robin_adapters.catalog.folder_layout`, the `folder-layout` catalog adapter. It finds
  recordings by the paths of the files under a local folder, with a pattern such as
  `site{site}/{date}/{recorder}_{date}_{time}.wav`, and makes each identity from the
  pattern's fields. A selection gives a text value, a list of text values, or a text
  range for each field. A folder that cannot be read stops the selection, so that no
  run gets a list with recordings missing.

### Changed

- The storage adapters move to `robin_adapters.storage`, with one module for each
  backend. `robin_adapters.storage.local` holds `LocalFileProvider` and
  `LocalArtifactWriter`. `robin_adapters.storage.ondio` holds `OndioFileProvider` and
  `OndioArtifactWriter`. `robin_adapters.location` is now
  `robin_adapters.storage.location`. The old paths `robin_adapters.file_provider`,
  `robin_adapters.artifact_writer`, and `robin_adapters.location` are removed.
- The protocols move from `robin_contracts.ports` to `robin_contracts.ports.storage`.
