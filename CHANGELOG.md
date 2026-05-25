# Changelog

All notable changes to `pymuvera` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.4.2] — 2026-05-25

### Changed
- Clarified `CALIBRATED_EIGENBASIS` documentation as an experimental
  SpectralQuant-inspired FDE/LSH adaptation.
- Added explicit SpectralQuant attribution and repository link.
- Restored the README reconstruction-error guidance and plots, including
  Eigenbasis-specific reconstruction-risk caveats.
- Added an Eigenbasis spectral-bias tradeoff plot for Error source 6.
- Added `docs/generate_readme_plots.py` to regenerate all README plot PNGs.
- Expanded `NOTICE` with current upstream and research attributions.
- Aligned package metadata with the `pymuvera` distribution name and Python 3.12+
  support.
- Fixed coverage configuration to point at `src/pymuvera`.

### Removed
- Removed tracked macOS `.DS_Store` metadata from the source tree.

## [0.4.1] — 2026-05-23

### Added
- `CALIBRATED_EIGENBASIS` experimental projection mode with calibration save/load support.
- Eigenvalue-weighted SimHash option for calibrated eigenbasis experiments.

### Changed
- Package metadata now targets Python 3.12 and newer only.
- Distribution name is `pymuvera`.

## [0.4.0] — 2026-05-19

### Added
- `LOW_RANK_GAUSSIAN`, `SRHT`, and `CROSS_POLYTOPE` projection modes.
- Densifying LSH empty-partition fill option.
- Additional tests for projection modes, fill strategies, and batch encoding.

## [0.1.0] — 2025-05-01

### Added
- Initial release.
- `MUVERAEncoder` high-level class with pre-cached per-repetition parameters.
- `generate_query_fde` and `generate_document_fde` low-level functional API.
- `FDEConfig` Pydantic v2 immutable configuration model.
- `ProjectionType` enum (`DEFAULT_IDENTITY`, `AMS_SKETCH` / Count Sketch).
- `fill_empty_partitions` support (document side) via SimHash Hamming-distance
  nearest-neighbour fill, operating in batches to bound peak memory.
- Optional final Count-Sketch compression (`final_projection_dimension`).
- Full `py.typed` marker for PEP 561 inline type annotations.
- GitHub Actions CI matrix.
- Trusted PyPI publishing via OIDC (no stored API tokens).
- Faithful attribution to Google's graph-mining Apache 2.0 upstream.

[Unreleased]: https://github.com/smarthi/pymuvera/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/smarthi/pymuvera/releases/tag/v0.4.2
[0.4.1]: https://github.com/smarthi/pymuvera/releases/tag/v0.4.1
[0.4.0]: https://github.com/smarthi/pymuvera/releases/tag/v0.4.0
[0.1.0]: https://github.com/smarthi/pymuvera/releases/tag/v0.1.0
