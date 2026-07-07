# Changelog

## v0.1.2 — 2026-07-07

### Fixed
- Build workflow now produces wheels on the OS/Python matrix and builds the `sdist` exactly once on Ubuntu, preventing corrupt/overwritten tarballs during artifact merge.
- Fixed PyPI trusted-publishing setup instructions in README (workflow name must be filename-only).

## v0.1.1 — 2026-07-07

### Fixed
- Synced documentation and README for v0.1.0 release content.
- Cleaned up lint issues surfaced by `ruff`.
- Cleaned up type issues surfaced by `mypy`.
- Fixed source distribution (`sdist`) packaging metadata and build configuration.

