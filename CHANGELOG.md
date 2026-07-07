# Changelog

## v0.1.4 — 2026-07-07

### Fixed
- Build workflow now produces wheels on Windows and macOS and builds the `sdist` exactly once on Ubuntu.
- Linux users install from the source distribution; PyPI does not accept plain `linux_x86_64` wheels, and the optional C++ extension can be compiled locally or falls back to pure Python.

## v0.1.3 — 2026-07-07

### Fixed
- Attempted to switch wheel builds to `pypa/cibuildwheel` for `manylinux` wheels. Reverted because the optional C++ extension failed to build in cibuildwheel's isolated environments on Windows and macOS.

## v0.1.2 — 2026-07-07

### Fixed
- Build workflow split into matrix wheel builds and a single Ubuntu `sdist` build, preventing corrupt/overwritten tarballs during artifact merge.
- Fixed PyPI trusted-publishing setup instructions in README (workflow name must be filename-only).

## v0.1.1 — 2026-07-07

### Fixed
- Synced documentation and README for v0.1.0 release content.
- Cleaned up lint issues surfaced by `ruff`.
- Cleaned up type issues surfaced by `mypy`.
- Fixed source distribution (`sdist`) packaging metadata and build configuration.

