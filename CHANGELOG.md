# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2026-08-11

### Added

- Add `NeutronMultigroupData` and hybrid neutron multigroup transport with material-local physical energy grids and energy-representation policies, together with the `Material.multigroup()` convenience interface, from [@ilhamv]
- Add manually triggered unit and serial regression compatibility testing for Python 3.11, 3.12, and 3.14 while retaining Python 3.13 for automatic testing, from [@ilhamv]
- Add PEP 561-compatible inline type information and strict Pyright checks for the public Python API, from [@ilhamv]

### Changed

- **Breaking:** Unify native and neutron multigroup materials under one non-polymorphic `Material` and runtime layout, replacing `MaterialMG` and the separate native and multigroup material structures, from [@ilhamv]
- Group transport techniques under `simulation.technique`, from [@ilhamv]
- **Breaking:** Encapsulate MC/DC model building and execution within an explicit `mcdc.Simulation` instance, replacing the global simulation state and interface from [@ilhamv]
- Modernize and reorganize the documentation around distinct user, API reference, theory, project, and developer paths; adopt the PyData Sphinx Theme; and expand the architecture and extension guidance from [@ilhamv]
- Move model-specific finalization into simulation compilation and reserve runtime preparation for framework-level packing and execution setup from [@ilhamv]
- Move regression tests from the custom `run.py` harness to pytest-based collection and reporting from [@massimolarsen]
- **Breaking:** Require Python 3.11 or newer and designate Python 3.13 as the primary automatic test version, from [@ilhamv]
- Run Black and Pyright with Python 3.14 while keeping Black output compatible with every supported Python version, from [@ilhamv]
- Adopt a three-month seasonal cycle for minor releases while continuing to publish patch releases as needed, from [@ilhamv]

### Removed

- Remove the legacy `install.sh` installation helper, from [@ilhamv]

## [0.14.2] - 2026-07-15

### Added

- Add layered documentation philosophy
- Tally spatial filter and scoring upgrades from [@massimolarsen] and [@ilhamv]
  - Add partial current scores
  - Add cell-filtered (and surface-cell-combo) support for current scores
  - Improved checks and error messages in tally-building user interface

### Changed

- Unit test upgrades
  - Combined `object_` and `transport` unit test for more efficient fixture reuse from [@massimolarsen]
  - Replace bare assert np.isclose with proper np.testing.assert_allclose from [@steps-re]

### Fixed

- Fix 2D-vector setter writes nothing (- instead of =) from [@steps-re]
- Fix delayed neutrons are never sampled (transport/physics/neutron/native.py, fission()) from [@steps-re]
- Fix delayed emission time uses β instead of λ (transport/physics/neutron/native.py, fission())from [@steps-re]
- Fix swapped transverse-basis branches (transport/distribution.py, sample_direction()) from [@steps-re]
- Fix divide-by-zero for a -z reference (transport/distribution.py, sample_white_direction()) from [@steps-re]
- Fix tally polar_reference corrupted (object_/tally.py) from [@steps-re]

## [0.14.1] - 2026-07-04

### Changed

- Documentation and packaging metadata fixes

## [0.14.0] - 2026-07-04

### Added

- CHANGELOG.md
- Data library generator upgrade from [@melekderman]
  - Electron data based on EPRDATA14 ACE-format
  - Improved organization for multi-particle data

### Changed

- **Breaking:** Redesigned the tabulated data infrastructure from [@ilhamv] and [@melekderman].
  - Added support for histogram, linear, semilog-x, semilog-y, and log-log interpolation
  - Added optional auxiliary arrays for helper data (e.g., CDFs in distributions)
  - Expanded use throughout distributions and reactions
- Unit test upgrade from [@massimolarsen]
  - Migrated from partial use of pytest to a fully pytest-based test suite
- Documentation updates
  - `mcdc.Source` documentation from [@ilhamv]
- Minor README and pull request template updates
- GitHub workflow for automatically marking stale issues
- Update CITATION.cff
- Automatic version updates on README, Sphinx docs, and pyproject. Version checker on CITATION

### Fixed

- Docker compatibility issue from [@melekderman]

## [0.13.0] - 2026-06-12

This release marks the final development phase under CEMeNT and the beginning of development under CARRE. A significant refactor was completed to improve ease of use, maintainability, and extensibility, restructuring the codebase to support future features and capabilities.

Major refactoring included:

- implementation of `code_factory`, which generates Numba- and GPU-compatible data structures from Python class objects; and
- reorganization of functions into a module-based architecture with well-defined interfaces.

GPU support is currently being updated to match the refactored architecture. The pre-refactor implementation, available in the `cement` branch, retains full GPU support. Complete GPU support for the refactored codebase (including both AMD and NVIDIA GPUs) is targeted for v0.15.0 or v0.16.0.

### Added

- `code_factory` from [@ilhamv]
- ACEtk-based data library generator from [@ilhamv]
- Multi-particle transport and physics model interfaces from [@ilhamv]
- ACE-based continuous-energy neutron physics from [@ilhamv]
- Element data library for electron transport from [@melekderman]
- Single-scattering electron transport from [@melekderman]
- Axis-aligned torus surfaces from [@Talen-Ayers]
- General cylinder surface from [@melekderman]
- Axis-aligned cone surfaces from [@melekderman]
- Docker support from [@melekderman]
- Pull request and issue templates from [@nglaser3]
- Unit tests for distribution sampling from [@massimolarsen]

### Changed

- Redesigned weight window implementation from [@nglaser3]
- Numba-optimized visualizer from [@gunnarrl]
- Documentation updates from [@melekderman]

### Removed

The following features were temporarily removed during the refactor:

- Domain decomposition
- iQMC
- UQ
- Compressed sensing
- Branchless collision
- Derivative Source Method
- Initial Condition bank

The pre-refactor implementation remains available in the `cement` branch as a reference for these features. They will be reintroduced incrementally in future releases.

### Fixed

- Multi-table distribution table selection sampling from [@melekderman]

[Unreleased]: https://github.com/mcdc-project/mcdc/tree/dev
[0.15.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.15.0
[0.14.2]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.2
[0.14.1]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.1
[0.14.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.14.0
[0.13.0]: https://github.com/mcdc-project/mcdc/releases/tag/v0.13.0
[@ilhamv]: https://github.com/ilhamv
[@melekderman]: https://github.com/melekderman
[@massimolarsen]: https://github.com/massimolarsen
[@nglaser3]: https://github.com/nglaser3
[@gunnarrl]: https://github.com/gunnarrl
[@Talen-Ayers]: https://github.com/Talen-Ayers
[@steps-re]: https://github.com/steps-re
