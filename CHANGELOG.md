# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2025-12-28

### Removed
- Removed bespoke RK4 backend - RoadRunner (CVODE) is now the only ODE solver
- Removed `--solver` CLI flag from `ssys-recast` and `recast_models.py`

### Changed
- Renamed `biomodels/` directory to `biomodels_batch/` for clarity

### Maintenance
- Added comprehensive unit tests (213 tests, 45% coverage)
- Fixed all ruff lint errors
- Added autonomy status detection and display

## [0.5.0] - 2025-12-27

### Added
- SBML-first parser architecture using reference Antimony library
- New `parse_antimony_via_sbml()` function for robust Antimony parsing
- Support for SBML `InitialAssignment` elements for parameter-dependent initial conditions
- All 118 test model files now parse correctly with standard Antimony library

### Changed
- Core dependencies now include `libroadrunner`, `antimony`, and `python-libsbml`
- Validator uses SBML-first parser for both original and recast files

### Fixed
- Fixed handling of symbolic expressions in S-system recasting
- Improved handling of rational functions and composite functions
- Corrected parameter substitution in direct S-system recasting
- Updated auxiliary variable naming process
- Enhanced Antimony output generation for complex expressions
- Fixed semicolon placement issues in Antimony parser
- Fixed reserved name conflicts (e.g., `gamma` → `gamma_rate`)

## [0.4.0] - 2025-11-21

### Changed
- Residual verification now intelligently skips models with function lifting
  - The pool-based residual formula is not applicable to systems with lifted auxiliaries
  - Reports "N/A" for residual with clear explanation directing users to trajectory comparison
  - Trajectory comparison remains the definitive verification for all models

### Fixed
- Resolved incorrect residual computation for models using function lifting (models #11, #13)
  - Models with lifted functions now display "N/A" instead of incorrect residual values
  - Documentation clarifies that trajectory comparison is the authoritative verification

## [0.3.0] - 2025-11-21

### Added
- Automatic composite function lifting: the recaster now automatically detects and lifts arbitrary composite functions (exp, sin, log, trig functions, special functions, etc.) to auxiliary variables before recasting
- Universal support for any differentiable function that sympy can differentiate, making the recaster applicable to a vastly wider range of ODE models
- Chain rule-based ODE generation for auxiliary variables representing composite functions

### Fixed
- **Critical fix**: Lifted auxiliaries (W_i for rational functions, Z_i for composite functions) now remain as single variables instead of being decomposed into pools
  - Lifted variables already have power-law ODEs from the lifting process, so pool construction is unnecessary and caused runtime errors
  - Previously, pool construction created equations with circular references (e.g., W_1 appearing in equations for its own pool auxiliaries)
  - Now lifted variables are kept as single variables with direct S-system equations (growth/decay form)
- **Critical fix**: Residual checker now uses original (pre-lifting) ODEs for verification
  - Previously used lifted system ODEs which reference lifted auxiliaries, causing incorrect residual computation
  - Now builds fresh symbolic system from original IR to get true original ODEs without lifted variable references
  - Residual correctly measures how well the recast reproduces the original model dynamics
- Lifted auxiliaries are no longer added to `factor_map`, preventing them from appearing in reconstruction plots
- Plotting in verification notebook now only shows original model variables, not internal lifted auxiliaries
- Models #11 (Monod chemostat) and #13 (composite function decomposition) now show correct residuals ~0 and execute successfully

## [0.2.0] - 2025-11-21

### Added
- Automatic rational function lifting: the recaster now automatically detects and lifts rational functions (e.g., `1/(X+1)`) to auxiliary variables before recasting, enabling S-system transformation of models with Monod kinetics and similar terms
- Chain rule-based ODE generation for auxiliary variables representing rational denominators

### Fixed
- Bug in `test_canonical_naming` test case where the test expected two auxiliaries for a simplified single-term equation. Updated the test to use non-combinable terms, ensuring proper testing of canonical naming with two distinct auxiliaries.
- Rational function lifting now correctly uses lifted ODEs (not original ODEs) when computing auxiliary variable derivatives, preventing circular denominator references

## [0.1.0] - 2025-11-21

### Added
- Initial release of ssys package
- Core recasting algorithm using pool-auxiliary construction
- Antimony parser for input models
- Symbolic ODE system builder
- Exact algebraic transformation to canonical S-system form
- CLI tool (`ssys-recast`) for batch processing
- Jupyter notebook report generation with:
  - Side-by-side trajectory comparison
  - LaTeX rendering of ODEs and S-systems
  - Algebraic residual verification
  - Factor mapping visualization
- Test suite with 10 biological/dynamical models
- BSD license
- Basic documentation in README
- CITATION.cff for academic citation
- Modern Python packaging with pyproject.toml

### Features
- Parse reactions, assignments, and explicit derivative rules from Antimony syntax
- Build symbolic ODE systems with parameters
- Recast arbitrary ODEs to canonical S-system form (growth - decay)
- Canonical auxiliary variable naming (X_1, X_2, ...)
- Export recast systems back to Antimony format
- Numerical integration and comparison (RK4)
- Residual checking for exactness verification

### Dependencies
- Python >=3.9
- sympy >=1.12
- numpy >=1.24
- matplotlib >=3.7
- nbformat >=5.9

[Unreleased]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.5.1...main
[0.5.1]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.5.0...v0.5.1
[0.5.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.4.0...v0.5.0
[0.4.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.3.0...v0.4.0
[0.3.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/tags/v0.1.0
