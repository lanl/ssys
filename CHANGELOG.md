# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- SBML parser now rejects, at the trust boundary with a structured
  `unsupported_feature` error, two input classes that were previously accepted
  but silently mis-integrated: variable reaction stoichiometry (a non-constant
  `<stoichiometryMath>`, or a `speciesReference` id driven by a
  rule/`InitialAssignment`) and time-varying compartment volume (a rate-rule
  compartment, or an assignment-rule compartment that does not fold to a
  constant, that owns a concentration species whose `-[S]·(dV/dt)/V` dilution
  term is unmodeled). Constant volumes, including assignment rules that fold to a
  constant, remain supported.
- Legacy Antimony parser (`ssys.parse_antimony` / `--parser legacy`) now fails
  closed at its trust boundary, with the same structured `unsupported_feature`
  error, on constructs its simplified single-unit-compartment subset cannot
  correctly interpret — a non-unit or multiple compartment, a `conversionFactor`,
  or a variable stoichiometric coefficient — instead of silently emitting wrong
  ODEs (the legacy-path companion to the SBML fixes in this release). A trivial
  unit `compartment cell = 1;` is still accepted, so every existing model is
  unaffected. `ssys.parse_antimony` also now emits a `DeprecationWarning` on
  direct use, and `build_sym_system` defensively rejects an incoming IR carrying a
  non-unit compartment size.
- Recast notebook report no longer routes simulation through the legacy parser:
  it re-simulates the original and recast Antimony through the SBML path, so a
  non-unit-volume model still renders a correct simulation section rather than
  inheriting the legacy fail-close.

### Fixed
- SBML L3 `conversionFactor` (a species' own or the Model default) is now applied
  to reaction-derived ODEs, `d(amount_S)/dt = cf_S·Σ stoich·kineticLaw`, composing
  with compartment-volume scaling. It was previously referenced nowhere in the
  parser, so any model declaring one integrated the unscaled rate. Verified
  against libRoadRunner.
- Constant reaction stoichiometry supplied through an L2 `<stoichiometryMath>` is
  now constant-folded over parameters and compartment sizes and used exactly.
  Previously only the static `getStoichiometry()` attribute was read (reported as
  1.0 when a `<stoichiometryMath>` was present), silently mis-integrating the
  reaction.

## [0.6.0] - 2026-07-04

### Release Policy
- Release maturity: alpha.
- Supported on Python 3.10, 3.11, and 3.12. Platform matrix automation is deferred until hosted project infrastructure exists.
- Input trust boundary: Antimony and SBML inputs are trusted local scientific model files, not safe untrusted uploads for multi-tenant or security-sensitive services.

### Changed
- Stopped redistributing the third-party BioModels benchmark SBML corpus;
  `biomodels_batch/data/` is now gitignored and regenerated locally via
  `step1_fetch.py`.
- Removed internal-only references (release punchlist notes and an internal
  issue-tracker URL) from public documentation ahead of the open-source release.
- Relicensed the project from BSD-3-Clause to the MIT License, adding the Triad
  National Security, LLC / U.S. Government copyright notice for the LANL open-source
  release (LANL reference O5066). Updated `LICENSE`, `pyproject.toml`, `CITATION.cff`,
  the README license section, and the release-metadata gate accordingly.
- Narrowed supported Python metadata from Python 3.10+ to Python 3.10-3.12
  while the package depends on NumPy 1.x and RoadRunner 2.7.x.
- Documented the trusted-input boundary for Antimony and SBML model files.

### Fixed
- Removed placeholder preferred-citation article metadata from `CITATION.cff`.
- Corrected the `CITATION.cff` license identifier so CFF validation passes.
- Added local artifact smoke expectations for built wheel and sdist checks.

### Added
- Added local release notes and a release checklist for pre-tag validation.
- Added `CORRECTNESS_SPEC.md` to define the local correctness contract.

## [0.5.5] - 2025-12-31

### Release Policy
- Release maturity: alpha.
- Supported on Python 3.10, 3.11, and 3.12. Platform matrix automation is deferred until hosted project infrastructure exists.
- Input trust boundary: Antimony and SBML inputs are trusted local scientific model files, not safe untrusted uploads for multi-tenant or security-sensitive services.

### Added
- **BioModels database benchmark results**: 289 validated models from 978 candidates
  - 66 General → GMA transformations (functional lifting)
  - 18 GMA → S-system transformations (sum-to-product reduction)
  - 1 General → S-system (full simplification)
- Reorganized `biomodels_batch/` scripts to sequential `stepN_` naming
- Consolidated report generation into `step6_report.py`

### Changed
- Updated paper.md with benchmark results
- Improved filter summary output with clear funnel visualization
- Default benchmark pipeline now uses numerical-only validation (faster)

### Fixed
- Antimony reserved keyword sanitization for `compartment`, `DNA`, `RNA`, etc.
- Validation file pattern matching in report generation

## [0.5.4] - 2025-12-30

### Added
- **BioModels batch validation workflow** with separate recasting and validation phases
- New `3b_validate_batch.py` script for standalone validation of existing recasts
- Partial validation support in `validate_recast_pair()`:
  - `run_symbolic`, `run_numerical`, `run_trajectory` parameters
  - `--symbolic-only` and `--numerical-only` CLI flags
- JAX autodiff backend option (`--use-jax`) for numerical validation
- JAX to `biomodels_batch/requirements.txt`

### Changed
- Validator `overall_pass` logic: only requires tests that were actually requested
- Batch scripts updated to use `antimony` library instead of `tellurium`

### Fixed
- Signal-based timeout now properly catches validation hangs
- Fixed `overall_pass` returning `False` when partial validation passed all requested tests

### Notes
- JAX numerical validation is **slower** than NumPy for batch processing due to JIT compilation
  overhead. Use NumPy backend (default) for batch validation. See DEVELOPMENT_NOTES.md.

## [0.5.3] - 2025-12-28

### Added
- All test models now pass 3-check validation (symbolic, numerical, trajectory)
- Regression tests for:
  - Symbolic exponent handling in pool construction
  - Variable 'I' vs SymPy `sp.I` imaginary unit collision
  - EPS_INIT factor_map expansion
  - Symbolic exponent parenthesization
  - sqrt(sum) auxiliary IC computation
  - Rational coefficient preservation

### Changed
- Trajectory comparison error tolerance increased from 1.5% to 3%
- Code quality: Fixed 81 ruff lint issues (whitespace, import order)

### Fixed
- **TypeError in pool construction** when exponents are symbolic (e.g., `X^h` with parameter `h`)
- **EPS_INIT expansion bug**: Factor_map is now expanded before checking for negative exponents,
  preventing incorrect EPS_INIT assignment when negative exponents cancel after expansion
- **Variable 'I' collision**: Validator now handles models using 'I' as a variable name
  (common in SIR epidemic models) without colliding with SymPy's imaginary unit `sp.I`
- **Symbolic IC regression**: Only use symbolic IC expressions when they don't depend on state variables
- **sqrt(sum) auxiliary ICs**: Fixed symbol identity mismatch in lift_composite_functions
  that caused wrong ICs for sqrt(X^2 + c) patterns
- **Symbolic exponent formatting**: Add expressions like `x^(-C - 1)` now properly parenthesized
  to avoid parsing as `(x^-C) - 1`
- **Rational coefficients**: Preserved as fractions (e.g., `(2/3)`) instead of decimals in output

## [0.5.2] - 2025-12-28

### Added
- User-configurable `EPS_INIT` via `@SIM` comment metadata
  - Users can now specify `// @SIM EPS_INIT=1e-6` in Antimony files
  - This controls the epsilon value used for zero IC approximation in pool construction
- Compartment filtering from params to avoid duplicate output

### Changed
- Default `T_END` changed from 20.0 to 1.0 for faster simulation feedback
- Default `N_STEPS` changed from 400 to 100 for faster simulation feedback
- Default `EPS_INIT` changed from 1e-9 to 1e-6 for better numerical stability

### Fixed
- Fixed all mypy type errors (21 errors resolved)
  - Fixed `Reaction` dataclass attribute names in roadrunner_backend.py
  - Added type annotations for validator.py list variables
  - Fixed untyped lambda in notebook_helpers.py

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

Comparison links are omitted until public project URLs exist.
