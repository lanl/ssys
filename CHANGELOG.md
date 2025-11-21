# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.2.0...main
[0.2.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://lisdi-git.lanl.gov/hlavacek/ssys/-/tags/v0.1.0
