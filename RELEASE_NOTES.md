# ssys 0.6.1 Release Notes
*Version: 0.6.1 | Release date: 2026-07-05*

## Release Status

Release maturity: alpha.

ssys 0.6.1 is supported on Python 3.10, 3.11, and 3.12. Python 3.13 is not advertised while ssys depends on the current NumPy 1.x and RoadRunner 2.7.x compatibility range. Platform matrix automation is deferred until hosted project infrastructure exists.

Input trust boundary: ssys treats Antimony and SBML inputs as trusted local scientific model files, not as safe untrusted uploads for multi-tenant or security-sensitive services.

## Validation Evidence

- A local release candidate must build both a source distribution and wheel from a clean worktree.
- Local artifact smoke checks must install the built wheel and sdist in clean environments, then run CLI recasting, validation-enabled recasting, and public API import checks.
- The source distribution includes `tests/` and `test_models1/` through `test_models4/`, covering the committed 117-model corpus fixtures.
- `CORRECTNESS_SPEC.md` defines the supported input classes, output classes, transformation postconditions, validation meaning, and current numerical limitations.
- Validation reports record the selected validation profile, include machine-readable reason codes for non-pass checks, and carry `schema_version`.
- Validation-report JSON Schema version `1.0` is packaged with the library and available through `ssys.load_validation_report_schema()`.
- Dependency and supply-chain review is local: `tools/check_dependency_risk.py` records lockfile status, exported requirements, environment metadata, and a `pip-audit` report under release evidence.
- Full release validation requires the DAE extra and `SSYS_REQUIRE_DAE_VALIDATION=1`.

## User-Visible Changes

- SBML parsing now interprets compartment volume, a species/model `conversionFactor`, and constant `stoichiometryMath` correctly, fixing silently wrong ODEs for models that use them (verified against libRoadRunner).
- The SBML parser fails closed with a structured `unsupported_feature` error on variable reaction stoichiometry and time-varying compartment volume, which are not power-law-recastable, instead of mis-integrating them.
- The deprecated legacy Antimony parser (`ssys.parse_antimony` / `--parser legacy`) now fails closed on the same constructs and emits a `DeprecationWarning`; the recast notebook no longer depends on it.
- Refreshed the BioModels benchmark: 848 transformations (86.7%) and 739 numerically validated models across 978 candidates.
