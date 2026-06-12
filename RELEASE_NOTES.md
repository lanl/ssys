# ssys 0.5.5 Release Notes
*Version: 0.5.5 | Release date: 2025-12-31*

## Release Status

Release maturity: alpha.

ssys 0.5.5 is supported on Python 3.10, 3.11, and 3.12. Python 3.13 is not advertised while ssys depends on the current NumPy 1.x and RoadRunner 2.7.x compatibility range. Platform matrix automation is deferred until hosted project infrastructure exists.

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

- Added BioModels benchmark reporting for 289 validated models from 978 candidates.
- Reorganized the BioModels batch workflow scripts into sequential `stepN_` commands.
- Consolidated BioModels report generation in `biomodels_batch/step6_report.py`.
- Fixed reserved-keyword sanitization and validation file pattern matching in benchmark reports.
- Added named validation profiles: `strict`, `structural`, `symbolic`, `numerical`, and `trajectory`.
