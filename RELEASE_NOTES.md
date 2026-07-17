# ssys 0.7.0 Release Notes
*Version: 0.7.0 | Release date: 2026-07-16*

## Release Status

Release maturity: alpha.

ssys 0.7.0 is supported on Python 3.10, 3.11, and 3.12. Python 3.13 is not advertised while ssys depends on the current NumPy 1.x and RoadRunner 2.7.x compatibility range. Platform matrix automation is deferred until hosted project infrastructure exists.

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

- **Breaking:** Removed the deprecated hand-rolled legacy Antimony parser (`ssys.parse_antimony`, `ssys.build_sym_system`), the `ssys-recast --parser` flag, and the `recast_file(..., parser="legacy")` mode. The SBML-first parser `ssys.parse_antimony_via_sbml` — the default every code path already used — is now the only Antimony parser. Because public API symbols were removed, this ships as a minor version bump.
- **Breaking:** Removed the now-unused `ModelIR` and `Reaction` dataclasses (`ssys.ModelIR`, `ssys.types.ModelIR`/`Reaction`, and their `ssys.recaster` re-exports). `parse_antimony_via_sbml` returns a `SymSystem`, the only model type the ODE/DAE backends ever received at runtime; the backend signatures now annotate `SymSystem` and read its native `vars`/`odes`/`initials` directly.
- **Recasting no longer silently corrupts negative initial conditions (GH #6).** S-system pool construction represents each state as a product of strictly-positive power-law auxiliaries, so a negative initial value has no valid representation. The builder previously substituted `0` silently, starting the recast from the wrong point. `recast_to_ssystem` now fails closed with a new `NegativeInitialConditionError` (exported from the top-level `ssys` namespace) that names every offending state. A zero initial value on a degenerate `X' = 0` state is now preserved exactly instead of being promoted to `1.0`.
- Refreshed the BioModels benchmark: 731 numerically validated models across 978 candidates. This is eight fewer than 0.6.1 — those eight models carry negative initial states (e.g. membrane-voltage models) and are now correctly rejected by the GH #6 fail-closed guard rather than silently recast from a corrupted initial point.
