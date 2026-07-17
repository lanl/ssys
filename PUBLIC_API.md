# ssys Public API

This document defines the intended local public API for the v0.6.1 alpha
release. Anything not listed here may change without compatibility guarantees
until the package leaves alpha.

## Stability Policy

- `ssys` top-level imports are the preferred stable API for users.
- Focused public modules listed below are supported for advanced use.
- `ssys.recaster` and `ssys.validator` are compatibility re-export shims for
  older imports. They intentionally expose additional helper names, including
  some underscored implementation helpers, and should not be treated as the
  stable API surface.
- New deprecations must emit a `DeprecationWarning`, be documented here, and
  include a planned removal release or release condition.
- No compatibility shim is scheduled for removal before a stable release. The
  current alpha release may still refine the stable API before that point.

## Top-Level Package API

Import these from `ssys`:

### Data Types

- `SymSystem`
- `RecastResult`
- `SBMLParseError`
- `SSysEquation`
- `SolverRequirement`
- `SystemClass`
- `ValidationProfile`

### Parsing And Recasting

- `parse_antimony_via_sbml`
- `parse_sbml`
- `recast_to_ssystem`
- `ssystem_to_antimony`
- `canonicalize_aux_names`

### Classification And Validation Utilities

- `classify_result`
- `classify_solver_requirement`
- `validation_profile_choices`
- `load_validation_report_schema`
- `VALIDATION_REPORT_SCHEMA_RESOURCE`
- `VALIDATION_REPORT_SCHEMA_VERSION`

### Release Metadata

- `ssys.__version__`
- `ssys.__release_date__`
- `ssys.__release_maturity__`

## Focused Public Modules

### `ssys.parsing`

- `parse_antimony_via_sbml`
- `parse_sbml`
- `parse_sbml_from_string`
- `expand_antimony_function_templates`

### `ssys.recasting`

- `recast_to_ssystem`
- `canonicalize_aux_names`
- `term_to_coeff_exps`

### `ssys.formatting`

- `ssystem_to_antimony`
- `gma_to_antimony`
- `product_to_antimony`
- `latex_odes`
- `latex_ssys`

### `ssys.lifting`

- `AutonomousLiftResult`
- `add_dummy_for_constants`
- `find_composite_functions`
- `find_rational_denominators`
- `find_sqrt_of_sums`
- `lift_composite_functions`
- `lift_rational_functions`
- `lift_squared_for_sqrt`
- `lift_time_functions_to_autonomous`

### `ssys.validator`

Preferred stable imports:

- `RecastValidator`
- `validate_recast_pair`
- `validate_generated_output_roundtrip`
- `EquivalenceTest`
- `ValidationReport`
- `ValidationResult`
- `ValidationProfile`
- `ValidationProfileSpec`
- `validation_profile_choices`
- `load_validation_report_schema`
- `VALIDATION_REPORT_SCHEMA_VERSION`

Compatibility note: `ssys.validator` also re-exports selected internal helpers
for historical tests and notebooks. Prefer the names above in new user code.

### `ssys.ode_backends`

- `simulate_model`
- `simulate_ode`
- `simulate_dae`
- `simulate_dae_projection`

These functions expose solver-level behavior. Users should usually prefer
validation profiles unless they need direct backend diagnostics.

## Command-Line API

The stable CLI entry point is:

```bash
ssys-recast --manifest models.manifest --outdir out
```

Supported options:

- `--manifest PATH`: required manifest with one Antimony file path per line.
- `--outdir PATH`: required output directory.
- `--mode {simplified,canonical}`: output mode; default is `simplified`.
- `--validate`: write validation JSON for each recast.
- `--validation-profile {strict,structural,symbolic,numerical,trajectory}`:
  profile used with `--validate`; default is `strict`.
- `--allow-validation-failures`: keep batch behavior when validation reports do
  not pass; without this flag validation failures exit nonzero.
- `--version`: print package version.

CLI output contracts:

- Successful recasts write `*_recast.ant`.
- Validation runs write `*_validation.json` matching schema version `1.0`.
- Successful batches write `recast_report.ipynb`.
- Manifest, recast, validation, and notebook failures exit nonzero with stable
  stderr diagnostics.

## Deprecated Or Compatibility-Only Behavior

- The legacy hand-rolled Antimony parser (`ssys.parse_antimony`,
  `ssys.build_sym_system`) and the `ssys-recast --parser` flag were removed after
  shipping a `DeprecationWarning` in 0.6.1. The SBML-first parser
  (`ssys.parse_antimony_via_sbml`) is now the only Antimony parser; it is the
  default every code path already used.
- `ssys.recaster` broad re-exports are compatibility-only. New code should use
  top-level `ssys` imports or focused public modules.
- Legacy validation boolean flags are compatibility-only when a named validation
  profile is not supplied. New code should pass `profile=...`.
