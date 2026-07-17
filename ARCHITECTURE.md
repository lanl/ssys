# ssys Architecture

This document maps the local codebase for maintainers and release reviewers. It
describes current implementation boundaries, not a future hosted service.

## Top-Level Flow

1. Input Antimony or SBML is parsed directly into `SymSystem`.
2. `SymSystem` stores symbolic state variables, parameters, ODEs, initial
   values, assignment rules, algebraic constraints, compartments, simulation
   metadata, and solver requirements.
3. The recaster transforms `SymSystem` into `RecastResult`.
4. Formatting writes `RecastResult` back to Antimony with mapping comments,
   auxiliary definitions, solver metadata, simulation metadata, and generated
   equations.
5. Validation reloads original and generated artifacts, checks structural and
   mathematical contracts, and writes a versioned JSON report.
6. CLI workflows batch the same operations over manifest entries and generate a
   notebook for local review.

## Parser Flow

Primary parser path:

1. `src/ssys/_recaster/parsing.py::parse_antimony_via_sbml()` sends Antimony
   text through the reference Antimony implementation and SBML.
2. SBML is read with libSBML.
3. `_parse_sbml_document()` extracts species, parameters, compartments,
   reactions, rate rules, assignment rules, initial assignments, algebraic
   constraints, and solver requirements.
4. SBML math is converted to SymPy expressions using explicit parser helpers
   and checked for unsupported or ambiguous features.
5. `_parse_sbml_document()` assembles the extracted data into the internal
   `SymSystem` symbolic representation, ready for recasting.

The SBML-first path is the only Antimony parser. The former hand-rolled
`parse_antimony()`/`build_sym_system()` compatibility parser and the
`--parser` flag were removed (see CHANGELOG); every code path uses the
SBML-first parser.

Unsupported input ownership:

- Events, delays, SBML constraints, unknown functions, invalid identifiers,
  missing formulas, ambiguous rate rules, and malformed Antimony are parser
  trust-boundary issues.
- Parser failures should raise structured diagnostics before any successful
  recast artifact is generated.

## Internal Representation

Core data types live in `src/ssys/types.py`.

- `SymSystem`: symbolic ODE model used by recasting and validation; the single
  model type produced by the parser and consumed by the ODE/DAE backends.
- `RecastResult`: generated recast model plus mapping and solver metadata.
- `SystemClass`: user-facing model classification.
- `SolverRequirement`: numerical backend class required for simulation or
  validation.

Public compatibility modules:

- `ssys.recaster` re-exports recasting APIs from focused internal modules.
- `ssys.validator` re-exports validation APIs from focused internal modules.
- `ssys.parsing`, `ssys.formatting`, `ssys.lifting`, and `ssys.recasting`
  provide compatibility import paths for user code.
- `PUBLIC_API.md` defines which of these imports are stable for user code and
  which broad re-export modules are compatibility-only.

## Recasting Pipeline

Main implementation files:

- `src/ssys/_recaster/algorithms.py`: high-level recasting algorithms and
  canonicalization.
- `src/ssys/_recaster/lifting.py`: rational, composite, trigonometric, and
  time-dependent lifting.
- `src/ssys/_recaster/antimony_formatting.py`: generated Antimony output.
- `src/ssys/classification.py`: output class and solver requirement
  classification.

Pipeline stages:

1. Classify the source ODEs and determine whether direct S-system, GMA, or
   lifting is required.
2. Lift rational denominators into auxiliary variables when needed.
3. Lift composite functions such as `exp`, `log`, and trigonometric functions
   with chain-rule ODEs or assignment rules.
4. Lift explicit time dependence with a generated clock state when needed.
5. Construct pools and power-law terms.
6. Fall back to GMA when strict S-system construction is not valid.
7. Canonicalize generated auxiliary names.
8. Emit factor maps, auxiliary definitions, solver requirements, assignment
   rules, algebraic constraints, initial values, and simulation metadata.

Key correctness contract:

- The mapping from recast variables back to original observables is `Phi`.
- Auxiliary definitions must remain auditable through comments, assignment
  rules, or algebraic constraints.
- DAE-required outputs must not be silently treated as ordinary ODE validation
  passes.

## Validation Pipeline

Main implementation files:

- `src/ssys/_validator/core.py`: orchestration and report assembly.
- `src/ssys/_validator/report.py`: validation result dataclasses, profiles,
  reason codes, and JSON serialization.
- `src/ssys/_validator/serialization.py`: generated-output roundtrip checks.
- `src/ssys/_validator/mapping.py`: mapping and auxiliary extraction checks.
- `src/ssys/_validator/symbolic.py`: exact symbolic equivalence checks.
- `src/ssys/_validator/numerical.py`: pointwise numerical checks.
- `src/ssys/_validator/trajectory.py`: trajectory and algebraic residual
  checks.
- `src/ssys/validation_schema.py` and `src/ssys/schemas/`: packaged JSON
  Schema for report automation.

Validation profiles:

- `strict`: release-grade local validation; all required check groups must pass.
- `structural`: generated-output roundtrip, parser, and mapping checks only.
- `symbolic`: structural checks plus symbolic and auxiliary identities.
- `numerical`: structural checks plus pointwise numerical and auxiliary checks.
- `trajectory`: structural checks plus solver-backed trajectory, algebraic
  residual, and auxiliary checks.

Fail-closed rule:

- `overall_pass` is true only when every required check for the selected profile
  returns `ValidationResult.PASS`.
- Unsupported, failed, timeout, inconclusive, and not-attempted required checks
  are not passes.
- Non-required profile-excluded checks are serialized as `not_attempted` with
  reason `profile_excluded`.

## Solver Selection

Solver selection is based on `SolverRequirement`.

- `ode_only`: libRoadRunner/CVODE trajectory simulation.
- `ode_with_assignment_rules`: libRoadRunner/CVODE with assignment-rule
  semantics.
- `dae_required`: optional IDA/SUNDIALS backend by default.
- Projection backend: explicit diagnostic fallback for selected DAE-like checks,
  not release-grade proof.

Main implementation files:

- `src/ssys/ode_backends/interface.py`: backend dispatch.
- `src/ssys/ode_backends/roadrunner_backend.py`: libRoadRunner/CVODE ODE
  simulation.
- `src/ssys/ode_backends/ida_sundials_backend.py`: optional IDA/SUNDIALS DAE
  simulation.
- `src/ssys/ode_backends/dae_backend.py`: projection backend.

Missing optional DAE dependencies must produce `unsupported` validation
results, never a pass.

## Artifact Generation

CLI ownership:

- `src/ssys/cli.py` reads manifests, invokes recasting, optionally writes
  validation JSON, and writes `recast_report.ipynb`.
- CLI failures should exit nonzero with stable stderr diagnostics, not raw
  tracebacks.
- `tests/test_cli.py` owns user-visible CLI contract tests.

Generated artifacts:

- `*_recast.ant`: generated Antimony model.
- `*_validation.json`: versioned validation report matching
  `validation-report-v1.schema.json`.
- `recast_report.ipynb`: local notebook review artifact.
- `release-evidence/`: local release evidence for artifact smoke, dependency
  risk, coverage, validation, and benchmark logs.

Release helper scripts:

- `tools/local_artifact_smoke.py`: builds wheel/sdist, installs artifacts, runs
  CLI/API smokes, records freezes and hashes.
- `tools/check_critical_coverage.py`: enforces critical module coverage.
- `tools/check_release_metadata.py`: verifies local release metadata
  consistency.
- `tools/check_dependency_risk.py`: records dependency evidence and runs
  `pip-audit`.

## Bug Triage Map

- Antimony/SBML parse error, unsupported input, unknown symbol, malformed math:
  `src/ssys/_recaster/parsing.py`, `tests/test_negative_corpus.py`.
- Incorrect generated equations, missing auxiliary, bad GMA/S-system
  classification: `src/ssys/_recaster/algorithms.py`,
  `src/ssys/_recaster/lifting.py`, `tests/test_theorem_transformations.py`,
  `tests/test_recaster.py`, `tests/test_lifting.py`.
- Bad generated Antimony semantics: `src/ssys/_recaster/antimony_formatting.py`,
  `tests/test_formatting.py`, `tests/test_golden_generated_artifacts.py`.
- Validation report shape, reason codes, or profile behavior:
  `src/ssys/_validator/report.py`, `tests/test_validation_report_schema.py`,
  `tests/test_validator.py`.
- Symbolic/numerical/trajectory validation mismatch:
  `src/ssys/_validator/symbolic.py`, `src/ssys/_validator/numerical.py`,
  `src/ssys/_validator/trajectory.py`, `tests/test_validator.py`.
- ODE or DAE backend behavior:
  `src/ssys/ode_backends/`, `tests/test_ode_backends.py`.
- CLI exit code, output files, manifests, notebook generation:
  `src/ssys/cli.py`, `tests/test_cli.py`.
- Packaging, release metadata, and local evidence:
  `pyproject.toml`, `RELEASE_CHECKLIST.md`, `tools/`,
  `tests/test_release_metadata.py`, `tests/test_local_artifact_smoke.py`.
