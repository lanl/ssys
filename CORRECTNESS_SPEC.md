# ssys Correctness Specification

*Source: CORRECTNESS_SPEC.md | v0.6.0 | 2026-07-04*

This document is the local correctness contract for the current `ssys`
implementation. It defines what the package claims to transform, what the
generated artifacts mean, what validation can prove, and which limitations are
outside the proven contract.

## Scope

`ssys` transforms deterministic finite-dimensional ODE models into S-system or
Generalized Mass Action (GMA) forms. The transformation is exact as an algebraic
change of variables on the invariant manifold defined by generated state
mappings and auxiliary definitions, subject to supported input classes,
consistent initial conditions, and the numerical limitations listed below.

This specification covers local source, wheel, and sdist artifacts. It does not
depend on hosted project infrastructure.

## Trust Boundary

Antimony and SBML inputs are trusted local scientific model files. They are not
safe untrusted uploads for multi-tenant or security-sensitive services. Parser
hardening rejects several malformed or ambiguous formulas, but security
sandboxing is outside this correctness contract.

## Core Objects

The implementation contract is expressed through these public and internal data
objects:

| Object | Contract |
| --- | --- |
| `ModelIR` | Parsed Antimony/SBML model: species, parameters, reactions, rate rules, assignment rules, algebraic constraints, compartments, initial values, simulation metadata, and solver requirement. |
| `SymSystem` | Symbolic ODE system over positive SymPy symbols with numeric parameters, initial conditions, optional assignment rules, algebraic constraints, and simulation metadata. |
| `RecastResult` | Generated recast model, including output status, S-system or GMA equations, generated state variables, initial values, `factor_map`, `auxiliary_defs`, assignment rules, algebraic constraints, solver requirement, and metadata. |
| `ValidationReport` | Fail-closed validation evidence: generated-artifact roundtrip, parser check, mapping check, symbolic check, numerical pointwise check, trajectory check, algebraic residual check, auxiliary identity checks, solver requirements, and overall verdict. |

## Supported Input Classes

An input model is inside the current proven contract when all of these are true:

- It is a deterministic continuous model expressible as reaction-derived ODEs,
  SBML rate rules, or explicit Antimony derivative rules.
- Floating species have finite numeric initial values after SBML
  `InitialAssignment` evaluation.
- Parameters, compartments, and local kinetic-law parameters have finite numeric
  values after supported initial-assignment evaluation.
- Mathematical expressions use identifiers known to the parser and functions in
  the current SBML formula allowlist: `Abs`, `Piecewise`, inverse trig,
  hyperbolic trig, `ceiling`, `cos`, `exp`, `floor`, `log`, `max`, `min`,
  `piecewise`, `pow`, `sin`, `sqrt`, `tan`, and `tanh`.
- The model is evaluated on a domain where every S-system base is positive.
  Users must translate variables with zero, negative, or sign-changing domains
  before recasting unless the generated model itself introduces a documented
  positive auxiliary.
- Assignment rules are algebraic definitions that can be parsed in terms of
  known state, parameter, compartment, assignment, or time symbols.
- Algebraic constraints, when present, are treated as DAE requirements rather
  than ordinary ODE equivalence.

## Unsupported Or Outside-Proof Input Classes

The following classes are outside the proven contract. They must be rejected
with structured diagnostics or treated as unsupported before a stable release
claim:

- Events, triggers, resets, delays, stochastic/hybrid semantics, and discrete
  state changes.
- Unknown formula identifiers or unknown functions.
- Initial assignments that cannot be evaluated to finite numeric values in the
  supported parameter/compartment/species context.
- Arbitrary untrusted text intended to exploit parser behavior.
- Piecewise or discontinuous dynamics beyond parse/roundtrip behavior unless
  future tests prove the intended semantics.
- Nonpositive or singular domains for power-law variables unless the user has
  supplied a valid positive-domain reformulation.
- Unsupported solver requirements, such as DAE-required validation when the
  optional IDA/SUNDIALS backend is unavailable.

Current tests already cover several structured parser diagnostics, but the
negative corpus task remains open until unsupported SBML/Antimony feature
fixtures are comprehensive.

## Output Classes

`recast_to_ssystem(sym, mode=...)` returns one of these output classes:

| Output class | Meaning |
| --- | --- |
| `CANONICAL_SSYSTEM` in `simplified` mode | Each equation is represented as production minus degradation. Zero coefficients and single-sided equations may remain. |
| `CANONICAL_SSYSTEM` in `canonical` mode | Formatting adds epsilon slack where needed so user-facing Antimony has strict two-term production/degradation form. |
| `GMA` | One or more equations require multiple production or degradation monomials with incompatible exponent patterns, or pool construction safety checks refused canonicalization. The GMA output preserves the flux terms instead of claiming canonical S-system form. |
| `GMA_TIME_VARYING` classification | GMA output with assignment rules that depend on the generated clock state `T`. |
| `FAILED` | Reserved for unrecoverable recasting failure. A failed result is not a valid recast. |

The CLI may call a model validated only when validation reports
`overall_pass: true`.

## State Mapping Contract

For original variables `X` and generated variables `Z`, the mapping `Phi` is
defined by:

- `factor_map`: original variables mapped to one generated variable or a product
  of generated pool variables.
- `auxiliary_defs`: generated auxiliary symbols mapped to their defining
  expressions in original variables, generated variables, parameters, or time.
- `assignment_rules`: generated or preserved algebraic definitions emitted with
  Antimony `:=`.
- `algebraic_constraints`: implicit equations that must be enforced by a DAE
  backend when coupled to differential states.

Correctness means that, on the manifold where all `factor_map`,
`auxiliary_defs`, assignment rules, and algebraic constraints hold with
consistent initial conditions, the generated dynamics reproduce the original
observable variables. The symbolic validation check attempts to prove
`J_Phi(Z) * f_recast(Z) = f_orig(Phi(Z))`. Numerical and trajectory checks are
supporting evidence, not exact proof.

## Transformation Rules

### Parser And Symbolic System Construction

Preconditions:

- Antimony text can be converted through the reference Antimony/SBML stack or
  parsed by the explicitly selected legacy parser.
- SBML formulas use known identifiers and supported function names.
- Initial assignments needed for parameters, compartments, or initial values can
  be evaluated numerically, unless exploratory warning mode is explicitly used.

Postconditions:

- Floating species become `SymSystem.vars`.
- Reaction stoichiometry and rate rules become `SymSystem.odes`.
- Assignment rules and algebraic rules are preserved separately from ODEs.
- Parameter, compartment, simulation, and solver metadata are propagated.
- SBML parsing avoids global simplification that would merge rational
  denominators and hide term structure needed by lifting.

Representative tests:

- `tests/test_recaster.py::TestSbmlParserIcHandling`
- `tests/test_recaster.py::test_same_named_local_parameters_are_scoped_by_reaction`
- `tests/test_recaster.py::test_unknown_formula_identifier_raises_structured_error`
- `tests/test_recaster.py::test_unknown_formula_function_raises_structured_error`
- `tests/test_recaster.py::test_malicious_formula_string_rejected_before_sympify`

### Rational Lifting

Preconditions:

- A denominator appears through a negative exponent or reciprocal form.
- Constant denominators can be substituted numerically.
- Simple symbol denominators are already power-law compatible.
- Nontrivial algebraic denominators depend on state variables and can be
  differentiated symbolically.

Postconditions:

- Each lifted denominator has an auxiliary definition in `auxiliary_defs`.
- ODEs are rewritten to use the auxiliary with a negative exponent.
- Auxiliary ODEs are generated by the chain rule.
- Initial conditions are computed from the defining expression and original
  initial values.

Representative tests:

- `tests/test_lifting.py::TestLiftRationalFunctions`
- `tests/test_lifting.py::test_create_auxiliary_for_denominator_uses_chain_rule`
- `tests/test_recaster.py::test_mm_to_gma_initial_conditions`
- `tests/test_validator.py::test_ode_auxiliary_lifted_denominator_matches_definition`

### Composite Lifting

Preconditions:

- A supported non-algebraic function application such as `exp`, `log`, `sin`,
  `cos`, `sqrt(sum)`, or `tanh` occurs in an ODE or assignment expression.
- The function is differentiable on the modeled domain.
- Required original initial values and parameter values are available.

Postconditions:

- State-dependent composites become generated auxiliaries with
  `auxiliary_defs`.
- Auxiliary ODEs are generated by the chain rule.
- Composite auxiliaries remain explicit variables; the implementation does not
  apply inverse mappings that would break the chain rule.
- Time-only composites may become assignment rules rather than state ODEs.

Representative tests:

- `tests/test_lifting.py::TestLiftCompositeFunctions`
- `tests/test_lifting.py::test_composite_inverse_mappings_cover_exp_log_and_nested_auxiliaries`
- `tests/test_recaster.py::TestSqrtSumIcComputation`
- `tests/test_recaster.py::test_composite_function_ic_from_params`
- `tests/test_validator.py::test_ode_auxiliary_composite_function_matches_definition`

### Trigonometric Lifting

Preconditions:

- Trigonometric functions are supported parser functions and are differentiable
  on the modeled interval.
- Coupled sine/cosine relationships have enough context to generate consistent
  ODEs or assignment rules.

Postconditions:

- State-dependent `sin(theta)` and `cos(theta)` are represented by coupled
  auxiliaries with chain-rule ODEs.
- Time-only trigonometric expressions may be assignment rules over clock state
  `T`.
- Auxiliary identity validation must prove or numerically support the generated
  trigonometric definitions.

Representative tests:

- `tests/test_lifting.py::test_lift_sin_cos_pair`
- `tests/test_lifting.py::test_autonomous_lift_helpers_cover_exp_harmonic_and_tanh`
- `tests/test_lifting.py::test_lift_time_functions_adds_clock_and_state_sqrt_auxiliary`
- `tests/test_validator.py::test_clock_auxiliary_matches_definition`

### Time Lifting

Preconditions:

- Time appears as `time` or `t` in supported ODE or assignment expressions.
- Time-dependent ODE semantics can be made autonomous with a generated clock.

Postconditions:

- A clock state `T` is introduced with `T' = 1` when needed.
- ODEs and assignment rules substitute `time`/`t` with `T`.
- Time-only functions are assignment rules when an ODE state would be
  unnecessary or numerically worse.
- Time-varying assignment rules may classify the output as `GMA_TIME_VARYING`.

Representative tests:

- `tests/test_lifting.py::TestLiftTimeFunctions`
- `tests/test_lifting.py::test_lift_clock_state`
- `tests/test_validator.py::TestTimeDependentValidation`
- `tests/test_formatting.py::test_classify_system_with_time_dependent_rule`

### Pool Construction

Preconditions:

- No rational/composite lifting is needed, or the system has no lifted
  auxiliaries requiring direct handling.
- Each ODE is a finite sum of monomial terms over state variables and
  parameters.
- Safety checks accept the term count, dimension expansion, product length, and
  exponent range.

Postconditions:

- Each original variable maps to a product of generated pool variables.
- Each ODE term gets a generated pool variable.
- Pool ODEs are production/degradation power laws derived from the original
  term exponents.
- Initial conditions satisfy the product mapping, except where `EPS_INIT`
  approximates zero for variables that would otherwise appear with negative
  exponents.
- Generated auxiliary names are canonicalized to `Z_1`, `Z_2`, ...

Representative tests:

- `tests/test_recaster.py::test_recast_exponential_decay`
- `tests/test_recaster.py::test_recast_two_term_ode`
- `tests/test_recaster.py::test_canonical_naming`
- `tests/test_recaster.py::TestEpsInitFactorMapExpansion`
- `tests/test_recaster.py::TestSymbolicExponents`

### GMA Fallback

Preconditions:

- Direct or pool S-system construction would require multiple production or
  degradation terms with incompatible exponent patterns, or pool safety checks
  refuse the transformation.

Postconditions:

- Output status is `GMA`.
- All growth and decay terms are preserved as GMA terms.
- The model is not reported as canonical S-system.
- `canonical_refusal_reason` records the reason when pool construction was
  refused or rejected.

Representative tests:

- `tests/test_edge_cases.py::test_classify_gma`
- `tests/test_formatting.py::test_classify_result_gma`
- `tests/test_validator.py::test_validator_with_refusal`
- `tests/test_integration.py::test_recast_all_models`

### Epsilon Slack

Preconditions:

- User selects `mode="canonical"` or generated Antimony needs strict two-term
  canonical presentation.
- One side of an equation would otherwise have zero coefficient.

Postconditions:

- Formatter emits `epsilon` slack terms using `EPS_SLACK` metadata when present,
  otherwise the package default.
- The added production/degradation slack terms cancel algebraically in the
  observable ODE.
- `EPS_SLACK` is recorded in simulation metadata when user-specified.

Representative tests:

- `tests/test_recaster.py::test_eps_slack_in_canonical_output`
- `tests/test_recaster.py::test_eps_slack_default_value_in_canonical`
- `tests/test_recaster.py::test_eps_slack_propagation_through_recast`
- `tests/test_formatting.py::TestSIMMetadataEmission`

### DAE And Assignment Handling

Preconditions:

- Assignment rules or algebraic constraints exist in the original or generated
  model, or lifted auxiliaries must remain on a state-dependent manifold.

Postconditions:

- `SolverRequirement.ODE_ONLY` is used only when ordinary ODE simulation is
  sufficient.
- `SolverRequirement.ODE_WITH_ASSIGNMENT_RULES` is used when an ODE backend that
  honors assignment rules is sufficient.
- `SolverRequirement.DAE_REQUIRED` is used when algebraic constraints or
  state-dependent auxiliary definitions must be enforced.
- Missing optional DAE dependencies produce `unsupported` validation results,
  not pass results.
- IDA/SUNDIALS residuals use differential equations, explicit assignment
  auxiliary equations, and implicit algebraic constraints.

Representative tests:

- `tests/test_formatting.py::TestSolverRequirementClassification`
- `tests/test_validator.py::TestSolverAwareValidation`
- `tests/test_ode_backends.py::test_dae_required_without_ida_dependency_fails_unsupported`
- `tests/test_ode_backends.py::test_ida_backend_enforces_explicit_assignment_auxiliary`
- `tests/test_ode_backends.py::test_ida_backend_rejects_inconsistent_user_algebraic_ic`

## Validation Contract

Validation is fail-closed:

- `overall_pass` is true only when every required test for the selected
  validation profile passes.
- Passing means `ValidationResult.PASS`, not unsupported, skipped, timeout, or
  inconclusive.
- Parser failures and generated-output roundtrip failures are required checks.
- Missing DAE support is `unsupported`.
- Solver failures cannot be counted as a pass.
- Reports record the selected profile, required test groups, and a
  machine-readable reason for every non-pass test result.
- Reports include `schema_version`; v0.6.0 emits schema version `1.0`.
- The packaged JSON Schema is available through
  `ssys.load_validation_report_schema()` and is validated against emitted
  pass/failure reports in local tests.
- Incompatible report-format changes must increment `schema_version`; additive
  stable fields must update the packaged schema and schema-validation tests.
- Profile-excluded checks are reported as `not_attempted` with reason
  `profile_excluded`, but they are not required for that profile's
  `overall_pass`.

Named validation profiles:

| Profile | Required checks |
| --- | --- |
| `strict` | Generated-output roundtrip, parser, mapping, symbolic, numerical, trajectory, algebraic residuals, auxiliary identities. |
| `structural` | Generated-output roundtrip, parser, mapping. |
| `symbolic` | Generated-output roundtrip, parser, mapping, symbolic, auxiliary identities. |
| `numerical` | Generated-output roundtrip, parser, mapping, numerical, auxiliary identities. |
| `trajectory` | Generated-output roundtrip, parser, mapping, trajectory, algebraic residuals, auxiliary identities. |

The CLI only describes a model as validated when `--validate` uses the `strict`
profile and `overall_pass` is true. Partial profiles are diagnostic evidence,
not release-grade validated claims.

Interpretation:

- Generated-output roundtrip, parser, and mapping checks are structural
  contracts. They prove artifact usability and observable reconstruction
  coverage, not mathematical equivalence by themselves.
- Symbolic checks are exact algebraic proof attempts over the supported
  expression class. A symbolic `pass` proves the chain-rule identity simplified
  to zero; a symbolic `failed`, `inconclusive`, or `not_attempted` result is not
  a pass.
- Numerical checks are sampled support over recorded domains and parameters.
  They use deterministic log-uniform sampling over positive domains, expand
  state ranges from finite positive model initial values, use simulation time
  metadata when present, and record the seed, sampled ranges, and parameter
  values. Invalid domains, non-finite sampled values, and singular surfaces are
  non-pass diagnostics. They can expose counterexamples but do not prove global
  equivalence.
- Trajectory checks are solver-backed behavioral support over the reported time
  grid and tolerances. Reports include absolute, relative, and peak-scaled
  errors, scaling method, worst variable/time, solver backends, solver
  tolerances, output-step diagnostics, and interpolation status. Missing
  required solver support is `unsupported`.
- Auxiliary identity and algebraic residual checks enforce generated variables,
  assignment rules, and DAE/constraint semantics needed to reconstruct original
  observables.

## Known Approximations And Numerical Limits

### `EPS_INIT`

S-system bases must remain positive when they appear with negative or
non-integer exponents. Pool construction preserves exact zero initial
conditions unless a generated variable appears with a negative exponent after
expanding the final `factor_map`. In that case, zero is replaced by `EPS_INIT`
to avoid division by zero. This is an approximation and is recorded in output
metadata when used.

Tests:

- `tests/test_recaster.py::test_eps_init_used_in_pool_construction`
- `tests/test_recaster.py::test_canceling_negative_exponents_get_zero_ic`
- `tests/test_recaster.py::test_true_negative_exponent_gets_eps_init`

### Trajectory Tolerance

Trajectory validation is numerical support, not proof. The default comparison
threshold is 3 percent peak-scaled error. This is a support threshold for
solver-backed behavior across locally supported models, not a mathematical
equivalence claim; exact claims require symbolic validation. Reports include
absolute, relative, and scaled max/mean errors, scaling method, worst
variable/time, solver backends, solver tolerances, output-step diagnostics,
interpolation status, and algebraic residual metadata when applicable.

Tests:

- `tests/test_validator.py::test_trajectory_comparison_fails_on_divergent_recast`
- `tests/test_validator.py::test_trajectory_comparison_interpolates_recast_time_grid`
- `tests/test_validator.py::test_simulate_model_failure_preserves_backend_metadata`

### Algebraic Manifold Drift

Lifted auxiliaries define a constrained manifold. ODE-mode integration can drift
from that manifold. Assignment-rule or DAE handling is required when algebraic
definitions must be enforced exactly during trajectory validation.

Tests:

- `tests/test_validator.py::test_algebraic_residual_detects_ode_mode_drift`
- `tests/test_validator.py::test_assignment_rule_auxiliary_residual_is_enforced`
- `tests/test_validator.py::test_algebraic_manifold_check_reports_pass_and_failure`

## Release Criteria Tied To This Spec

Before a stable local release claim:

- Every transformation rule above must have behavior tests, not only execution
  tests.
- `tests/test_theorem_transformations.py` contains theorem-style analytic tests
  that prove observable equivalence and auxiliary chain-rule identities for
  pool construction, rational lifting, composite lifting, trigonometric/GMA
  fallback, time lifting, `EPS_INIT`, symbolic exponents, constant terms,
  assignment rules, and canonical epsilon slack.
- Unsupported input classes must have negative fixtures and structured
  diagnostics.
- Validation profiles must remain named, documented, and fail-closed.
- The release artifact smoke and full model validation gates in
  `RELEASE_CHECKLIST.md` must pass from fresh artifacts.
