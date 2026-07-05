# ODE → GMA or S‑System Recast (Antimony → Antimony, with SBML library support)
*Source: README.md | v0.6.0 | 2026-07-04*

This toolkit converts ordinary differential equation (ODE) models into **S‑System** or **GMA** form and writes the result back to Antimony. The command-line interface batch-processes **Antimony** models and generates **Jupyter notebook** verification reports; the Python library can also parse SBML files directly.

**Release maturity:** ssys is alpha software. ssys supports Python 3.10, 3.11, and 3.12. Python 3.13 and NumPy 2.x are not advertised for this release because the RoadRunner-backed validation stack currently requires NumPy 1.x. Treat the APIs, generated Antimony details, and validation-report format as subject to change until the release gates in `RELEASE_CHECKLIST.md` are closed.

**Scope:** current release work is local-first. Local artifact builds, local validation reports, local benchmark evidence, and local release-evidence directories are the source of truth. Hosted documentation, hosted CI, public issue links, and release uploads are deferred until public project infrastructure exists.

**Trust boundary:** ssys treats Antimony and SBML inputs as trusted local scientific model files, not as safe untrusted uploads. Do not expose the CLI or parser directly to arbitrary user-submitted model text in a multi-tenant or security-sensitive service.

---

## Contents

```
ssys/
  src/ssys/
    __init__.py           # Package interface
    cli.py                # Command-line interface
    recaster.py           # Core library: parse → ODE → recast → Antimony
    validator.py          # Mathematical correctness validation
    notebook_helpers.py   # Jupyter notebook generation utilities
    ode_backends/         # ODE solver backend (RoadRunner/CVODE)
  test_models1/           # 29 integration testing models
  test_models2/           # 28 models from Savageau & Voit (1987)
  test_models3/           # 40 models with published recastings
  test_models4/           # 20 systems biology models
  tests/                  # Unit tests
  PUBLIC_API.md           # Stable public API and compatibility policy
  CORRECTNESS_SPEC.md     # Supported correctness contract and validation limits
  PARSER_TRUST_BOUNDARY.md # Parser threat model and trusted-input audit
  ARCHITECTURE.md         # Local architecture and ownership map
  RECASTING.md            # Recasting theory and rules
  TEST_MODELS.md          # Test model collection documentation
  README.md               # This file
```

---

## Installation

Use [uv](https://astral.sh/uv) for local development and release checks.

### Local Checkout

Create a local environment from the checkout:

```bash
uv sync --python 3.12 --extra dev
uv run ssys-recast --help
```

Run a validation-enabled recast with the strict profile:

```bash
uv run ssys-recast --manifest test_models1/models.manifest \
  --outdir out_test_models1 \
  --mode simplified \
  --validate \
  --validation-profile strict
```

`strict` is the release-grade local validation profile. A model is not counted
as validated when a required check is `failed`, `unsupported`,
`not_attempted`, `timeout`, or `inconclusive`.

### Named Development Environment

The repository also keeps a convenience script for a named `ssys_dev`
environment:

```bash
./setup_dev_env.sh
source ssys_dev/bin/activate
```

That script installs the `dev` extra. Use `uv sync` directly when you need
optional extras such as `dae` or `jax`.

### Optional Extras

```bash
# Development tools: pytest, ruff, mypy, jsonschema, Jupyter/IPython
uv sync --extra dev

# DAE trajectory validation through scikit-sundae/SUNDIALS IDA
uv sync --extra dev --extra dae

# Optional JAX numerical diagnostics
uv sync --extra dev --extra jax
```

The base install includes the SBML-first parser and ODE trajectory dependencies:
libRoadRunner, Antimony, and python-libsbml. The `dae` extra is required for
DAE-required trajectory validation; missing DAE dependencies are reported as
`unsupported`, not as validation passes. The `jax` extra is capped below
versions that require NumPy 2.x and is optional diagnostic acceleration, not a
release-critical dependency.

---

## Requirements And Limits

- Python 3.10, 3.11, and 3.12
- Supported platforms: Linux and macOS (Windows is not tested or supported)
- sympy ≥1.12 (symbolic math)
- numpy ≥1.24,<2 (RoadRunner 2.7.x compatibility)
- scipy ≥1.10
- matplotlib ≥3.7 (plots in generated notebooks)
- nbformat ≥5.9 (notebook generation)
- **libroadrunner ≥2.5,<2.8** (ODE simulation, SBML)
- **antimony ≥2.13** (Antimony parsing via SBML)
- **python-libsbml ≥5.20** (SBML model representation)

## Dependency And Supply-Chain Checks

Local release review should record dependency evidence with:

```bash
python tools/check_dependency_risk.py \
  --evidence-dir release-evidence/dependency-risk
```

The command verifies that `uv.lock` is up to date, exports all dependency
extras to `requirements-all-extras.txt`, records Python/platform metadata, and
runs `pip-audit` to write a local vulnerability report. Use
`--skip-pip-audit` only for offline development checks; it is not sufficient
for release evidence.

After the local gates have written their evidence directories, create a hashed
manifest for the release-candidate evidence tree:

```bash
uv run python tools/archive_release_evidence.py \
  --evidence-dir release-evidence \
  --require artifact-smoke \
  --require dependency-risk \
  --require biomodels \
  --require performance
```

The manifest records file sizes and SHA-256 hashes for local evidence files and
artifacts under `dist/`. The `release-evidence/` directory is ignored by git and
is intended for local release-candidate records, not source control.

## Input Trust Boundary

ssys treats Antimony and SBML inputs as trusted scientific model files, not as safe untrusted uploads. Parser hardening rejects malformed symbolic expressions where practical, but the supported threat model is trusted research inputs from local files.

See [PARSER_TRUST_BOUNDARY.md](PARSER_TRUST_BOUNDARY.md) for the local parser
audit and parser-mode decision.

---

## Command-Line Interface

The CLI tool batch-recasts models and generates verification notebooks:

```bash
ssys-recast --manifest test_models1/models.manifest \
            --outdir out_test_models1 \
            --mode simplified \
            --validate
```

### Arguments

- `--manifest`: Path to manifest file (one Antimony `.ant` file path per line)
- `--outdir`: Output directory for recast models and notebook
- `--mode`: Output mode (default: `simplified`)
  - `simplified`: Flexible S-system form, preserves zeros
  - `canonical`: Strict 2-term form with epsilon slack variables
- `--parser`: Antimony parser to use (default: `sbml`)
  - `sbml`: Parses Antimony by converting through SBML with the reference Antimony implementation
  - `legacy`: Hand-rolled parser (deprecated)
- `--validate`: Run mathematical correctness validation on each recast
- `--validation-profile`: Named validation profile to run with `--validate`
  (default: `strict`)

### Manifest Format

Plain text file with one Antimony file path per line:
```
m01_exp_decay.ant
m02_logistic.ant
# Lines starting with # are ignored
```

Raw SBML files are supported through the Python API (`ssys.parse_sbml(...)`) and the BioModels batch workflow, not through `ssys-recast --manifest`.

### Output

For each input model:
- Recast `.ant` file in `--outdir`
- `*_validation.json` with validation results (if `--validate`)
- Verification notebook: `recast_report.ipynb`

---

## Validation

Validation reports are fail-closed: `overall_pass` is true only when every
required check for the selected profile returns `pass`. Unsupported,
not-attempted, timed-out, inconclusive, and failed required checks are not valid
passes. JSON reports include a machine-readable `reason` for every non-pass
test result, record the selected validation profile, and include
`schema_version`.

Validation-report JSON is covered by a packaged JSON Schema. The current schema
version is `1.0` and can be loaded from Python with:

```python
import ssys

schema = ssys.load_validation_report_schema()
```

Incompatible report-format changes must increment `schema_version`. Additive
stable fields must update the packaged schema and local schema-validation tests.

### Profiles

| Profile | Intended use | Required checks |
| --- | --- | --- |
| `strict` | Release-grade local validation and the default for `--validate`. The CLI only reports models as "validated" for this profile. | Generated-output roundtrip, parser, mapping, symbolic, numerical, trajectory, algebraic residuals, auxiliary identities |
| `structural` | Fast artifact or parser smoke where solver-backed evidence is not needed. | Generated-output roundtrip, parser, mapping |
| `symbolic` | Exact symbolic proof work without numerical simulation. | Generated-output roundtrip, parser, mapping, symbolic, auxiliary identities |
| `numerical` | Pointwise numerical support without trajectories. | Generated-output roundtrip, parser, mapping, numerical, auxiliary identities |
| `trajectory` | Solver-backed trajectory support. | Generated-output roundtrip, parser, mapping, trajectory, algebraic residuals, auxiliary identities |

### Test Families

1. **Generated-output roundtrip**: Parses the emitted Antimony/SBML artifact and
   rejects invalid generated files.

2. **Parser and mapping checks**: Confirm that original and recast models parse
   and that every original observable has a reconstruction mapping.

3. **Symbolic Test**: Proves exact equivalence using the Jacobian chain rule. Tests whether `J_Φ(Z) · f_recast(Z) = f_orig(Φ(Z))` simplifies to zero symbolically.

4. **Numerical Test**: Validates equivalence at 1000 deterministic random
   sample points with ε = 10⁻⁵ threshold. Sampling is log-uniform over positive
   domains, expands state ranges from finite positive model initial values, uses
   `@SIM` time metadata when present, and records the seed, sampled ranges, and
   parameter values in the validation report.

5. **Trajectory Test**: Simulates both original and recast models, compares
   trajectories with a 3.0% peak-scaled error threshold, and reports algebraic
   residual evidence when constraints or lifted auxiliaries require it. Reports
   include absolute, relative, and scaled errors, the scaling method, worst
   variable/time, solver backends, solver tolerances, output-step diagnostics,
   and whether recast trajectories were interpolated to the original time grid.

6. **Auxiliary identity checks**: Verify generated auxiliary definitions, observable
   assignment rules, and algebraic residuals needed to reconstruct original variables.

### Interpretation

- `symbolic` evidence is an exact algebraic proof when SymPy simplification can
  reduce the chain-rule identity to zero. A symbolic failure is a failed proof,
  not a numerical tolerance issue.
- `numerical` evidence is pointwise support over sampled domains. It can find
  counterexamples but does not prove global equivalence. Invalid domains,
  non-finite sampled values, and singular surfaces are non-pass diagnostics.
- `trajectory` evidence is solver-backed behavioral support over a time grid. It
  depends on solver availability, solver tolerances, model metadata, and the
  documented trajectory threshold. The default 3.0% threshold is a support
  threshold for solver-backed behavior, not proof; exact claims require
  symbolic validation.
- `structural` evidence proves generated-output roundtrip, parser, and mapping
  contracts only. It is useful for smoke tests, not for a validated scientific
  claim.
- Auxiliary identity and algebraic residual checks protect the generated
  variables and assignment rules that connect recast state back to original
  observables.

Failed report excerpt:

```json
{
  "schema_version": "1.0",
  "validation_profile": {"name": "strict"},
  "overall_pass": false,
  "overall_result": "failed",
  "summary": "Validation FAILED: at least one required check failed",
  "tests": {
    "symbolic": {
      "name": "symbolic_equivalence",
      "result": "failed",
      "reason": "failed",
      "details": "Symbolic expressions differ",
      "counterexamples": [{"variable": "X"}],
      "metadata": {}
    }
  }
}
```

Unsupported report excerpt:

```json
{
  "schema_version": "1.0",
  "validation_profile": {"name": "strict"},
  "overall_pass": false,
  "overall_result": "unsupported",
  "summary": "Validation UNSUPPORTED: a required backend is unavailable",
  "tests": {
    "trajectory": {
      "name": "trajectory_comparison",
      "result": "unsupported",
      "reason": "unsupported",
      "details": "libRoadRunner/CVODE or IDA/SUNDIALS backend unavailable",
      "counterexamples": [],
      "metadata": {"required": true}
    }
  }
}
```

**Example output**:
```
Test        Result   Max Error
symbolic    ✓ pass   N/A
numerical   ✓ pass   6.34e-16
trajectory  ✓ pass   1.17e-02
Overall: ✓ PASS
```

---

## Running Test Models

### Quick Start

```bash
# Run test_models1 (29 core models)
python recast_models.py test_models1

# Run a specific mode; this helper always runs validation
python recast_models.py test_models1 --mode canonical
```

### Helper Script Usage

The `recast_models.py` script simplifies batch processing:

```bash
python recast_models.py <directory> [options]
```

Options:
- `--mode {simplified,canonical}`: Output mode (default: simplified)
- `--both`: Run both simplified and canonical modes
- `--outdir DIR`: Output directory (default: `out_<input_dir>`)
- `--parser {sbml,legacy}`: Antimony parser (default: `sbml`)

### Test Model Sets

| Directory | Description | Count |
|-----------|-------------|-------|
| `test_models1/` | Integration testing models | 29 |
| `test_models2/` | Savageau & Voit (1987) examples | 28 |
| `test_models3/` | Models with published recastings | 40 |
| `test_models4/` | Systems biology models | 20 |
| **Total** | | **117** |

See [TEST_MODELS.md](TEST_MODELS.md) for complete model documentation,
[RECASTING.md](RECASTING.md) for recasting theory, and
[CORRECTNESS_SPEC.md](CORRECTNESS_SPEC.md) for the supported correctness contract.
Public API stability and compatibility shims are documented in
[PUBLIC_API.md](PUBLIC_API.md).

### pytest Integration Tests

The full integration test suite validates all 117 models in both modes:

```bash
# Run full integration suite (~3 min; requires the DAE extra)
uv sync --extra dev --extra dae
SSYS_REQUIRE_DAE_VALIDATION=1 pytest tests/test_integration.py -m slow -v

# Run backend cross-checks for representative ODE/DAE fixtures
SSYS_REQUIRE_DAE_VALIDATION=1 pytest tests/test_solver_cross_checks.py -v

# Skip slow integration tests during rapid development
pytest -m "not slow"

# Run a specific model directory
pytest tests/test_integration.py -k "test_models1"
```

### Source Distribution Tests

The source distribution includes `tests/` and the four committed model fixture directories
(`test_models1/` through `test_models4/`). From a freshly unpacked sdist, run the fast
artifact smoke test with:

```bash
python -m pip install . pytest
python -m pytest -o addopts= tests/test_integration.py -m "integration and not slow" -v
```

To run the full 117-model validation suite from the unpacked sdist, install the DAE extra
and use the slow integration command above.

### Local Artifact Smoke

To build local release artifacts and verify both installed distribution formats:

```bash
python tools/local_artifact_smoke.py --all-supported-pythons \
  --evidence-dir release-evidence/artifact-smoke
```

The command enforces a clean git tree by default, builds wheel and sdist artifacts, installs
each artifact in clean virtual environments, runs `ssys-recast --version`, a tiny recast,
a validation-enabled recast, a public API import smoke, and the unpacked-sdist fast test.
It writes logs, dependency freezes, artifact hashes, and `summary.json` under the evidence
directory.

### BioModels Benchmark Evidence

The BioModels benchmark has a local evidence wrapper around
`biomodels_batch/run_benchmark.sh`. For release-candidate evidence, use the
wheel built by the local artifact smoke gate so the benchmark imports from the
installed artifact rather than the source checkout. Replace the artifact path
with the wheel path recorded in `release-evidence/artifact-smoke/summary.json`:

```bash
uv run python tools/run_biomodels_benchmark.py \
  --artifact release-evidence/artifact-smoke/dist/ssys-0.6.0-py3-none-any.whl \
  --evidence-dir release-evidence/biomodels \
  --from-stage filter \
  --force \
  --min-candidates 900 \
  --min-recasts 800 \
  --min-validation-reports 200 \
  --min-validated 200
```

The wrapper copies the tracked BioModels snapshot into the evidence directory,
installs the artifact in an isolated virtual environment, records the benchmark
command, runtime, return code, dependency freeze, Python/platform metadata,
model counts, status classifications, validation profile/result/reason counts,
representative validation reports, and key benchmark outputs under the evidence
directory. Use `--skip-run` only to summarize existing local benchmark outputs
during development; release evidence should run the benchmark command from the
installed artifact.

### Critical Coverage Gate

For local release-candidate review, run the non-slow suite with branch coverage and
enforce the critical-module thresholds:

```bash
uv run python tools/check_critical_coverage.py --run-pytest \
  --coverage-json release-evidence/critical-coverage.json
```

The gate requires at least 90% statement coverage and 85% branch coverage for the
critical recasting, parsing, formatting, validation, and solver-backend modules.

### Maintainability Gate

Critical modules also have a local maintainability baseline for module length,
maximum function length, and a simple cyclomatic complexity score:

```bash
uv run python tools/check_maintainability.py
```

The baseline is stored in `tools/maintainability_baseline.json`. Lower metrics
are allowed and should be committed after refactors; increases fail the local
gate until they are reduced or intentionally reviewed.

### Performance Budget Gate

Representative recast and validation workflows have local performance budgets:

```bash
uv run python tools/check_performance_budgets.py \
  --evidence-dir release-evidence/performance
```

The budgets are stored in `tools/performance_budgets.json`. Each task runs in a
separate Python subprocess with a deterministic timeout, records stdout/stderr
logs and task metadata, and writes `summary.json` under the evidence directory.
The gate fails on task errors, timeouts, or budget overruns.

### Viewing the Notebook Reports

```bash
jupyter notebook out_test_models1/recast_report.ipynb
```

Each model entry shows:
- Original and recast Antimony code
- LaTeX equations (original and S-system)
- Numerical simulation comparison plots
- Validation profile results
- System classification

---

## Output Modes: Simplified vs. Canonical

### Simplified Mode (`--mode simplified`)

**Default behavior.** Produces a flexible S-system form:
- Preserves mathematical structure of original equations
- Zero coefficients remain zero (single-term equations allowed)
- Cleaner output for visualization and analysis

**Use when:**
- You need readable output
- Analyzing system behavior
- Visualizing dynamics

### Canonical Mode (`--mode canonical`)

**Strict S-system form** following Savageau & Voit (1987):
- Guarantees exactly 2 terms per equation (growth + decay)
- Adds epsilon slack variables to ensure both terms present
- Required for certain theoretical analyses

**Use when:**
- Publishing S-system analyses
- Applying S-system-specific algorithms
- Theoretical work requiring canonical form

---

## Library Usage (Programmatic)

### Basic Example

```python
import ssys
from ssys.recaster import parse_antimony_via_sbml

# Load and parse model (SBML-based parser)
text = open("test_models1/m01_exp_decay.ant").read()
sym = parse_antimony_via_sbml(text)

# Recast to S-system (simplified mode)
result = ssys.recast_to_ssystem(sym, mode="simplified")

# Generate Antimony output
out = ssys.ssystem_to_antimony(result, 
                                model_name="m01_exp_decay_recast",
                                mode="simplified")

# Save
open("m01_exp_decay_recast.ant", "w").write(out)
```

### SBML Input

```python
import ssys

# Parse an SBML file directly through libSBML
sym = ssys.parse_sbml("model.xml")
result = ssys.recast_to_ssystem(sym, mode="simplified")
out = ssys.ssystem_to_antimony(result, model_name="model_recast", mode="simplified")
```

### Classification

```python
from ssys.recaster import classify_system, classify_result

# Classify input system
input_class = classify_system(sym)
print(f"Input: {input_class.value}")
# Output: "General", "S-system", "Canonical S-system", "GMA",
# or "GMA with time-varying coefficients"

# Classify output (mode-aware)
output_class = classify_result(result, mode="simplified")
print(f"Output: {output_class.value}")
```

### Validation

```python
from ssys.validator import validate_recast_pair

report = validate_recast_pair(
    original_file="test_models1/m01_exp_decay.ant",
    recast_file="out_test_models1/m01_exp_decay_recast.ant",
    profile="strict",
)

print(f"Overall pass: {report.overall_pass}")
print(f"Summary: {report.summary}")
```

---

## GMA and S-system Forms

### Generalized Mass Action (GMA) Form

A **GMA system** has equations where each right-hand side is a sum of power-law monomials:

```
Ẋᵢ = Σₖ cₖ ∏ⱼ Xⱼ^(eₖⱼ)
```

*Note:* If the coefficients cₖ depend on time (via a clock state T), the system is classified as **GMA (time-varying)**.

### S-system Form

An **S-system** has at most two terms per equation—one production term and one degradation term:

```
Ẋᵢ = αᵢ ∏ⱼ Xⱼ^(gᵢⱼ) - βᵢ ∏ⱼ Xⱼ^(hᵢⱼ)
```

*Note:* The form is **S-system** if αᵢ, βᵢ ≥ 0. The form is **Strict canonical S-system** if both terms are present with αᵢ, βᵢ > 0 (achieved via ε-splitting).

See [RECASTING.md](RECASTING.md) for detailed recasting theory, rules, and worked examples.

---

## Antimony Input Subset Supported

**Supported:**
- Reactions: `A + B -> C; k*A*B`
- Initializations: `X = 2.5`, `k = 0.1`
- Explicit rate rules: `X' = ...`
- Boundary species: `$X` (not dynamic)
- Parameters (treated as positive constants)
- **Elementary functions**: `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`
- **Rational functions**: `X/(Y+1)`, `1/(X+Y+Z)`
- **Assignment rules**: `Z := X + Y`
  - As input: substituted into ODEs before recasting
  - As output: generated as observable variables to reconstruct original variables from auxiliaries

**Not yet supported:**
- Modules
- Events and piecewise functions
- Non-positive variables (requires preprocessing)

**Simulation metadata:**
- `@SIM` comments specify simulation parameters:
  ```antimony
  // @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6
  ```
  - `T_START`: Simulation start time (default: 0.0)
  - `T_END`: Simulation end time (default: 1.0)
  - `N_STEPS`: Number of steps (default: 100)
  - `EPS_INIT`: Epsilon value for zero initial condition approximation in pool construction (default: 1e-6). Use smaller values for higher precision or larger values if numerical instability occurs.

---

## Algorithm Overview

1. **Parse** Antimony via SBML (reference implementation), or parse SBML directly through the Python API
2. **Build** SymPy ODEs from reactions and rate rules
3. **Lift composite functions**: exp, sin, log, etc. → auxiliary variables (chain rule)
4. **Lift rational functions**: denominators → auxiliary variables (exact S-system form)
5. **Expand** RHS into sum of monomials (products of powers)
6. **Split sums**: iteratively introduce auxiliaries to express as single growth - single decay
7. **Mode-specific formatting**:
   - Simplified: preserve zeros
   - Canonical: add epsilon slack for strict 2-term form
8. **Emit** Antimony with auxiliary S‑system variables and canonical rate rules

---

## System Classifications

The tool classifies both input and output systems:

- **General**: Contains non-monomial terms (arbitrary functions)
- **GMA** (Generalized Mass Action): All monomial terms, may have multiple terms
- **S-system**: 1-2 monomial terms per equation (growth and/or decay), α, β ≥ 0
- **Strict canonical S-system**: Exactly 2 terms per equation (1 growth + 1 decay), α, β > 0

---

## Handling Zero Initial Conditions

S-systems require positive state variables due to power-law terms with potentially negative exponents. If a model has a state variable with initial condition `x(0) = 0`, the recaster uses **ε-regularization** to handle this:

### Current Behavior

Zero initial conditions are automatically replaced with a small positive value (`EPS_INIT`):
- Default: `EPS_INIT = 1e-6`
- User-configurable via `@SIM` metadata in the Antimony file

### Configuration

Specify `EPS_INIT` in your model's `@SIM` comment:
```antimony
// @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6
// Note: Zero-valued initial conditions are replaced with EPS_INIT during recasting.
```

### Choosing EPS_INIT

- **Smaller values** (e.g., `1e-9`): Higher accuracy if the solution immediately leaves zero
- **Larger values** (e.g., `1e-4`): Better numerical stability for stiff systems
- **Scale-aware**: If your variables have typical magnitude S, use `ε ≈ 1e-6 * S`

### Limitations

The ε-regularization approach:
- Solves a slightly different IVP (the original had `x(0)=0`, the recast has `x(0)=ε`)
- May introduce sensitivity near `t=0`
- May break exact conservation laws that depend on zeros

Alternative strategies for exact zero handling should be tracked in local design notes until
public project infrastructure exists.

---

## Examples

See [RECASTING.md](RECASTING.md) for detailed worked examples covering:
- Exponential decay (trivial S-system)
- Central t-distribution (sum lifting)
- Van der Pol oscillator (GMA → S-system)
- Monod chemostat (Michaelis-Menten → GMA)
- Brusselator (product auxiliaries)
- SIR epidemic model (ε-splitting)
- Two-body orbit problem (parameter family)

---

## Troubleshooting

**Environment or import errors:**
- From a `uv sync` environment, run commands through `uv run ...`.
- From the named helper environment, activate with
  `source ssys_dev/bin/activate`.
- If the named environment is stale or corrupted, recreate it with
  `./setup_dev_env.sh --force`.

**Parser errors:**
- The default `--parser sbml` path parses Antimony through the reference
  Antimony implementation and then through SBML/libSBML.
- Unsupported features such as SBML events, delays, constraints, unknown
  functions, duplicate rate rules, malformed Antimony, and missing math
  formulas are rejected before a successful recast artifact is produced.
- `--parser legacy` is compatibility-only and deprecated. Prefer `--parser
  sbml` for local release evidence.
- Inputs are trusted local scientific model files, not safe untrusted uploads;
  see [PARSER_TRUST_BOUNDARY.md](PARSER_TRUST_BOUNDARY.md).

**Missing RoadRunner/CVODE:**
- ODE and assignment-rule trajectory validation require libRoadRunner/CVODE.
- If RoadRunner is unavailable, required trajectory checks report
  `unsupported` instead of passing. Inspect the validation JSON `tests`
  entries for `result`, `reason`, `details`, and backend metadata.
- Re-sync the base environment with `uv sync --extra dev` before debugging
  trajectory failures.

**Missing DAE backend:**
- DAE-required trajectory validation requires the `dae` extra:
  `uv sync --extra dev --extra dae`.
- Local release-candidate DAE checks should run with
  `SSYS_REQUIRE_DAE_VALIDATION=1`; missing IDA/SUNDIALS support is then a
  failing release-gate condition.
- Normal development runs may report missing DAE support as `unsupported`, but
  `unsupported` is never a validation pass.

**Numerical sampling failures:**
- `invalid_sampling_domain` means the model metadata did not provide a usable
  finite positive sampling domain for a required numerical check.
- `nonfinite_sample` means a sampled point produced a non-finite value, often
  because the sampled domain touched a singular surface.
- Validation reports record the deterministic seed, sampled ranges, parameter
  values, threshold, and counterexample metadata needed to reproduce the
  failure.

**Trajectory validation mismatches:**
- Trajectory validation is solver-backed support, not symbolic proof.
- Check the report metadata for absolute, relative, and scaled errors, worst
  variable/time/value, solver backend, tolerances, output-step diagnostics, and
  interpolation status.
- A `failed` trajectory check can indicate a recasting error, parameter/domain
  issue, solver failure, or a model that needs symbolic rather than
  trajectory-based evidence. Confirm with the `symbolic` or `numerical`
  profile before treating it as a transformation bug.

**Artifact install or release-gate failures:**
- Run `python tools/local_artifact_smoke.py --allow-dirty` during development
  to inspect local wheel/sdist smoke logs.
- Release evidence should omit `--allow-dirty` and should archive command logs,
  dependency freezes, validation reports, artifact hashes, and
  `summary.json` under a local `release-evidence/` directory.
- Run `python3 tools/check_release_metadata.py` after changing README,
  changelog, release notes, citation metadata, package metadata, or maturity
  wording.

---

## References

- Savageau, M. A., & Voit, E. O. (1987). Recasting nonlinear differential equations as S‑systems: a canonical nonlinear form. *Mathematical Biosciences*, 87(1), 83-115.
- Smith, L. P., Bergmann, F. T., Chandran, D., & Sauro, H. M. (2009). Antimony: A modular model definition language. *Bioinformatics*, 25(18), 2452-2454.

---

## Contributing

See `CONTRIBUTING.md` for guidelines.

## License

ssys is open-source software distributed under the **MIT License**. See the `LICENSE`
file for the full license text and the accompanying Triad National Security, LLC /
U.S. Government copyright notice.

This software was produced at Los Alamos National Laboratory (LANL) under U.S.
Government contract 89233218CNA000001 and is released under LANL reference number
**O# (O5066)**.

© 2026. Triad National Security, LLC. All rights reserved.

## Citation

See `CITATION.cff` for citation information.
