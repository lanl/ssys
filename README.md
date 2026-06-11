# ODE → GMA or S‑System Recast (Antimony → Antimony, with SBML library support)
*Source: README.md | v0.5.5 | 2025-12-31*

This toolkit converts ordinary differential equation (ODE) models into **S‑System** or **GMA** form and writes the result back to Antimony. The command-line interface batch-processes **Antimony** models and generates **Jupyter notebook** verification reports; the Python library can also parse SBML files directly.

**Release maturity:** ssys is currently published as alpha software. Treat the APIs and generated report format as subject to change until the release gates in `dev/punchlist.md` are closed.

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
  RECASTING.md            # Recasting theory and rules
  TEST_MODELS.md          # Test model collection documentation
  README.md               # This file
```

---

## Installation

### Quick Setup with uv (Recommended)

Use [uv](https://astral.sh/uv) for fast, reliable Python environment management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# One-time setup - creates ssys_dev environment
./setup_dev_env.sh

# Activate environment
source ssys_dev/bin/activate
```

The script creates a `ssys_dev` virtual environment using uv and installs ssys with all required dependencies including libRoadRunner, Antimony, and python-libsbml for SBML-based parsing.

### Manual Installation (Alternative)

```bash
# Create virtual environment
uv venv ssys_dev
source ssys_dev/bin/activate

# Install in editable mode with all development dependencies
uv pip install -e ".[dev]"
```

---

## Requirements

- Python 3.10+
- sympy ≥1.12 (symbolic math)
- numpy ≥1.24,<2 (libroadrunner compatibility)
- scipy ≥1.10
- matplotlib ≥3.7 (plots in generated notebooks)
- nbformat ≥5.9 (notebook generation)
- **libroadrunner ≥2.5** (ODE simulation, SBML)
- **antimony ≥2.13** (Antimony parsing via SBML)
- **python-libsbml ≥5.20** (SBML model representation)

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

The validator performs three independent tests to verify mathematical correctness:

### Test Suite

1. **Symbolic Test**: Proves exact equivalence using the Jacobian chain rule. Tests whether `J_Φ(Z) · f_recast(Z) = f_orig(Φ(Z))` simplifies to zero symbolically.

2. **Numerical Test**: Validates equivalence at 1000 random sample points with ε = 10⁻⁵ threshold.

3. **Trajectory Test**: Simulates both original and recast models, compares trajectories with 3.0% error threshold.

### Validation Logic

**Default verdict uses AND logic**: A recast is considered **valid only if all requested tests pass**. The CLI and default API call request all three tests, so symbolic, numerical, and trajectory checks must all pass.

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

See [TEST_MODELS.md](TEST_MODELS.md) for complete model documentation and [RECASTING.md](RECASTING.md) for recasting theory.

### pytest Integration Tests

The full integration test suite validates all 117 models in both modes:

```bash
# Run full integration suite (~3 min; requires the DAE extra)
uv sync --extra dev --extra dae
pytest tests/test_integration.py -m slow -v

# Skip slow integration tests during rapid development
pytest -m "not slow"

# Run a specific model directory
pytest tests/test_integration.py -k "test_models1"
```

### Viewing the Notebook Reports

```bash
jupyter notebook out_test_models1/recast_report.ipynb
```

Each model entry shows:
- Original and recast Antimony code
- LaTeX equations (original and S-system)
- Numerical simulation comparison plots
- Validation results (symbolic + numerical + trajectory)
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
    recast_file="out_test_models1/m01_exp_decay_recast.ant"
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

Alternative strategies for exact zero handling are documented as issues at the online repository.

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

**"Module not found" error:**
- Ensure environment is activated: `source ssys_dev/bin/activate`

**Parser errors:**
- Check for unsupported constructs (modules, events)

**Validation failures:**
- Check the validation JSON for which test failed
- Symbolic failures may be due to SymPy simplification limits
- Trajectory failures may indicate numerical instability

**Simulation failures:**
- RoadRunner is required for SBML-based simulation
- Check parameter values for reasonableness

---

## References

- Savageau, M. A., & Voit, E. O. (1987). Recasting nonlinear differential equations as S‑systems: a canonical nonlinear form. *Mathematical Biosciences*, 87(1), 83-115.
- Smith, L. P., Bergmann, F. T., Chandran, D., & Sauro, H. M. (2009). Antimony: A modular model definition language. *Bioinformatics*, 25(18), 2452-2454.

---

## Contributing

See `CONTRIBUTING.md` for guidelines.

## License

See `LICENSE` file.

## Citation

See `CITATION.cff` for citation information.
