# ODE → S‑System Recast (Antimony → Antimony)

**Version:** v0.5.2
**Date:** 2025-12-28

This toolkit converts ordinary differential equation (ODE) models written in **Antimony** into **S‑System** or **GMA** form and writes the result back to Antimony. It provides both a **Python library** and a **command-line interface** for batch processing models and generating **Jupyter notebook** verification reports.

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
  test_models1/           # 29 core test models
  test_models2a/          # Literature models (passing)
  test_models2b/          # Literature models (advanced)
  test_models3/           # BioModels-derived test cases
  pathological_models/    # Numerically challenging models
  tests/                  # Unit tests
  literature/             # Reference papers on S-systems
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
            --parser sbml \
            --validate
```

### Arguments

- `--manifest`: Path to manifest file (one `.ant` file path per line)
- `--outdir`: Output directory for recast models and notebook
- `--mode`: Output mode (default: `simplified`)
  - `simplified`: Flexible S-system form, preserves zeros
  - `canonical`: Strict 2-term form with epsilon slack variables
- `--parser`: Antimony parser to use (default: `sbml`)
  - `sbml`: SBML-based parser (recommended, uses reference Antimony implementation)
  - `legacy`: Hand-rolled parser (deprecated)
- `--validate`: Run mathematical correctness validation on each recast

### Manifest Format

Plain text file with one Antimony file path per line:
```
test_models1/01_exp_decay.ant
test_models1/02_logistic.ant
# Lines starting with # are ignored
```

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

3. **Trajectory Test**: Simulates both original and recast models, compares trajectories with 1.5% error threshold.

### Validation Logic

**Overall verdict uses AND logic**: A recast is considered **valid only if ALL THREE tests pass**. This strict criterion ensures high confidence in recasting correctness.

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
# Activate environment
source ssys_dev/bin/activate

# Run test_models1 (29 core models)
python recast_models.py test_models1 --parser sbml

# Run with validation
python recast_models.py test_models1 --parser sbml --validate
```

### Helper Script Usage

The `recast_models.py` script simplifies batch processing:

```bash
python recast_models.py <directory> [options]
```

Options:
- `--parser {sbml,legacy}`: Parser to use (default: sbml)
- `--mode {simplified,canonical}`: Output mode (default: simplified)

### Test Model Sets

| Directory | Description | Count |
|-----------|-------------|-------|
| `test_models1/` | Core test suite | 29 |
| `test_models2a/` | Literature models (passing) | ~40 |
| `test_models2b/` | Literature models (advanced) | ~15 |
| `test_models3/` | BioModels-derived | ~17 |
| `pathological_models/` | Numerically challenging | 3 |

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

# Load and parse model (SBML-based parser)
text = open("test_models1/01_exp_decay.ant").read()
sym = ssys.parse_antimony_via_sbml(text)

# Recast to S-system (simplified mode)
result = ssys.recast_to_ssystem(sym, mode="simplified")

# Generate Antimony output
out = ssys.ssystem_to_antimony(result, 
                                model_name="exp_decay_recast",
                                mode="simplified")

# Save
open("exp_decay_recast.ant", "w").write(out)
```

### Classification

```python
from ssys.recaster import classify_system, classify_result

# Classify input system
input_class = classify_system(sym)
print(f"Input: {input_class.value}")
# Output: "General", "S-system", "Canonical S-system", or "GMA"

# Classify output (mode-aware)
output_class = classify_result(result, mode="simplified")
print(f"Output: {output_class.value}")
```

### Validation

```python
from ssys.validator import validate_recast_pair

report = validate_recast_pair(
    original_file="test_models1/01_exp_decay.ant",
    recast_file="out_test_models1/01_exp_decay_recast.ant",
    parser="sbml"
)

print(f"Overall pass: {report.overall_pass}")
print(f"Summary: {report.summary}")
```

---

## What "Recast" Means

Given an ODE of the form:
```
Ẋᵢ = Σₖ cᵢₖ ∏ⱼ Zⱼ^(pᵢₖⱼ)
```

We iteratively **split sums into products** by introducing **auxiliary variables** until every derivative is a **difference of two single power‑law products**:

```
Ẋᵢ = αᵢ ∏ⱼ Xⱼ^(gᵢⱼ) - βᵢ ∏ⱼ Xⱼ^(hᵢⱼ)
```

Original variables are **replaced by products of auxiliaries**. The **product constraints** are enforced via initial conditions so that, at t₀, the product equals the original initial value.

This follows Savageau & Voit (1987): positive orthant assumption, decomposition of composite functions, and sum‑splitting into canonical S‑systems.

---

## Antimony Subset Supported

✅ **Supported:**
- Reactions: `A + B -> C; k*A*B`
- Initializations: `X = 2.5`, `k = 0.1`
- Explicit rate rules: `X' = ...`
- Boundary species: `$X` (not dynamic)
- Parameters (treated as positive constants)
- **Elementary functions**: `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`
- **Rational functions**: `X/(Y+1)`, `1/(X+Y+Z)`
- **Assignment rules**: `Z := X + Y` (substituted into ODEs)

❌ **Not yet supported:**
- Modules
- Events and piecewise functions
- Non-positive variables (requires preprocessing)

✅ **Simulation metadata:**
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

1. **Parse** Antimony via SBML (reference implementation) or legacy parser
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
- **S-system**: 1-2 monomial terms per equation (growth and/or decay)
- **Canonical S-system**: Exactly 2 terms per equation (1 growth + 1 decay)

---

## Examples

### Example 1: Exponential Decay

**Input** (`01_exp_decay.ant`):
```
model exponential_decay()
  X = 1.0
  k = 0.5
  X' = -k*X
end
```

**Simplified output**:
```
X' = 0 - 0.5*X^1
```

### Example 2: Michaelis-Menten

**Input**:
```
S' = -k1*S*E/(Km + S)
P' = k1*S*E/(Km + S) - k2*P
```

**Output** (after lifting):
```
S' = 0 - k1*S*E*Y_1^-1
P' = k1*S*E*Y_1^-1 - k2*P

Y_1' = ... # Auxiliary for (Km + S)
```

---

## Troubleshooting

**"Module not found" error:**
- Ensure environment is activated: `source ssys_dev/bin/activate`

**Parser errors:**
- Use `--parser sbml` (default) for better Antimony compatibility
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
- Voit, E. O. (2013). Biochemical systems theory: A review. *ISRN Biomathematics*, 2013.
- Sauro, H. M., et al. Antimony: A modular model definition language. *Bioinformatics*, 2009.

---

## Contributing

See `CONTRIBUTING.md` for guidelines.

## License

See `LICENSE` file.

## Citation

See `CITATION.cff` for citation information.
