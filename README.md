# ODE → S‑System Recast (Antimony → Antimony)

**Version:** v0.4.0  
**Date:** 2025-11-25

This toolkit converts ordinary differential equation (ODE) models written in **Antimony** into **canonical S‑System** form and writes the result back to Antimony. It provides both a **Python library** and a **command-line interface** for batch processing models and generating **Jupyter notebook** verification reports.

---

## Contents

```
ssys/
  src/ssys/
    __init__.py           # Package interface
    cli.py                # Command-line interface
    recaster.py           # Core library: parse → ODE → recast → Antimony
    notebook_helpers.py   # Jupyter notebook generation utilities
  tests/                  # Example Antimony models (22 test cases)
  literature/             # Reference papers on S-systems
  README.md               # This file
```

---

## Installation

### Quick Setup with uv (Recommended)

Use the provided setup script to create a development environment with [uv](https://astral.sh/uv):

```bash
# One-time setup - creates ssys_dev environment
./setup_env.sh

# Activate environment
source ssys_dev/bin/activate
```

The script:
- Creates a `ssys_dev` virtual environment
- Installs ssys with all development dependencies
- Includes libRoadRunner for ODE simulation (if available)

**Options:**
```bash
./setup_env.sh --minimal  # Skip libroadrunner (uses RK4 fallback)
```

### Manual Installation

```bash
pip install -e .
```

Or use directly with PYTHONPATH:
```bash
export PYTHONPATH=/path/to/ssys/src:$PYTHONPATH
```

---

## Requirements

- Python 3.9+
- sympy (symbolic math)
- matplotlib (plots in generated notebooks)
- nbformat (notebook generation)
- jupyter (to run generated notebooks)

---

## Command-Line Interface

The CLI tool batch-recasts models and generates verification notebooks:

```bash
ssys-recast --manifest tests/tests.manifest \
            --outdir out_simplified \
            --mode simplified
```

### Arguments

- `--manifest`: Path to manifest file (one `.ant` file path per line)
- `--outdir`: Output directory for recast models and notebook
- `--mode`: Output mode (default: `simplified`)
  - `simplified`: Flexible S-system form, preserves zeros
  - `canonical`: Strict 2-term form with epsilon slack variables
- `--validate`: Run mathematical correctness validation on each recast
  - Performs symbolic equivalence test (Jacobian chain rule)
  - Performs numerical pointwise test (1000 random samples)
  - Generates JSON validation reports
  - Displays validation results in notebook

### Validation Logic & Interpreting Results

The validator uses two independent tests to verify mathematical correctness:

1. **Symbolic Test**: Proves exact equivalence using the Jacobian chain rule. Tests whether `J_Φ(Z) · f_recast(Z) = f_orig(Φ(Z))` simplifies to zero symbolically.

2. **Numerical Test**: Validates equivalence at 1000 random sample points with ε = 10⁻⁵ threshold.

**Overall verdict uses OR logic**: A recast is considered **valid if EITHER test passes**. This is because:

- **Symbolic test is the gold standard** but may fail due to SymPy's simplification limitations. Complex expressions that are mathematically zero may not fully simplify (e.g., `exp(X*k) - exp(X*k)` might remain unsimplified due to symbol identity issues).

- **Numerical test provides empirical validation** but may fail for lifted systems if auxiliary variables are sampled off their constraint manifold (e.g., for `Z := exp(X*k)`, sampling Z independently instead of computing it from X).

**When to be concerned**:
- ✓ **Symbolic PASS, Numerical FAIL**: Excellent! Symbolic proof guarantees correctness.
- ✓ **Symbolic FAIL, Numerical PASS**: Good! Empirical validation with 1000 samples confirms correctness despite simplification challenges.
- ✗ **Both FAIL**: This indicates a potential issue. Review the model and validation details.

**Example output**:
```
Test       Result   Max Error
symbolic   ✓ pass   N/A
numerical  ✗ fail   8.94e+00
Overall: ✓ PASS
```
This is valid! The symbolic test proved exact equivalence, which is sufficient.

### Manifest Format

Plain text file with one Antimony file path per line:
```
tests/01_exp_decay.ant
tests/02_logistic.ant
# Lines starting with # are ignored
```

### Output

For each input model:
- Recast `.ant` file in `--outdir`
- Verification notebook: `recast_report.ipynb`

The notebook shows for each model:
- Original and recast Antimony code
- LaTeX ODEs (original and S-system)
- Numerical simulation comparison
- System classification (S-system, Canonical S-system, or GMA)

---

## Output Modes: Simplified vs. Canonical

### Simplified Mode (`--mode simplified`)

**Default behavior.** Produces a flexible S-system form:
- Preserves mathematical structure of original equations
- Zero coefficients remain zero (single-term equations allowed)
- Cleaner output for visualization and analysis
- Example: Pure decay `X' = -k*X` becomes `X' = 0 - k*X`

**Use when:**
- You need readable output
- Analyzing system behavior
- Visualizing dynamics
- Zero terms are acceptable

### Canonical Mode (`--mode canonical`)

**Strict S-system form** following Savageau & Voit (1987):
- Guarantees exactly 2 terms per equation (growth + decay)
- Adds epsilon slack variables to ensure both terms present
- Required for certain theoretical analyses
- Example: Pure decay `X' = -k*X` becomes `X' = ε*X - (ε+k)*X`

**Use when:**
- Publishing S-system analyses
- Applying S-system-specific algorithms
- Theoretical work requiring canonical form
- Comparing with literature using strict definition

**Technical details:**
- Epsilon (ε) is a small positive constant (default: 1.0)
- Transformation preserves dynamics: `ε*X - (ε+k)*X = -k*X`
- Both growth and decay terms are positive monomials

---

## ODE Solver Selection

### Overview

Verification notebooks support two ODE solvers for trajectory simulation:
- **libRoadRunner** (recommended): Production-quality CVODE integrator with adaptive stepping
- **RK4** (fallback): Simple fixed-step 4th-order Runge-Kutta

### Installation

Install libRoadRunner in your conda environment:
```bash
conda activate ssys_env
conda install -c conda-forge libroadrunner
```

Or via pip:
```bash
pip install libroadrunner
```

### Usage in Notebooks

Generated verification notebooks automatically use libRoadRunner. To change solver:

```python
# Use libRoadRunner (default in generated notebooks)
load_and_report('model.ant', 'model_recast.ant', solver='roadrunner')

# Use RK4
load_and_report('model.ant', 'model_recast.ant', solver='rk4')
```

### Why libRoadRunner?

- **Adaptive stepping**: Automatically adjusts step size for accuracy and stability
- **Robust for stiff systems**: CVODE handles challenging dynamics
- **Widely used**: Standard tool in systems biology
- **Direct Antimony support**: No SBML conversion needed

### Fallback Behavior

If a solver fails, the notebook automatically falls back to the basic RK4 method with a warning message.

---

## Library Usage (Programmatic)

### Basic Example

```python
import ssys

# Load and parse model
text = open("tests/01_exp_decay.ant").read()
ir = ssys.parse_antimony(text)

# Build symbolic ODE system
sym = ssys.build_sym_system(ir)

# Recast to S-system (simplified mode)
result = ssys.recast_to_ssystem(sym, mode="simplified")

# Generate Antimony output
out = ssys.ssystem_to_antimony(result, 
                                model_name="exp_decay_recast",
                                mode="simplified")

# Save
open("exp_decay_recast.ant", "w").write(out)
```

### Canonical Mode Example

```python
import ssys

text = open("tests/14_Michaelis_Menten_prod_deg.ant").read()
ir = ssys.parse_antimony(text)
sym = ssys.build_sym_system(ir)

# Canonical mode with epsilon slack
result = ssys.recast_to_ssystem(sym, mode="canonical")
out = ssys.ssystem_to_antimony(result,
                                model_name="MM_canonical",
                                mode="canonical")

open("MM_canonical.ant", "w").write(out)
```

### Classification

```python
from ssys.recaster import classify_system, classify_result

# Classify input system
input_class = classify_system(sym)
print(f"Input: {input_class.value}")
# Output: "General", "S-system", "Canonical S-system", or "GMA"

# Classify output (mode-aware)
output_class = classify_result(result, mode="canonical")
print(f"Output: {output_class.value}")
```

### LaTeX Output

```python
from ssys.recaster import latex_odes, latex_ssys

# Original ODEs in LaTeX
print(latex_odes(sym))

# S-system in LaTeX
print(latex_ssys(result))
```

### Factor Map

The `factor_map` allows reconstruction of original variables from auxiliaries:

```python
result = ssys.recast_to_ssystem(sym)

# result.factor_map: {X: [X_1, X_2, ...]}
# Meaning: X = X_1 * X_2 * ...

# Reconstruct during simulation
for orig, aux_list in result.factor_map.items():
    reconstructed = 1.0
    for aux in aux_list:
        reconstructed *= aux_state[aux]
    # reconstructed now equals original variable X
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
- Time-dependent forcing
- Non-positive variables (requires preprocessing)

---

## Algorithm Overview

1. **Parse** Antimony → intermediate representation (species, parameters, reactions, rate rules)
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
- **GMA** (Generalized Mass Action): All monomial terms, may have multiple incompatible terms
- **S-system**: 1-2 monomial terms per equation (growth and/or decay)
- **Canonical S-system**: Exactly 2 terms per equation (1 growth + 1 decay)

Classification is mode-aware:
- Simplified mode: counts actual non-zero terms
- Canonical mode: accounts for epsilon transformation

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

**Canonical output**:
```
X' = epsilon*X^1 - (epsilon + 0.5)*X^1
```

### Example 2: Michaelis-Menten

**Input** (`14_Michaelis_Menten_prod_deg.ant`):
```
S' = -k1*S*E/(Km + S)
P' = k1*S*E/(Km + S) - k2*P
```

**Output** (both modes, after lifting):
```
S' = 0 - k1*S*E*Y_1^-1
P' = k1*S*E*Y_1^-1 - k2*P

Y_1' = ... # Auxiliary for (Km + S)
```

---

## Troubleshooting

**"Module not found" error:**
- Ensure `src/` is in PYTHONPATH or install with `pip install -e .`

**"Unexpected symbol" during parsing:**
- Check for unsupported Antimony constructs (modules, events)
- Simplify model or extend parser

**Flat lines in reconstructed plots:**
- Check initial conditions—product constraints require correct auxiliary initial values
- Verify factor_map is applied correctly

**Numerical blow-ups:**
- Large exponents can cause rapid growth
- Reduce simulation time or use adaptive solver
- Check parameter values for reasonableness

**Classification mismatch:**
- Ensure mode parameter is passed to both `recast_to_ssystem()` and `classify_result()`
- Simplified mode preserves zeros, canonical mode adds epsilon

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
