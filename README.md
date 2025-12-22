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

Use [uv](https://astral.sh/uv) for fast, reliable Python environment management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# One-time setup - creates ssys_dev environment
./setup_env.sh

# Activate environment
source ssys_dev/bin/activate
```

The script:
- Creates a `ssys_dev` virtual environment using uv
- Installs ssys with all development dependencies
- Includes libRoadRunner for ODE simulation (if available)

**Options:**
```bash
./setup_env.sh --minimal  # Skip libroadrunner (uses RK4 fallback)
```

### Manual Installation (Alternative)

If you prefer not to use the setup script, you can install manually:

```bash
# Create virtual environment
uv venv ssys_dev
source ssys_dev/bin/activate

# Install in editable mode with all development dependencies
uv pip install -e ".[dev,notebook,simulation]"

# Or minimal (without libroadrunner simulation backend):
uv pip install -e ".[dev,notebook]"
```

### Installing Additional Packages

With the environment active, use uv for package management:

```bash
uv pip install <package_name>
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

## Running Test Problems (tests2)

The `tests2/` directory contains 69 ODE models from published literature demonstrating S-system recasting techniques. These models cover a wide range of mathematical structures from Savageau & Voit (1987) and other foundational papers.

### Quick Start

```bash
# Ensure environment is set up and activated
source ssys_dev/bin/activate

# Run all test problems in both modes
python run_tests2.py --both
```

### Usage Options

```bash
python run_tests2.py                    # Simplified mode only (default)
python run_tests2.py --mode canonical   # Canonical mode only
python run_tests2.py --both             # Both modes (for comparison)
```

### Output

The script generates:
- `out_tests2_simplified/` - Recast models + notebook (simplified mode)
- `out_tests2_canonical/` - Recast models + notebook (canonical mode)

Each directory contains:
- Individual `*_recast.ant` files for each model
- `*_validation.json` files with validation results
- `recast_report.ipynb` - Interactive notebook with all results

### Viewing the Notebook Reports

```bash
# View simplified mode results
jupyter notebook out_tests2_simplified/recast_report.ipynb

# View canonical mode results  
jupyter notebook out_tests2_canonical/recast_report.ipynb
```

Each model entry in the notebook shows:
- Original and recast Antimony code
- LaTeX equations (original and S-system)
- Numerical simulation comparison plots
- Validation results (symbolic + numerical tests)
- System classification

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

Install libRoadRunner in your environment:
```bash
source ssys_dev/bin/activate
uv pip install libroadrunner
```

Note: The `./setup_env.sh` script installs libRoadRunner by default.

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
- Non-positive variables (requires preprocessing)

✅ **Simulation metadata:**
- `@SIM` comments specify simulation time parameters:
  ```antimony
  // @SIM T_START=0 T_END=100 N_STEPS=500
  ```
  - `T_START`: Simulation start time (default: 0.0)
  - `T_END`: Simulation end time (default: 20.0)
  - `N_STEPS`: Number of steps (default: 400)

---

## Antimony Parser Limitations and Remedies

The ssys Antimony parser supports a **subset** of the full Antimony language. This section documents common parser limitations and how to remedy them by reformulating your model.

### Summary of Limitations

| Limitation | Error Symptom | Remedy |
|------------|---------------|--------|
| Parameterized functions | `FunctionClass error` | Inline expressions into ODEs |
| Multi-line expressions | `could not parse '( 1'` | Reformat to single line |
| `piecewise()` function | `unsupported function` | Use smooth approximations (see below) |
| Events | Parse error | Not recastable; approximate or remove |
| Modules | Parse error | Flatten to single model |
| Reserved names | SymPy conflicts | Rename variables (e.g., `I` → `I_var`) |

### Parameterized Functions → Inline Expressions

**Problem:** Antimony allows user-defined functions with parameters like `f(u, K) := ...`. The ssys parser does not support these.

**Before (unsupported):**
```antimony
// Parameterized Hill function - NOT SUPPORTED
f(u, K) := u^H / (K^H + u^H);

Y' = b_1 * f(X, K_xy) - a_1 * Y;
```

**After (supported):**
```antimony
// Inline the Hill function directly into the ODE
Y' = b_1 * X^H / (K_xy^H + X^H) - a_1 * Y;
```

**Example:** Goldbeter1996 used `V1(C) := V_M1 * C / (K_c + C)`. This was fixed by defining `V_1 := V_M1 * C / (K_c + C)` (simple assignment rule without parameters) and using `V_1` in the ODEs.

### Multi-Line Expressions → Single Line

**Problem:** The parser may fail on expressions split across multiple lines within assignment rules.

**Before (may fail):**
```antimony
x011 := ( 1
          - (x000*(1 + IP3/d1) + x001*(1 + IP3/d3))
        ) / (1 + IP3/d3);
```

**After (works):**
```antimony
x011 := (1 - (x000*(1 + IP3/d_1) + x001*(1 + IP3/d_3))) / (1 + IP3/d_3);
```

### Reserved Names and LaTeX-Friendly Nomenclature

**Problem:** Some variable names may conflict with SymPy reserved symbols or render poorly in LaTeX.

**Best practices:**
- Use underscores for subscripts: `k_1` instead of `k1` (renders as $k_1$)
- Avoid single-letter names that conflict with SymPy: `I` → `I_var` or `I_0`
- Use descriptive subscripts: `K_xy` instead of `Kxy` (renders as $K_{xy}$)

---

## Replacing Step Functions with Smooth Approximations

The ssys recaster requires **differentiable** expressions because it uses symbolic differentiation (chain rule) during lifting. Discontinuous functions like `piecewise()`, Heaviside steps, and ReLU cannot be processed directly.

However, these can be replaced with **smooth (C∞) approximations** that match the original behavior within a small tolerance.

### Smooth ReLU: max(0, x)

**Original (discontinuous):**
```antimony
J := piecewise(0, x < 0, x);   // = max(0, x)
```

**Smooth approximation:**
```antimony
const eps = 0.01;  // Smoothing parameter

// Smooth max(0, x) using sqrt
J := 0.5 * (x + sqrt(x^2 + eps^2));
```

**Properties:**
- Matches `max(0, x)` within ε/2 everywhere
- C∞ smooth (infinitely differentiable)
- As ε → 0, converges to exact max(0, x)

**Example:** Fink2000 used `piecewise(0, Ca <= Ca_c, g0*(Ca - Ca_c))` for calcium extrusion with a threshold. This was replaced with:
```antimony
const eps_pm = 0.01;
J_pm := 0.5 * g_0 * ((Ca - Ca_c) + sqrt((Ca - Ca_c)^2 + eps_pm^2));
```

### Smooth Heaviside Step Function

**Original (discontinuous):**
```antimony
H := piecewise(0, t < t_0, 1);   // = H(t - t_0) Heaviside step
```

**Smooth approximation using tanh:**
```antimony
const k_steep = 5.0;  // Steepness parameter (larger = sharper transition)

// Smooth step: 0 → 1 transition at t = t_0
H := 0.5 * (1 + tanh(k_steep * (time - t_0)));
```

**Properties:**
- Transitions smoothly from 0 to 1 around t = t₀
- Width of transition ≈ 2/k_steep
- C∞ smooth

### Time Windows (On/Off Pulses)

**Original (discontinuous):**
```antimony
// Active only during t_1 < t < t_2
Y := piecewise(0, time <= t_1, 1 - cos(2*pi*time/30), time < t_2, 0);
```

**Smooth approximation:**
```antimony
const k_steep = 5.0;

// Smooth turn-on at t_1, turn-off at t_2
H_on  := 0.5 * (1 + tanh(k_steep * (time - t_1)));
H_off := 0.5 * (1 + tanh(k_steep * (t_2 - time)));
Y := (1 - cos(2*pi*time/30)) * H_on * H_off;
```

**Example:** Weber2018 used a piecewise time window for 5 < t < 70. This was replaced with:
```antimony
const k_steep = 5.0;
const eps_k = 0.01;

H_on  := 0.5*(1 + tanh(k_steep*(time - 5)));
H_off := 0.5*(1 + tanh(k_steep*(70 - time)));
Y_1   := (1 - cos(2*pi*time/30)) * H_on * H_off;

// Also needed smooth ReLU for k_23 = max(0, k_23_0 + beta*Y_1)
k_23_raw := k_23_0 + beta*Y_1;
k_23 := 0.5 * (k_23_raw + sqrt(k_23_raw^2 + eps_k^2));
```

### Choosing Smoothing Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `eps` (ReLU) | 0.01 | Smaller = sharper corner, larger = smoother |
| `k_steep` (tanh) | 5.0 | Larger = sharper transition, smaller = gradual |

**Guidelines:**
- Start with default values (eps = 0.01, k_steep = 5.0)
- Verify that dynamics are qualitatively unchanged
- Decrease eps or increase k_steep if sharper transitions are needed
- Ensure smoothing scale is small compared to system timescales

### Summary of Smooth Approximations

| Original | Smooth Approximation |
|----------|---------------------|
| `max(0, x)` | `0.5*(x + sqrt(x² + ε²))` |
| `H(t - t₀)` (Heaviside) | `0.5*(1 + tanh(k*(t - t₀)))` |
| `abs(x)` | `sqrt(x² + ε²)` |
| `sign(x)` | `tanh(k*x)` |
| `min(a, b)` | `0.5*(a + b - sqrt((a-b)² + ε²))` |

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
