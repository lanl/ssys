
# ODE → S‑System Recast (Antimony → Antimony)

**Version:** v0.3.0  
**Date:** 2025-11-21

This toolkit converts ordinary differential equation (ODE) models written in **Antimony** into **canonical S‑System** form and writes the result back to Antimony. It also ships with a **testing harness** that batch‑recasts a manifest of models and generates a **Jupyter notebook** that shows Antimony (before/after), LaTeX ODEs (before/after), and numerical simulations for both the original and the recast system.

---

## Contents

```
/mnt/data/ode_to_ssys/
  ssys_recaster.py        # Core library: parse → build ODEs → recast → emit Antimony
  harness.py              # Test harness: manifest → recast outputs → notebook report
  README.md               # This file
  /tests                  # Example Antimony inputs
  /out                    # Example recast outputs
```

---

## Requirements

- Python 3.9+
- sympy (symbolic math)
- matplotlib (plots in the generated notebook)
- nbformat (the harness writes notebooks)

> If you only use the library (no harness), you can skip nbformat.

---

## What “recast” means (summary)

Given an ODE of the form
\[
\dot X_i = \sum_k c_{ik} \prod_j Z_j^{p_{ikj}},
\]
we iteratively **split sums into products** by introducing **auxiliary variables** until every derivative is a **difference of two single power‑law products**:
\[
\dot X_i = \alpha_i\,\prod_j X_j^{g_{ij}}\; -\; \beta_i\,\prod_j X_j^{h_{ij}}.
\]
Original variables are **replaced by products of auxiliaries**. The **product constraints** are enforced via initial conditions so that, at \(t_0\), the product equals the original initial value.

This follows Savageau & Voit (1987): positive orthant assumption, decomposition of composite functions (future work), and sum‑splitting into canonical S‑systems.

---

## Antimony subset supported (v0.3.0)

- Reactions: `A + B -> C; k*A*B`  
- Initializations: `X = 2.5`, `k = 0.1`  
- Explicit rate rules: `X' = ...`  
- Boundary species marked with `$X` on reaction sides (not dynamic)  
- Parameters are treated as positive constants
- Elementary functions: `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`
- Rational functions: e.g., `X/(Y+1)`

> Not yet supported: modules, reversible shorthand `<->` semantics beyond two one‑way reactions, general assignment rules `:=` with symbolic RHS, piecewise/conditionals.

---

## Library usage (programmatic)

```python
from ssys_recaster import parse_antimony, build_sym_system, recast_to_ssystem, ssystem_to_antimony

text = open("your_model.ant").read()
ir   = parse_antimony(text)          # parse reactions/inits/rate rules
sys  = build_sym_system(ir)          # SymPy ODEs
rec  = recast_to_ssystem(sys)        # recast into canonical S-System (aux variables + factor map)
out  = ssystem_to_antimony(rec, model_name="your_model_recast")

open("your_model_recast.ant","w").write(out)

# Optional: LaTeX pretty‑printing of ODEs
from ssys_recaster import latex_odes, latex_ssys
print(latex_odes(sys))
print(latex_ssys(rec))
```

### Factor map
`recast_to_ssystem` returns a `factor_map` so you can reconstruct original variables as products of auxiliary variables during simulation/analysis.

---

## Testing harness

The harness consumes a **manifest** (plain text, one path to a `.ant` file per line), recasts each model, writes the recast `.ant`, and generates a **Jupyter notebook** report.

### Manifest format
```
/absolute/path/to/model1.ant
/absolute/path/to/model2.ant
# Lines starting with # are ignored
```

### Run
```bash
python /mnt/data/ode_to_ssys/harness.py   --manifest /path/to/manifest.txt   --outdir   /path/to/outdir   --module   /mnt/data/ode_to_ssys/ssys_recaster.py
```

### Outputs
- Recast `.ant` files in `--outdir`
- `recast_report.ipynb` in `--outdir`  
  Each test section shows:
  - Original Antimony and recast Antimony (as code blocks)
  - Original ODEs (LaTeX) and S‑System ODEs (LaTeX)
  - Two plots: original system trajectories, and **reconstructed** original trajectories obtained by multiplying the auxiliary S‑system states according to the factor map

> Plots are rendered with one figure per panel (no subplots) for clarity.

---

## Algorithm sketch

1. Parse Antimony → intermediate representation (species, parameters, reactions, explicit `S' = …` rules, inits).  
2. Build SymPy ODEs from reactions + rate rules.  
3. Expand each RHS into a sum of monomials (products of powers).  
4. Iterative sum‑splitting: introduce auxiliary variables to express sums through products so that each derivative is a single power‑law product (growth) minus a single product (decay).  
5. Initial conditions enforce the product constraints at \(t_0\).  
6. Emit Antimony containing only auxiliary S‑system variables with canonical rate rules.

---

## Limitations / roadmap

- **Elementary functions** (`exp`, `log`, `sin`, `cos`, `tan`, `sqrt`): ✅ **Now supported!** (v0.3.0) Composite functions are automatically lifted into auxiliary variables using the chain rule, allowing models with transcendental functions to be recast into S-system form.
- **Rational functions**: ✅ **Now supported!** Expressions like `X/(Y+1)` are automatically lifted into auxiliary variables.
- **Positive orthant**: S‑systems assume positivity. Add a preprocessing translation for variables that can cross zero (paper's Step B).  
- **Stiff systems**: built‑in RK4 in the notebook is fine for moderate problems. If stiffness appears, plug in SciPy's adaptive solvers.  
- **Antimony grammar**: extend to modules, reversible shorthands, and assignment rules with non‑numeric RHS.

---

## Troubleshooting

- “Unexpected symbol” during parsing: likely an unsupported Antimony construct (module, `:=`, piecewise). Simplify the model or extend the parser.  
- Flat lines in reconstructed plots: check initial conditions—product constraints require correct initial values of auxiliary factors.  
- Numerical blow‑ups: exponents and parameter values can create large growth. Reduce `tmax`, or switch to an adaptive solver.

---

## References

- Savageau, M. A., & Voit, E. O. (1987). Recasting nonlinear differential equations as S‑systems: a canonical nonlinear form.  
- Sauro, H. M., et al. Antimony language papers.
