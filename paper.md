---
title: 'ssys: Exact recasting of ODE models into canonical S-system form with reproducible reports'
tags:
  - systems biology
  - dynamical systems
  - symbolic computation
  - Antimony
  - model transformation
authors:
  - name: Your Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Your Institution, Department, City, Country
    index: 1
date: 2025-11-18
bibliography: paper.bib
---

# Summary

`ssys` is a small, focused toolkit that **recasts** ordinary differential equation (ODE) models written in **Antimony** into **canonical S-system** form in a mathematically exact way. For each state variable \(X_i\) with right-hand side \(f_i(X)\) decomposed into a finite sum of signed monomials \(f_i=\sum_j s_{ij}\), the tool constructs auxiliary “pool” variables \(\{V_{ij}\}\) such that \(X_i=\prod_j V_{ij}\) and
\\[\dot V_{ij} = \pm \alpha_{ij}\, \frac{\prod_k X_k^{g_{ijk}}}{\prod_{\ell\neq j} V_{i\ell}} ,\\]
which guarantees \( \frac{d}{dt}\prod_j V_{ij}=\sum_j s_{ij}=f_i(X) \) identically. The recast is therefore **exact**, not an approximation. The software emits (i) an Antimony file for the S-system, (ii) a stable **mapping** from original states to the product of auxiliaries, and (iii) a Jupyter report that prints the original and recast models in LaTeX, integrates both systems, and verifies an algebraic residual \(\|\dot X_{\text{recast}}-\dot X_{\text{orig}}\|\) at random test points.

Two use-cases motivate releasing this as a stand-alone, tested tool. **(A)** It provides **ground truth** S-systems to **evaluate structure-learning methods** that target power-law kinetics (e.g., generalized SINDy or Bayesian symbolic learners). **(B)** It exposes a **normal form** in which **steady states reduce to linear algebra in log-space**, enabling rank-based **identifiability diagnostics** for exponents and rate ratios. The package is lightweight (pure Python with SymPy/NumPy/Matplotlib), scriptable (CLI + manifest harness), and designed to be embedded in other workflows.

# Statement of need

Power-law (“S-system”) representations have long been used in biochemical systems analysis [@Savageau1969; @Voit2013]. Yet, researchers lack an **open, maintained, and reproducible** path to take an ODE model—as it is commonly authored today in **Antimony**—and obtain an **exact** S-system together with a **provable mapping** back to the original variables. Existing simulators or legacy codes either approximate the mapping, require bespoke formats, or are not readily scriptable for batch testing and benchmarking. `ssys` fills that gap:

- **Exactness by construction.** The pool-auxiliary factorization guarantees algebraic equivalence.
- **Reproducible reporting.** A harness consumes a manifest of Antimony files, produces recast models and a JupyterLab report with side-by-side integrations and LaTeX renderings.
- **Instrumentation for methods research.** The emitted mapping and a built-in residual check make it straightforward to create **ground-truth test suites** for learners of S-system structure and parameters (e.g., [@Daniels2015]).
- **Leverage of the normal form.** In log-space, steady-state equations are linear, enabling rank tests for parameter/elasticity identifiability without resorting to heavy nonlinear profiling.

# State of the field

Antimony is a widely used, human-readable modeling language for systems biology [@Smith2009Antimony], with transparent interop to SBML [@Hucka2003]. The S-system formalism is classical [@Savageau1969; @Voit2013], but modern, general-purpose **recasting tools** are scarce. We are not aware of an open-source package that (i) ingests arbitrary Antimony ODEs, (ii) returns an **exact** S-system, and (iii) emits the explicit **state-mapping** and a **residual** certificate. `ssys` aims to be that missing bridge.

# Functionality

- **Parser & IR.** A minimal Antimony parser builds a symbolic IR (variables, parameters, reactions/rules, explicit \(X'=\cdot\) rate rules). Parameters and initials are carried forward.
- **Exact recast.** Each \(f_i\) is expanded into signed monomials; per term \(s_{ij}\) we create one auxiliary \(V_{ij}\) and define \(\dot V_{ij}\) so that \(X_i=\prod_j V_{ij}\) and \( \dot X_i = \sum_j s_{ij}\) hold identically.
- **Canonical names.** Auxiliaries are renamed \(X_1,X_2,\dots\) in first-appearance order so reports are stable.
- **Mapping.** The emitted S-system Antimony file includes comments of the form `// X = X_3*X_7*X_8` and the notebook prints the mapping in LaTeX.
- **Reports.** The harness (`harness.py`) accepts a manifest of `.ant` files and produces `out/*.ant` plus a `recast_report.ipynb` with: files, LaTeX of original and recast ODEs, an algebraic residual, and **side-by-side** time-course plots (original vs reconstructed from auxiliaries).
- **Verification.** A residual function draws random positive test points, evaluates both \(\dot X\) and the reconstructed \(\sum_j \dot V_{ij}\prod_{\ell\ne j}V_{i\ell}\), and reports a relative max-norm error (typically ~1e-14–1e-12).

# Example

Original Antimony (logistic growth):
```antimony
model m2()
  X = 1
  r = 1.2
  K = 5
  X' = r*X*(1 - X/K)
end
```

Excerpt of recast (mapping and canonical S-system):
```antimony
model m2_recast()
  // Mapping from original variables to canonical auxiliaries (product form)
  // X = X_1*X_2
  X_1 = 1
  X_2 = 1
  r = 1.2
  K = 5
  X_1' = r*X_2 - 0
  X_2' = 0 - (r/K)*X_1*X_2^2
end
```

The report notebook integrates both systems and overlays \(X(t)\) (from the original) with \(X_1(t)X_2(t)\) (from the recast).

# Design and implementation

`ssys` is implemented in ~300 lines of pure Python on top of **SymPy** [@Meurer2017SymPy] for symbolic manipulation and **NumPy/Matplotlib/Jupyter** [@Harris2020NumPy; @Hunter2007Matplotlib; @Kluyver2016Jupyter] for numerical demonstration and reporting. The key routine `recast_to_ssystem` performs:

1. **Symbolic expansion** of each \(f_i\) to a sum of monomials.
2. **Term-wise auxiliaries** \(V_{ij}\) with exponents over original variables and negative exponents over all *other* auxiliaries in the same pool (never over \(V_{ij}\) itself), ensuring exact product-rule cancellation.
3. **Canonicalization** of auxiliary names and emission of mapping + initials (first auxiliary seeded to \(X_i(0)\), others to 1.0).

# Quality control

- Unit tests cover parsing, exactness of the recast (residual ≈ 0), mapping consistency, and numeric equivalence on a curated test suite (decay, logistic, Lotka–Volterra, branched pathway, Bertalanffy, SIR, van der Pol, etc.).
- The harness runs headlessly in CI and stores the notebook as an artifact.
- The algebraic residual is reported per test in the notebook to catch regressions.

# Availability and installation

`ssys` is a single-file module plus a harness script. It targets Python ≥3.9 and depends on SymPy/NumPy/Matplotlib/Jupyter.

```bash
pip install sympy numpy matplotlib jupyterlab
# clone your repo and run the harness
python harness.py --manifest tests/tests.manifest --outdir out --module ssys_recaster.py
```

# Usage

- **CLI/harness:** Provide a text manifest of `.ant` files (one per line). The harness writes recast `.ant` outputs and a `recast_report.ipynb` containing LaTeX, mapping, residual, and plots.
- **Library:** `import ssys_recaster as ss; rec = ss.recast_to_ssystem(sym)` where `sym` is built from `parse_antimony` → `build_sym_system`.

# Limitations and future work

- **Grammar coverage.** The current parser supports reactions, assignments, and explicit \(X'=\cdot\) rules; events and piecewise constructs are not yet implemented. Assignment rules could be inlined; events would require a lifted hybrid representation.
- **Elementary functions.** A planned “aux-lifting” pass will introduce positive auxiliaries for `exp`, `log`, `sin`, etc., and prove exactness under the lifting identities.
- **Stiffness.** The demo integrator is RK4 for clarity; we will add an optional stiff solver (e.g., BDF/LSODA via SciPy) in the notebook harness.
- **Positivity.** The recast assumes positive orthant states; we guard numerical evaluation with a small floor when needed. Documenting this explicitly with invariants would tighten guarantees.

# Acknowledgements

We thank colleagues and mentors in biochemical systems analysis for foundational ideas and feedback. Any errors are our own.

# References
