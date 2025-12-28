# BioModels Benchmark Analysis TODO

## Current Status
- **Total models analyzed**: 937
- **Recast success**: 788 (84.1%)
- **Timeouts**: 102 (at 15s)
- **Other errors**: 47

---

## 1. Recast Type Classification

Count from among successful validated recastings:

- [ ] **1.1** General → GMA recastings
- [ ] **1.2** General → S-system recastings
- [ ] **1.3** GMA → GMA (no change in form, already power-law)
- [ ] **1.4** GMA → S-system recastings
- [ ] **1.5** Aborted General → S-system (attempted but couldn't achieve)
- [ ] **1.6** Aborted GMA → S-system (attempted but couldn't achieve)

### Implementation Notes
- Need to parse recast output to determine result type
- Check `result.status` or output structure from `recast_to_ssystem()`
- May need to run both `simplified` and `canonical` modes to detect differences

---

## 2. Size Distribution Analysis

- [ ] **2.1** Size distribution of models that timed out
  - Species count, reaction count, parameter count
  - Histogram/density plot
  
- [ ] **2.2** Size distribution of successfully recast models
  - Same metrics
  - Histogram/density plot
  
- [ ] **2.3** Statistical comparison of distributions
  - KL divergence (Kullback-Leibler)
  - Kolmogorov-Smirnov test
  - Mann-Whitney U test
  - Effect size (Cohen's d)

### Implementation Notes
- Merge `batch_recast_results.csv` with `candidates.csv` for size info
- Use `scipy.stats` for statistical tests
- Use `scipy.special.rel_entr` for KL divergence

---

## 3. Numerical Validation

- [ ] **3.1** Trajectory comparison code
  - Integrate original SBML with libRoadRunner
  - Integrate recast Antimony with libRoadRunner
  - Compute trajectory error metrics (RMSE, max error, relative error)
  
- [ ] **3.2** Numerical instability detection
  - Monitor for NaN/Inf during integration
  - Detect stiffness indicators
  - Track solver failures and step rejections
  
- [ ] **3.3** Condition number analysis
  - Compute condition number of Jacobian at initial conditions
  - Compute condition number along trajectories
  - Identify ill-conditioned recastings

### Implementation Notes
- Use `ssys.notebook_helpers` or extend with new validation functions
- Need to handle parameter extraction from SBML
- Consider multiple time scales

---

## 4. Ideas to Improve Coverage

- [ ] **4.1** Increase timeout for complex models (60s, 120s)
- [ ] **4.2** Handle symbolic parameter inequalities
  - The `TypeError: cannot determine truth value of Relational` errors
  - Add symbolic parameter bounds/assumptions
- [ ] **4.3** Support for additional functions
  - Currently: sin, cos, exp, log
  - Missing: tan, tanh, sinh, cosh, sqrt with symbolic arguments
- [ ] **4.4** Piecewise function approximation
  - Smooth approximations (sigmoid, tanh)
- [ ] **4.5** Better algebraic constraint handling
- [ ] **4.6** Partial recasting
  - Recast what's possible, leave problematic terms
- [ ] **4.7** Profile slow models to find bottlenecks
  - Symbolic simplification overhead?
  - Large expression manipulation?

---

## 5. Manuscript Analyses

### 5.1 Conservation Law Preservation
- [ ] Detect moiety conservation laws in original model
- [ ] Verify conservation laws preserved in recast
- [ ] Report: how many models with conservation laws correctly handled?

### 5.2 Steady-State Analysis
- [ ] Compare steady states: original vs recast
- [ ] Compute Jacobian eigenvalues at steady states
- [ ] Verify stability classification preserved (stable/unstable)

### 5.3 Structural Properties
- [ ] Sparsity comparison: S-system/GMA matrices vs original ODE Jacobian
- [ ] Stoichiometric matrix rank preservation
- [ ] Deficiency analysis (CRNT metric)

### 5.4 Parameter Sensitivity
- [ ] Local sensitivity analysis comparison
- [ ] Does S-system form reveal clearer parameter dependencies?
- [ ] Sensitivity indices in original vs recast form

### 5.5 Complexity Metrics
- [ ] Term count: original vs recast
- [ ] Auxiliary variable overhead
- [ ] Effective dimension increase (n_original vs n_recast)
- [ ] Expression complexity (tree depth, operation count)

### 5.6 Failure Mode Taxonomy
- [ ] Categorize failure reasons:
  - Symbolic parameters in exponents
  - Unsupported functions (tan, tanh)
  - Timeout (complexity)
  - Parse errors (SBML issues)
  - Algebraic constraints
- [ ] Feature importance: which model features predict failure?
- [ ] Decision tree/random forest classifier

---

## 6. Code to Implement

### 6.1 `5_classify_recasts.py`
```python
# Count recast types from successful runs
# Parse output .ant files or re-run with diagnostic mode
```

### 6.2 `6_size_distribution.py`
```python
# Compare size distributions of timeout vs success
# Statistical tests and visualizations
```

### 6.3 `7_trajectory_validation.py`
```python
# Numerical integration comparison
# Instability detection
# Condition number analysis
```

### 6.4 `8_structural_analysis.py`
```python
# Conservation laws
# Steady states
# Jacobian analysis
```

---

## Priority Order

1. **High**: Recast type classification (1.1-1.6)
2. **High**: Trajectory validation (3.1-3.2)
3. **Medium**: Size distribution analysis (2.1-2.3)
4. **Medium**: Condition number analysis (3.3)
5. **Medium**: Conservation law preservation (5.1)
6. **Low**: Steady-state analysis (5.2)
7. **Low**: Complexity metrics (5.5)
8. **Low**: Failure mode taxonomy (5.6)

---

## Data Files Needed

- `results/batch_recast_results.csv` - Current recast results
- `results/candidates.csv` - Model metadata (size, features)
- `results/recasts/*.ant` - Successful recast outputs
- `data/sbml_candidates/*.xml` - Original SBML files

---

## Notes

- Current timeout: 15s (may need 60s for ~100 models)
- 47 TypeError failures mostly from symbolic parameter comparisons
- 788 successful recasts available for analysis
