# Development Notes

This document contains development plans and investigation records for the ssys project.

---

## Table of Contents

1. [Work Plans](#work-plans)
   - [GMA→S-System Condensation](#gma-to-s-system-condensation)
2. [Bug Fixes & Investigations](#bug-fixes--investigations)
   - [SymPy Sign Comparison Bug](#sympy-sign-comparison-bug)
   - [Validation Analysis: 0.0 Error Cases](#validation-analysis-00-error-cases)

---

# Work Plans

## GMA→S-System Condensation

**Status:** Planned (future work)

### Objective

Implement BST-style "condensation" to convert exact GMA models into approximate S-systems by matching log-derivatives at a steady-state reference point.

### Background

**GMA (Generalized Mass Action):**
```
dX_i/dt = P_i(X) − Q_i(X)
P_i(X) = Σ_k a_ik * Π_j X_j^{b_ijk}
Q_i(X) = Σ_ℓ a_iℓ * Π_j X_j^{c_ijℓ}
```

**S-System (special case):**
```
dX_i/dt = α_i * Π_j X_j^{g_ij} − β_i * Π_j X_j^{h_ij}
```
One production monomial, one degradation monomial per equation.

**Condensation formula:**
- At reference state X*, compute weights: `w_ik = M_ik(X*) / Σ_k M_ik(X*)`
- Condensed exponents: `g_ij = Σ_k w_ik * b_ijk`
- Condensed coefficient: `α_i = P_i(X*) / Π_j (X*_j)^{g_ij}`

### Implementation Steps

1. **Extend RecastResult** with `is_condensed: bool` and `ref_state: Dict`
2. **Build numeric GMA RHS** from GMAEquation structures
3. **Steady-state solver** using log-space root-finding (ensures positivity)
4. **Condensation core** - per-equation production/degradation condensation
5. **Antimony emission** with comments noting approximate nature
6. **Validation hooks** - compare GMA vs condensed S-system at X*

### Key Functions to Implement

```python
def find_steady_state_gma(result: RecastResult, tol=1e-10) -> Dict[Symbol, float]
def condense_gma_to_ssystem(gma_result, x_ref, exp_tol=1e-8) -> RecastResult
```

---

# Bug Fixes & Investigations

## SymPy Sign Comparison Bug

**Status:** Fixed

### Problem

Error in `_direct_ssystem_recast`:
```
TypeError: cannot determine truth value of Relational: -sign(sin(t)) >= 0
```

### Root Cause

```python
if sp.sign(coeff) >= 0:  # WRONG - coeff may be symbolic!
```

When `coeff = -sign(sin(t))`, `sp.sign(coeff)` returns a symbolic `Sign(...)` object, not -1/0/+1. Python's `if` cannot evaluate a symbolic inequality.

### Solution

Use only the numeric part of the coefficient:
```python
numeric_coeff, rest = coeff.as_coeff_Mul()
if numeric_coeff >= 0:  # numeric_coeff is a plain number
    growth_terms.append((coeff, exps))
else:
    decay_terms.append((sp.Abs(coeff), exps))
```

Or reuse existing `_get_coefficient_sign()` / `_analyze_ode_terms()` which already handle this correctly.

---

## Validation Analysis: 0.0 Error Cases

**Status:** Investigated (not a bug)

### Summary

Models showing `max_error: 0.0` fall into three categories:

**Category 1: Identity Mappings (Legitimate)**
- Model 14: `X_1 = X_1` with auxiliary `Y_1 := K_2 + X_1`
- Model 17: `f = f, t = t` with auxiliary

**Category 2: Simple Renamings (Legitimate)**
- Model 1: `S → Z_1`
- Model 9: `A→Z_1, B→Z_2, C→Z_3`
- Model 16: `f→Z_1, t→Z_2`

**Category 3: GMA with Auxiliaries (Verified Legitimate)**
- Model 11: GMA with `Y_1 := X2 + 1`
- Model 23: Large GMA system (11 variables)
- Model 28: GMA with auxiliaries for `Z_1² + Z_2²`

### Conclusion

The 0.0 errors are NOT bugs - they indicate mathematical equivalence:
- Auxiliary computation IS working correctly (verified)
- When Jacobian is identity/permutation, dynamics are identical
- GMA systems with exact auxiliary constraints CAN have 0.0 error

---

# References

- Savageau & Voit 1987: "Recasting Nonlinear Differential Equations as S-Systems"
- Marin-Sanguino et al. 2007: "Optimization of Biotechnological Systems through Geometric Programming"
- Voit 1988: "New Nonlinear Methodologies for Modeling Molecular and Cellular Systems"
