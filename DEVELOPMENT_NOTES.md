# Development Notes

This document consolidates development plans, design notes, and investigation records for the ssys project.

---

## Table of Contents

1. [Work Plans](#work-plans)
   - [GMA→S-System Condensation](#gma-to-s-system-condensation)
   - [ODE Solver Integration (Tellurium/RoadRunner)](#ode-solver-integration)
   - [Sign-Changing Function Transformations](#sign-changing-function-transformations)
2. [Design Notes](#design-notes)
   - [BioModels Fetching Pipeline](#biomodels-fetching-pipeline)
3. [Bug Fixes & Investigations](#bug-fixes--investigations)
   - [SymPy Sign Comparison Bug](#sympy-sign-comparison-bug)
   - [Validation Analysis: 0.0 Error Cases](#validation-analysis-00-error-cases)

---

# Work Plans

## GMA→S-System Condensation

**Status:** Planned

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

## ODE Solver Integration

**Status:** Partially Implemented (RoadRunner backend exists)

### Goal

Replace in-house RK4 with Tellurium + libRoadRunner while keeping Antimony as user-facing format.

### Architecture

```python
class OdeSimulator:
    def simulate(self, antimony: str, start: float, end: float, 
                 n_points: int, integrator="cvode", ...) -> SimulationResult

class RoadRunnerSimulator(OdeSimulator):
    # Primary implementation using Tellurium

class Rk4Simulator(OdeSimulator):
    # Optional fallback for debugging
```

### Configuration

```python
# config.py additions
DEFAULT_SOLVER = "roadrunner"
DEFAULT_INTEGRATOR = "cvode"
DEFAULT_REL_TOL = 1e-7
DEFAULT_ABS_TOL = 1e-9
DEFAULT_MAX_STEPS = 10000
```

### Key Implementation Notes

1. **Model loading:** `te.loada(antimony)` handles Antimony→SBML→RoadRunner internally
2. **Integrator config:** Set tolerances via `r.getIntegrator().setValue(...)`
3. **Error handling:** Custom exceptions (ModelLoadError, SimulationError)
4. **Batch processing:** Never crash whole run on single bad model

---

## Sign-Changing Function Transformations

**Status:** In Progress (Phase 1 Complete)

### Goal

Implement Savageau 1987 transformations `X = Z + c` for sign-changing functions.

### Background

For `cos(t)` which ranges [-1, 1]:
- Transform: `Z_cos = cos(t) + 2` → range [1, 3] (always positive)
- Similarly: `Z_sin = sin(t) + 2` → range [1, 3]

### Coupled Derivatives

```
Z_sin = sin(t) + 2  →  Z_sin' = cos(t) = Z_cos - 2
Z_cos = cos(t) + 2  →  Z_cos' = -sin(t) = 2 - Z_sin
```

### Implementation Phases

**Phase 1: Detection (✅ Complete)**
- `_requires_positivity_transform()` detects sin/cos → returns (True, 2.0)

**Phase 2: Offset Transformation (In Progress)**
- Replace `f(X)` with `(Z - offset)` in substitutions
- Generate coupled ODEs for sin/cos pairs
- Adjust initial conditions: `Z(0) = f(X(0)) + offset`

**Phase 3: Difference Expression Handling (Planned)**
- Handle expressions like `1 - X` that can be negative

---

# Design Notes

## BioModels Fetching Pipeline

### Pipeline Overview

1. **Discovery & Fetch:** Query BioModels for curated ODE models, download SBML
2. **Convert to Antimony:** `antimony.loadSBMLString()` → `getAntimonyString()`
3. **Structural Filters:** Reject models with events, delays, DAEs, sign-changing trig
4. **Tag Candidates:** S-system candidate vs GMA candidate
5. **Output:** CSV/JSON index with tags and blockers

### Filter Criteria

**KEEP (GMA):** rational/Hill/MM/Holling, mass-action, smooth algebraic rules

**DROP EARLY:**
- Events, delays, DAEs
- Piecewise-heavy switching, min/max
- Sign-changing transcendentals (sin, cos, tanh)

**S-SYSTEM CANDIDATE only if:**
- Each state has ≤1 production channel AND ≤1 loss channel
- Every kinetic law is a single monomial

### Key Helper Functions

```python
def rough_antimony_filters(ant: str) -> Dict[str, bool]:
    """Detect events, delays, piecewise, transcendentals, explicit time"""

def ssystem_candidate_heuristic(ant: str) -> bool:
    """Each species in ≤2 reactions, all rates monomial-like"""

def gma_candidate_heuristic(ant: str) -> bool:
    """No events/delays - rationals OK"""
```

---

# Bug Fixes & Investigations

## SymPy Sign Comparison Bug

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
