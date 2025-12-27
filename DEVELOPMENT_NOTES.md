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

## Autonomous Lifting for Strict GMA/S-System

**Status:** COMPLETED (Clock Approach)

### Objective

Transform nonautonomous ODE systems (with explicit `time` dependence) into equivalent autonomous systems by lifting time-dependent functions to state variables with their own ODEs.

### Problem Statement

Current recaster produces output like:
```antimony
Z_1 := cos(pi*time/15) + 2.0;  // Assignment rule - NOT GMA!
```

This is **not strict GMA** because:
1. GMA means sums of power-law monomials in **state variables only**
2. Assignment rules depending on `time` make the system nonautonomous
3. The RHS still contains transcendental functions (cos, tanh, sqrt)

### Solution Implemented: Clock Approach

For time-dependent models, we use a **clock state variable** approach:

1. **Add clock state**: T' = 1, T(0) = 0
2. **Substitute time → T** in all ODEs and assignment rules
3. **Keep assignment rules as-is** (not expanded to ODEs)

This produces output classified as **"GMA with time-varying coefficients"**:
- ODE RHS has power-law structure in state variables
- Coefficients are functions of clock state T via assignment rules
- System is autonomous (no explicit `time` in ODEs)

**Example (Weber2018):**
```
Input: S_1' = -k_12*S_1 + k_21*S_2 where k_23 := f(time)
Output:
  T' = 1;  // Clock
  S_1' = -k_12*S_1 + k_21*S_2;  // Same structure
  k_23 := f(T);  // Assignment rule uses clock T
```

**Classification hierarchy:**
1. `CANONICAL_SSYSTEM` - exactly 2 terms per equation, constant coefficients
2. `SSYSTEM` - 1-2 terms per equation, constant coefficients
3. `GMA` - multiple monomials, constant coefficients
4. `GMA_TIME_VARYING` - power-law structure, time-varying coefficients via assignment rules
5. `GENERAL` - non-power-law terms

### Alternative: Pure GMA Lifting (Deferred)

The original approach of lifting time-functions to ODEs:

### Phase 1: Pattern-Based Lifting

#### 1.1 Exponential Decay: `exp(-k*time)`
```
f(t) = exp(-k*t)
→ E' = -k*E      (GMA: one decay term)
  E(0) = 1
```

#### 1.2 Harmonic Oscillator: `cos(ω*time)`, `sin(ω*time)`
```
f(t) = cos(ωt), g(t) = sin(ωt)
→ c' = -ω*s     (GMA: one decay term)
  s' = ω*c      (GMA: one growth term)
  c(0) = 1, s(0) = 0
```

For `cos(ωt + φ)`: use `c(0) = cos(φ)`, `s(0) = sin(φ)`

#### 1.3 Logistic Sigmoid: `tanh(k*(time - a))`

The sigmoid function `σ(x) = 1/(1 + exp(-x))` satisfies:
```
σ'(x) = σ(x)*(1 - σ(x))
```

For `h(t) = 1/(1 + exp(-2k*(t-a)))`:
```
→ h' = 2k*h - 2k*h²   (GMA: growth - decay)
  h(0) = 1/(1 + exp(2k*a))
```

Connection to tanh:
```
tanh(k*(t-a)) = 2*sigmoid(2k*(t-a)) - 1 = 2*h - 1
```

So: `H_on = 0.5*(1 + tanh(k*(t-5)))` becomes simply `h` after lifting.

### Phase 2: Squared Variable Lifting for sqrt

For smooth ReLU approximations like:
```
k23 = 0.5*(raw + sqrt(raw² + ε²))
```

Lift `u = raw² + ε²` with ODE:
```
u' = 2*raw*raw'   (chain rule)
u(0) = raw(0)² + ε²
```

Then use `u^(0.5)` which is a GMA monomial.

### Implementation Functions

```python
def lift_exp_decay(expr: sp.Expr, state_vars: Set[sp.Symbol]) 
    -> Optional[Tuple[sp.Symbol, sp.Expr, float]]
    """Detect exp(-k*time) → (E, -k*E, 1.0)"""

def lift_harmonic(expr: sp.Expr, state_vars: Set[sp.Symbol])
    -> Optional[Tuple[sp.Symbol, sp.Symbol, sp.Expr, sp.Expr, float, float]]
    """Detect cos/sin(ω*time) → (c, s, -ω*s, ω*c, c0, s0)"""

def lift_logistic(expr: sp.Expr, state_vars: Set[sp.Symbol])
    -> Optional[Tuple[sp.Symbol, sp.Expr, float]]
    """Detect tanh(k*(time±a)) → (h, 2k*h - 2k*h², h0)"""

def lift_squared_for_sqrt(expr: sp.Expr, sym: SymSystem)
    -> Optional[Tuple[sp.Symbol, sp.Expr, float]]
    """Detect sqrt(X² + c) → (u, 2*X*X', u0)"""
```

### Test Cases

1. **Fink2000**: `exp(-k_0 * time)` → single exponential decay
2. **Weber2018**: 
   - `cos(2*pi*time/30)` → harmonic oscillator
   - `tanh(k_steep*(time - 5))` → logistic h
   - `tanh(k_steep*(70 - time))` → logistic w (complement)
   - `sqrt(raw² + eps_k²)` → squared lifting

### Future Work (Phase 4)

General nonautonomous → autonomous transformation:
- Given arbitrary `f(time)`, attempt to find ODE such that `y(t) = f(t)` is a solution
- Use SymPy's ODE matching capabilities
- Fallback to error/warning if no ODE found

### References

- Weber, Raymond, Munsky (2018): Model 2, Lambda_A
- Fink et al. (2000): "An image-based model of calcium waves"
- Savageau & Voit (1987): Recasting nonlinear differential equations

---

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
