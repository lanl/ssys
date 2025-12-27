# Development Notes

This document contains development plans and investigation records for the ssys project.

---

## Table of Contents

1. [Work Plans](#work-plans)
   - [SBML-First Parser Refactoring](#sbml-first-parser-refactoring) ← **ACTIVE**
   - [GMA→S-System Condensation](#gma-to-s-system-condensation)
   - [Autonomous Lifting for Strict GMA/S-System](#autonomous-lifting-for-strict-gmas-system)
2. [Bug Fixes & Investigations](#bug-fixes--investigations)
   - [SymPy Sign Comparison Bug](#sympy-sign-comparison-bug)
   - [Validation Analysis: 0.0 Error Cases](#validation-analysis-00-error-cases)

---

# Work Plans

## SBML-First Parser Refactoring

**Status:** ACTIVE  
**Branch:** `fix/antimony-parser`  
**Created:** 2024-12-27

### Problem Statement

The hand-rolled Antimony parser (`parse_antimony()`) is fragile and fails on valid Antimony syntax:

- **test_models3**: 18/18 passing (uses semicolons at line endings)
- **test_models1**: FAILING (uses `J_1: -> X_1; rate` format where semicolon is in middle)

The line-continuation logic assumes statements end with `;`, but reaction lines like:
```antimony
J_1: -> X_1; a - c*X_2
```
have semicolons in the MIDDLE (separating stoichiometry from rate law), not at the end.

**Root Cause:** We built our own parser instead of using the reference implementation.

### Solution: SBML-First Architecture

Replace the fragile hand-rolled parser with a pipeline using established tools:

```
Antimony text → RoadRunner (reference Antimony parser) → SBML → libSBML → SymSystem
```

**Key insight:** `parse_sbml()` already exists and works! The biomodels scripts use it directly.

### Implementation Checklist

#### Phase 1: Dependencies & Environment

- [ ] **1.1** Update `pyproject.toml` - move simulation deps to core
  - Move `libroadrunner>=2.5`, `antimony>=2.13`, `python-libsbml>=5.20` from `[simulation]` to main `dependencies`
  - Remove `[sbml]` optional group (redundant)
  - Remove `[simulation]` optional group (now core)
  - Keep `[notebook]` optional (jupyter for interactive use)
  - Keep `[dev]` for testing tools only

- [ ] **1.2** Update `setup_env.sh`
  - Change `uv pip install -e ".[dev,notebook,simulation]"` to `uv pip install -e ".[dev]"`
  - Update comments to reflect simpler install

- [ ] **1.3** Verify environment
  - Run `./setup_env.sh --force` to recreate environment
  - Confirm `import roadrunner, antimony, libsbml` all work

#### Phase 2: New Parser Function

- [ ] **2.1** Add `parse_antimony_via_sbml()` in `recaster.py`
  ```python
  def parse_antimony_via_sbml(antimony_text: str) -> SymSystem:
      """
      Parse Antimony text using RoadRunner's reference parser.
      
      Pipeline: Antimony → RoadRunner → SBML string → libSBML → SymSystem
      
      This replaces the fragile hand-rolled parse_antimony() + build_sym_system().
      """
      import roadrunner
      rr = roadrunner.RoadRunner()
      rr.loadAntimonyString(antimony_text)
      sbml_string = rr.getSBML()
      return parse_sbml_from_string(sbml_string)
  ```

- [ ] **2.2** Add `parse_sbml_from_string()` in `recaster.py`
  - Copy logic from existing `parse_sbml()` 
  - Accept string instead of file path
  - Use `libsbml.readSBMLFromString(sbml_string)` instead of `readSBML(path)`

- [ ] **2.3** Preserve @SIM metadata extraction
  - The @SIM metadata (T_START, T_END, N_STEPS) is in Antimony comments
  - RoadRunner/SBML won't preserve these
  - Extract before conversion and attach to SymSystem

#### Phase 3: Update CLI & Entry Points

- [ ] **3.1** Update `cli.py:recast_file()` 
  - Replace: `ir = parse_antimony(txt); sym = build_sym_system(ir)`
  - With: `sym = parse_antimony_via_sbml(txt)`
  - Extract @SIM metadata separately

- [ ] **3.2** Update `__init__.py` exports
  - Add `parse_antimony_via_sbml` to `__all__`
  - Keep `parse_antimony` temporarily for backwards compatibility (deprecate later)

- [ ] **3.3** Update `notebook_helpers.py` if needed
  - Check if it uses parse_antimony directly
  - Update to use new function

#### Phase 4: Testing

**Primary Test Suites (verified recastable - MUST ALL PASS):**

- [ ] **4.1** Test against test_models3 (18 models - baseline guard)
  - MUST still pass 18/18
  - If any regression, STOP and debug immediately
  - Run: `python recast_models.py test_models3 --solver roadrunner`

- [ ] **4.2** Test against test_models1 (29 models - the failing suite)
  - All models verified recastable with correct output
  - Should now work with reference Antimony parser
  - Run: `python recast_models.py test_models1 --solver roadrunner`
  - Target: 29/29 validated

**Secondary (NOT part of acceptance criteria):**

- [ ] **4.3** (OPTIONAL) Test against test_models2
  - **NOTE:** This directory contains literature examples that have NOT been systematically verified as recastable
  - Run if time permits, but failures here do NOT block the refactor
  - Document any successes for future reference

- [ ] **4.4** Verify biomodels scripts still work
  - They use `parse_sbml()` directly, should be unaffected
  - Quick sanity check: `python biomodels/3_recast_batch.py --limit 5`

#### Phase 5: Cleanup (After Tests Pass)

- [ ] **5.1** Remove deprecated code from `recaster.py`
  - Delete `parse_antimony()` (~300 lines)
  - Delete `build_sym_system()` (~100 lines)
  - Delete `ModelIR` dataclass (~30 lines)
  - Delete helper functions only used by above

- [ ] **5.2** Update docstrings and comments
  - Document the SBML-first architecture
  - Update module docstring

- [ ] **5.3** Update README.md
  - Note dependency on libroadrunner
  - Update installation instructions if needed

- [ ] **5.4** Final validation run
  - Primary test suites: test_models3 (18/18), test_models1 (29/29)
  - Run pytest unit tests
  - Verify notebook generation still works

### Rollback Procedure

If something goes wrong:
```bash
git checkout main -- src/ssys/recaster.py src/ssys/cli.py pyproject.toml
```

The old parser code can be restored instantly from git history.

### Benefits

1. **No more parser bugs** - Using reference Antimony parser
2. **All Antimony syntax supported** - Not just what we implemented
3. **~400 lines of code deleted** - Less maintenance burden
4. **Canonicalized SBML** - Clear semantics for reactions, rules, etc.
5. **Consistent with biomodels workflow** - Same pipeline

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| RoadRunner not installed | Error with clear message pointing to install |
| SBML conversion loses info | Extract @SIM metadata before conversion |
| Performance regression | Unlikely; RoadRunner is fast |
| Unit tests break | Fix tests to use new API |

### Dependencies After Refactor

**Core (required):**
```toml
dependencies = [
    "sympy>=1.12",
    "numpy>=1.24,<2",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "nbformat>=5.9",
    "libroadrunner>=2.5",
    "antimony>=2.13",
    "python-libsbml>=5.20",
]
```

**Dev (testing only):**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.7",
    "jupyter>=1.0",
]
```

---

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
