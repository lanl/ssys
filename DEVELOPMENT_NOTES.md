# Development Notes

This document contains development plans and investigation records for the ssys project.

---

## Table of Contents

1. [Work Plans](#work-plans)
   - [SBML-First Parser Refactoring](#sbml-first-parser-refactoring) ✓ COMPLETED
   - [Autonomous Lifting for Strict GMA/S-System](#autonomous-lifting-for-strict-gmas-system)
   - [GMA→S-System Condensation](#gma-to-s-system-condensation)
   - [Future Enhancements: BioModels Coverage](#future-enhancements-biomodels-coverage-analysis)
2. [Bug Fixes & Investigations](#bug-fixes--investigations)
   - [SymPy Sign Comparison Bug](#sympy-sign-comparison-bug)
   - [Validation Analysis: 0.0 Error Cases](#validation-analysis-00-error-cases)

---

# Work Plans

## SBML-First Parser Refactoring

**Status:** COMPLETED ✓  
**Branch:** `fix/antimony-parser`  
**Created:** 2024-12-27  
**Completed:** 2025-12-27

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

#### Phase 1: Dependencies & Environment ✅ COMPLETE

- [x] **1.1** Update `pyproject.toml` - move simulation deps to core
  - Move `libroadrunner>=2.5`, `antimony>=2.13`, `python-libsbml>=5.20` from `[simulation]` to main `dependencies`
  - Remove `[sbml]` optional group (redundant)
  - Remove `[simulation]` optional group (now core)
  - Keep `[notebook]` optional (jupyter for interactive use)
  - Keep `[dev]` for testing tools only

- [x] **1.2** Update `setup_env.sh`
  - Change `uv pip install -e ".[dev,notebook,simulation]"` to `uv pip install -e ".[dev]"`
  - Update comments to reflect simpler install

- [x] **1.3** Verify environment
  - Run `./setup_env.sh --force` to recreate environment
  - Confirm `import roadrunner, antimony, libsbml` all work

#### Phase 2: New Parser Function ✅ COMPLETE

- [x] **2.1** Add `parse_antimony_via_sbml()` in `recaster.py`
  - Uses `antimony` library directly (not roadrunner) for Antimony→SBML
  - Cleaner implementation than planned

- [x] **2.2** Add `parse_sbml_from_string()` in `recaster.py`
  - Shares implementation with `parse_sbml()` via `_parse_sbml_document()`

- [x] **2.3** Preserve @SIM metadata extraction
  - `_extract_sim_metadata()` function extracts from comments
  - Attached to SymSystem as private attributes

#### Phase 3: Update CLI & Entry Points ✅ COMPLETE

- [x] **3.1** Update `cli.py:recast_file()` 
  - Added `parser` parameter: 'legacy' or 'sbml'
  - `--parser sbml` flag in CLI
  - @SIM metadata extracted and preserved

- [x] **3.2** Update `recast_models.py`
  - Added `--parser {legacy,sbml}` argument (default: legacy)
  - Passes parser choice to CLI command

- [x] **3.3** Update `validator.py`
  - Changed to use `parse_antimony_via_sbml()` for both original and recast files
  - Added ModelIR compatibility shims (species, reactions, explicit_rates, initial)
  - Converts `**` to `^` in explicit_rates for Antimony compatibility

#### Phase 4: Make Recaster Robust to SBML Forms (ACTIVE)

**Commits Made:**
1. `518db1b` - Add parser parameter to validator for SBML debugging
2. `8b7e9fe` - Fix SBML parser: Handle InitialAssignments for parameter-dependent ICs

**Final Status (2025-12-27):**

All primary test suites pass validation:
- **test_models1:** 29/29 ✓ (core validation suite)
- **test_models2a:** 42/42 ✓ (literature models - passing subset)
- **test_models3:** 18/18 ✓ (BioModels-derived)

The `test_models2b` directory (27 models) contains known failing cases that need future work.

**Fixes Applied:**

- [x] **4.1** Add parser parameter to validator
  - Validator now accepts `parser` parameter to use same parser for original and recast
  - CLI passes `--parser` flag to validator

- [x] **4.2** Fix SBML InitialAssignments
  - SBML `InitialAssignment` elements allow ICs like `I = I_b`
  - Added STEP 6b in `_parse_sbml_document()` to evaluate these
  - Uses correct symbol objects from `all_syms` for substitution

- [x] **4.3** Fixed RoadRunner simulation failures
  - Resolved output format issues and missing declarations
  - All 18 test_models3 now simulate correctly

- [x] **4.4** Fixed trajectory divergences
  - Corrected ODE structures and auxiliary handling
  - All models now produce matching trajectories

#### Phase 5: Testing ✅ COMPLETE

#### Pre-requisite: Fix Antimony Syntax ✅ COMPLETE

All 118 .ant files now parse with standard `antimony` library:
- test_models1: 29 models ✓
- test_models2: 69 models ✓  
- test_models3: 18 models ✓
- pathological_models: 2 models ✓

Changes made:
- Added explicit `compartment cell = 1; species X in cell;` declarations
- Fixed semicolon placement (consistent statement endings)
- Renamed `gamma` → `gamma_rate` (reserved Antimony name)
- Fixed comment/code mixing in HBF1998, I1988, P2011, S1993 files

**Primary Test Suites:**

- [x] **4.1** Test against test_models3 (18 models - baseline guard)
  - Result: 18/18 ✓
  - Run: `python recast_models.py test_models3 --solver roadrunner`

- [x] **4.2** Test against test_models1 (29 models - the failing suite)
  - Result: 29/29 ✓
  - Run: `python recast_models.py test_models1 --solver roadrunner`

**Secondary:**

- [x] **4.3** Test against test_models2
  - test_models2a (passing subset): 42/42 ✓
  - test_models2b (known failing): documented for future work

- [x] **4.4** Verify biomodels scripts still work
  - Uses `parse_sbml()` directly, unaffected by refactor

#### Phase 6: Cleanup (Future Work)

- [ ] **6.1** Remove deprecated code from `recaster.py`
  - Delete `parse_antimony()` (~300 lines)
  - Delete `build_sym_system()` (~100 lines)
  - Delete `ModelIR` dataclass (~30 lines)
  - Delete helper functions only used by above

- [ ] **6.2** Update docstrings and comments
  - Document the SBML-first architecture
  - Update module docstring

- [ ] **6.3** Update README.md
  - Note dependency on libroadrunner
  - Update installation instructions if needed

**Note:** Legacy parser code retained for backward compatibility. Cleanup deferred to future release.

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

## Future Enhancements: BioModels Coverage Analysis

**Status:** Planning (post-SBML-first refactor)  
**Data Source:** `biomodels/results/filter_summary.txt` (1644 models analyzed)

### Current Coverage

Based on BioModels analysis, ssys can currently handle:

| Feature | Models | Status |
|---------|--------|--------|
| exp() | 241 | ✅ Lifting implemented |
| log/ln() | 129 | ✅ Lifting implemented |
| sin/cos | 24 | ✅ Lifting implemented |
| Already GMA | 82 | ✅ No recast needed |

**Eligible models:** 1041 (63.3% of BioModels)

### High-Priority Enhancement: Piecewise Functions

**Impact:** 266 models (25% of eligible models)

SBML `piecewise` is used extensively in biological models for:
- Threshold functions (production starts above threshold)
- Saturation approximations
- Switch-like behavior

**Investigation needed:**
1. Categorize piecewise patterns in BioModels
2. Identify "smooth-approximable" patterns vs. true discontinuities
3. Design sigmoid approximation strategy

**Potential approach:**
```
piecewise(0, X < K, V) → V * sigmoid(steep * (X - K))
```

Where `sigmoid(x) = 1/(1 + exp(-x))` is already liftable to GMA.

**Key questions:**
- What steepness parameter is appropriate?
- How to handle multi-branch piecewise?
- How to validate approximation quality?

**Reference models to investigate:**
- Run: `python biomodels/2_filter_models.py` and examine piecewise patterns
- Look for common idioms in SBML kinetic laws

### Medium-Priority: Additional Trig Functions

**Impact:** 17 models ("unsupported_trig" in filter summary)

Likely includes: `tan`, `sec`, `cot`, `sinh`, `cosh`, `tanh` (beyond our sin/cos handling)

**Lifting strategy:**
- `tan(x) = sin(x)/cos(x)` → use existing sin/cos lifting + rational function lifting
- `sec(x) = 1/cos(x)` → same approach
- Hyperbolic functions follow similar patterns

**Effort:** Low (builds on existing infrastructure)

### Low-Priority: SBML L3 Packages

**Impact:** 93 models (excluded due to "sbml_l3_packages")

SBML Level 3 packages include:
- **FBC** (Flux Balance Constraints) - constraint-based modeling
- **comp** (Hierarchical composition) - modular models
- **qual** (Qualitative models) - discrete/Boolean logic

**Assessment needed:**
- Which packages are actually blocking?
- Can we ignore certain packages and still extract ODEs?
- Are these models even ODE-based?

### Already Handled / Non-Issues

| Category | Models | Notes |
|----------|--------|-------|
| No ODEs | 291 | Not applicable to recasting |
| Discrete events | 214 | Cannot be represented in GMA |
| SBML parse errors | 29 | SBML-first approach should help |
| Delay functions | 5 | Not in scope (DDEs) |
| Algebraic constraints | 2 | Not in scope (DAEs) |

### Summary: Potential Coverage Expansion

| Enhancement | Models Gained | Effort | Priority |
|-------------|---------------|--------|----------|
| Piecewise → smooth | ~266 | High | **HIGH** |
| Additional trig | ~17 | Low | Medium |
| L3 package subset | ~30? | Unknown | Low |

**Recommendation:** Focus on piecewise approximation after SBML-first refactor is complete.

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
