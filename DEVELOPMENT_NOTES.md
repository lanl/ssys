# Development Notes

Development notes for the ssys project. Deferred feature requests and enhancements are
tracked as issues in the remote repository.

---

## Current Status (v0.5.2)

**Release Date:** 2025-12-28

The ssys recaster is functional with the following capabilities:
- SBML-first parser architecture (Antimony → SBML → SymPy)
- Exact algebraic recasting to S-system/GMA form
- Composite function lifting (exp, log, sin, cos, etc.)
- Rational function lifting
- Nonautonomous → autonomous transformation via clock variable
- Three-test validation suite (symbolic, numerical, trajectory)

**Test Coverage:**
- test_models1: 29/29 ✓
- test_models2a: 42/42 ✓
- test_models3: 18/18 ✓
- 213 unit tests passing

---

## Planned: Bespoke Parser Removal

The hand-rolled Antimony parser (`parse_antimony()`, `build_sym_system()`, 
`ModelIR`) is retained for backward compatibility. Removal of this parser is planned for a near-term future release.

**Files affected:** `src/ssys/recaster.py`

**Lines to remove (estimated):**
- `parse_antimony()` (~300 lines)
- `build_sym_system()` (~100 lines)  
- `ModelIR` dataclass (~30 lines)
- Helper functions only used by above

---

## Planned: JAX Dependency Evaluation

The optional `[jax]` extra in `pyproject.toml` provides JAX-based numerical validation 
using automatic differentiation. This functionality appears largely redundant with the 
SymPy-based numerical validation that is already implemented.

**Investigation needed:**
- Profile JAX vs SymPy validation performance on representative models
- Identify any cases where JAX autodiff provides unique value (e.g., higher-dimensional models)
- Measure JIT compilation overhead vs. evaluation speed

**Current status:** JAX validation path is disabled by default due to observed slowdowns 
(JIT compilation overhead). SymPy numerical validation provides equivalent coverage.

**Files affected:**
- `src/ssys/validator.py` - Contains disabled JAX validation code
- `pyproject.toml` - `[jax]` optional dependency group

**Decision:** If investigation shows no compelling use case, remove JAX dependency entirely.

---

## Planned: Handling Zero Initial Conditions

This section documents strategies for recasting models where a state variable has a 
legitimate zero initial condition (e.g., Bergman minimal model).

### The Zero-IC Problem

S-system and GMA forms assume positive state variables because:
- **S-system** (strict BST): each ODE is a difference of two monomials with potentially 
  **negative exponents** (e.g., `Z^-1`) during lifting/factorization
- **GMA**: each ODE is a sum/difference of monomials

Negative powers and log-based reasoning require strict positivity. If an original model 
has a legitimate state with `x(0)=0` (or a state that can hit 0), a strict S-system 
recast with `x^-1` terms is not well-defined at that point.

### Approach 1: Strict S-system with ε-regularized ICs

**Idea:** Produce a strict S-system recast (BST S-system). If any recast variable appears 
with negative exponent (e.g., `Z^-1`), keep that variable positive by replacing zero ICs:
- `x(0)=0` → `x(0)=ε`
- similarly for any auxiliary variable that depends on that state

**Exactness:**
- **Not exact** as an IVP when the original IC is truly zero (solving a different IVP)
- If the original solution immediately leaves 0 (e.g., `x'(0)>0`), small ε gives a good 
  approximation for `t>0`, but error can be very sensitive near `t=0`
- If the original solution stays at 0 (invariant manifold), ε destroys that invariance

**Numerical behavior:**
- Can be stable if ε chosen sensibly and resulting system is not excessively stiff
- ε can introduce stiffness when it appears in denominators or negative powers

**How to choose ε:**
- Use **scale-aware ε**, not universal constant
- If variable has typical scale S, pick `ε ≈ 1e-12*S` to `1e-8*S` depending on tolerances
- Ensure ε is above solver's absolute tolerance scale

**Pros:**
- Produces pure S-system (useful if downstream tools assume S-system)
- Keeps everything in ODE form (no assignment rules/DAEs)

**Cons:**
- Not exact for true zero ICs
- Can introduce stiffness and extreme sensitivity
- Can break invariants and conservation laws that rely on exact zeros

### Approach 2: Softplus/Exponential Positivity Transform (Regularization)

**Idea:** Replace each positive-constrained variable `x` with a smooth positive function 
of an unconstrained variable `u`:
```
x = softplus(u) = log(1 + exp(u))
x = exp(u)
```
Then recast the transformed system.

**About zero ICs:**
- With softplus: represent values close to 0 by taking `u << 0`, but never exactly 0
- With exp: can never represent 0 at all
- Choose `u(0)` so that `x(0) ≈ ε`

**Exactness:**
- **Not an algebraically exact recast** of the original ODE in `x`
- Get a **different dynamical system** whose state is `u`, with observable `x(u)`
- For softplus, near zero you are smoothing/clipping behavior

**Numerical behavior:**
- Can prevent solver-induced negative values
- Can help with wide dynamic ranges (especially log/exp)
- Can create **new stiffness**: exponential maps amplify derivatives when `u` is large;
  softplus introduces regions with very small derivative (saturation)

**Pros:**
- Enforces positivity without hard clamps
- Can make some systems more numerically robust

**Cons:**
- Not exact relative to original model with true zeros
- Complicates interpretation and validation (must compare observables)
- Often breaks strict GMA/S-system structure unless lifted further

### Approach 3: GMA with Assignment Rules (Recommended)

**Idea:** Avoid forcing strict S-system structure where it causes singularities. Instead:
- Keep **ODEs** in **GMA** form (sum/difference of monomials where possible)
- Represent problematic composites (denominators, Hill functions, etc.) using 
  **assignment rules** rather than differential "lift" equations

**Example pattern:**
```
// Define helper algebraic expression
Z := 1 + x^2;

// Use in ODEs via assignment (do NOT integrate Z' as an ODE)
y' = k * Z * y;
```

**Exactness:**
- **Algebraically exact** as a recast of the original system (same IVP)
- Dynamics unchanged—only introduced named algebraic subexpressions

**Numerical behavior:**
- Usually **more stable** than differentiating algebraic constraints
- Avoids "drift off the constraint manifold" failure mode (McMillen-type issues)
- In Antimony/RoadRunner, assignment rules are handled natively

**Pros:**
- Can remain exact while avoiding singularities at zero
- Avoids manifold drift when lifted variable is really an algebraic constraint
- Simplifies validation: identical RHS when you substitute the rules

**Cons:**
- Not a strict S-system (BST) anymore
- Some downstream tooling may not support assignment rules

### Decision Table

| Goal | Best Approach | Rationale |
|------|---------------|-----------|
| Exact equivalence to original IVP | **3** | Assignment rules preserve algebra exactly |
| Must output strict BST S-system | **1** | Only way without DAEs; accept ε as approximation |
| Avoid manifold drift from lifted denominators | **3** | Keep constraints algebraic |
| Strong positivity guarantees for numerics | **2** or **3** | 2 enforces; 3 avoids singularities |

### Recommendation

For models where a state can be exactly zero and that zero is meaningful (e.g., Bergman):

1. **Prefer Approach 3** (GMA + assignment rules) for an **exact**, robust recast
2. Use Approach 1 only if you **must** deliver a strict S-system—then treat ε as part 
   of the model specification and **report it** as a modeling choice
3. Treat Approach 2 (softplus/log/exp) as an **approximation/regularization technique**, 
   not as an exact recast

---

## Issue Tracking

Notable deferred feature requests are tracked in the remote repository:

- **GMA→S-System Condensation** - BST-style aggregation for approximate S-systems
- **Piecewise Function Support** - Smooth sigmoid approximations for SBML piecewise

See: https://lisdi-git.lanl.gov/hlavacek/ssys/-/issues

---

## References

- Savageau & Voit 1987: "Recasting Nonlinear Differential Equations as S-Systems"
- Marin-Sanguino et al. 2007: "Optimization of Biotechnological Systems"
- Voit 1988: "New Nonlinear Methodologies for Modeling"
