# Development Notes

Development notes for the ssys project. Deferred feature requests and enhancements are
tracked as issues in the remote repository.

---

## Current Status (v0.5.1)

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
