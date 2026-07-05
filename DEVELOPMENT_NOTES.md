# Development Notes

Development notes for the ssys project. Deferred feature requests and enhancements are
tracked as issues in the remote repository.

---

## Current Status (v0.5.4)

**Release Date:** 2025-12-30

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

## Progress: BioModels Batch Re-run (2025-12-30)

Work has begun on re-running the BioModels batch processing with full validation.

### Scripts Modified/Created

1. **`3_recast_batch.py`** - Updated to use real validation
   - Replaced stub validation with actual `validate_recast_pair()` calls
   - Switched from `tellurium` to `antimony` library for SBML→Antimony conversion
   - Added `--no-validate` flag for recast-only mode

2. **`3b_validate_batch.py`** - NEW: Separate validation script
   - Allows validation to run as a separate phase after recasting
   - Features:
     - `--symbolic-only` - Fast validation (no trajectory simulation)
     - `--timeout N` - Per-model timeout in seconds (signal-based)
     - `--resume` - Skip models already validated
     - Progress indicators with running totals (pass/fail/error counts)

3. **`3_recast_batch_old.py`** - Archive of previous version

### Validator API Changes

Updated `validate_recast_pair()` and `RecastValidator.validate()` to support partial validation:

```python
def validate_recast_pair(
    original_file, recast_file,
    run_symbolic=True,    # NEW
    run_numerical=True,   # NEW  
    run_trajectory=True,  # NEW
    ...
)
```

Fixed `overall_pass` logic: now only requires tests that were actually requested.
Previously required all three tests to pass even if some weren't run.

### Recommended Workflow

```bash
cd biomodels_batch

# Phase 1: Recast all models (fast, ~1 hour)
python 3_recast_batch.py --mode simplified --timeout 15 --no-validate
python 3_recast_batch.py --mode simplified --timeout 60 --resume --no-validate

# Phase 2: Validate separately
python 3b_validate_batch.py --mode simplified --symbolic-only --timeout 30
# Or for full validation (slower):
python 3b_validate_batch.py --mode simplified --timeout 60

# Phase 3: Analyze results
python 4_analyze_results.py
```

### Known Limitations

**Signal-based timeout cannot interrupt SymPy's C-level operations:**
- The timeout mechanism uses `signal.SIGALRM` which only triggers when Python
  regains control between operations
- SymPy's simplification routines run in C and can't be interrupted
- Some complex models hang indefinitely during symbolic simplification
- Workaround: Use `--symbolic-only` for fast first pass, identify problem models

**Potential solutions (not yet implemented):**
- Use `multiprocessing` with hard process termination
- Set internal sympy timeout (if available)
- Pre-filter models by complexity before validation

### Current Data

- Old results archived in `results_old_20251230/`
- 244 recast files already exist in `results/recasts/` (from earlier run)
- 937 candidate SBML files in `data/sbml_candidates/`

### JAX vs NumPy Numerical Validation (2025-12-30)

Comparison of numerical validation backends on 20 BioModels:

| Backend | Time | Pass | Fail | Error | Notes |
|---------|------|------|------|-------|-------|
| NumPy/SymPy | 97s | 4 | 10 | 6 | Immediate evaluation |
| JAX | 220s | 2 | 12 | 6 | JIT compilation overhead |

**Key findings:**

1. **JAX is slower for batch validation** - JIT compilation overhead dominates.
   Models 26 and 30 timed out with JAX but passed with NumPy.

2. **JIT compilation is the bottleneck** - Each model produces different expression
   trees, requiring separate JIT compilation. No opportunity to reuse compiled code.

3. **GPUs wouldn't help** for this workload:
   - XLA compilation runs on CPU (timeout occurs during compilation, not evaluation)
   - Problem sizes are too small (~10x10 matrices, 1000 sample points)
   - Serial workflow (one model at a time) - no batched parallelism
   - GPU kernel launch overhead would dominate

4. **Where JAX/GPU would help:**
   - Vectorizing validation across many models simultaneously
   - Large parameter sweeps (thousands of IC combinations)
   - Very large models (100+ variables)

**Recommendation:** Use NumPy backend (`--numerical-only` without `--use-jax`) for
batch validation. JAX is not beneficial for serial processing of diverse models.

**Silencing JAX TPU warning:**
```bash
export JAX_PLATFORMS=cpu
```

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

## Planned: BioModels Batch Re-run with Full Validation

A batch processing run against BioModels database models was completed with an earlier version
of ssys. The ssys codebase has changed significantly since then, so the batch run needs to be
repeated with the current version and full validation.

### Previous Results (reference only)

The previous run processed **937 filtered candidate models** from BioModels:
- **788 successful recasts** (84.1%)
- **149 failures** (102 timeouts, 47 other errors)
- Total time: ~52 minutes
- Average recast time: 1.87 seconds per model

**Important:** Previous validation was a stub (returned success without actual validation).

### Current Data

Data from the previous run is preserved in `biomodels_batch/`:
- `data/sbml_candidates/` - 937 filtered SBML files (ready to use)
- `results/` - Previous results (to be archived)

No re-filtering is needed; the candidate models were filtered based on SBML structure
(absence of events, delays, algebraic rules) which remains valid.

### Required Changes to `3_recast_batch.py`

The batch script needs to be updated to call the real validator instead of returning
a stub success response.

**Current stub code (to be replaced):**
```python
def validate_recast_wrapper(model_id: str, mode: str) -> dict | None:
    recast_path = Path(config.RECASTS_DIR) / f"{model_id}_{mode}.ant"
    if not recast_path.exists():
        return None
    return {
        "model_id": model_id,
        "mode": mode,
        "overall_pass": True,
        "note": "Validation skipped (SBML-based workflow)",
    }
```

**New code using real validation:**
```python
def validate_recast_wrapper(model_id: str, mode: str) -> dict | None:
    from ssys.validator import validate_recast_pair
    
    sbml_path = Path(config.SBML_CANDIDATES_DIR) / f"{model_id}.xml"
    recast_path = Path(config.RECASTS_DIR) / f"{model_id}_{mode}.ant"
    
    if not sbml_path.exists() or not recast_path.exists():
        return None
    
    try:
        report = validate_recast_pair(
            str(sbml_path),
            str(recast_path),
            mode=mode,
            parser="sbml"
        )
        return report.to_dict()
    except Exception as e:
        return {
            "model_id": model_id,
            "mode": mode,
            "overall_pass": False,
            "error": str(e),
        }
```

### Execution Plan

1. **Archive old results:**
   ```bash
   cd biomodels_batch
   mv results results_old_YYYYMMDD
   mkdir -p results/{recasts,validation,failures}
   ```

2. **Archive and update script:**
   ```bash
   cp 3_recast_batch.py 3_recast_batch_old.py
   # Edit 3_recast_batch.py to use real validation
   ```

3. **Test on small sample:**
   ```bash
   source ../activate_dev_env.sh
   python 3_recast_batch.py --limit 10 --timeout 15
   ```

4. **Full run - first pass (simplified mode only):**
   ```bash
   python 3_recast_batch.py --mode simplified --timeout 15
   ```

5. **Full run - second pass (retry timeouts with longer timeout):**
   ```bash
   python 3_recast_batch.py --mode simplified --timeout 60 --resume
   ```

6. **Generate analysis:**
   ```bash
   python 4_analyze_results.py
   ```

### Time Estimate

- First pass (15s timeout): ~1 hour estimated
- Second pass (60s timeout, timeouts only): depends on first pass results
- Per-model timeout: 15 seconds initially, 60 seconds for retries

### Expected Outcomes

- Updated success/failure statistics with current ssys version
- Full validation reports (symbolic, numerical, trajectory) for each successful recast
- Updated figures and summary statistics for the paper

---

## Issue Tracking

Notable deferred feature requests:

- **GMA→S-System Condensation** - BST-style aggregation for approximate S-systems
- **Piecewise Function Support** - Smooth sigmoid approximations for SBML piecewise
- **Handling Zero Initial Conditions** - Alternative approaches

See the project issue tracker at https://github.com/lanl/ssys/issues.

---

## References

- Savageau & Voit 1987: "Recasting Nonlinear Differential Equations as S-Systems"
- Marin-Sanguino et al. 2007: "Optimization of Biotechnological Systems"
- Voit 1988: "New Nonlinear Methodologies for Modeling"
