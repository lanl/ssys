# BioModels Benchmark Suite

This directory contains tools for benchmarking the ssys recaster against real-world models from the BioModels database.

## Overview

The benchmark suite operates in four phases:

1. **Fetch**: Download models from BioModels (SBML → Antimony)
2. **Filter**: Apply heuristics to identify transformation candidates
3. **Transform**: Attempt S-system transformation and validate results
4. **Analyze**: Generate statistics and reports

## Terminology

- **Candidate**: Model that passed heuristic filters (no blockers like events/delays)
- **Transformation completed**: The algorithm produced output without crash/timeout
- **Validated**: Numerical or symbolic tests confirmed mathematical equivalence

**Important:** "Transformation completed" does NOT mean the output is correct!
The transformation algorithm can produce syntactically valid output that is
mathematically wrong. Only **validated** models are confirmed equivalent to the original.

## Setup

Install the benchmark dependencies using uv:

```bash
cd biomodels_batch
uv pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import bioservices; import antimony; import roadrunner; print('✓ Ready!')"
```

**Note**: The numpy<2.0 constraint in requirements.txt is critical for compatibility with libroadrunner.

## Usage

### Phase 1: Fetch Models

**Initial fetch** (1000 random models):
```bash
python 1_fetch_models.py --n 1000 --strategy random
```

**Expand** (add 2000 more, avoiding duplicates):
```bash
python 1_fetch_models.py --n 2000 --strategy random --mode expand
```

**Expand to target** (fetch until you have 5000 total):
```bash
python 1_fetch_models.py --target-total 5000 --strategy random
```

**Sequential filling**:
```bash
python 1_fetch_models.py --n 100 --strategy sequential --mode expand
```

### Phase 2: Filter Models

```bash
python 2_filter_models.py
```

Output: `results/candidates.csv` and `results/filter_summary.txt`

### Phase 3: Batch Transform

```bash
# Test with 10 S-system candidates
python 3_recast_batch.py --filter s_system --limit 10

# All candidates, simplified mode
python 3_recast_batch.py --mode simplified

# Skip validation for speed (transformation only)
python 3_recast_batch.py --no-validate
```

Output: `results/recasts/` (transformation output), `results/validation/`, `results/failures/`

### Phase 4: Analyze Results

```bash
python 4_analyze_results.py
```

Output: `results/summary.json`, `results/figures/`, `results/report.ipynb`

## Incremental Fetching

The fetch system is designed to be incremental and expandable:

- **No duplicates**: Tracks all fetched models in `data/fetch_history.json`
- **Resumable**: Can stop and restart without re-downloading
- **Expandable**: Run again with larger N to add more models
- **Auditable**: Complete history of fetch sessions

### Example Workflow

```bash
# Day 1: Start with 100 models for testing
python 1_fetch_models.py --n 100 --strategy random

# Day 2: Expand to 1000 models
python 1_fetch_models.py --target-total 1000 --strategy random

# Day 3: Add 500 more specific models
python 1_fetch_models.py --n 500 --strategy sequential --mode expand
```

## Directory Structure

```
benchmarks/
  config.py              # Configuration settings
  utils.py               # Shared utilities
  1_fetch_models.py      # Download from BioModels
  2_filter_models.py     # Apply heuristics (TODO)
  3_recast_batch.py      # Batch recast with validation (TODO)
  4_analyze_results.py   # Generate reports (TODO)
  
  data/                  # Downloaded models (git-ignored)
    sbml/                # Original SBML files
    antimony/            # Converted Antimony files
    fetch_history.json   # Fetch tracking
    model_registry.json  # BioModels catalog cache
  
  results/               # Outputs (git-ignored)
    candidates.csv       # Filtered models
    recasts/             # Successful recasts
    validation/          # Validation reports
    failures/            # Failed models
    summary.json         # Statistics
    report.ipynb         # Analysis notebook
```

## Configuration

Edit `config.py` to adjust:
- Timeout settings
- Complexity filters
- Parallel processing (N_WORKERS)
- Feature detection thresholds

## Status

- [x] Phase 1: Fetch models (complete with incremental support)
- [x] Phase 2: Filter models (complete)
- [x] Phase 3: Batch recast with validation (complete)
- [x] Phase 4: Analyze results and visualization (complete)

**All phases complete!** Ready for production use.

## Notes

- BioModels API has rate limits - the fetch script includes delays
- SBML → Antimony conversion requires Antimony and libroadrunner
- Large model collections may take hours to fetch
- All data files are git-ignored to avoid repository bloat

## Download Success Rate

Not all models in BioModels can be downloaded. Typical download success is ~60-65%:

| BioModels ODE Models | ~1,700 |
|----------------------|--------|
| Successfully downloaded | ~1,056 (62%) |
| Download failures | ~644 (38%) |

**Reasons for download failures:**

1. **COMBINE Archive Format Variations**: BioModels provides models as OMEX/COMBINE archives (zip files). The script looks for SBML files named `{model_id}_url.xml` or files containing "sbml". Some archives use different naming conventions.

2. **API/Network Issues**: Rate limiting, temporary 404 errors, or server issues. The script includes 50ms delays between requests to avoid rate limits.

3. **Missing SBML Content**: Some BioModels entries are curated metadata without downloadable SBML files. Others may be deprecated or use alternative formats (CellML, etc.).

4. **Model ID Filtering**: Only models classified as "ordinary differential equation" in BioModels metadata are fetched. Non-ODE models (stochastic, spatial, etc.) are filtered out.

**The ~1,056 downloadable models represent the "actually usable" subset** for S-system recasting. This is a strong benchmark - many published validation studies use far fewer models.

## Re-Running the Full Pipeline

To regenerate all results from scratch:

```bash
# Activate environment
source ../ssys_dev/bin/activate

# ============================================================
# TRANSFORMATION PHASE
# ============================================================

# Step 1a: First pass - quick transformation with 15s timeout (~90% of models)
python 3_recast_batch.py --clean --mode simplified --timeout 15 --no-validate

# Step 1b: Second pass - retry only timeout failures with 60s timeout
python 3_recast_batch.py --mode simplified --timeout 60 --retry-timeouts --no-validate

# ============================================================
# VALIDATION PHASE (3-stage pipeline)
# ============================================================

# Stage 1: Fast numerical screening (all CPUs)
python 3b_validate_batch.py --numerical-only --timeout 60 --workers -1

# Stage 2: JAX numerical cross-check (passed models only)
python 3b_validate_batch.py --numerical-only --use-jax --passed-only \
    --timeout 120 --workers -1

# Stage 3: Symbolic proof (passed models, subprocess isolation)
python 3b_validate_batch.py --symbolic-only --passed-only --subprocess \
    --timeout 120 --workers 4

# ============================================================
# COLLECTION PHASE
# ============================================================

# Collect validated models
python 6_collect_validated.py

# Rebuild results CSV
python 5_rebuild_results_csv.py
```

**Time estimates (on 8-core machine):**
- Transformation Phase: ~15-25 minutes total
  - First pass (15s): ~10-15 minutes
  - Second pass (timeouts only): ~5-10 minutes
- Validation Phase: ~30-60 minutes total
  - Stage 1 (numerical): ~10-15 minutes (parallelized)
  - Stage 2 (JAX): ~5-10 minutes (only ~20% of models)
  - Stage 3 (symbolic): ~10-20 minutes (only passed models)

## 3-Stage Validation Explained

**Stage 1: Non-JAX Numerical (fastest, most robust)**
- Tests: Pointwise numerical comparison of ODEs
- Speed: ~0.5s per model
- Purpose: Fast screening to filter out incorrect transformations
- Flags: `--numerical-only --workers -1`

**Stage 2: JAX Numerical (independent implementation)**
- Tests: Same numerical test using JAX autodiff
- Speed: ~1-2s per model (JAX compilation overhead)
- Purpose: Cross-validate with independent code
- Flags: `--numerical-only --use-jax --passed-only --workers -1`

**Stage 3: Symbolic Proof (can hang on complex models)**
- Tests: Algebraic simplification to prove equivalence
- Speed: Variable (can hang indefinitely on SymPy)
- Purpose: Mathematical proof of correctness
- Flags: `--symbolic-only --passed-only --subprocess --workers 4`
- Note: `--subprocess` enables hard kill for SymPy hangs

## Parallel Processing

The validation script supports parallel execution:

```bash
# Use all CPUs
--workers -1

# Use 4 workers (good for memory-intensive symbolic)
--workers 4

# Single-threaded (default)
--workers 1
```

For symbolic validation, use `--subprocess --workers 4` to:
1. Run each model in isolated subprocess
2. Hard-kill stuck SymPy processes
3. Limit parallelism to avoid memory pressure

## Understanding Failure Logs

When a model transformation fails, an explanatory log is created in `results/failures/`:

```
Model: BIOMD0000000123
Mode: simplified
Timestamp: 2025-12-31T12:00:00
Category: TIMEOUT
Error: Timeout

--- Explanation ---
Model took too long to recast. Try with --timeout 60 or higher.
Complex models with many species/reactions or deeply nested functions
may require longer processing time.
```

**Error categories:**
- `TIMEOUT` - Model took too long (try with higher --timeout)
- `UNSUPPORTED_CONSTRUCT` - Piecewise functions, events, or delays
- `PARSE_ERROR` - SBML/Antimony syntax issues
- `NEGATIVITY` - Variables that become negative
- `COMPLEXITY` - Deep nesting or recursion limits
- `OTHER` - See error message for details
