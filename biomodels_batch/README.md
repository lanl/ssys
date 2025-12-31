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

Generate statistics and visualizations:

```bash
python 4_analyze_results.py
```

**Output files:**
- `results/summary.json` - Overall statistics (success rates, timing, etc.)
- `results/report.ipynb` - Jupyter notebook with embedded visualizations
- `results/figures/` - PNG visualizations (see below)

**Generated figures:**

| Figure | Description |
|--------|-------------|
| `pipeline_funnel.png` | Data flow: Downloaded → Filtered → Transformed → Validated |
| `transformation_types.png` | Bar chart of General→S-system, GMA→S-system, etc. |
| `error_breakdown.png` | Pie chart of failure categories (timeout, piecewise, etc.) |
| `validation_errors.png` | Histogram of numerical validation error magnitudes |
| `model_size_distribution.png` | Histograms of species/reactions/parameters counts |
| `time_vs_complexity.png` | Scatter plots of transformation time vs model size |
| `feature_prevalence.png` | Bar chart of exp/log/sin/cos usage in models |
| `success_rates.png` | Overview pie/bar of success vs failure |
| `timing_distribution.png` | Histogram of transformation times |
| `complexity_analysis.png` | Success rate vs model complexity |

**Note:** Some figures require validation data (`manifest.csv`) and will be skipped if validation hasn't been run yet.

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
biomodels_batch/
  run_benchmark.sh        # Main pipeline runner (recommended)
  config.py               # Configuration settings
  utils.py                # Shared utilities
  
  # Pipeline scripts
  1_fetch_models.py       # Download from BioModels
  2_filter_models.py      # Apply heuristic filters
  3_recast_batch.py       # Batch transformation
  3b_validate_batch.py    # Batch validation (3-stage)
  4_analyze_results.py    # Generate reports/plots
  5_rebuild_results_csv.py # Rebuild CSV from files
  6_collect_validated.py  # Collect validated models
  7_generate_results_md.py # Generate RESULTS.md
  
  # Documentation
  README.md               # This file
  RESULTS.md              # Auto-generated results summary
  PARSING_ERRORS.md       # Known parsing issues
  
  data/                   # Downloaded models (git-ignored)
    sbml_downloads/       # Original SBML files
    sbml_candidates/      # Filtered candidates
    fetch_history.json    # Fetch tracking
  
  results/                # Outputs (git-ignored)
    candidates.csv        # Filtered models metadata
    recasts/              # Successful transformation .ant files
    validation/           # Validation JSON reports
    validated/            # Final validated model pairs
    failures/             # Failure log files
    summary.json          # Statistics
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

| BioModels ODE Models | 1,644 |
|----------------------|-------|
| Successfully downloaded | 1,644 (100%) |

**Note:** Earlier runs (~2023) had ~60% download success due to API changes. The current fetch script handles COMBINE archive formats correctly.

## Quick Start: run_benchmark.sh

The easiest way to run the full pipeline:

```bash
# Activate environment
source ../ssys_dev/bin/activate

# Run entire pipeline (auto-skips completed stages)
./run_benchmark.sh

# Check status
./run_benchmark.sh --status

# Start from a specific stage
./run_benchmark.sh --from validate

# Run only one stage
./run_benchmark.sh --only report

# Force re-run even if stage appears ready
./run_benchmark.sh --force

# Clean all outputs and start fresh
./run_benchmark.sh --clean --force
```

**Pipeline stages:**
1. `fetch` - Download SBML from BioModels
2. `filter` - Identify transformation candidates
3. `recast` - Convert to S-system form (two-pass)
4. `validate_numerical` - Fast numerical screening
5. `validate_jax` - JAX cross-check
6. `validate_symbolic` - Symbolic proof
7. `collect` - Collect validated models
8. `report` - Generate RESULTS.md (optional, not in default pipeline)

## Manual Pipeline Commands

For more control, run individual Python scripts:

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

# Stage 1: Fast numerical screening (8 workers)
python 3b_validate_batch.py --numerical-only --timeout 60 --workers 8

# Stage 2: JAX numerical cross-check (passed models only)
python 3b_validate_batch.py --numerical-only --use-jax --passed-only \
    --timeout 120 --workers 8

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
- Flags: `--numerical-only --workers 8`

**Stage 2: JAX Numerical (independent implementation)**
- Tests: Same numerical test using JAX autodiff
- Speed: ~1-2s per model (JAX compilation overhead)
- Purpose: Cross-validate with independent code
- Flags: `--numerical-only --use-jax --passed-only --workers 8`

**Stage 3: Symbolic Proof (can hang on complex models)**
- Tests: Algebraic simplification to prove equivalence
- Speed: Variable (can hang indefinitely on SymPy)
- Purpose: Mathematical proof of correctness
- Flags: `--symbolic-only --passed-only --subprocess --workers 4`
- Note: `--subprocess` enables hard kill for SymPy hangs

## Parallel Processing

The validation script supports parallel execution:

```bash
# Use 8 workers (recommended for 10-core machines)
--workers 8

# Use 4 workers (good for memory-intensive symbolic)
--workers 4

# Single-threaded (default)
--workers 1
```

**Warning:** `--workers -1` uses all CPUs and can freeze the machine. Use explicit counts like `--workers 8` instead.

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
