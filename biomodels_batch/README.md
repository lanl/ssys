# BioModels Benchmark Suite

This directory contains tools for benchmarking the ssys recaster against real-world models from the BioModels database.

## Overview

The benchmark suite operates in four phases:

1. **Fetch**: Download models from BioModels (SBML → Antimony)
2. **Filter**: Apply heuristics to identify recast candidates
3. **Recast**: Attempt recasts with validation
4. **Analyze**: Generate statistics and reports

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

### Phase 3: Batch Recast

```bash
# Test with 10 S-system candidates
python 3_recast_batch.py --filter s_system --limit 10

# All candidates, simplified mode
python 3_recast_batch.py --mode simplified

# Skip validation for speed
python 3_recast_batch.py --no-validate
```

Output: `results/recasts/`, `results/validation/`, `results/failures/`

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

# Step 1: Re-recast all models (applies latest fixes)
python 3_recast_batch.py --mode simplified --timeout 60

# Step 2: Re-run validation
python 3b_validate_batch.py --numerical-only --timeout 60

# Step 3: Collect validated models
python 6_collect_validated.py

# Step 4: Rebuild results CSV
python 5_rebuild_results_csv.py
```

**Time estimates:**
- Recasting: ~15-30 minutes (894 models)
- Validation: ~30-60 minutes (depends on model complexity)
