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
