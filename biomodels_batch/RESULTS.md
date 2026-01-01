# BioModels Batch Transformation Results

*Auto-generated on 2025-12-31 17:12*

## Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| BioModels ODE models | 1,644 | Matched fetch query |
| SBML downloads | 1,644 | Successfully downloaded |
| Filtered out | 666 | No ODEs, events, delays, etc. |
| **Transformation candidates** | **978** | Passed heuristic filters |
| Successful transformations | 895 | 91.5% success rate |
| Failed (timeout) | 0 | >60s processing time |
| Failed (error) | 105 | Various parsing/transformation errors |
| Validation reports | 895 | Numerical validation attempted |
| **Validated models** | **106** | Numerical validation passed |

## Transformation Achievements

The following table shows successful transformations by type.
**Highlights:** General → S-system/GMA transformations demonstrate ssys lifting
non-polynomial ODEs to canonical forms.

| Transformation | Count | Significance |
|----------------|-------|--------------|
| General → S-system | 1 | Full simplification achieved |
| General → GMA | 44 | Functional lifting to GMA form |
| GMA → S-system | 14 | Sum-to-product reduction |
| GMA → GMA | 39 | Already in GMA form (identity) |
| Canonical S-system → S-system | 4 | - |
| General → General | 4 | - |
| **Total validated** | **106** | |

## Sample Validated Models

Examples of successfully transformed and validated models:

| Model ID | Original Type | Transformed Type | Max Error |
|----------|---------------|------------------|-----------|
| BIOMD0000000732 | General | S-system | 0.00e+00 |
| BIOMD0000000461 | General | GMA | 1.90e-10 |
| BIOMD0000000298 | General | GMA | 2.21e-07 |
| BIOMD0000000299 | General | GMA | 1.49e-07 |
| BIOMD0000000305 | General | GMA | 0.00e+00 |
| BIOMD0000000325 | General | GMA | 1.37e-07 |
| BIOMD0000000328 | General | GMA | 5.56e-07 |
| BIOMD0000000330 | General | GMA | 2.50e-07 |
| BIOMD0000000331 | General | GMA | 8.93e-07 |
| BIOMD0000000354 | General | GMA | 4.09e-07 |

See `results/validated/manifest.csv` for the complete list of 106 validated models.

## Transformation Failures

| Failure Category | Count |
|------------------|-------|
| Other errors | 105 |
| **Total failures** | **105** |

## Files

- `results/batch_recast_results.csv` - Complete results for all candidates
- `results/recasts/` - Successful transformation .ant files
- `results/failures/` - Failure log files
- `results/validation/` - Validation JSON reports
- `results/validated/` - Validated model pairs (SBML + Antimony)

## Reproducing Results

```bash
# Activate environment
source activate_dev_env.sh

# Run full pipeline
cd biomodels_batch
./run_benchmark.sh

# Or run individual stages
./run_benchmark.sh --only fetch
./run_benchmark.sh --only filter
./run_benchmark.sh --only recast
./run_benchmark.sh --from validate

# Regenerate this file
python 7_generate_results_md.py --write
```