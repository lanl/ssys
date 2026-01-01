# BioModels Batch Transformation Results

*Auto-generated on 2025-12-31 21:13*

## Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| SBML downloads | 1,644 | Successfully downloaded |
| Filtered out | 666 | No ODEs, events, delays, etc. |
| **Transformation candidates** | **978** | Passed heuristic filters |
| Successful transformations | 896 | 91.6% success rate |
| Failed (error) | 103 | Various errors |
| **Validated models** | **204** | Numerical validation passed |

## Transformation Achievements

| Transformation | Count | Significance |
|----------------|-------|--------------|
| General → S-system | 1 | Full simplification achieved |
| General → GMA | 66 | Functional lifting to GMA form |
| GMA → S-system | 18 | Sum-to-product reduction |
| GMA → GMA | 74 | Already in GMA form |
| S-system → S-system | 4 | Already in S-system form |
| Canonical S-system → S-system | 4 | - |
| General → General | 37 | - |
| **Total validated** | **204** | |

## Sample Validated Models

| Model ID | Original Type | Transformed Type | Max Error |
|----------|---------------|------------------|-----------|
| BIOMD0000000732 | General | S-system | 0.00e+00 |
| MODEL8236520494 | General | GMA | 4.71e-16 |
| BIOMD0000000274 | General | GMA | 3.00e-16 |
| MODEL1108260015 | General | GMA | 0.00e+00 |
| BIOMD0000000414 | General | GMA | 0.00e+00 |
| BIOMD0000000423 | General | GMA | 1.76e-15 |
| BIOMD0000000424 | General | GMA | 4.49e-16 |
| BIOMD0000000254 | General | GMA | 2.80e-16 |
| BIOMD0000000448 | General | GMA | 1.16e-16 |
| BIOMD0000000245 | General | GMA | 4.65e-16 |

See `results/validated/manifest.csv` for complete list (204 models).

## Reproducing Results

```bash
cd biomodels_batch
./run_benchmark.sh              # Default: numerical validation only
./run_benchmark.sh --full       # Full validation (+ JAX + symbolic)
./run_benchmark.sh --from recast  # Re-run from transformation step
python step6_report.py --figures  # Regenerate this file with figures
```