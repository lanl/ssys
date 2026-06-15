# BioModels Batch Transformation Results

*Auto-generated on 2026-06-15 14:52*

## Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| SBML downloads | 1,644 | Successfully downloaded |
| Filtered out | 666 | No ODEs, events, delays, etc. |
| **Transformation candidates** | **978** | Passed heuristic filters |
| Successful transformations | 848 | 86.7% success rate |
| Failed (error) | 131 | Various errors |
| **Validated models** | **738** | Numerical validation passed |

## Transformation Achievements

| Transformation | Count | Significance |
|----------------|-------|--------------|
| General → S-system | 2 | Full simplification achieved |
| General → GMA | 281 | Functional lifting to GMA form |
| GMA → S-system | 77 | Sum-to-product reduction |
| GMA → GMA | 182 | Already in GMA form |
| S-system → S-system | 4 | Already in S-system form |
| Canonical S-system → Canonical S-system | 8 | - |
| General → Canonical S-system | 5 | - |
| General → General | 179 | - |
| **Total validated** | **738** | |

## Sample Validated Models

| Model ID | Original Type | Transformed Type | Max Error |
|----------|---------------|------------------|-----------|
| BIOMD0000000914 | General | S-system | 0.00e+00 |
| MODEL1108260010 | General | S-system | 0.00e+00 |
| BIOMD0000000730 | General | GMA | 8.76e-16 |
| BIOMD0000000545 | General | GMA | 4.10e-16 |
| MODEL1203220000 | General | GMA | 2.62e-16 |
| BIOMD0000000520 | General | GMA | 3.82e-15 |
| MODEL1204280001 | General | GMA | 1.93e-16 |
| MODEL1204280002 | General | GMA | 1.93e-16 |
| MODEL1204280003 | General | GMA | 2.98e-16 |
| MODEL1204280004 | General | GMA | 2.47e-16 |

Rerunning the benchmark writes the complete validated manifest to `results/validated/manifest.csv` (738 models).

## Reproducing Results

```bash
cd biomodels_batch
./run_benchmark.sh              # Default: numerical validation only
./run_benchmark.sh --full       # Full validation (+ JAX + symbolic)
./run_benchmark.sh --from recast  # Re-run from transformation step
python step6_report.py --figures  # Regenerate this file with figures
```
