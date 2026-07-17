# BioModels Batch Transformation Results

*Auto-generated on 2026-07-16 21:39*

## Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| SBML downloads | 1,644 | Successfully downloaded |
| Filtered out | 666 | No ODEs, events, delays, etc. |
| **Transformation candidates** | **978** | Passed heuristic filters |
| Successful transformations | 840 | 85.9% success rate |
| Failed (error) | 138 | Various errors |
| **Validated models** | **731** | Numerical validation passed |

> **Note (issue #6 — fail-closed on negative initial conditions):** S-system pool construction maps each state to a product of strictly-positive power-law auxiliaries, so a state with a negative initial value has no valid representation. ssys now rejects such models at recast with `NegativeInitialConditionError` instead of silently substituting 0 and recasting from a corrupted initial point. In this corpus that fail-closes 8 candidates with negative initial states (e.g. membrane-voltage models starting at V = -60 mV), lowering the validated count from 739 (pre-#6) to 731. These were previously counted as passes only because the numerical profile samples the RHS over a positive domain and never integrates from the initial condition, so the silent corruption was invisible to it.

## Transformation Achievements

| Transformation | Count | Significance |
|----------------|-------|--------------|
| General → S-system | 2 | Full simplification achieved |
| General → GMA | 282 | Functional lifting to GMA form |
| GMA → S-system | 75 | Sum-to-product reduction |
| GMA → GMA | 182 | Already in GMA form |
| S-system → S-system | 4 | Already in S-system form |
| Canonical S-system → Canonical S-system | 8 | - |
| General → Canonical S-system | 5 | - |
| General → General | 173 | - |
| **Total validated** | **731** | |

## Sample Validated Models

| Model ID | Original Type | Transformed Type | Max Error |
|----------|---------------|------------------|-----------|
| MODEL1108260010 | General | S-system | 0.00e+00 |
| BIOMD0000000914 | General | S-system | 0.00e+00 |
| BIOMD0000000477 | General | GMA | 2.22e-16 |
| BIOMD0000000560 | General | GMA | 4.31e-16 |
| MODEL1203220000 | General | GMA | 2.62e-16 |
| BIOMD0000000545 | General | GMA | 4.10e-16 |
| MODEL1204280001 | General | GMA | 1.93e-16 |
| MODEL1204280002 | General | GMA | 1.93e-16 |
| MODEL1204280003 | General | GMA | 2.98e-16 |
| MODEL1204280004 | General | GMA | 2.47e-16 |

Rerunning the benchmark writes the complete validated manifest to `results/validated/manifest.csv` (731 models).

## Reproducing Results

```bash
cd biomodels_batch
./run_benchmark.sh              # Default: numerical validation only
./run_benchmark.sh --full       # Full validation (+ JAX + symbolic)
./run_benchmark.sh --from recast  # Re-run from transformation step
python step6_report.py --figures  # Regenerate this file with figures
```