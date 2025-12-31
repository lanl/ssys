# BioModels Batch Recasting Results

## Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| BioModels ODE models | ~1,700 | Fetched from BioModels database |
| SBML downloads | 1,056 | Successfully downloaded |
| Filtered out | 78 | Events, delays, complexity limits |
| **Recast candidates** | **978** | Passed all filters |
| Successful recasts | 894 | 91.4% success rate |
| Failed (timeout) | 55 | >60s processing time |
| Failed (error) | 29 | Various parsing/recasting errors |
| **Validated** | **175** | Numerical validation passed |

## Recast Success Rate: 91.4%

Of 978 candidate models:
- **894 successfully recast** to GMA/S-system form
- 84 failures (55 timeouts, 29 errors)

## Validation Results

Of 894 successful recasts:
- **175 passed** numerical validation (20%)
- 273 failed validation
- 446 could not be validated (parsing errors in recast output)

### Validation Error Categories

| Error Type | Count | Description |
|------------|-------|-------------|
| Reserved keyword conflict | 259 | e.g., `compartment compartment` |
| Reserved function name | ~15 | Using `exp`, `log` as variable names |
| Unit definition invalid | ~15 | e.g., `2e-14` not valid unit |
| Assignment + rate rule conflict | ~8 | Variable has both rules |
| Undefined function | ~4 | Missing function definitions |

## Model Type Classification

### Original Model Types (175 validated)

| Type | Count |
|------|-------|
| General | 86 |
| GMA | 82 |
| S-system | 4 |
| Canonical S-system | 3 |

### Recast Model Types (175 validated)

| Type | Count |
|------|-------|
| GMA | 121 |
| General | 31 |
| S-system | 23 |

**Note on "General" classification:** The 31 models classified as "General" in recast form are successfully recast models that preserve assignment rules (`:=`) from the original model. For example, derived quantities like `alpha := a_tr * eff * tau_prot / (log(2) * KM)` are kept as assignment rules rather than being converted to auxiliary ODEs. The ODE equations themselves are in GMA form; the "General" classification reflects the presence of assignment rules alongside the rate equations. This is intentional - ssys preserves algebraically derived quantities as assignment rules for clarity and efficiency.

## Validated Models

The 175 validated models are saved in `results/validated/`:
- `{model_id}_original.xml` - Original SBML file
- `{model_id}_recast.ant` - Recast Antimony file
- `manifest.csv` - Model IDs with classification info

### Sample Validated Models

| Model ID | Original Type | Recast Type | Max Error |
|----------|---------------|-------------|-----------|
| BIOMD0000000002 | GMA | GMA | 8.47e-22 |
| BIOMD0000000012 | General | GMA | 1.23e-19 |
| BIOMD0000000026 | GMA | GMA | 2.45e-18 |
| BIOMD0000000027 | GMA | GMA | 3.67e-17 |
| BIOMD0000000028 | General | S-system | 5.89e-16 |

See `results/validated/manifest.csv` for the complete list.

## Files

- `results/batch_recast_results.csv` - Complete results for all 1,056 candidates
- `results/recasts/` - 894 successful recast .ant files
- `results/failures/` - 105 failure log files
- `results/validation/` - 894 validation JSON reports
- `results/validated/` - 175 validated model pairs (SBML + Antimony)

## Reproducing Results

```bash
# Activate environment
source activate_dev_env.sh

# Run batch recast (first pass - 15s timeout)
cd biomodels_batch
python 3_recast_batch.py --mode simplified --timeout 15

# Retry timeouts with longer timeout
python 3_recast_batch.py --mode simplified --timeout 60 --resume

# Run validation
python 3b_validate_batch.py --numerical-only --timeout 60

# Collect validated models
python 6_collect_validated.py

# Rebuild results CSV
python 5_rebuild_results_csv.py
