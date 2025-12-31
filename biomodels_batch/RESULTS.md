# BioModels Batch Transformation Results

*Auto-generated on 2025-12-31 14:59*

## Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| BioModels ODE models | 1,644 | Fetched from BioModels database |
| SBML downloads | 1,644 | Successfully downloaded |
| Filtered out | 666 | Events, delays, parse errors |
| **Transformation candidates** | **978** | Passed all filters |
| Successful transformations | 884 | 90.4% success rate |
| Failed (timeout) | 76 | >60s processing time |
| Failed (error) | 29 | Various parsing/transformation errors |

## Transformation Achievements

*Validation pending - run `./run_benchmark.sh --from validate`*


## Sample Validated Models

*Validation pending*


## Transformation Failures

| Failure Category | Count |
|------------------|-------|
| Timeout (model too complex) | 76 |
| Unsupported: piecewise functions | 22 |
| Parse errors | 4 |
| Other errors | 3 |
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