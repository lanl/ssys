# BioModels Benchmark Suite

Benchmarking ssys recaster against real-world models from the BioModels database.

## Quick Start

```bash
# Activate development environment
source ../activate_dev_env.sh

# Run complete pipeline (default: numerical validation only)
./run_benchmark.sh

# With optional JAX/symbolic validation
./run_benchmark.sh --full
```

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Fetch | `step1_fetch.py` | Download SBML from BioModels API |
| 2. Filter | `step2_filter.py` | Apply heuristics to identify candidates |
| 3. Recast | `step3_recast.py` | Transform to S-system form |
| 4. Validate | `step4_validate.py` | Numerical/JAX/symbolic validation |
| 5. Collect | `step5_collect.py` | Collect validated model pairs |
| 6. Report | `step6_report.py` | Generate CSV + RESULTS.md + figures |

## Usage

### Automated Pipeline (Recommended)

```bash
./run_benchmark.sh              # Default: numerical validation only
./run_benchmark.sh --jax        # Add JAX cross-check
./run_benchmark.sh --symbolic   # Add symbolic proof
./run_benchmark.sh --full       # All validation stages
./run_benchmark.sh --status     # Show pipeline status
./run_benchmark.sh --from recast  # Resume from specific stage
./run_benchmark.sh --force      # Force re-run
./run_benchmark.sh --clean      # Delete outputs first
```

### Manual Execution

```bash
# Transform only (fast)
uv run python step3_recast.py --mode simplified --timeout 15 --no-validate

# Validate transformed models
uv run python step4_validate.py --numerical-only --timeout 60 --workers 8

# Generate reports
uv run python step6_report.py --figures
```

## Generated Artifacts

The checked-in SBML files under `data/sbml_downloads/` and `data/sbml_candidates/` are an intentional benchmark snapshot. New downloads, filtered candidates, validation results, figures, logs, coverage files, and cache directories are ignored so local benchmark runs do not add repository noise.

To regenerate the batch outputs from the tracked scripts and snapshot, run:

```bash
uv run python step2_filter.py
uv run python step3_recast.py --mode simplified --timeout 120
uv run python step4_validate.py --numerical-only --timeout 30 --workers 4
uv run python step5_collect.py
uv run python step6_report.py
```

## Directory Structure

```
biomodels_batch/
├── run_benchmark.sh     # Main pipeline orchestration
├── step1_fetch.py       # Download SBML
├── step2_filter.py      # Filter candidates
├── step3_recast.py      # Transform to S-system
├── step4_validate.py    # Validate transformations
├── step5_collect.py     # Collect validated pairs
├── step6_report.py      # Generate reports
├── config.py            # Shared configuration
├── utils.py             # Shared utilities
├── README.md            # This file
├── RESULTS.md           # Auto-generated results
├── requirements.txt     # Dependencies (biomodels API)
├── .gitignore           # Ignore data/results
│
├── data/                # Downloads (gitignored)
│   ├── sbml_downloads/  # Original SBML files
│   └── sbml_candidates/ # Filtered candidates
│
└── results/             # Outputs (gitignored)
    ├── recasts/         # Transformed .ant files
    ├── validation/      # Validation JSON reports
    ├── validated/       # Final validated pairs
    ├── failures/        # Failure log files
    └── figures/         # PNG visualizations
```

## Validation Pipeline

```
step4_validate.py supports three validation modes:

Stage 1: Numerical (default)
  --numerical-only --workers 8
  Fast pointwise ODE comparison (~0.5s/model)

Stage 2: JAX Cross-Check (optional: --jax)
  --numerical-only --use-jax --passed-only --workers 8
  Independent implementation validation

Stage 3: Symbolic Proof (optional: --symbolic)  
  --symbolic-only --passed-only --subprocess --workers 4
  Algebraic equivalence proof (can hang, use --subprocess)
```

## Terminology

- **Candidate**: Model passed filters (no events, delays, piecewise)
- **Transformed**: Algorithm completed without crash/timeout
- **Validated**: Numerical/symbolic tests confirmed equivalence

**Important**: "Transformed" ≠ "Correct". Only **validated** models are confirmed equivalent.

## Configuration

Edit `config.py` for:
- Timeout settings
- Complexity thresholds
- Default parallelism (WORKERS_ALL=8, WORKERS_SYMBOLIC=4)

## Notes

- BioModels API has rate limits - fetch includes delays
- Transformation is single-threaded (SymPy memory issues)
- Validation uses parallel workers (default: 8)
- Large collections take hours to process
- All data files gitignored to avoid repo bloat
