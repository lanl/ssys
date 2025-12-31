#!/bin/bash
#
# BioModels Benchmark Pipeline
# ============================
#
# Runs the complete S-system recasting benchmark on BioModels:
# 1. Fetch SBML models from BioModels
# 2. Filter candidates for recasting
# 3. Recast to S-system form (two-pass: quick then retry timeouts)
# 4. Validate (three-stage: numerical, JAX, symbolic)
# 5. Collect validated models
#
# Usage:
#   ./run_benchmark.sh              # Full pipeline (auto-skip completed)
#   ./run_benchmark.sh --from recast     # Start from recasting
#   ./run_benchmark.sh --from validate   # Start from validation
#   ./run_benchmark.sh --only fetch      # Run only fetch step
#   ./run_benchmark.sh --force           # Force re-run all steps
#   ./run_benchmark.sh --clean           # Delete all outputs first
#   ./run_benchmark.sh --help            # Show this help
#
set -euo pipefail

# ===========================================================================
# Configuration
# ===========================================================================

# Minimum thresholds to consider a stage "complete"
MIN_SBML_FILES=100
MIN_RECAST_FILES=50
MIN_VALIDATION_FILES=50

# Timeouts
TIMEOUT_QUICK=15
TIMEOUT_LONG=60
TIMEOUT_VALIDATION=60
TIMEOUT_JAX=120
TIMEOUT_SYMBOLIC=120

# Parallelism
WORKERS_ALL=-1      # All CPUs
WORKERS_SYMBOLIC=4  # Limited for memory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===========================================================================
# Paths (relative to script directory)
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data"
RESULTS_DIR="results"
SBML_DIR="$DATA_DIR/sbml_downloads"
CANDIDATES_DIR="$DATA_DIR/sbml_candidates"
RECASTS_DIR="$RESULTS_DIR/recasts"
VALIDATION_DIR="$RESULTS_DIR/validation"
VALIDATED_DIR="$RESULTS_DIR/validated"
FAILURES_DIR="$RESULTS_DIR/failures"
CANDIDATES_CSV="$RESULTS_DIR/candidates.csv"

# ===========================================================================
# Helpers
# ===========================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[DONE]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_stage() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

count_files() {
    local dir="$1"
    local pattern="${2:-*}"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# ===========================================================================
# Stage Completion Checks
# ===========================================================================

is_fetch_complete() {
    local count
    count=$(count_files "$SBML_DIR" "*.xml")
    [[ $count -ge $MIN_SBML_FILES ]]
}

is_filter_complete() {
    [[ -f "$CANDIDATES_CSV" ]] && [[ $(count_files "$CANDIDATES_DIR" "*.xml") -ge $MIN_SBML_FILES ]]
}

is_recast_complete() {
    local count
    count=$(count_files "$RECASTS_DIR" "*.ant")
    [[ $count -ge $MIN_RECAST_FILES ]]
}

is_validate_numerical_complete() {
    local count
    count=$(count_files "$VALIDATION_DIR" "*_validation.json")
    [[ $count -ge $MIN_VALIDATION_FILES ]]
}

is_validate_jax_complete() {
    # Check if validation reports contain JAX results
    # This is a heuristic - look for a few files
    if [[ ! -d "$VALIDATION_DIR" ]]; then
        return 1
    fi
    # JAX validation updates existing reports, so we just check if files exist
    # and are recent (within last hour) - simplified check
    is_validate_numerical_complete
}

is_validate_symbolic_complete() {
    # Check if validation reports contain symbolic results
    if [[ ! -d "$VALIDATION_DIR" ]]; then
        return 1
    fi
    # Look for symbolic test results in any validation file
    if grep -l '"symbolic":' "$VALIDATION_DIR"/*.json &>/dev/null; then
        return 0
    fi
    return 1
}

is_collect_complete() {
    local count
    count=$(count_files "$VALIDATED_DIR" "*.ant")
    [[ $count -ge 10 ]]
}

is_report_complete() {
    # Report is never "complete" - always regenerate on request
    return 1
}

# ===========================================================================
# Stage Implementations
# ===========================================================================

stage_fetch() {
    log_stage "STAGE: Fetch SBML models from BioModels"
    
    log_info "Fetching ODE models from BioModels API..."
    python 1_fetch_models.py --target-total 2000 --strategy random
    
    local count
    count=$(count_files "$SBML_DIR" "*.xml")
    log_success "Downloaded $count SBML files"
}

stage_filter() {
    log_stage "STAGE: Filter candidates for S-system recasting"
    
    log_info "Applying heuristic filters..."
    python 2_filter_models.py
    
    local count
    count=$(count_files "$CANDIDATES_DIR" "*.xml")
    log_success "Filtered to $count candidate models"
}

stage_recast() {
    log_stage "STAGE: Recast to S-system form"
    
    # Pass 1: Quick timeout
    log_info "Pass 1: Quick recast (${TIMEOUT_QUICK}s timeout)..."
    python 3_recast_batch.py --mode simplified --timeout "$TIMEOUT_QUICK" --no-validate
    
    local pass1_count
    pass1_count=$(count_files "$RECASTS_DIR" "*.ant")
    log_info "Pass 1 complete: $pass1_count recasts"
    
    # Pass 2: Retry timeouts only
    log_info "Pass 2: Retry timeouts (${TIMEOUT_LONG}s timeout)..."
    python 3_recast_batch.py --mode simplified --timeout "$TIMEOUT_LONG" --retry-timeouts --no-validate
    
    local final_count
    final_count=$(count_files "$RECASTS_DIR" "*.ant")
    log_success "Recasting complete: $final_count total recasts"
}

stage_validate_numerical() {
    log_stage "STAGE: Validate (numerical, non-JAX)"
    
    log_info "Running numerical validation on all recasts..."
    python 3b_validate_batch.py --numerical-only --timeout "$TIMEOUT_VALIDATION" --workers "$WORKERS_ALL"
    
    local count
    count=$(count_files "$VALIDATION_DIR" "*_validation.json")
    log_success "Numerical validation complete: $count reports"
}

stage_validate_jax() {
    log_stage "STAGE: Validate (numerical, JAX cross-check)"
    
    log_info "Running JAX numerical validation on passed models..."
    python 3b_validate_batch.py --numerical-only --use-jax --passed-only --timeout "$TIMEOUT_JAX" --workers "$WORKERS_ALL"
    
    log_success "JAX validation complete"
}

stage_validate_symbolic() {
    log_stage "STAGE: Validate (symbolic proof)"
    
    log_info "Running symbolic validation on passed models (subprocess isolation)..."
    python 3b_validate_batch.py --symbolic-only --passed-only --subprocess --timeout "$TIMEOUT_SYMBOLIC" --workers "$WORKERS_SYMBOLIC"
    
    log_success "Symbolic validation complete"
}

stage_collect() {
    log_stage "STAGE: Collect validated models"
    
    log_info "Collecting validated model pairs..."
    python 6_collect_validated.py
    
    log_info "Rebuilding results CSV..."
    python 5_rebuild_results_csv.py
    
    local count
    count=$(count_files "$VALIDATED_DIR" "*.ant")
    log_success "Collection complete: $count validated models"
}

stage_report() {
    log_stage "STAGE: Generate RESULTS.md report"
    
    log_info "Regenerating RESULTS.md from current data..."
    python 7_generate_results_md.py --write
    
    log_success "Report generated: RESULTS.md"
}

# ===========================================================================
# Clean Function
# ===========================================================================

clean_outputs() {
    log_stage "CLEANING: Removing all output files"
    
    local dirs=("$RECASTS_DIR" "$VALIDATION_DIR" "$VALIDATED_DIR" "$FAILURES_DIR")
    for dir in "${dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            log_info "Deleted $dir"
        fi
    done
    
    local files=("$RESULTS_DIR/batch_recast_results.csv" "$RESULTS_DIR/validation_summary.txt" "$RESULTS_DIR/batch_recast_summary.txt")
    for f in "${files[@]}"; do
        if [[ -f "$f" ]]; then
            rm "$f"
            log_info "Deleted $f"
        fi
    done
    
    log_success "Cleanup complete"
}

# ===========================================================================
# Main Runner
# ===========================================================================

run_stage() {
    local stage="$1"
    local force="${2:-false}"
    
    case "$stage" in
        fetch)
            if ! $force && is_fetch_complete; then
                log_warn "Fetch already complete ($(count_files "$SBML_DIR" "*.xml") files)"
            else
                stage_fetch
            fi
            ;;
        filter)
            if ! $force && is_filter_complete; then
                log_warn "Filter already complete ($(count_files "$CANDIDATES_DIR" "*.xml") candidates)"
            else
                stage_filter
            fi
            ;;
        recast)
            if ! $force && is_recast_complete; then
                log_warn "Recast already complete ($(count_files "$RECASTS_DIR" "*.ant") recasts)"
            else
                stage_recast
            fi
            ;;
        validate_numerical)
            if ! $force && is_validate_numerical_complete; then
                log_warn "Numerical validation already complete ($(count_files "$VALIDATION_DIR" "*_validation.json") reports)"
            else
                stage_validate_numerical
            fi
            ;;
        validate_jax)
            # JAX always runs on passed models (idempotent)
            stage_validate_jax
            ;;
        validate_symbolic)
            # Symbolic always runs on passed models (idempotent)
            stage_validate_symbolic
            ;;
        collect)
            stage_collect
            ;;
        report)
            stage_report
            ;;
        *)
            log_error "Unknown stage: $stage"
            exit 1
            ;;
    esac
}

show_help() {
    cat << 'EOF'
BioModels Benchmark Pipeline
============================

Usage:
  ./run_benchmark.sh [OPTIONS]

Options:
  --from STAGE    Start from a specific stage (skip earlier stages)
                  Stages: fetch, filter, recast, validate, validate_jax, 
                          validate_symbolic, collect, report

  --only STAGE    Run only a specific stage (no continuation)

  --force         Force re-run stages even if already complete

  --clean         Delete all output files before running

  --status        Show completion status of each stage

  --help          Show this help message

Examples:
  ./run_benchmark.sh              # Full pipeline (auto-skip completed)
  ./run_benchmark.sh --from recast     # Start from recasting
  ./run_benchmark.sh --from validate   # Start from validation (stage 1)
  ./run_benchmark.sh --only fetch      # Only fetch models
  ./run_benchmark.sh --force           # Re-run everything
  ./run_benchmark.sh --clean --force   # Fresh start

Stages (in order):
  1. fetch              - Download SBML from BioModels
  2. filter             - Identify recast candidates
  3. recast             - Convert to S-system form
  4. validate           - Numerical validation (non-JAX)
  5. validate_jax       - Numerical validation (JAX cross-check)
  6. validate_symbolic  - Symbolic equivalence proof
  7. collect            - Collect validated models
  8. report             - Generate RESULTS.md (optional, not in default pipeline)

The pipeline auto-detects completed stages and skips them unless --force is used.
EOF
}

show_status() {
    log_stage "Pipeline Status"
    
    local stages=("fetch" "filter" "recast" "validate_numerical" "validate_jax" "validate_symbolic" "collect" "report")
    
    for stage in "${stages[@]}"; do
        local check_func="is_${stage}_complete"
        local status
        
        if $check_func 2>/dev/null; then
            status="${GREEN}✓ READY${NC}"
        else
            status="${YELLOW}○ PENDING${NC}"
        fi
        
        printf "  %-20s %b\n" "$stage" "$status"
    done
    
    echo ""
    echo "File counts:"
    echo "  SBML downloads:     $(count_files "$SBML_DIR" "*.xml")"
    echo "  Candidates:         $(count_files "$CANDIDATES_DIR" "*.xml")"
    echo "  Transforms:         $(count_files "$RECASTS_DIR" "*.ant")"
    echo "  Validation reports: $(count_files "$VALIDATION_DIR" "*_validation.json")"
    echo "  Validated models:   $(count_files "$VALIDATED_DIR" "*.ant")"
}

# ===========================================================================
# Argument Parsing
# ===========================================================================

FORCE=false
CLEAN=false
START_FROM=""
ONLY_STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            START_FROM="$2"
            shift 2
            ;;
        --only)
            ONLY_STAGE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --status)
            show_status
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ===========================================================================
# Main Execution
# ===========================================================================

log_stage "BioModels Benchmark Pipeline"
echo "Working directory: $SCRIPT_DIR"
echo "Force mode: $FORCE"
echo ""

# Clean if requested
if $CLEAN; then
    clean_outputs
fi

# Run only a single stage if requested
if [[ -n "$ONLY_STAGE" ]]; then
    run_stage "$ONLY_STAGE" "$FORCE"
    log_success "Single stage '$ONLY_STAGE' complete!"
    exit 0
fi

# Determine which stages to run
STAGES=()

case "${START_FROM:-all}" in
    all|"")
        STAGES=(fetch filter recast validate_numerical validate_jax validate_symbolic collect)
        ;;
    fetch)
        STAGES=(fetch filter recast validate_numerical validate_jax validate_symbolic collect)
        ;;
    filter)
        STAGES=(filter recast validate_numerical validate_jax validate_symbolic collect)
        ;;
    recast)
        STAGES=(recast validate_numerical validate_jax validate_symbolic collect)
        ;;
    validate|validate_numerical)
        STAGES=(validate_numerical validate_jax validate_symbolic collect)
        ;;
    validate_jax)
        STAGES=(validate_jax validate_symbolic collect)
        ;;
    validate_symbolic)
        STAGES=(validate_symbolic collect)
        ;;
    collect)
        STAGES=(collect)
        ;;
    report)
        STAGES=(report)
        ;;
    *)
        log_error "Unknown start point: $START_FROM"
        echo "Valid stages: fetch, filter, recast, validate, validate_jax, validate_symbolic, collect, report"
        exit 1
        ;;
esac

# Run the pipeline
for stage in "${STAGES[@]}"; do
    run_stage "$stage" "$FORCE"
done

log_stage "Pipeline Complete!"
show_status
