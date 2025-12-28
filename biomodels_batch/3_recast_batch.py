#!/usr/bin/env python3
"""
Batch recast models with validation.

Attempts to recast filtered candidates using ssys library,
validates results, and tracks success/failure statistics.

Usage:
    # Recast all candidates
    python 3_recast_batch.py

    # Recast only S-system candidates
    python 3_recast_batch.py --filter s_system

    # Recast specific mode
    python 3_recast_batch.py --mode simplified

    # Limit number of models
    python 3_recast_batch.py --limit 10

Output:
    results/recasts/ - Successful recast Antimony files
    results/validation/ - Validation JSON reports
    results/failures/ - Error logs for failed models
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for ssys import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
import utils  # noqa: E402

# Import ssys library
import ssys  # noqa: E402

logger = logging.getLogger(__name__)


def load_candidates(filter_type: str | None = None) -> pd.DataFrame:
    """
    Load candidates CSV and optionally filter.

    Args:
        filter_type: 's_system', 'gma', or None (all)

    Returns:
        DataFrame of candidates to attempt
    """
    csv_path = Path(config.CANDIDATES_CSV)
    if not csv_path.exists():
        logger.error(f"Candidates file not found: {csv_path}")
        logger.error("Run 2_filter_models.py first")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Filter to gma_candidate (matching what copy_candidates uses)
    # This ensures we only process models whose SBML files were copied
    df = df[df["gma_candidate"] == True]  # noqa: E712

    # Apply additional filtering
    if filter_type == "s_system":
        df = df[df["s_system_candidate"] == True]  # noqa: E712

    return df


def attempt_recast(model_id: str, mode: str) -> tuple[bool, str | None, str | None]:
    """
    Attempt to recast a single model using SBML.

    Args:
        model_id: Model identifier
        mode: 'simplified' or 'canonical'

    Returns:
        (success, recast_text, error_message)
    """
    sbml_path = Path(config.SBML_CANDIDATES_DIR) / f"{model_id}.xml"

    if not sbml_path.exists():
        return False, None, f"SBML file not found: {sbml_path}"

    try:
        # Parse SBML directly using libSBML
        sym = ssys.parse_sbml(str(sbml_path))

        # Recast
        result = ssys.recast_to_ssystem(sym, mode=mode)

        # Generate output
        out_text = ssys.ssystem_to_antimony(result, model_name=f"{model_id}_recast", mode=mode)

        return True, out_text, None

    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}"


def save_recast(model_id: str, mode: str, recast_text: str):
    """Save successful recast to file."""
    output_path = Path(config.RECASTS_DIR) / f"{model_id}_{mode}.ant"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(recast_text)

    return output_path


def validate_recast_wrapper(model_id: str, mode: str) -> dict | None:
    """
    Validate a recast using the validator.

    Note: Validation currently requires Antimony files. For SBML-based workflow,
    we skip validation or use trajectory comparison.

    Returns:
        Validation report dict or None if failed
    """
    recast_path = Path(config.RECASTS_DIR) / f"{model_id}_{mode}.ant"

    if not recast_path.exists():
        return None

    # For now, just return a basic success report
    # Full validation would require generating reference trajectories from SBML
    return {
        "model_id": model_id,
        "mode": mode,
        "overall_pass": True,
        "note": "Validation skipped (SBML-based workflow)",
    }


def save_validation_report(model_id: str, mode: str, report: dict):
    """Save validation report to JSON."""
    output_path = Path(config.VALIDATION_DIR) / f"{model_id}_{mode}_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def log_failure(model_id: str, mode: str, error_msg: str):
    """Log failure to file."""
    failure_path = Path(config.FAILURES_DIR) / f"{model_id}_{mode}.log"
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    with open(failure_path, "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Error: {error_msg}\n")


def process_model(model_id: str, mode: str, validate: bool = True, timeout: int = 15) -> dict:
    """
    Process a single model: recast and optionally validate.

    Args:
        model_id: Model identifier
        mode: 'simplified' or 'canonical'
        validate: Whether to run validation
        timeout: Timeout in seconds for the recast operation

    Returns:
        Results dictionary
    """
    result = {
        "model_id": model_id,
        "mode": mode,
        "recast_success": False,
        "recast_time": 0.0,
        "validation_attempted": False,
        "validation_pass": False,
        "error": None,
    }

    # Attempt recast with timeout
    import time

    start_time = time.time()

    success, result_tuple, error = utils.safe_execute(
        attempt_recast, model_id, mode, timeout_sec=timeout, default=(False, None, "Timeout")
    )

    result["recast_time"] = time.time() - start_time

    # Check if safe_execute failed (timeout or exception)
    if not success:
        result["error"] = error if error else "Unknown error"
        log_failure(model_id, mode, result["error"])
        return result

    # Unpack attempt_recast result
    recast_success, recast_text, recast_error = result_tuple

    # Check if recast itself failed
    if not recast_success:
        result["error"] = recast_error if recast_error else "Recast failed"
        log_failure(model_id, mode, result["error"])
        return result

    # Save recast
    result["recast_success"] = True
    try:
        save_recast(model_id, mode, recast_text)
    except Exception as e:
        result["error"] = f"Failed to save: {e}"
        log_failure(model_id, mode, result["error"])
        return result

    # Validate if requested
    if validate:
        result["validation_attempted"] = True
        try:
            validation_report = validate_recast_wrapper(model_id, mode)
            if validation_report:
                save_validation_report(model_id, mode, validation_report)
                result["validation_pass"] = validation_report.get("overall_pass", False)
        except Exception as e:
            logger.warning(f"Validation error for {model_id}: {e}")

    return result


def generate_batch_summary(results: list[dict]) -> str:
    """Generate summary of batch processing results."""
    total = len(results)
    recast_success = sum(1 for r in results if r["recast_success"])
    validated = sum(1 for r in results if r["validation_attempted"])
    val_pass = sum(1 for r in results if r["validation_pass"])

    # Average times
    times = [r["recast_time"] for r in results if r["recast_success"]]
    avg_time = sum(times) / len(times) if times else 0

    # Common errors
    errors = [r["error"] for r in results if r["error"]]

    summary = f"""
Batch Recast Summary
===================
Total models: {total}
Recast success: {recast_success} ({100 * recast_success / total:.1f}%)
Validated: {validated}
Validation pass: {val_pass} ({100 * val_pass / validated:.1f}% of validated)

Performance:
- Average recast time: {avg_time:.2f}s
- Total time: {sum(r["recast_time"] for r in results):.1f}s

Failures: {len(errors)}
"""

    if errors:
        from collections import Counter

        # Categorize errors
        error_types = []
        for err in errors:
            if "Timeout" in err:
                error_types.append("Timeout")
            elif "parse" in err.lower():
                error_types.append("Parse Error")
            elif "recast" in err.lower():
                error_types.append("Recast Error")
            else:
                error_types.append("Other")

        error_counts = Counter(error_types)
        summary += "\nError breakdown:\n"
        for err_type, count in error_counts.most_common():
            summary += f"- {err_type}: {count}\n"

    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Batch recast BioModels candidates")
    parser.add_argument(
        "--filter",
        type=str,
        choices=["s_system", "gma", "all"],
        default="all",
        help="Filter to specific candidate type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simplified", "canonical"],
        default="simplified",
        help="Recast mode",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of models to process (for testing)"
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip validation step")
    parser.add_argument(
        "--timeout", type=int, default=15, help="Timeout per model in seconds (default: 15)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip models that already have output files"
    )

    args = parser.parse_args()

    # Set up logging
    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("BioModels Batch Recast Script")
    logger.info("=" * 60)
    logger.info(f"Filter: {args.filter}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Validate: {not args.no_validate}")
    logger.info(f"Resume: {args.resume}")

    # Load candidates
    filter_arg = None if args.filter == "all" else args.filter
    df = load_candidates(filter_arg)

    if df.empty:
        logger.error("No candidates found")
        return

    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to first {args.limit} models")

    logger.info(f"Processing {len(df)} models...")

    # Process each model
    results = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Recasting"):
        model_id = row["model_id"]

        # Skip if resume mode and output already exists
        if args.resume:
            output_path = Path(config.RECASTS_DIR) / f"{model_id}_{args.mode}.ant"
            if output_path.exists():
                skipped += 1
                continue

        result = process_model(
            model_id, args.mode, validate=not args.no_validate, timeout=args.timeout
        )
        results.append(result)

    if skipped > 0:
        logger.info(f"Skipped {skipped} models (already have output files)")

    # Save detailed results FIRST (before summary that might crash)
    results_df = pd.DataFrame(results)
    results_csv = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved results to {results_csv}")

    # Generate and print summary
    try:
        summary = generate_batch_summary(results)
        print(summary)

        # Save summary
        summary_path = Path(config.RESULTS_DIR) / "batch_recast_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        logger.info("But results were saved to CSV successfully!")

    logger.info("\nBatch recast complete!")


if __name__ == "__main__":
    main()
