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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    Calls the real validate_recast_pair() function to run three validation tests:
    - Symbolic equivalence
    - Numerical pointwise comparison
    - Trajectory comparison

    Returns:
        Validation report dict or None if files don't exist
    """
    import tempfile

    import antimony

    from ssys.validator import validate_recast_pair

    sbml_path = Path(config.SBML_CANDIDATES_DIR) / f"{model_id}.xml"
    recast_path = Path(config.RECASTS_DIR) / f"{model_id}_{mode}.ant"

    if not sbml_path.exists() or not recast_path.exists():
        return None

    try:
        # Convert SBML to Antimony for validation (validator expects Antimony files)
        antimony.clearPreviousLoads()
        result = antimony.loadSBMLFile(str(sbml_path))
        if result == -1:
            raise ValueError(f"Failed to load SBML: {antimony.getLastError()}")
        antimony_text = antimony.getAntimonyString()

        # Write to temp file for validation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ant", delete=False
        ) as tmp:
            tmp.write(antimony_text)
            original_ant_path = tmp.name

        try:
            report = validate_recast_pair(
                original_ant_path,
                str(recast_path),
                mode=mode,
                parser="sbml"
            )
            return report.to_dict()
        finally:
            # Clean up temp file
            Path(original_ant_path).unlink(missing_ok=True)

    except Exception as e:
        return {
            "model_id": model_id,
            "mode": mode,
            "overall_pass": False,
            "error": str(e),
        }


def save_validation_report(model_id: str, mode: str, report: dict):
    """Save validation report to JSON."""
    output_path = Path(config.VALIDATION_DIR) / f"{model_id}_{mode}_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def categorize_error(error_msg: str) -> tuple[str, str]:
    """
    Categorize an error message and provide a human-readable explanation.

    Returns:
        (category, explanation)
    """
    error_lower = error_msg.lower()

    if "timeout" in error_lower:
        return ("TIMEOUT",
                "Model took too long to recast. Try with --timeout 60 or higher. "
                "Complex models with many species/reactions or deeply nested functions "
                "may require longer processing time.")

    if "piecewise" in error_lower or "event" in error_lower:
        return ("UNSUPPORTED_CONSTRUCT",
                "Model contains piecewise functions or events, which are not supported "
                "by algebraic recasting. These models have discontinuous dynamics that "
                "cannot be represented in S-system/GMA form.")

    if "delay" in error_lower:
        return ("UNSUPPORTED_CONSTRUCT",
                "Model contains time delays (delay differential equations). "
                "S-system recasting only supports ODEs, not DDEs.")

    if "parse" in error_lower or "syntax" in error_lower:
        return ("PARSE_ERROR",
                "Failed to parse the SBML/Antimony model. The model may have "
                "syntax errors or use constructs not supported by the parser.")

    if "sbml" in error_lower and ("load" in error_lower or "read" in error_lower):
        return ("SBML_ERROR",
                "Failed to load SBML file. The file may be corrupted, "
                "use an unsupported SBML level/version, or contain invalid XML.")

    if "negative" in error_lower or "non-positive" in error_lower:
        return ("NEGATIVITY",
                "Model has variables that can become negative, which violates "
                "the positivity requirement for S-system power-law terms. "
                "Consider preprocessing to ensure positive variables.")

    if "symbol" in error_lower and "undefined" in error_lower:
        return ("UNDEFINED_SYMBOL",
                "Model references an undefined symbol (species, parameter, or function). "
                "This may indicate an incomplete model or missing dependencies.")

    if "recursion" in error_lower or "maximum recursion" in error_lower:
        return ("COMPLEXITY",
                "Model is too complex for symbolic processing. "
                "Deep nesting or circular dependencies caused recursion limit.")

    if "memory" in error_lower:
        return ("RESOURCE",
                "Model exceeded memory limits during processing. "
                "Very large models may require more system resources.")

    # Generic fallback
    return ("OTHER",
            f"Recast failed with error: {error_msg[:200]}...")


def log_failure(model_id: str, mode: str, error_msg: str):
    """Log failure to file with categorization and explanation."""
    failure_path = Path(config.FAILURES_DIR) / f"{model_id}_{mode}.log"
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    category, explanation = categorize_error(error_msg)

    with open(failure_path, "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Error: {error_msg}\n")
        f.write("\n--- Explanation ---\n")
        f.write(f"{explanation}\n")


def worker_error_result(model_id: str, mode: str, exc: BaseException) -> dict:
    """Record a worker-level failure in the same shape as process_model()."""
    error_msg = f"Worker error: {type(exc).__name__}: {exc}"
    log_failure(model_id, mode, error_msg)
    return {
        "model_id": model_id,
        "mode": mode,
        "recast_success": False,
        "recast_time": 0.0,
        "validation_attempted": False,
        "validation_pass": False,
        "error": error_msg,
    }


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


def should_process_model(model_id: str, mode: str, *, resume: bool, retry_timeouts: bool) -> bool:
    """Return whether a candidate should be processed for the selected mode."""
    if resume:
        output_path = Path(config.RECASTS_DIR) / f"{model_id}_{mode}.ant"
        if output_path.exists():
            return False

    if retry_timeouts:
        failure_path = Path(config.FAILURES_DIR) / f"{model_id}_{mode}.log"
        if not failure_path.exists():
            return False
        try:
            failure_content = failure_path.read_text()
        except Exception:
            return False
        if "Category: TIMEOUT" not in failure_content:
            return False

    return True


def process_models(
    model_ids: list[str],
    *,
    mode: str,
    validate: bool,
    timeout: int,
    workers: int,
) -> list[dict]:
    """Process candidate models sequentially or with a process pool."""
    if workers <= 1:
        return [
            process_model(model_id, mode, validate=validate, timeout=timeout)
            for model_id in tqdm(model_ids, total=len(model_ids), desc="Recasting")
        ]

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_model, model_id, mode, validate, timeout): model_id
            for model_id in model_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Recasting"):
            model_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(worker_error_result(model_id, mode, exc))
    return results


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

    # Safe percentage calculation
    val_pct = f"{100 * val_pass / validated:.1f}%" if validated > 0 else "N/A"

    summary = f"""
Batch Recast Summary
===================
Total models: {total}
Recast success: {recast_success} ({100 * recast_success / total:.1f}%)
Validated: {validated}
Validation pass: {val_pass} ({val_pct}'

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
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Only retry models that previously failed with timeout (requires prior run)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete old results (recasts/, failures/, validation/, validated/) before starting"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel recast worker processes (default: 1)",
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
    logger.info(f"Retry timeouts only: {args.retry_timeouts}")
    logger.info(f"Workers: {args.workers}")

    # Clean old results if requested
    if args.clean:
        import shutil
        logger.info("Cleaning old results...")
        dirs_to_clean = [
            Path(config.RECASTS_DIR),
            Path(config.FAILURES_DIR),
            Path(config.VALIDATION_DIR),
            Path(config.RESULTS_DIR) / "validated",
        ]
        for d in dirs_to_clean:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"  Deleted {d}")

        # Also reset the results CSV
        results_csv = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
        if results_csv.exists():
            results_csv.unlink()
            logger.info(f"  Deleted {results_csv}")

        logger.info("Cleanup complete.")

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

    # Select models to process
    model_ids: list[str] = []
    skipped = 0
    for _, row in df.iterrows():
        model_id = row["model_id"]
        if should_process_model(
            model_id,
            args.mode,
            resume=args.resume,
            retry_timeouts=args.retry_timeouts,
        ):
            model_ids.append(model_id)
        else:
            skipped += 1

    # Process selected models
    results = process_models(
        model_ids,
        mode=args.mode,
        validate=not args.no_validate,
        timeout=args.timeout,
        workers=max(1, args.workers),
    )

    if skipped > 0:
        logger.info(f"Skipped {skipped} models (already have output files)")

    # Save detailed results - MERGE with existing CSV instead of overwriting
    results_csv = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    new_results_df = pd.DataFrame(results)

    if results_csv.exists() and not new_results_df.empty:
        # Load existing results and merge
        existing_df = pd.read_csv(results_csv)

        # Create a key for merging (model_id + mode)
        new_results_df["_key"] = new_results_df["model_id"] + "_" + new_results_df["mode"]
        existing_df["_key"] = existing_df["model_id"] + "_" + existing_df["mode"]

        # Remove existing entries that will be replaced by new results
        existing_df = existing_df[~existing_df["_key"].isin(new_results_df["_key"])]

        # Combine old (non-overlapping) + new results
        merged_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        merged_df = merged_df.drop(columns=["_key"])

        # Sort by model_id for consistency
        merged_df = merged_df.sort_values("model_id").reset_index(drop=True)

        merged_df.to_csv(results_csv, index=False)
        logger.info(f"Merged {len(new_results_df)} new results with {len(existing_df)} existing")
        logger.info(f"Total results in {results_csv}: {len(merged_df)}")
    else:
        # No existing file or no new results - just save
        new_results_df.to_csv(results_csv, index=False)
        logger.info(f"Saved {len(new_results_df)} results to {results_csv}")

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
