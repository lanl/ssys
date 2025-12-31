#!/usr/bin/env python3
"""
Rebuild batch_recast_results.csv from actual recast/failure files.

This script scans the results directories to reconstruct an accurate
results CSV, avoiding issues with the CSV being corrupted during resume operations.

Usage:
    python 5_rebuild_results_csv.py
"""

import csv
import json
import logging
from pathlib import Path

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild_results_csv(mode: str = "simplified") -> None:
    """
    Rebuild batch_recast_results.csv from recast and failure files.

    Args:
        mode: Recast mode ("simplified" or "canonical")
    """
    recasts_dir = Path(config.RECASTS_DIR)
    failures_dir = Path(config.FAILURES_DIR)
    validation_dir = Path(config.VALIDATION_DIR)
    candidates_file = Path(config.RESULTS_DIR) / "candidates.csv"
    output_file = Path(config.RESULTS_DIR) / "batch_recast_results.csv"

    logger.info("=" * 60)
    logger.info("Rebuilding Results CSV")
    logger.info("=" * 60)

    # Get all candidate model IDs from SBML files (more accurate than CSV)
    sbml_candidates_dir = Path(config.SBML_CANDIDATES_DIR)
    if sbml_candidates_dir.exists():
        sbml_files = list(sbml_candidates_dir.glob("*.xml"))
        candidate_ids = {f.stem for f in sbml_files}
        logger.info(f"Found {len(candidate_ids)} SBML candidate files")
    elif candidates_file.exists():
        # Fallback to candidates.csv
        candidate_ids = set()
        with open(candidates_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate_ids.add(row["model_id"])
        logger.info(f"Found {len(candidate_ids)} candidates from CSV")
    else:
        logger.error(f"No candidates found")
        return

    # Scan recast files
    recast_files = list(recasts_dir.glob(f"*_{mode}.ant"))
    recast_ids = {f.stem.replace(f"_{mode}", "") for f in recast_files}
    logger.info(f"Found {len(recast_ids)} successful recast files")

    # Scan failure files (may be .txt or .log extension)
    failure_files = list(failures_dir.glob(f"*_{mode}.txt"))
    failure_files += list(failures_dir.glob(f"*_{mode}.log"))
    failure_ids = {f.stem.replace(f"_{mode}", "") for f in failure_files}
    logger.info(f"Found {len(failure_ids)} failure files")

    # Scan validation files
    validation_files = list(validation_dir.glob(f"*_{mode}_validation.json"))
    validation_data = {}
    for vf in validation_files:
        model_id = vf.stem.replace(f"_{mode}_validation", "")
        try:
            with open(vf) as f:
                validation_data[model_id] = json.load(f)
        except Exception:
            pass
    logger.info(f"Found {len(validation_data)} validation files")

    # Build results
    results = []
    for model_id in sorted(candidate_ids):
        # Determine status
        if model_id in recast_ids:
            status = "success"
        elif model_id in failure_ids:
            # Check if it's a timeout or other error
            failure_file = failures_dir / f"{model_id}_{mode}.log"
            if not failure_file.exists():
                failure_file = failures_dir / f"{model_id}_{mode}.txt"
            try:
                with open(failure_file) as f:
                    error_text = f.read().strip()
                    if "TimeoutError" in error_text or "timed out" in error_text:
                        status = "timeout"
                    else:
                        status = "error"
            except Exception:
                status = "error"
        else:
            status = "filtered"

        row = {
            "model_id": model_id,
            "mode": mode,
            "status": status,
            "recast_success": model_id in recast_ids,
            "recast_time": "",  # Not available from files
            "validation_attempted": model_id in validation_data,
            "validation_pass": False,
            "error": "",
        }

        # Check validation
        if model_id in validation_data:
            vdata = validation_data[model_id]
            row["validation_pass"] = vdata.get("overall_pass", False)

        # Check failure reason
        if model_id in failure_ids:
            failure_file = failures_dir / f"{model_id}_{mode}.log"
            if not failure_file.exists():
                failure_file = failures_dir / f"{model_id}_{mode}.txt"
            try:
                with open(failure_file) as f:
                    error_text = f.read().strip()
                    # Extract first line or first 100 chars
                    first_line = error_text.split("\n")[0][:100]
                    row["error"] = first_line
            except Exception:
                row["error"] = "Unknown error"

        results.append(row)

    # Write CSV
    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "model_id",
            "mode",
            "status",
            "recast_success",
            "recast_time",
            "validation_attempted",
            "validation_pass",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Wrote {len(results)} rows to {output_file}")

    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")
    filtered_count = sum(1 for r in results if r["status"] == "filtered")
    processed_count = success_count + timeout_count + error_count
    validated_count = sum(1 for r in results if r["validation_attempted"])
    validation_pass_count = sum(1 for r in results if r["validation_pass"])

    print()
    print("Rebuild Summary")
    print("=" * 60)
    print(f"Total SBML candidates: {len(results)}")
    print(f"  Filtered out: {filtered_count}")
    print(f"  Processed: {processed_count}")
    print()
    print(f"Processing Results:")
    if processed_count > 0:
        print(f"  Success: {success_count} ({100*success_count/processed_count:.1f}%)")
        print(f"  Timeout: {timeout_count}")
        print(f"  Error: {error_count}")
    print()
    print(f"Validation:")
    print(f"  Attempted: {validated_count}")
    print(f"  Passed: {validation_pass_count}")
    print()
    print(f"CSV written to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild results CSV from files")
    parser.add_argument(
        "--mode",
        choices=["simplified", "canonical"],
        default="simplified",
        help="Recast mode (default: simplified)",
    )
    args = parser.parse_args()

    rebuild_results_csv(args.mode)


if __name__ == "__main__":
    main()
