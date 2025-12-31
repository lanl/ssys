#!/usr/bin/env python3
"""
Validate existing recasts against their original SBML files.

Runs validation on recasts produced by 3_recast_batch.py.
This allows recasting and validation to be run as separate phases.

Three-Stage Validation Pipeline:
Stage 1 (fast, robust):
    python 3b_validate_batch.py --numerical-only --timeout 60 --workers 8

Stage 2 (JAX cross-check on passed models):
    python 3b_validate_batch.py --numerical-only --use-jax --passed-only \
        --timeout 120 --workers 8

Stage 3 (symbolic proof on passed models, subprocess isolation):
    python 3b_validate_batch.py --symbolic-only --passed-only --subprocess \
        --timeout 120 --workers 4

Usage:
    # Validate all existing recasts
    python 3b_validate_batch.py

    # Validate specific mode
    python 3b_validate_batch.py --mode simplified

    # Limit number of models
    python 3b_validate_batch.py --limit 10

    # Skip models already validated
    python 3b_validate_batch.py --resume

    # Only validate models that passed numerical validation
    python 3b_validate_batch.py --passed-only --symbolic-only

Output:
    results/validation/ - Validation JSON reports
    results/validation_summary.txt - Summary statistics
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import warnings
from pathlib import Path

import antimony
import numpy as np

# Suppress NumPy warnings about invalid values (e.g., negative^fractional)
# These occur during lambdified function evaluation and are handled by
# detecting NaN in the results.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(invalid="ignore", divide="ignore")

# Add parent directory to path for ssys import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
import utils  # noqa: E402

from ssys.validator import validate_recast_pair  # noqa: E402

logger = logging.getLogger(__name__)


def find_recast_files(mode: str) -> list[tuple[str, Path, Path]]:
    """
    Find all recast files and their matching SBML originals.

    Returns:
        List of (model_id, sbml_path, recast_path) tuples
    """
    recasts_dir = Path(config.RECASTS_DIR)
    sbml_dir = Path(config.SBML_CANDIDATES_DIR)

    if not recasts_dir.exists():
        logger.error(f"Recasts directory not found: {recasts_dir}")
        return []

    pairs = []
    for recast_file in sorted(recasts_dir.glob(f"*_{mode}.ant")):
        # Extract model_id from filename: BIOMD0000000146_simplified.ant
        model_id = recast_file.stem.replace(f"_{mode}", "")

        # Find matching SBML
        sbml_path = sbml_dir / f"{model_id}.xml"
        if sbml_path.exists():
            pairs.append((model_id, sbml_path, recast_file))
        else:
            logger.warning(f"SBML not found for {model_id}")

    return pairs


def filter_passed_models(
    pairs: list[tuple[str, Path, Path]], mode: str
) -> list[tuple[str, Path, Path]]:
    """
    Filter to only models that passed numerical validation.
    
    Checks existing validation reports for numerical test pass.
    """
    filtered = []
    for model_id, sbml_path, recast_path in pairs:
        val_path = Path(config.VALIDATION_DIR) / f"{model_id}_{mode}_validation.json"
        if not val_path.exists():
            continue
        
        try:
            with open(val_path) as f:
                report = json.load(f)
            
            # Check if numerical test passed
            tests = report.get("tests", {})
            num_test = tests.get("numerical", {})
            if num_test.get("result") == "pass":
                filtered.append((model_id, sbml_path, recast_path))
        except Exception:
            continue
    
    return filtered


def validate_model(
    model_id: str, sbml_path: Path, recast_path: Path, mode: str,
    symbolic_only: bool = False, numerical_only: bool = False,
    use_jax: bool = False
) -> dict:
    """
    Validate a single recast against its original SBML.

    Args:
        model_id: Model identifier
        sbml_path: Path to original SBML file
        recast_path: Path to recast Antimony file
        mode: Recast mode ('simplified' or 'canonical')
        symbolic_only: If True, only run symbolic test
        numerical_only: If True, only run numerical test (fast, no symbolic)
        use_jax: Use JAX autodiff for numerical validation

    Returns:
        Validation report dict
    """
    try:
        # Convert SBML to Antimony for validation
        antimony.clearPreviousLoads()
        result = antimony.loadSBMLFile(str(sbml_path))
        if result == -1:
            raise ValueError(
                f"Failed to load SBML: {antimony.getLastError()}"
            )
        antimony_text = antimony.getAntimonyString()

        # Write to temp file for validation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ant", delete=False
        ) as tmp:
            tmp.write(antimony_text)
            original_ant_path = tmp.name

        try:
            # Determine which tests to run
            if numerical_only:
                run_sym, run_num, run_traj = False, True, False
            elif symbolic_only:
                run_sym, run_num, run_traj = True, False, False
            else:
                run_sym, run_num, run_traj = True, True, True

            report = validate_recast_pair(
                original_ant_path,
                str(recast_path),
                mode=mode,
                parser="sbml",
                run_symbolic=run_sym,
                run_numerical=run_num,
                run_trajectory=run_traj,
                use_jax=use_jax,
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


def format_test_result(report: dict) -> str:
    """Format a short test result summary."""
    if "error" in report:
        return f"ERROR: {report['error'][:50]}..."

    tests = report.get("tests", {})
    
    sym_test = tests.get("symbolic")
    num_test = tests.get("numerical")
    traj_test = tests.get("trajectory")
    
    symbolic = sym_test.get("result", "?") if sym_test else "-"
    numerical = num_test.get("result", "?") if num_test else "-"
    trajectory = traj_test.get("result", "?") if traj_test else "-"

    s = "✓" if symbolic == "pass" else ("✗" if symbolic == "fail" else "-")
    n = "✓" if numerical == "pass" else ("✗" if numerical == "fail" else "-")
    t = "✓" if trajectory == "pass" else ("✗" if trajectory == "fail" else "-")

    return f"sym:{s} num:{n} traj:{t}"


def _subprocess_validate(args_tuple):
    """
    Run validation in a subprocess (for reliable timeout of SymPy).
    
    This function runs in a separate process to allow hard kills
    when SymPy hangs during symbolic simplification.
    """
    (model_id, sbml_path, recast_path, mode, 
     symbolic_only, numerical_only, use_jax, timeout) = args_tuple
    
    result_queue = mp.Queue()
    
    def worker():
        try:
            report = validate_model(
                model_id, sbml_path, recast_path, mode,
                symbolic_only, numerical_only, use_jax
            )
            result_queue.put(("success", report))
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    p = mp.Process(target=worker)
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        # Force kill the subprocess
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()  # SIGKILL as last resort
            p.join()
        return {
            "model_id": model_id,
            "mode": mode,
            "overall_pass": False,
            "error": f"Timeout after {timeout}s (subprocess killed)",
        }
    
    try:
        status, result = result_queue.get_nowait()
        if status == "success":
            return result
        else:
            return {
                "model_id": model_id,
                "mode": mode,
                "overall_pass": False,
                "error": result,
            }
    except Exception:
        return {
            "model_id": model_id,
            "mode": mode,
            "overall_pass": False,
            "error": "Subprocess communication failed",
        }


def _parallel_validate_worker(args):
    """Worker function for parallel validation."""
    # Suppress NumPy warnings in worker process
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(invalid="ignore", divide="ignore")
    
    (model_id, sbml_path, recast_path, mode,
     symbolic_only, numerical_only, use_jax, timeout, use_subprocess) = args
    
    if use_subprocess:
        report = _subprocess_validate((
            model_id, sbml_path, recast_path, mode,
            symbolic_only, numerical_only, use_jax, timeout
        ))
    else:
        # Use thread-based timeout (may not work for SymPy hangs)
        success, report, error = utils.safe_execute(
            validate_model,
            model_id, sbml_path, recast_path, mode,
            symbolic_only, numerical_only, use_jax,
            timeout_sec=timeout,
            default=None
        )
        
        if not success or report is None:
            error_msg = error if error else f"Timeout after {timeout}s"
            report = {
                "model_id": model_id,
                "mode": mode,
                "overall_pass": False,
                "error": error_msg,
            }
    
    return model_id, report


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Validate existing recasts against original SBML files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simplified", "canonical"],
        default="simplified",
        help="Recast mode to validate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of models to validate (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have validation reports"
    )
    parser.add_argument(
        "--symbolic-only",
        action="store_true",
        help="Only run symbolic validation (can hang on complex models)"
    )
    parser.add_argument(
        "--numerical-only",
        action="store_true",
        help="Only run numerical validation (fast, no symbolic simplification)"
    )
    parser.add_argument(
        "--use-jax",
        action="store_true",
        help="Use JAX autodiff for numerical validation (faster)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per model in seconds (default: 60)"
    )
    parser.add_argument(
        "--passed-only",
        action="store_true",
        help="Only validate models that passed numerical validation (for stage 2+3)"
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each validation in subprocess (reliable timeout for SymPy hangs)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (-1 for all CPUs, default: 1)"
    )

    args = parser.parse_args()

    # Set up logging
    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("BioModels Validation Script")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbolic only: {args.symbolic_only}")
    logger.info(f"Numerical only: {args.numerical_only}")
    logger.info(f"Use JAX: {args.use_jax}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Passed only: {args.passed_only}")
    logger.info(f"Subprocess isolation: {args.subprocess}")
    logger.info(f"Workers: {args.workers}")

    # Find recast files
    pairs = find_recast_files(args.mode)

    if not pairs:
        logger.error("No recast files found")
        return

    logger.info(f"Found {len(pairs)} recast files to validate")

    # Apply limit if specified
    if args.limit:
        pairs = pairs[:args.limit]
        logger.info(f"Limited to first {args.limit} models")

    # Filter to passed models if requested (for stage 2+3 validation)
    if args.passed_only:
        original_count = len(pairs)
        pairs = filter_passed_models(pairs, args.mode)
        logger.info(
            f"Filtered to {len(pairs)} passed models (from {original_count})"
        )

    # Filter out already validated if resuming
    if args.resume:
        filtered = []
        for model_id, sbml_path, recast_path in pairs:
            val_path = (
                Path(config.VALIDATION_DIR) / f"{model_id}_{args.mode}_validation.json"
            )
            if not val_path.exists():
                filtered.append((model_id, sbml_path, recast_path))
        skipped = len(pairs) - len(filtered)
        pairs = filtered
        if skipped > 0:
            logger.info(f"Skipped {skipped} already validated models")

    if not pairs:
        logger.info("Nothing to validate")
        return

    # Determine number of workers
    n_workers = args.workers
    if n_workers == -1:
        n_workers = os.cpu_count() or 1
    
    # Validate each model with progress indicators
    pass_count = 0
    fail_count = 0
    error_count = 0

    print(f"\nValidating {len(pairs)} models with {n_workers} workers...\n")
    print("-" * 70)

    if n_workers > 1:
        # Parallel execution
        from joblib import Parallel, delayed
        
        # Prepare arguments for parallel workers
        work_items = [
            (model_id, str(sbml_path), str(recast_path), args.mode,
             args.symbolic_only, args.numerical_only, args.use_jax,
             args.timeout, args.subprocess)
            for model_id, sbml_path, recast_path in pairs
        ]
        
        # Run in parallel with progress reporting
        results = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(_parallel_validate_worker)(item) for item in work_items
        )
        
        # Process results
        for model_id, report in results:
            save_validation_report(model_id, args.mode, report)
            
            if "error" in report:
                error_count += 1
            elif report.get("overall_pass", False):
                pass_count += 1
            else:
                fail_count += 1
    else:
        # Sequential execution (original behavior)
        for i, (model_id, sbml_path, recast_path) in enumerate(pairs, 1):
            if args.subprocess:
                # Subprocess isolation for reliable timeout
                report = _subprocess_validate((
                    model_id, sbml_path, recast_path, args.mode,
                    args.symbolic_only, args.numerical_only, args.use_jax,
                    args.timeout
                ))
            else:
                # Thread-based timeout (may not work for SymPy)
                success, report, error = utils.safe_execute(
                    validate_model,
                    model_id, sbml_path, recast_path, args.mode,
                    args.symbolic_only, args.numerical_only, args.use_jax,
                    timeout_sec=args.timeout,
                    default=None
                )
                
                if not success or report is None:
                    error_msg = error if error else f"Timeout after {args.timeout}s"
                    report = {
                        "model_id": model_id,
                        "mode": args.mode,
                        "overall_pass": False,
                        "error": error_msg,
                    }

            # Save report
            save_validation_report(model_id, args.mode, report)

            # Update counts
            if "error" in report:
                error_count += 1
                status = "⚠ ERROR"
            elif report.get("overall_pass", False):
                pass_count += 1
                status = "✓ PASS"
            else:
                fail_count += 1
                status = "✗ FAIL"

            # Print progress line
            test_summary = format_test_result(report)
            total = pass_count + fail_count + error_count
            pct = 100 * pass_count / total if total > 0 else 0

            print(
                f"[{i:3d}/{len(pairs)}] {model_id:25s} {status:8s} ({test_summary})"
            )

            # Print running totals every 10 models or at the end
            if i % 10 == 0 or i == len(pairs):
                print(
                    f"         Running: ✓ {pass_count} pass | "
                    f"✗ {fail_count} fail | ⚠ {error_count} error | "
                    f"{pct:.1f}% pass rate"
                )
                print("-" * 70)

    # Final summary
    total = pass_count + fail_count + error_count
    pass_rate = 100 * pass_count / total if total > 0 else 0

    summary = f"""
Validation Summary
==================
Total validated: {total}
Pass: {pass_count} ({pass_rate:.1f}%)
Fail: {fail_count}
Error: {error_count}
"""

    print(summary)

    # Save summary
    summary_path = Path(config.RESULTS_DIR) / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    logger.info(f"Saved summary to {summary_path}")
    logger.info("\nValidation complete!")


if __name__ == "__main__":
    main()
