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

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add parent directory to path for ssys import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
import utils  # noqa: E402

# Import ssys library
import ssys  # noqa: E402

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

RECAST_POLICY_FIELDS = [
    "recast_attempt_role",
    "recast_attempt_count",
    "recast_base_timeout_seconds",
    "recast_retry_timeout_seconds",
    "recast_final_attempt_timeout_seconds",
    "recast_retry_policy",
    "recast_recovered_by_retry",
]

RECAST_TELEMETRY_FIELDS = [
    "recast_last_phase",
    "recast_dominant_phase_attribution",
]

RESULT_COLUMNS = [
    "model_id",
    "mode",
    "recast_success",
    "recast_time",
    "recast_phase_history",
    "recast_phase_seconds",
    "recast_dominant_phase",
    "recast_dominant_phase_seconds",
    *RECAST_TELEMETRY_FIELDS,
    *RECAST_POLICY_FIELDS,
    "validation_attempted",
    "validation_pass",
    "error",
]


def _read_result_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _result_fieldnames(
    existing_fieldnames: list[str],
    rows: list[dict],
) -> list[str]:
    fieldnames: list[str] = []
    for field in [*existing_fieldnames, *RESULT_COLUMNS]:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def _write_result_rows(
    path: Path,
    rows: list[dict],
    *,
    existing_fieldnames: list[str] | None = None,
) -> None:
    fieldnames = _result_fieldnames(existing_fieldnames or [], rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_candidates(filter_type: str | None = None) -> pd.DataFrame:
    """
    Load candidates CSV and optionally filter.

    Args:
        filter_type: 's_system', 'gma', or None (all)

    Returns:
        DataFrame of candidates to attempt
    """
    import pandas as pd

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


def _dominant_phase_from_seconds(phase_seconds: dict[str, float]) -> tuple[str, float] | None:
    """Return the slowest recorded recast phase."""
    if not phase_seconds:
        return None
    phase, seconds = max(phase_seconds.items(), key=lambda item: item[1])
    return phase, seconds


def _last_phase_from_history(phase_history: list[dict]) -> str:
    for item in reversed(phase_history):
        if isinstance(item, dict) and item.get("phase"):
            return str(item["phase"])
    return ""


def _open_phase_interval(
    phase_history: list[dict],
    *,
    elapsed_seconds: float,
) -> tuple[str, float] | None:
    open_starts: dict[str, float] = {}
    for item in phase_history:
        if not isinstance(item, dict) or not item.get("phase"):
            continue
        try:
            elapsed = float(item.get("elapsed_seconds"))
        except (TypeError, ValueError):
            continue
        phase = str(item["phase"])
        event = str(item.get("event", "phase_start"))
        if event == "phase_start":
            open_starts[phase] = elapsed
        elif event == "phase_end":
            open_starts.pop(phase, None)

    if not open_starts:
        return None
    phase, started = max(open_starts.items(), key=lambda item: item[1])
    if elapsed_seconds <= started:
        return None
    return phase, elapsed_seconds - started


def _apply_recast_timing(
    result: dict,
    recast_timing: dict,
    *,
    elapsed_seconds: float | None = None,
    timed_out: bool = False,
) -> None:
    phase_history = recast_timing.get("phase_history", [])
    if not isinstance(phase_history, list):
        phase_history = []
    phase_seconds = recast_timing.get("phase_seconds", {})
    if not isinstance(phase_seconds, dict):
        phase_seconds = {}

    result["recast_phase_history"] = json.dumps(phase_history, separators=(",", ":"))
    result["recast_phase_seconds"] = json.dumps(phase_seconds, separators=(",", ":"))
    result["recast_last_phase"] = recast_timing.get(
        "last_phase",
        _last_phase_from_history(phase_history),
    )

    if timed_out and elapsed_seconds is not None:
        open_interval = _open_phase_interval(phase_history, elapsed_seconds=elapsed_seconds)
        if open_interval is not None:
            phase, seconds = open_interval
            result["recast_dominant_phase"] = phase
            result["recast_dominant_phase_seconds"] = round(seconds, 6)
            result["recast_last_phase"] = phase
            result["recast_dominant_phase_attribution"] = "inferred_open_interval"
            return

    dominant_phase = recast_timing.get("dominant_phase", "")
    dominant_phase_seconds = recast_timing.get("dominant_phase_seconds", "")
    if not dominant_phase:
        dominant = _dominant_phase_from_seconds({
            str(phase): float(seconds)
            for phase, seconds in phase_seconds.items()
            if isinstance(seconds, (int, float))
        })
        if dominant is not None:
            dominant_phase, dominant_phase_seconds = dominant

    result["recast_dominant_phase"] = dominant_phase
    result["recast_dominant_phase_seconds"] = dominant_phase_seconds
    if dominant_phase:
        result["recast_dominant_phase_attribution"] = recast_timing.get(
            "dominant_phase_attribution",
            "measured",
        )


def attempt_recast(
    model_id: str,
    mode: str,
    phase_recorder: dict | None = None,
) -> tuple[bool, str | None, str | None, dict]:
    """
    Attempt to recast a single model using SBML.

    Args:
        model_id: Model identifier
        mode: 'simplified' or 'canonical'

    Returns:
        (success, recast_text, error_message, timing_metadata)
    """
    sbml_path = Path(config.SBML_CANDIDATES_DIR) / f"{model_id}.xml"
    import time

    started = time.perf_counter()
    phase_history = []
    phase_seconds = {}

    def update_phase_recorder() -> None:
        if phase_recorder is not None:
            phase_recorder.clear()
            phase_recorder.update(timing_metadata())

    def start_phase(phase: str) -> float:
        elapsed = time.perf_counter() - started
        phase_history.append({
            "event": "phase_start",
            "phase": phase,
            "elapsed_seconds": round(elapsed, 6),
        })
        update_phase_recorder()
        return time.perf_counter()

    def end_phase(phase: str, phase_start: float) -> None:
        elapsed = time.perf_counter() - started
        seconds = time.perf_counter() - phase_start
        phase_seconds[phase] = seconds
        phase_history.append({
            "event": "phase_end",
            "phase": phase,
            "elapsed_seconds": round(elapsed, 6),
            "phase_seconds": round(seconds, 6),
        })
        update_phase_recorder()

    def timing_metadata() -> dict:
        dominant = _dominant_phase_from_seconds(phase_seconds)
        metadata = {
            "phase_history": phase_history,
            "phase_seconds": {key: round(value, 6) for key, value in phase_seconds.items()},
            "last_phase": _last_phase_from_history(phase_history),
        }
        if dominant is not None:
            metadata["dominant_phase"] = dominant[0]
            metadata["dominant_phase_seconds"] = round(dominant[1], 6)
            metadata["dominant_phase_attribution"] = "measured"
        return metadata

    if not sbml_path.exists():
        return False, None, f"SBML file not found: {sbml_path}", timing_metadata()

    try:
        # Parse SBML directly using libSBML
        phase_start = start_phase("parse_sbml")
        sym = ssys.parse_sbml(str(sbml_path))
        end_phase("parse_sbml", phase_start)

        # Recast
        phase_start = start_phase("recast_to_ssystem")
        result = ssys.recast_to_ssystem(sym, mode=mode)
        end_phase("recast_to_ssystem", phase_start)

        # Generate output
        phase_start = start_phase("ssystem_to_antimony")
        out_text = ssys.ssystem_to_antimony(result, model_name=f"{model_id}_recast", mode=mode)
        end_phase("ssystem_to_antimony", phase_start)

        return True, out_text, None, timing_metadata()

    except utils.TimeoutError:
        raise
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}", timing_metadata()


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

    if "recastcomplexityerror" in error_lower or "recast_complexity:" in error_lower:
        return ("RECAST_COMPLEXITY",
                "Model exceeded a bounded direct-recast symbolic complexity budget. "
                "The recaster failed closed before generating an artifact, preserving "
                "structured stage, operation, threshold, and expression-preview details.")

    if "piecewise" in error_lower or "event" in error_lower:
        return ("UNSUPPORTED_CONSTRUCT",
                "Model contains piecewise functions or events, which are not supported "
                "by algebraic recasting. These models have discontinuous dynamics that "
                "cannot be represented in S-system/GMA form.")

    if (
        "unsupported_generated_output" in error_lower
        or "unsupported composite derivative" in error_lower
        or "unsupported derivative" in error_lower
    ):
        return ("UNSUPPORTED_CONSTRUCT",
                "Model contains an unsupported derivative of a nonsmooth or unknown "
                "function. Recasting fails closed before generating invalid Antimony.")

    if "delay" in error_lower:
        return ("UNSUPPORTED_CONSTRUCT",
                "Model contains time delays (delay differential equations). "
                "S-system recasting only supports ODEs, not DDEs.")

    if (
        "parse" in error_lower
        or "parsing" in error_lower
        or "syntax" in error_lower
    ):
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


def log_failure(model_id: str, mode: str, error_msg: str, metadata: dict | None = None):
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

    if metadata:
        metadata_path = Path(config.FAILURES_DIR) / f"{model_id}_{mode}_recast_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def failure_metadata_from_result(result: dict) -> dict:
    """Return sidecar metadata that reconciles a failure log with CSV output."""
    keys = [
        "model_id",
        "mode",
        "recast_success",
        "recast_time",
        "recast_phase_history",
        "recast_phase_seconds",
        "recast_dominant_phase",
        "recast_dominant_phase_seconds",
        *RECAST_TELEMETRY_FIELDS,
        *RECAST_POLICY_FIELDS,
        "error",
    ]
    return {key: result.get(key) for key in keys}


def recast_policy_metadata(
    *,
    attempt_role: str,
    timeout: int,
    base_timeout: int | None,
    retry_timeout: int | None,
    recast_policy: str,
    recast_success: bool = False,
) -> dict:
    """Return CSV-stable metadata for the selected recast attempt policy."""
    return {
        "recast_attempt_role": attempt_role,
        "recast_attempt_count": 2 if attempt_role == "retry" else 1,
        "recast_base_timeout_seconds": (
            base_timeout if base_timeout is not None else timeout
        ),
        "recast_retry_timeout_seconds": retry_timeout if retry_timeout is not None else "",
        "recast_final_attempt_timeout_seconds": timeout,
        "recast_retry_policy": recast_policy,
        "recast_recovered_by_retry": attempt_role == "retry" and recast_success,
    }


def worker_error_result(
    model_id: str,
    mode: str,
    exc: BaseException,
    *,
    attempt_role: str = "base",
    timeout: int = 15,
    base_timeout: int | None = None,
    retry_timeout: int | None = None,
    recast_policy: str = "base_timeout_only",
) -> dict:
    """Record a worker-level failure in the same shape as process_model()."""
    error_msg = f"Worker error: {type(exc).__name__}: {exc}"
    log_failure(model_id, mode, error_msg)
    return {
        "model_id": model_id,
        "mode": mode,
        "recast_success": False,
        "recast_time": 0.0,
        "recast_phase_history": "",
        "recast_phase_seconds": "",
        "recast_dominant_phase": "",
        "recast_dominant_phase_seconds": "",
        "recast_last_phase": "",
        "recast_dominant_phase_attribution": "",
        **recast_policy_metadata(
            attempt_role=attempt_role,
            timeout=timeout,
            base_timeout=base_timeout,
            retry_timeout=retry_timeout,
            recast_policy=recast_policy,
        ),
        "validation_attempted": False,
        "validation_pass": False,
        "error": error_msg,
    }


def process_model(
    model_id: str,
    mode: str,
    validate: bool = True,
    timeout: int = 15,
    *,
    attempt_role: str = "base",
    base_timeout: int | None = None,
    retry_timeout: int | None = None,
    recast_policy: str = "base_timeout_only",
) -> dict:
    """
    Process a single model: recast and optionally validate.

    Args:
        model_id: Model identifier
        mode: 'simplified' or 'canonical'
        validate: Whether to run validation
        timeout: Timeout in seconds for the recast operation
        attempt_role: Whether this is the base attempt or timeout retry
        base_timeout: Historical/base timeout budget in seconds
        retry_timeout: Retry timeout budget in seconds, if configured
        recast_policy: Stable policy label recorded in CSV output

    Returns:
        Results dictionary
    """
    result = {
        "model_id": model_id,
        "mode": mode,
        "recast_success": False,
        "recast_time": 0.0,
        "recast_phase_history": "",
        "recast_phase_seconds": "",
        "recast_dominant_phase": "",
        "recast_dominant_phase_seconds": "",
        "recast_last_phase": "",
        "recast_dominant_phase_attribution": "",
        **recast_policy_metadata(
            attempt_role=attempt_role,
            timeout=timeout,
            base_timeout=base_timeout,
            retry_timeout=retry_timeout,
            recast_policy=recast_policy,
        ),
        "validation_attempted": False,
        "validation_pass": False,
        "error": None,
    }

    # Attempt recast with timeout
    import time

    start_time = time.time()
    recast_timing: dict = {}

    success, result_tuple, error = utils.safe_execute(
        attempt_recast,
        model_id,
        mode,
        phase_recorder=recast_timing,
        timeout_sec=timeout,
        default=(False, None, "Timeout"),
    )

    result["recast_time"] = time.time() - start_time
    if recast_timing:
        _apply_recast_timing(
            result,
            recast_timing,
            elapsed_seconds=result["recast_time"],
            timed_out=not success and "timeout" in str(error).lower(),
        )

    # Check if safe_execute failed (timeout or exception)
    if not success:
        result["error"] = error if error else "Unknown error"
        log_failure(
            model_id,
            mode,
            result["error"],
            metadata=failure_metadata_from_result(result),
        )
        return result

    # Unpack attempt_recast result
    if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
        recast_success, recast_text, recast_error, recast_timing = result_tuple
    else:
        recast_success, recast_text, recast_error = result_tuple
    if recast_timing:
        _apply_recast_timing(result, recast_timing)

    # Check if recast itself failed
    if not recast_success:
        result["error"] = recast_error if recast_error else "Recast failed"
        log_failure(
            model_id,
            mode,
            result["error"],
            metadata=failure_metadata_from_result(result),
        )
        return result

    # Save recast
    result["recast_success"] = True
    result["recast_recovered_by_retry"] = attempt_role == "retry"
    try:
        save_recast(model_id, mode, recast_text)
    except Exception as e:
        result["error"] = f"Failed to save: {e}"
        log_failure(
            model_id,
            mode,
            result["error"],
            metadata=failure_metadata_from_result(result),
        )
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
    attempt_role: str,
    base_timeout: int | None,
    retry_timeout: int | None,
    recast_policy: str,
) -> list[dict]:
    """Process candidate models sequentially or with a process pool."""
    if workers <= 1:
        return [
            process_model(
                model_id,
                mode,
                validate=validate,
                timeout=timeout,
                attempt_role=attempt_role,
                base_timeout=base_timeout,
                retry_timeout=retry_timeout,
                recast_policy=recast_policy,
            )
            for model_id in tqdm(model_ids, total=len(model_ids), desc="Recasting")
        ]

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_model,
                model_id,
                mode,
                validate,
                timeout,
                attempt_role=attempt_role,
                base_timeout=base_timeout,
                retry_timeout=retry_timeout,
                recast_policy=recast_policy,
            ): model_id
            for model_id in model_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Recasting"):
            model_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    worker_error_result(
                        model_id,
                        mode,
                        exc,
                        attempt_role=attempt_role,
                        timeout=timeout,
                        base_timeout=base_timeout,
                        retry_timeout=retry_timeout,
                        recast_policy=recast_policy,
                    )
                )
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
        "--attempt-role",
        choices=["base", "retry"],
        default=None,
        help=(
            "Policy role for this recast attempt. Defaults to 'retry' with "
            "--retry-timeouts and 'base' otherwise."
        ),
    )
    parser.add_argument(
        "--base-timeout",
        type=int,
        default=None,
        help="Base/first-attempt timeout budget in seconds for policy metadata.",
    )
    parser.add_argument(
        "--retry-timeout",
        type=int,
        default=None,
        help="Retry timeout budget in seconds for policy metadata.",
    )
    parser.add_argument(
        "--recast-policy",
        default=None,
        help="Stable recast retry policy label recorded in CSV output.",
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

    attempt_role = args.attempt_role or ("retry" if args.retry_timeouts else "base")
    base_timeout = args.base_timeout if args.base_timeout is not None else args.timeout
    retry_timeout = args.retry_timeout
    if retry_timeout is None and attempt_role == "retry":
        retry_timeout = args.timeout
    recast_policy = args.recast_policy
    if recast_policy is None:
        recast_policy = (
            "quick_then_retry_timeouts"
            if args.retry_timeouts or attempt_role == "retry"
            else "base_timeout_only"
        )
    logger.info(f"Recast attempt role: {attempt_role}")
    logger.info(f"Recast policy: {recast_policy}")
    logger.info(f"Base timeout: {base_timeout}s")
    logger.info(
        "Retry timeout: %s",
        f"{retry_timeout}s" if retry_timeout is not None else "not configured",
    )

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
        attempt_role=attempt_role,
        base_timeout=base_timeout,
        retry_timeout=retry_timeout,
        recast_policy=recast_policy,
    )

    if skipped > 0:
        logger.info(f"Skipped {skipped} models (already have output files)")

    # Save detailed results - MERGE with existing CSV instead of overwriting
    results_csv = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    if results_csv.exists() and not results:
        logger.info(f"No new results; preserved existing {results_csv}")
    elif results_csv.exists():
        existing_rows, existing_fieldnames = _read_result_rows(results_csv)
        new_keys = {(row["model_id"], row["mode"]) for row in results}
        retained_rows = [
            row
            for row in existing_rows
            if (row.get("model_id"), row.get("mode")) not in new_keys
        ]
        merged_rows = sorted(
            [*retained_rows, *results],
            key=lambda row: str(row.get("model_id", "")),
        )
        _write_result_rows(
            results_csv,
            merged_rows,
            existing_fieldnames=existing_fieldnames,
        )
        logger.info(f"Merged {len(results)} new results with {len(retained_rows)} existing")
        logger.info(f"Total results in {results_csv}: {len(merged_rows)}")
    else:
        # No existing file or no new results - just save
        _write_result_rows(results_csv, results, existing_fieldnames=RESULT_COLUMNS)
        logger.info(f"Saved {len(results)} results to {results_csv}")

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
