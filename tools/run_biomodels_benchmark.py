#!/usr/bin/env python3
"""Run or summarize the local BioModels benchmark with release evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_DIR = Path("release-evidence/biomodels")
DEFAULT_RECAST_QUICK_TIMEOUT_SECONDS = 15
DEFAULT_RECAST_TIMEOUT_SECONDS = 60
DEFAULT_VALIDATION_TIMEOUT_SECONDS = 60
DEFAULT_NEAR_TIMEOUT_FRACTION = 0.85
DEFAULT_RECAST_RETRY_POLICY = "quick_then_retry_timeouts"
VALIDATION_PATTERNS = ("*_numerical.json", "*_validation.json", "*_symbolic.json")
BENCHMARK_COPY_IGNORES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "results",
    "exports",
    "benchmark.log",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_fraction(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _count_files(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern))) if directory.exists() else 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed <= 0:
        return None
    return int(parsed)


def _bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validation_files(validation_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in VALIDATION_PATTERNS:
        files.extend(validation_dir.glob(pattern))
    return sorted(set(files))


def _iter_validation_tests(report: dict[str, Any]):
    for test in report.get("tests", {}).values():
        if isinstance(test, dict):
            yield test
        elif isinstance(test, list):
            for item in test:
                if isinstance(item, dict):
                    yield item


def _phase_intervals_from_history(
    phase_history: Any,
    *,
    final_elapsed_seconds: float | int | None = None,
) -> list[dict[str, Any]]:
    """Derive phase intervals from phase-start history records."""
    if not isinstance(phase_history, list):
        return []

    starts: list[tuple[str, float]] = []
    for item in phase_history:
        if not isinstance(item, dict) or not item.get("phase"):
            continue
        elapsed = _float_or_none(item.get("elapsed_seconds"))
        if elapsed is None:
            continue
        starts.append((str(item["phase"]), elapsed))

    intervals: list[dict[str, Any]] = []
    for (phase, start), (next_phase, end) in zip(starts, starts[1:]):
        if end < start:
            continue
        intervals.append({
            "phase": phase,
            "start_seconds": round(start, 6),
            "end_seconds": round(end, 6),
            "duration_seconds": round(end - start, 6),
            "next_phase": next_phase,
        })

    final_elapsed = _float_or_none(final_elapsed_seconds)
    if final_elapsed is not None and starts:
        phase, start = starts[-1]
        if final_elapsed > start:
            intervals.append({
                "phase": phase,
                "start_seconds": round(start, 6),
                "end_seconds": round(final_elapsed, 6),
                "duration_seconds": round(final_elapsed - start, 6),
                "next_phase": "timeout_budget",
            })

    return intervals


def _dominant_phase_interval(phase_intervals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not phase_intervals:
        return None
    return max(phase_intervals, key=lambda item: item.get("duration_seconds", 0.0))


def _json_value_or_none(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _recast_phase_seconds(row: dict[str, str]) -> dict[str, float]:
    raw = _json_value_or_none(row.get("recast_phase_seconds"))
    if not isinstance(raw, dict):
        return {}
    phase_seconds: dict[str, float] = {}
    for phase, seconds in raw.items():
        parsed = _float_or_none(seconds)
        if parsed is not None:
            phase_seconds[str(phase)] = parsed
    return phase_seconds


def _dominant_recast_phase_from_seconds(
    phase_seconds: dict[str, float],
) -> tuple[str, float] | None:
    if not phase_seconds:
        return None
    phase, seconds = max(phase_seconds.items(), key=lambda item: item[1])
    return phase, seconds


def _recast_phase_progress(
    row: dict[str, str],
    *,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    """Return measured and open recast phase intervals from CSV timing history."""
    raw_history = _json_value_or_none(row.get("recast_phase_history"))
    if not isinstance(raw_history, list):
        return {
            "last_phase": None,
            "open_phase_interval": None,
            "completed_intervals": [],
        }

    open_starts: dict[str, float] = {}
    completed_intervals: list[dict[str, Any]] = []
    last_phase: str | None = None
    for item in raw_history:
        if not isinstance(item, dict) or not item.get("phase"):
            continue
        phase = str(item["phase"])
        elapsed = _float_or_none(item.get("elapsed_seconds"))
        if elapsed is None:
            continue
        event = str(item.get("event", "phase_start"))
        last_phase = phase
        if event == "phase_start":
            open_starts[phase] = elapsed
        elif event == "phase_end":
            start = open_starts.pop(phase, None)
            if start is None:
                phase_seconds = _float_or_none(item.get("phase_seconds"))
                if phase_seconds is None:
                    continue
                start = elapsed - phase_seconds
            if elapsed < start:
                continue
            completed_intervals.append({
                "phase": phase,
                "start_seconds": round(start, 6),
                "end_seconds": round(elapsed, 6),
                "duration_seconds": round(elapsed - start, 6),
            })

    open_phase_interval = None
    final_elapsed = _float_or_none(elapsed_seconds)
    if final_elapsed is not None and open_starts:
        phase, start = max(open_starts.items(), key=lambda item: item[1])
        if final_elapsed > start:
            open_phase_interval = {
                "phase": phase,
                "start_seconds": round(start, 6),
                "end_seconds": round(final_elapsed, 6),
                "duration_seconds": round(final_elapsed - start, 6),
            }
            last_phase = phase

    return {
        "last_phase": last_phase,
        "open_phase_interval": open_phase_interval,
        "completed_intervals": completed_intervals,
    }


def _recast_row_timed_out(row: dict[str, str]) -> bool:
    status = str(row.get("status", "")).lower()
    error = str(row.get("error", "")).lower()
    recast_success = _bool_from_csv(row.get("recast_success"))
    return status == "timeout" or (not recast_success and "timeout" in error)


def _model_id_from_recast_failure_path(path: Path) -> str:
    stem = path.stem
    for marker in ("_simplified", "_canonical", "_gma"):
        if stem.endswith(marker):
            return stem[: -len(marker)]
    return stem


def _recast_failure_log_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in {"Model", "Mode", "Category", "Error"}:
            fields[key.lower()] = value.strip()
    return fields


def _parse_recast_complexity_error(message: str) -> dict[str, Any] | None:
    if "recast_complexity:" not in message and "RecastComplexityError" not in message:
        return None

    parsed: dict[str, Any] = {}
    patterns = {
        "stage": r"stage=([^;]+)",
        "operation": r"operation=([^;]+)",
        "expression_label": r"expression_label=([^;]+)",
        "operation_count": r"operation_count=(\d+)",
        "max_ops": r"max_ops=(\d+)",
        "free_symbol_count": r"free_symbol_count=(\d+)",
        "max_free_symbol_count": r"max_free_symbol_count=(\d+)",
        "expression_preview": r"expression_preview=(.*)$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, message)
        if match is None:
            continue
        value: str | int = match.group(1).strip()
        if key in {
            "operation_count",
            "max_ops",
            "free_symbol_count",
            "max_free_symbol_count",
        }:
            value = int(value)
        parsed[key] = value
    return parsed or None


def _recast_failure_category(fields: dict[str, str]) -> str:
    category = fields.get("category", "").strip().lower()
    error = fields.get("error", "")
    if _parse_recast_complexity_error(error) is not None:
        return "recast_complexity"
    error_lower = error.lower()
    if category in {"", "other"} and (
        "parse" in error_lower
        or "parsing" in error_lower
        or "syntax" in error_lower
    ):
        return "parse_error"
    if not category:
        return "unknown"
    return category.lower()


def _load_recast_failure_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name(f"{path.stem}_recast_metadata.json")
    if not metadata_path.exists():
        return {}
    return _read_json(metadata_path)


def _recast_failure_summaries(
    failures_dir: Path,
    *,
    active_failed_keys: set[tuple[str, str]] | None = None,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    category_counts: Counter[str] = Counter()
    complexity_models: list[dict[str, Any]] = []
    discovered_paths = sorted(
        [
            *failures_dir.glob("*_simplified.log"),
            *failures_dir.glob("*_canonical.log"),
            *failures_dir.glob("*_gma.log"),
            *failures_dir.glob("*_simplified.txt"),
            *failures_dir.glob("*_canonical.txt"),
            *failures_dir.glob("*_gma.txt"),
        ]
    )
    failure_paths_by_key: dict[tuple[str, str], Path] = {}
    for path in discovered_paths:
        mode = "unknown"
        for marker in ("_simplified", "_canonical", "_gma"):
            if marker in path.stem:
                mode = marker.removeprefix("_")
                break
        key = (_model_id_from_recast_failure_path(path), mode)
        if active_failed_keys is not None and key not in active_failed_keys:
            continue
        existing = failure_paths_by_key.get(key)
        if existing is None or (existing.suffix != ".log" and path.suffix == ".log"):
            failure_paths_by_key[key] = path
    failure_paths = sorted(failure_paths_by_key.values())

    for path in failure_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            category_counts["unreadable"] += 1
            continue
        fields = _recast_failure_log_fields(text)
        category = _recast_failure_category(fields)
        category_counts[category] += 1
        parsed = _parse_recast_complexity_error(fields.get("error", ""))
        if parsed is None:
            continue
        metadata = _load_recast_failure_metadata(path)
        entry: dict[str, Any] = {
            "model_id": fields.get("model") or _model_id_from_recast_failure_path(path),
            "mode": fields.get("mode"),
            "category": category,
            "failure_log": str(path),
            **parsed,
        }
        metadata_path = path.with_name(f"{path.stem}_recast_metadata.json")
        if metadata_path.exists():
            entry["metadata_path"] = str(metadata_path)
        elapsed = _float_or_none(metadata.get("recast_time"))
        if elapsed is not None:
            entry["elapsed_seconds"] = round(elapsed, 6)
        for key, value in _recast_policy_entry_fields({
            field: str(metadata.get(field, ""))
            for field in [
                "recast_attempt_role",
                "recast_attempt_count",
                "recast_base_timeout_seconds",
                "recast_retry_timeout_seconds",
                "recast_final_attempt_timeout_seconds",
                "recast_retry_policy",
                "recast_recovered_by_retry",
            ]
        }).items():
            entry[key] = value
        for key in [
            "recast_last_phase",
            "recast_dominant_phase",
            "recast_dominant_phase_seconds",
            "recast_dominant_phase_attribution",
        ]:
            if metadata.get(key) not in {None, ""}:
                entry[key] = metadata[key]
        complexity_models.append(entry)

    complexity_models.sort(key=lambda item: item["model_id"])
    return category_counts, complexity_models


def _recast_phase_attribution(
    row: dict[str, str],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Describe recast phase attribution without mislabeling killed timeouts."""
    phase_seconds = _recast_phase_seconds(row)
    progress = _recast_phase_progress(row, elapsed_seconds=elapsed_seconds)
    is_timeout = _recast_row_timed_out(row)

    if is_timeout:
        open_interval = progress.get("open_phase_interval")
        if isinstance(open_interval, dict):
            return {
                "dominant_phase": open_interval.get("phase", "unknown"),
                "dominant_phase_seconds": open_interval.get("duration_seconds"),
                "dominant_phase_attribution": "inferred_open_interval",
                "last_phase": open_interval.get("phase", "unknown"),
            }
        row_attribution = row.get("recast_dominant_phase_attribution") or ""
        row_dominant_phase = row.get("recast_dominant_phase") or ""
        if row_attribution == "inferred_open_interval" and row_dominant_phase:
            return {
                "dominant_phase": row_dominant_phase,
                "dominant_phase_seconds": _float_or_none(
                    row.get("recast_dominant_phase_seconds")
                ),
                "dominant_phase_attribution": row_attribution,
                "last_phase": row.get("recast_last_phase") or row_dominant_phase,
            }
        return {
            "dominant_phase": "unknown",
            "dominant_phase_seconds": None,
            "dominant_phase_attribution": "unknown",
            "last_phase": progress.get("last_phase") or row.get("recast_last_phase") or "unknown",
        }

    dominant_phase = row.get("recast_dominant_phase") or None
    dominant_phase_seconds = _float_or_none(row.get("recast_dominant_phase_seconds"))
    if not dominant_phase:
        dominant = _dominant_recast_phase_from_seconds(phase_seconds)
        if dominant is not None:
            dominant_phase, dominant_phase_seconds = dominant

    return {
        "dominant_phase": dominant_phase or "unknown",
        "dominant_phase_seconds": (
            round(dominant_phase_seconds, 6)
            if dominant_phase_seconds is not None
            else None
        ),
        "dominant_phase_attribution": (
            row.get("recast_dominant_phase_attribution")
            or ("measured" if dominant_phase else "unknown")
        ),
        "last_phase": progress.get("last_phase") or row.get("recast_last_phase") or "unknown",
    }


def _model_id_from_validation_path(path: Path) -> str:
    stem = path.stem
    for marker in ("_simplified_", "_canonical_", "_gma_"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def _validation_timing(
    report: dict[str, Any],
    *,
    timeout_seconds: int | None,
) -> dict[str, Any] | None:
    """Return the best available wrapper timing metadata for a validation report."""
    for test in _iter_validation_tests(report):
        metadata = test.get("metadata")
        if not isinstance(metadata, dict):
            continue

        history = metadata.get("phase_history")
        if not isinstance(history, list):
            continue

        metadata_timeout = _float_or_none(metadata.get("timeout_seconds"))
        selected_timeout = timeout_seconds
        if selected_timeout is None and metadata_timeout is not None:
            selected_timeout = int(metadata_timeout)

        final_elapsed: float | None = None
        if report.get("overall_result") == "timeout" and selected_timeout is not None:
            final_elapsed = float(selected_timeout)

        intervals = _phase_intervals_from_history(
            history,
            final_elapsed_seconds=final_elapsed,
        )
        dominant = _dominant_phase_interval(intervals)
        elapsed = final_elapsed
        if elapsed is None and history:
            last_elapsed = _float_or_none(history[-1].get("elapsed_seconds"))
            if last_elapsed is not None:
                elapsed = last_elapsed

        return {
            "elapsed_seconds": round(elapsed, 6) if elapsed is not None else None,
            "phase_history": history,
            "phase_intervals": intervals,
            "dominant_phase_interval": dominant,
            "last_phase": _phase_from_metadata(metadata),
            "timeout_seconds": selected_timeout,
        }
    return None


def _near_timeout_entry(
    *,
    model_id: str,
    elapsed_seconds: float,
    timeout_seconds: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "model_id": model_id,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "timeout_seconds": timeout_seconds,
        "budget_fraction": round(elapsed_seconds / timeout_seconds, 6),
    }
    if extra:
        entry.update(extra)
    return entry


def _recast_policy_entry_fields(row: dict[str, str]) -> dict[str, Any]:
    """Return populated recast policy fields for a summary model entry."""
    fields: dict[str, Any] = {}
    text_fields = {
        "recast_attempt_role": "recast_attempt_role",
        "recast_retry_policy": "recast_retry_policy",
    }
    int_fields = {
        "recast_attempt_count": "recast_attempt_count",
        "recast_base_timeout_seconds": "recast_base_timeout_seconds",
        "recast_retry_timeout_seconds": "recast_retry_timeout_seconds",
        "recast_final_attempt_timeout_seconds": "recast_final_attempt_timeout_seconds",
    }
    for source, destination in text_fields.items():
        value = row.get(source)
        if value not in {None, ""}:
            fields[destination] = value
    for source, destination in int_fields.items():
        value = _positive_int_or_none(row.get(source))
        if value is not None:
            fields[destination] = value
    recovered = row.get("recast_recovered_by_retry")
    if recovered not in {None, ""}:
        fields["recast_recovered_by_retry"] = _bool_from_csv(recovered)
    return fields


def _phase_from_metadata(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    phase = metadata.get("validation_phase")
    if phase:
        return str(phase)
    history = metadata.get("phase_history")
    if isinstance(history, list):
        for item in reversed(history):
            if isinstance(item, dict) and item.get("phase"):
                return str(item["phase"])
    return None


def _validation_timeout_phase(report: dict[str, Any]) -> str | None:
    """Return the last known validation phase for a timeout report."""
    for test in _iter_validation_tests(report):
        if test.get("reason") != "validator_timeout" and test.get("result") != "timeout":
            continue
        phase = _phase_from_metadata(test.get("metadata"))
        if phase:
            return phase
    if report.get("overall_result") == "timeout":
        return "unknown"
    if "error" in report and "timeout after" in str(report.get("error", "")).lower():
        return "unknown"
    return None


def _test_result(report: dict[str, Any], name: str) -> str | None:
    test = report.get("tests", {}).get(name)
    if isinstance(test, dict):
        result = test.get("result")
        return str(result) if result is not None else None
    return None


def _test_reason(report: dict[str, Any], name: str) -> str | None:
    test = report.get("tests", {}).get(name)
    if isinstance(test, dict):
        reason = test.get("reason")
        return str(reason) if reason is not None else None
    return None


def _has_failed_auxiliary_identity(report: dict[str, Any]) -> bool:
    """Return True when the auxiliary identity check is the failed requirement."""
    auxiliaries = report.get("tests", {}).get("auxiliaries")
    if not isinstance(auxiliaries, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("result") == "failed"
        and str(item.get("name", "")).endswith("_auxiliary_identity")
        for item in auxiliaries
    ) or any(
        isinstance(item, dict)
        and item.get("result") == "failed"
        and str(item.get("name", "")).startswith("ode_auxiliary_identity:")
        for item in auxiliaries
    )


def _classify_validation_non_pass(report: dict[str, Any]) -> str:
    """Return the model-level validation category for a report."""
    if "_read_error" in report:
        return "validation_report_unreadable"
    if report.get("overall_pass") is True or report.get("overall_result") == "pass":
        return "validation_pass"

    if "error" in report:
        message = str(report.get("error", "")).lower()
        if "timeout after" in message:
            return "validator_timeout"
        if "codec can't decode" in message:
            return "input_decode_failed"
        if "failed to load sbml" in message:
            return "input_sbml_load_failed"
        return "validation_worker_failed"

    reasons = {str(test.get("reason")) for test in _iter_validation_tests(report) if test.get("reason")}
    results = {str(test.get("result")) for test in _iter_validation_tests(report) if test.get("result")}

    if report.get("overall_result") == "timeout" or "validator_timeout" in reasons:
        return "validator_timeout"
    if reasons & {"input_decode_failed", "input_load_failed", "input_sbml_load_failed"}:
        return sorted(reasons & {"input_decode_failed", "input_load_failed", "input_sbml_load_failed"})[0]
    if reasons & {"validator_subprocess_failed", "validation_worker_failed"}:
        return sorted(reasons & {"validator_subprocess_failed", "validation_worker_failed"})[0]

    generated_result = _test_result(report, "generated_output")
    if generated_result in {"failed", "inconclusive"}:
        return "artifact_roundtrip_failed"

    parser_reason = _test_reason(report, "parser")
    if parser_reason in {"parser_failed", "failed"}:
        return "validator_parser_failed"

    if reasons & {"unsupported_feature", "unsupported"}:
        return "unsupported_validation_feature"

    numerical_result = _test_result(report, "numerical")
    if numerical_result == "failed":
        return "confirmed_failed_correctness"
    if numerical_result == "not_attempted":
        return "validation_blocked"
    if _has_failed_auxiliary_identity(report):
        return "auxiliary_identity_failed"

    if "nonfinite_sample" in reasons:
        return "validation_blocked"
    if "unresolved_parameter" in reasons or "unresolved_symbol" in reasons:
        return "validation_blocked"
    if "parser_failed" in reasons:
        return "validator_parser_failed"
    if "failed" in results:
        return "validation_failed"
    if "not_attempted" in results:
        return "validation_blocked"
    return "validation_unclassified"


def summarize_biomodels_outputs(
    benchmark_dir: Path,
    subset_size: int = 10,
    *,
    validation_timeout_seconds: int | None = DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    recast_timeout_seconds: int | None = DEFAULT_RECAST_TIMEOUT_SECONDS,
    near_timeout_fraction: float = DEFAULT_NEAR_TIMEOUT_FRACTION,
) -> dict[str, Any]:
    results_dir = benchmark_dir / "results"
    data_dir = benchmark_dir / "data"
    validation_dir = results_dir / "validation"
    validated_dir = results_dir / "validated"
    recasts_dir = results_dir / "recasts"
    failures_dir = results_dir / "failures"

    result_rows = _read_csv_rows(results_dir / "batch_recast_results.csv")
    manifest_rows = _read_csv_rows(validated_dir / "manifest.csv")
    validation_files = _validation_files(validation_dir)
    validation_result_counts: Counter[str] = Counter()
    validation_profile_counts: Counter[str] = Counter()
    validation_reason_counts: Counter[str] = Counter()
    validation_model_category_counts: Counter[str] = Counter()
    validation_timeout_phase_counts: Counter[str] = Counter()
    validation_timeout_dominant_phase_counts: Counter[str] = Counter()
    validation_near_timeout_models: list[dict[str, Any]] = []
    recast_near_timeout_models: list[dict[str, Any]] = []
    recast_near_timeout_status_counts: Counter[str] = Counter()
    recast_near_timeout_attempt_role_counts: Counter[str] = Counter()
    recast_policy_counts: Counter[str] = Counter()
    recast_attempt_role_counts: Counter[str] = Counter()
    recast_first_attempt_successes = 0
    recast_retry_recovered_successes = 0
    recast_unrecovered_timeouts = 0
    active_failed_keys = {
        (str(row.get("model_id", "")), str(row.get("mode", "")))
        for row in result_rows
        if not _bool_from_csv(row.get("recast_success"))
        and str(row.get("status", "")).lower() in {"error", "timeout"}
    }
    recast_failure_category_counts, recast_complexity_models = (
        _recast_failure_summaries(
            failures_dir,
            active_failed_keys=active_failed_keys,
        )
    )

    for row in result_rows:
        role = row.get("recast_attempt_role") or "unspecified"
        policy = row.get("recast_retry_policy") or "unspecified"
        recast_attempt_role_counts[role] += 1
        recast_policy_counts[policy] += 1
        recast_success = _bool_from_csv(row.get("recast_success"))
        recovered_by_retry = _bool_from_csv(row.get("recast_recovered_by_retry"))
        if recast_success and role == "base":
            recast_first_attempt_successes += 1
        if recast_success and (recovered_by_retry or role == "retry"):
            recast_retry_recovered_successes += 1
        if _recast_row_timed_out(row):
            recast_unrecovered_timeouts += 1

    for path in validation_files:
        report = _read_json(path)
        if "_read_error" in report:
            validation_result_counts["unreadable"] += 1
            category = _classify_validation_non_pass(report)
            validation_model_category_counts[category] += 1
            continue
        result = str(report.get("overall_result") or report.get("overall_pass"))
        validation_result_counts[result] += 1
        category = _classify_validation_non_pass(report)
        validation_model_category_counts[category] += 1
        if category == "validator_timeout":
            validation_timeout_phase_counts[_validation_timeout_phase(report) or "unknown"] += 1
        timing = _validation_timing(
            report,
            timeout_seconds=validation_timeout_seconds,
        )
        if category == "validator_timeout":
            dominant = timing.get("dominant_phase_interval") if timing else None
            phase = str(dominant.get("phase")) if isinstance(dominant, dict) else "unknown"
            validation_timeout_dominant_phase_counts[phase] += 1
        if timing and validation_timeout_seconds:
            elapsed = timing.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                if elapsed / validation_timeout_seconds >= near_timeout_fraction:
                    dominant = timing.get("dominant_phase_interval")
                    validation_near_timeout_models.append(_near_timeout_entry(
                        model_id=_model_id_from_validation_path(path),
                        elapsed_seconds=float(elapsed),
                        timeout_seconds=validation_timeout_seconds,
                        extra={
                            "overall_result": str(report.get("overall_result")),
                            "last_phase": timing.get("last_phase") or "unknown",
                            "dominant_phase": (
                                dominant.get("phase")
                                if isinstance(dominant, dict)
                                else "unknown"
                            ),
                            "dominant_phase_seconds": (
                                dominant.get("duration_seconds")
                                if isinstance(dominant, dict)
                                else None
                            ),
                            "report_file": path.name,
                        },
                    ))
        profile = report.get("validation_profile")
        if isinstance(profile, dict):
            validation_profile_counts[str(profile.get("name", "unknown"))] += 1
        elif profile:
            validation_profile_counts[str(profile)] += 1
        for test in _iter_validation_tests(report):
            if test.get("reason"):
                validation_reason_counts[str(test["reason"])] += 1

    if recast_timeout_seconds:
        for row in result_rows:
            elapsed = _float_or_none(row.get("recast_time"))
            if elapsed is None:
                continue
            row_timeout_seconds = (
                _positive_int_or_none(row.get("recast_final_attempt_timeout_seconds"))
                or recast_timeout_seconds
            )
            if elapsed / row_timeout_seconds < near_timeout_fraction:
                continue
            status = row.get("status", "unknown")
            recast_near_timeout_status_counts[str(status)] += 1
            attempt_role = row.get("recast_attempt_role") or "unspecified"
            recast_near_timeout_attempt_role_counts[attempt_role] += 1
            phase_attribution = _recast_phase_attribution(
                row,
                elapsed_seconds=elapsed,
            )
            recast_near_timeout_models.append(_near_timeout_entry(
                model_id=row.get("model_id", "unknown"),
                elapsed_seconds=elapsed,
                timeout_seconds=row_timeout_seconds,
                extra={
                    "mode": row.get("mode", "unknown"),
                    "status": status,
                    "recast_success": _bool_from_csv(row.get("recast_success")),
                    **phase_attribution,
                    **_recast_policy_entry_fields(row),
                    "error": row.get("error") or None,
                },
            ))

    recast_near_timeout_models.sort(
        key=lambda item: item["budget_fraction"],
        reverse=True,
    )
    validation_near_timeout_models.sort(
        key=lambda item: item["budget_fraction"],
        reverse=True,
    )

    status_counts = Counter(row.get("status", "unknown") for row in result_rows)
    validation_pass_count = sum(row.get("validation_pass") == "True" for row in result_rows)
    representative = manifest_rows[:subset_size]

    return {
        "benchmark_dir": str(benchmark_dir),
        "counts": {
            "sbml_downloads": _count_files(data_dir / "sbml_downloads", "*.xml"),
            "candidate_models": _count_files(data_dir / "sbml_candidates", "*.xml"),
            "recast_artifacts": _count_files(recasts_dir, "*.ant"),
            "failure_logs": _count_files(failures_dir, "*.log")
            + _count_files(failures_dir, "*.txt"),
            "validation_reports": len(validation_files),
            "validated_artifacts": _count_files(validated_dir, "*.ant"),
            "validated_manifest_rows": len(manifest_rows),
            "result_rows": len(result_rows),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "recast_failure_category_counts": dict(
            sorted(recast_failure_category_counts.items())
        ),
        "recast_complexity_models": recast_complexity_models,
        "recast_policy": {
            "policy_counts": dict(sorted(recast_policy_counts.items())),
            "attempt_role_counts": dict(sorted(recast_attempt_role_counts.items())),
            "first_attempt_successes": recast_first_attempt_successes,
            "retry_recovered_successes": recast_retry_recovered_successes,
            "unrecovered_timeouts": recast_unrecovered_timeouts,
            "rows_with_policy_metadata": sum(
                count
                for role, count in recast_attempt_role_counts.items()
                if role != "unspecified"
            ),
        },
        "validation_pass_count_from_results_csv": validation_pass_count,
        "validation_result_counts": dict(sorted(validation_result_counts.items())),
        "validation_profile_counts": dict(sorted(validation_profile_counts.items())),
        "validation_reason_counts": dict(validation_reason_counts.most_common(20)),
        "validation_model_category_counts": dict(
            validation_model_category_counts.most_common()
        ),
        "validation_timeout_phase_counts": dict(
            validation_timeout_phase_counts.most_common()
        ),
        "validation_timeout_dominant_phase_counts": dict(
            validation_timeout_dominant_phase_counts.most_common()
        ),
        "near_timeout": {
            "threshold_fraction": near_timeout_fraction,
            "recast": {
                "timeout_seconds": recast_timeout_seconds,
                "count": len(recast_near_timeout_models),
                "status_counts": dict(sorted(recast_near_timeout_status_counts.items())),
                "attempt_role_counts": dict(
                    sorted(recast_near_timeout_attempt_role_counts.items())
                ),
                "models": recast_near_timeout_models,
            },
            "validation": {
                "timeout_seconds": validation_timeout_seconds,
                "count": len(validation_near_timeout_models),
                "models": validation_near_timeout_models,
            },
        },
        "representative_validated_subset": representative,
    }


def _benchmark_command(args: argparse.Namespace, benchmark_dir: Path) -> list[str]:
    command = [str(benchmark_dir / "run_benchmark.sh")]
    if args.only_stage:
        command.extend(["--only", args.only_stage])
    elif args.from_stage:
        command.extend(["--from", args.from_stage])
    if args.force:
        command.append("--force")
    if args.jax:
        command.append("--jax")
    if args.symbolic:
        command.append("--symbolic")
    if args.full:
        command.append("--full")
    return command


def _selected_validation_timeout(args: argparse.Namespace) -> tuple[int, str]:
    if args.validation_timeout is not None:
        return args.validation_timeout, "cli"
    env_value = os.environ.get("TIMEOUT_VALIDATION")
    if env_value:
        try:
            timeout = _positive_int(env_value)
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise SystemExit(f"invalid TIMEOUT_VALIDATION={env_value!r}: {exc}") from exc
        return timeout, "environment"
    return DEFAULT_VALIDATION_TIMEOUT_SECONDS, "default"


def _selected_recast_timeout(args: argparse.Namespace) -> tuple[int, str]:
    if args.recast_timeout is not None:
        return args.recast_timeout, "cli"
    env_value = os.environ.get("TIMEOUT_RECAST")
    if env_value:
        try:
            timeout = _positive_int(env_value)
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise SystemExit(f"invalid TIMEOUT_RECAST={env_value!r}: {exc}") from exc
        return timeout, "environment"
    return DEFAULT_RECAST_TIMEOUT_SECONDS, "default"


def _selected_recast_quick_timeout(args: argparse.Namespace) -> tuple[int, str]:
    if args.recast_quick_timeout is not None:
        return args.recast_quick_timeout, "cli"
    env_value = os.environ.get("TIMEOUT_QUICK")
    if env_value:
        try:
            timeout = _positive_int(env_value)
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise SystemExit(f"invalid TIMEOUT_QUICK={env_value!r}: {exc}") from exc
        return timeout, "environment"
    return DEFAULT_RECAST_QUICK_TIMEOUT_SECONDS, "default"


def _benchmark_env(
    *,
    base_env: dict[str, str] | None,
    recast_quick_timeout_seconds: int,
    recast_timeout_seconds: int,
    validation_timeout_seconds: int,
) -> dict[str, str]:
    env = dict(base_env) if base_env is not None else os.environ.copy()
    env["TIMEOUT_QUICK"] = str(recast_quick_timeout_seconds)
    env["TIMEOUT_RECAST"] = str(recast_timeout_seconds)
    env["TIMEOUT_VALIDATION"] = str(validation_timeout_seconds)
    return env


def _copy_benchmark_tree(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in BENCHMARK_COPY_IGNORES}

    shutil.copytree(source, destination, ignore=ignore)
    return destination


def _run_setup_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join([
            "$ " + " ".join(command),
            f"cwd: {cwd}",
            f"returncode: {result.returncode}",
            "",
            "[stdout]",
            result.stdout,
            "",
            "[stderr]",
            result.stderr,
        ]),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(f"command failed; see {log_path}: {' '.join(command)}")


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _venv_bin(venv: Path) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin")


def _prepare_artifact_benchmark(
    *,
    source_benchmark_dir: Path,
    artifact: Path,
    evidence_dir: Path,
    python_executable: str,
) -> tuple[Path, Path, dict[str, Any]]:
    work_root = evidence_dir / "artifact-work"
    benchmark_copy = _copy_benchmark_tree(source_benchmark_dir, work_root / "biomodels_batch")
    venv = work_root / "venv"
    logs = evidence_dir / "artifact-setup-logs"
    artifact = artifact.resolve()

    if venv.exists():
        shutil.rmtree(venv)
    _run_setup_command(
        [python_executable, "-m", "venv", str(venv)],
        cwd=evidence_dir,
        log_path=logs / "venv.log",
    )
    python_exe = _venv_python(venv)
    _run_setup_command(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=evidence_dir,
        log_path=logs / "pip-upgrade.log",
    )
    _run_setup_command(
        [str(python_exe), "-m", "pip", "install", str(artifact)],
        cwd=evidence_dir,
        log_path=logs / "install-artifact.log",
    )
    requirements = benchmark_copy / "requirements.txt"
    if requirements.exists():
        _run_setup_command(
            [str(python_exe), "-m", "pip", "install", "-r", str(requirements)],
            cwd=benchmark_copy,
            log_path=logs / "install-biomodels-requirements.log",
        )

    metadata = {
        "artifact": str(artifact),
        "benchmark_copy": str(benchmark_copy),
        "venv": str(venv),
        "python": str(python_exe),
    }
    return benchmark_copy, python_exe, metadata


def _run_benchmark(
    command: list[str],
    benchmark_dir: Path,
    evidence_dir: Path,
    timeout: int | None,
    env: dict[str, str] | None = None,
    log_prefix: str = "benchmark",
) -> dict[str, Any]:
    started = _utc_now()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=benchmark_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"benchmark timed out after {timeout} seconds",
        )
        timed_out = True

    duration = time.perf_counter() - start
    (evidence_dir / f"{log_prefix}_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (evidence_dir / f"{log_prefix}_stderr.log").write_text(completed.stderr, encoding="utf-8")
    return {
        "command": command,
        "started_at": started,
        "finished_at": _utc_now(),
        "duration_seconds": round(duration, 3),
        "returncode": completed.returncode,
        "timed_out": timed_out,
    }


def _record_dependency_freeze(evidence_dir: Path) -> None:
    _record_dependency_freeze_for_python(evidence_dir, Path(sys.executable))


def _record_dependency_freeze_for_python(evidence_dir: Path, python_exe: Path) -> None:
    freeze = subprocess.run(
        [str(python_exe), "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    (evidence_dir / "dependency-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    metadata = {
        "python": subprocess.run(
            [str(python_exe), "-c", "import sys; print(sys.version)"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "executable": str(python_exe),
        "platform": platform.platform(),
        "pip_freeze_returncode": freeze.returncode,
    }
    (evidence_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _archive_key_outputs(benchmark_dir: Path, evidence_dir: Path, subset_size: int) -> None:
    results_dir = benchmark_dir / "results"
    _copy_if_exists(benchmark_dir / "RESULTS.md", evidence_dir / "RESULTS.md")
    _copy_if_exists(results_dir / "batch_recast_results.csv", evidence_dir / "batch_recast_results.csv")
    _copy_if_exists(results_dir / "validation_summary.txt", evidence_dir / "validation_summary.txt")
    _copy_if_exists(results_dir / "validated" / "manifest.csv", evidence_dir / "validated_manifest.csv")

    subset_dir = evidence_dir / "representative-validation"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for path in _validation_files(results_dir / "validation")[:subset_size]:
        shutil.copy2(path, subset_dir / path.name)


def _threshold_failures(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    counts = summary["counts"]
    failures = []
    checks = {
        "candidate_models": args.min_candidates,
        "recast_artifacts": args.min_recasts,
        "validation_reports": args.min_validation_reports,
        "validated_manifest_rows": args.min_validated,
    }
    for key, minimum in checks.items():
        if counts.get(key, 0) < minimum:
            failures.append(f"{key}={counts.get(key, 0)} is below required minimum {minimum}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local BioModels benchmark and archive release evidence."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=Path("biomodels_batch"))
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--skip-run", action="store_true", help="Summarize existing outputs only.")
    parser.add_argument("--from-stage", default="filter", help="Pipeline stage for --from.")
    parser.add_argument("--only-stage", default="", help="Run only one pipeline stage.")
    parser.add_argument("--force", action="store_true", help="Pass --force to run_benchmark.sh.")
    parser.add_argument("--jax", action="store_true", help="Pass --jax to run_benchmark.sh.")
    parser.add_argument("--symbolic", action="store_true", help="Pass --symbolic to run_benchmark.sh.")
    parser.add_argument("--full", action="store_true", help="Pass --full to run_benchmark.sh.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=(
            "Wheel or sdist to install into an isolated benchmark environment before running. "
            "The BioModels tree is copied into the evidence directory so imports resolve from "
            "the installed artifact, not the source checkout."
        ),
    )
    parser.add_argument(
        "--artifact-python",
        default=sys.executable,
        help="Python executable used to create the isolated artifact benchmark venv.",
    )
    parser.add_argument("--timeout", type=int, default=None, help="Optional whole-run timeout.")
    parser.add_argument(
        "--recast-quick-timeout",
        type=_positive_int,
        default=None,
        help=(
            "Per-model BioModels quick/base recast timeout in seconds. Defaults to "
            "TIMEOUT_QUICK if set, otherwise 15."
        ),
    )
    parser.add_argument(
        "--recast-timeout",
        type=_positive_int,
        default=None,
        help=(
            "Per-model BioModels long recast retry timeout in seconds. Defaults to "
            "TIMEOUT_RECAST if set, otherwise 60."
        ),
    )
    parser.add_argument(
        "--validation-timeout",
        type=_positive_int,
        default=None,
        help=(
            "Per-model BioModels validation timeout in seconds. Defaults to "
            "TIMEOUT_VALIDATION if set, otherwise 60. This is distinct from "
            "--timeout, which limits the whole benchmark command."
        ),
    )
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument(
        "--near-timeout-fraction",
        type=_positive_fraction,
        default=DEFAULT_NEAR_TIMEOUT_FRACTION,
        help=(
            "Fraction of a per-model timeout budget used to flag near-timeout "
            "recast and validation models in summary.json (default: 0.85)."
        ),
    )
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--min-recasts", type=int, default=1)
    parser.add_argument("--min-validation-reports", type=int, default=1)
    parser.add_argument("--min-validated", type=int, default=1)
    args = parser.parse_args(argv)

    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = args.benchmark_dir.resolve()
    benchmark_env = None
    benchmark_python: Path | None = None
    artifact_metadata: dict[str, Any] | None = None
    recast_quick_timeout_seconds, recast_quick_timeout_source = (
        _selected_recast_quick_timeout(args)
    )
    recast_timeout_seconds, recast_timeout_source = _selected_recast_timeout(args)
    validation_timeout_seconds, validation_timeout_source = _selected_validation_timeout(args)

    if args.artifact and not args.skip_run:
        benchmark_dir, benchmark_python, artifact_metadata = _prepare_artifact_benchmark(
            source_benchmark_dir=benchmark_dir,
            artifact=args.artifact,
            evidence_dir=evidence_dir,
            python_executable=args.artifact_python,
        )
        benchmark_env = os.environ.copy()
        benchmark_env["PATH"] = (
            f"{benchmark_python.parent}{os.pathsep}{benchmark_env.get('PATH', '')}"
        )
        benchmark_env.pop("PYTHONPATH", None)

    benchmark_env = _benchmark_env(
        base_env=benchmark_env,
        recast_quick_timeout_seconds=recast_quick_timeout_seconds,
        recast_timeout_seconds=recast_timeout_seconds,
        validation_timeout_seconds=validation_timeout_seconds,
    )

    command = _benchmark_command(args, benchmark_dir)
    run_metadata: dict[str, Any]
    report_metadata: dict[str, Any] | None = None
    if args.skip_run:
        run_metadata = {
            "command": command,
            "skipped": True,
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "duration_seconds": 0.0,
            "returncode": 0,
            "timed_out": False,
        }
    else:
        run_metadata = _run_benchmark(
            command,
            benchmark_dir,
            evidence_dir,
            args.timeout,
            env=benchmark_env,
        )
        if run_metadata["returncode"] == 0:
            report_command = [str(benchmark_dir / "run_benchmark.sh"), "--only", "report"]
            report_metadata = _run_benchmark(
                report_command,
                benchmark_dir,
                evidence_dir,
                args.timeout,
                env=benchmark_env,
                log_prefix="report",
            )

    summary = summarize_biomodels_outputs(
        benchmark_dir,
        subset_size=args.subset_size,
        validation_timeout_seconds=validation_timeout_seconds,
        recast_timeout_seconds=recast_timeout_seconds,
        near_timeout_fraction=args.near_timeout_fraction,
    )
    threshold_failures = _threshold_failures(summary, args)
    if benchmark_python is not None:
        _record_dependency_freeze_for_python(evidence_dir, benchmark_python)
    else:
        _record_dependency_freeze(evidence_dir)
    _archive_key_outputs(benchmark_dir, evidence_dir, args.subset_size)

    payload = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "run": run_metadata,
        "report_run": report_metadata,
        "artifact_environment": artifact_metadata,
        "benchmark_parameters": {
            "near_timeout_fraction": args.near_timeout_fraction,
            "recast_quick_timeout_seconds": recast_quick_timeout_seconds,
            "recast_quick_timeout_source": recast_quick_timeout_source,
            "recast_retry_policy": DEFAULT_RECAST_RETRY_POLICY,
            "recast_timeout_seconds": recast_timeout_seconds,
            "recast_timeout_source": recast_timeout_source,
            "validation_timeout_seconds": validation_timeout_seconds,
            "validation_timeout_source": validation_timeout_source,
            "whole_run_timeout_seconds": args.timeout,
        },
        "summary": summary,
        "threshold_failures": threshold_failures,
    }
    (evidence_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_failed = report_metadata is not None and report_metadata["returncode"] != 0
    if run_metadata["returncode"] != 0 or report_failed or threshold_failures:
        print("BioModels benchmark evidence failed:", file=sys.stderr)
        if run_metadata["returncode"] != 0:
            print(f"- command returned {run_metadata['returncode']}", file=sys.stderr)
        if report_failed:
            print(f"- report command returned {report_metadata['returncode']}", file=sys.stderr)
        for failure in threshold_failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    counts = summary["counts"]
    print(
        "BioModels benchmark evidence written to "
        f"{evidence_dir} ({counts['candidate_models']} candidates, "
        f"{counts['recast_artifacts']} recasts, {counts['validation_reports']} "
        f"validation reports, {counts['validated_manifest_rows']} validated)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
