#!/usr/bin/env python3
"""Audit final BioModels validation-blocked reports for release evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ssys

PROFILE_EXCLUDED = "profile_excluded"
KNOWN_BLOCKED_REASONS = {
    "auxiliary_complexity",
    "blocked_by_wrapper_failure",
    "failed",
    "nonfinite_sample",
    "numerical_complexity",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_id_from_validation_path(path: Path) -> str:
    stem = path.stem
    suffixes = (
        "_simplified_numerical",
        "_canonical_numerical",
        "_gma_numerical",
        "_simplified_symbolic",
        "_canonical_symbolic",
        "_gma_symbolic",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _iter_validation_tests(report: dict[str, Any]):
    for name, value in (report.get("tests") or {}).items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item.get("name", name), item
        elif isinstance(value, dict):
            yield name, value


def _test_result(report: dict[str, Any], name: str) -> str | None:
    value = (report.get("tests") or {}).get(name)
    return value.get("result") if isinstance(value, dict) else None


def _test_reason(report: dict[str, Any], name: str) -> str | None:
    value = (report.get("tests") or {}).get(name)
    return value.get("reason") if isinstance(value, dict) else None


def _has_failed_auxiliary_identity(report: dict[str, Any]) -> bool:
    auxiliaries = (report.get("tests") or {}).get("auxiliaries") or []
    return any(
        isinstance(item, dict)
        and item.get("result") == "failed"
        and str(item.get("name", "")).startswith("ode_auxiliary_identity:")
        for item in auxiliaries
    )


def _classify_validation_non_pass(report: dict[str, Any]) -> str:
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

    reasons = {
        str(test.get("reason"))
        for _name, test in _iter_validation_tests(report)
        if test.get("reason")
    }
    results = {
        str(test.get("result"))
        for _name, test in _iter_validation_tests(report)
        if test.get("result")
    }

    if report.get("overall_result") == "timeout" or "validator_timeout" in reasons:
        return "validator_timeout"
    input_reasons = {"input_decode_failed", "input_load_failed", "input_sbml_load_failed"}
    if reasons & input_reasons:
        return sorted(reasons & input_reasons)[0]
    worker_reasons = {"validator_subprocess_failed", "validation_worker_failed"}
    if reasons & worker_reasons:
        return sorted(reasons & worker_reasons)[0]
    if _test_result(report, "generated_output") in {"failed", "inconclusive"}:
        return "artifact_roundtrip_failed"
    if _test_reason(report, "parser") in {"parser_failed", "failed"}:
        return "validator_parser_failed"
    if reasons & {"unsupported_feature", "unsupported"}:
        return "unsupported_validation_feature"
    if _test_result(report, "numerical") == "failed":
        return "confirmed_failed_correctness"
    if _test_result(report, "numerical") == "not_attempted":
        return "validation_blocked"
    if _has_failed_auxiliary_identity(report):
        return "auxiliary_identity_failed"
    if reasons & {"nonfinite_sample", "unresolved_parameter", "unresolved_symbol"}:
        return "validation_blocked"
    if "parser_failed" in reasons:
        return "validator_parser_failed"
    if "failed" in results:
        return "validation_failed"
    if "not_attempted" in results:
        return "validation_blocked"
    return "validation_unclassified"


def _compact_mapping(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value.get(key) for key in keys if key in value}


def _numerical_diagnostic_complete(diagnostic: dict[str, Any]) -> bool:
    phase = diagnostic.get("phase")
    subphase = diagnostic.get("active_subphase")
    if diagnostic.get("wrapper_timeout_phase"):
        return bool(
            phase
            and subphase
            and diagnostic.get("wrapper_timeout_seconds") is not None
            and diagnostic.get("limit_seconds") is not None
        )
    if phase == "numerical_preflight":
        return bool(
            subphase
            and diagnostic.get("expression_label")
            and diagnostic.get("expression_ops") is not None
            and diagnostic.get("max_expression_ops") is not None
        )
    if phase == "numerical_sample_evaluation":
        return bool(
            subphase
            and diagnostic.get("sample_index") is not None
            and diagnostic.get("samples_completed") is not None
            and diagnostic.get("n_samples") is not None
            and diagnostic.get("elapsed_seconds") is not None
            and diagnostic.get("limit_seconds") is not None
        )
    return bool(phase and subphase)


def _summarize_numerical_test(test: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = test.get("metadata") or {}
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        diagnostics = [metadata]

    fields = (
        "side",
        "reason",
        "phase",
        "active_subphase",
        "expression_label",
        "active_expression_label",
        "expression_ops",
        "max_expression_ops",
        "operation_count",
        "operation_threshold",
        "free_symbol_count",
        "sample_index",
        "samples_completed",
        "n_samples",
        "elapsed_seconds",
        "limit_seconds",
        "wrapper_timeout_phase",
        "wrapper_timeout_seconds",
    )
    summarized = []
    for diagnostic in diagnostics:
        entry = _compact_mapping(diagnostic, fields)
        entry["diagnostic_complete"] = _numerical_diagnostic_complete(diagnostic)
        summarized.append(entry)
    return summarized


def _auxiliary_threshold_status(metadata: dict[str, Any]) -> str:
    has_canonical = (
        "operation_threshold" in metadata or "free_symbol_threshold" in metadata
    )
    has_legacy = (
        "max_operation_count" in metadata or "max_free_symbol_count" in metadata
    )
    if has_canonical:
        return "canonical"
    if has_legacy:
        return "legacy_task68_schema_gap"
    return "missing"


def _summarize_auxiliary_test(name: str, test: dict[str, Any]) -> dict[str, Any]:
    metadata = test.get("metadata") or {}
    fields = (
        "context",
        "candidate_index",
        "active_subphase",
        "operation_count",
        "free_symbol_count",
        "operation_threshold",
        "free_symbol_threshold",
        "max_operation_count",
        "max_free_symbol_count",
        "risky_symbolic_power_guard",
        "elapsed_seconds",
    )
    entry = {"name": name, **_compact_mapping(metadata, fields)}
    threshold_status = _auxiliary_threshold_status(metadata)
    entry["threshold_metadata_status"] = threshold_status
    entry["diagnostic_complete"] = bool(
        metadata.get("active_subphase")
        and metadata.get("candidate_index") is not None
        and metadata.get("operation_count") is not None
        and metadata.get("free_symbol_count") is not None
        and threshold_status in {"canonical", "legacy_task68_schema_gap"}
    )
    return entry


def _summarize_nonfinite_test(test: dict[str, Any]) -> dict[str, Any]:
    metadata = test.get("metadata") or {}
    sampling = metadata.get("sampling")
    state_variables = {}
    if isinstance(sampling, dict) and isinstance(sampling.get("state_variables"), dict):
        state_variables = sampling["state_variables"]
    return {
        "result": test.get("result"),
        "details": test.get("details"),
        "n_samples": metadata.get("n_samples"),
        "validation_phase": metadata.get("validation_phase"),
        "sampling_variable_count": len(state_variables),
        "has_sampling_metadata": bool(state_variables),
        "has_parameter_values": isinstance(metadata.get("parameter_values"), dict),
        "dominant_phase_interval": metadata.get("dominant_phase_interval"),
        "diagnostic_complete": bool(
            state_variables
            and isinstance(metadata.get("parameter_values"), dict)
            and metadata.get("validation_phase")
            and metadata.get("n_samples") is not None
        ),
    }


def _dominant_release_decision(reasons: set[str]) -> str:
    if "nonfinite_sample" in reasons and "failed" in reasons:
        return "task24_nonfinite_sampling_mixed_auxiliary_residuals"
    if "nonfinite_sample" in reasons:
        return "task24_nonfinite_sampling"
    if "numerical_complexity" in reasons:
        return "accepted_numerical_complexity"
    if "auxiliary_complexity" in reasons:
        return "accepted_auxiliary_complexity"
    return "needs_named_followup"


def _audit_blocked_report(model_id: str, report: dict[str, Any]) -> dict[str, Any]:
    reason_classes = {
        str(test.get("reason"))
        for _name, test in _iter_validation_tests(report)
        if test.get("reason") and test.get("reason") != PROFILE_EXCLUDED
    }
    numerical_diagnostics = []
    auxiliary_entries = []
    nonfinite_entries = []
    failed_auxiliary_entries = []

    for name, test in _iter_validation_tests(report):
        reason = test.get("reason")
        if reason == "numerical_complexity":
            numerical_diagnostics.extend(_summarize_numerical_test(test))
        elif reason == "auxiliary_complexity":
            auxiliary_entries.append(_summarize_auxiliary_test(name, test))
        elif reason == "nonfinite_sample":
            nonfinite_entries.append(_summarize_nonfinite_test(test))
        elif reason == "failed" and str(name).startswith("ode_auxiliary_identity:"):
            failed_auxiliary_entries.append(
                {"name": name, "result": test.get("result"), "details": test.get("details")}
            )

    release_decision = _dominant_release_decision(reason_classes)
    unhandled_reasons = sorted(reason_classes - KNOWN_BLOCKED_REASONS)
    diagnostic_complete = all(
        item.get("diagnostic_complete", False)
        for item in numerical_diagnostics + auxiliary_entries + nonfinite_entries
    )
    if release_decision == "task24_nonfinite_sampling_mixed_auxiliary_residuals":
        diagnostic_complete = diagnostic_complete and bool(failed_auxiliary_entries)
    if not (numerical_diagnostics or auxiliary_entries or nonfinite_entries):
        diagnostic_complete = False
    if unhandled_reasons or release_decision == "needs_named_followup":
        diagnostic_complete = False

    return {
        "model_id": model_id,
        "overall_result": report.get("overall_result"),
        "reason_classes": sorted(reason_classes),
        "dominant_release_decision": release_decision,
        "diagnostic_complete": diagnostic_complete,
        "numerical_complexity_diagnostics": numerical_diagnostics,
        "auxiliary_complexity_entries": auxiliary_entries,
        "nonfinite_sample_entries": nonfinite_entries,
        "failed_auxiliary_entries": failed_auxiliary_entries,
        "unhandled_reason_classes": unhandled_reasons,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_payload(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None:
        return None
    payload = _load_json(summary_path)
    return payload.get("summary", payload)


def audit_validation_blocked(args: argparse.Namespace) -> dict[str, Any]:
    validation_dir = Path(args.validation_dir).resolve()
    reports = []
    category_counts: Counter[str] = Counter()
    for path in sorted(validation_dir.glob("*.json")):
        model_id = _model_id_from_validation_path(path)
        try:
            report = _load_json(path)
        except Exception as exc:  # pragma: no cover - defensive evidence path
            report = {"_read_error": str(exc), "overall_result": "failed"}
        category = _classify_validation_non_pass(report)
        category_counts[category] += 1
        if category == "validation_blocked":
            reports.append(_audit_blocked_report(model_id, report))

    blocked_model_ids = [item["model_id"] for item in reports]
    reason_model_counts = Counter()
    release_decision_counts = Counter()
    for item in reports:
        release_decision_counts[item["dominant_release_decision"]] += 1
        for reason in item["reason_classes"]:
            reason_model_counts[reason] += 1

    needs_models = [
        item["model_id"]
        for item in reports
        if not item["diagnostic_complete"]
        or item["dominant_release_decision"] == "needs_named_followup"
    ]

    task63_comparison = None
    if args.task63_audit:
        task63 = _load_json(Path(args.task63_audit))
        task63_models = set(task63.get("validation_blocked_model_ids", []))
        current_models = set(blocked_model_ids)
        task63_comparison = {
            "task63_blocked_count": len(task63_models),
            "current_blocked_count": len(current_models),
            "preserved_models": sorted(task63_models & current_models),
            "entered_after_task63": sorted(current_models - task63_models),
            "exited_after_task63": sorted(task63_models - current_models),
        }

    summary = _summary_payload(Path(args.summary)) if args.summary else None
    summary_category_counts = None
    if summary is not None:
        summary_category_counts = summary.get("validation_model_category_counts")

    threshold_status_counts = Counter()
    for item in reports:
        for entry in item["auxiliary_complexity_entries"]:
            threshold_status_counts[entry["threshold_metadata_status"]] += 1

    regression_failures = []
    forbidden_categories = {
        "confirmed_failed_correctness",
        "validation_unclassified",
        "validation_worker_failed",
        "validator_subprocess_failed",
        "validator_timeout",
    }
    for category in sorted(forbidden_categories):
        if category_counts.get(category, 0):
            regression_failures.append(f"{category}: {category_counts[category]}")
    if args.expected_blocked_count is not None and len(reports) != args.expected_blocked_count:
        regression_failures.append(
            f"expected {args.expected_blocked_count} validation_blocked reports, "
            f"found {len(reports)}"
        )
    if needs_models:
        regression_failures.append(
            "models need diagnostic or optimization follow-up: "
            + ", ".join(needs_models)
        )

    ssys_file = Path(getattr(ssys, "__file__", "")).resolve()
    environment = {
        "profile": args.profile,
        "python_executable": sys.executable,
        "pythonpath_present": bool(os.environ.get("PYTHONPATH")),
        "ssys_file": str(ssys_file),
        "ssys_file_is_site_packages": "site-packages" in ssys_file.parts,
    }
    if args.require_site_packages and not environment["ssys_file_is_site_packages"]:
        regression_failures.append(f"ssys import is not installed-artifact clean: {ssys_file}")

    return {
        "created_at": _utc_now(),
        "task": "75",
        "description": "Final Task 73-74 validation_blocked audit.",
        "validation_dir": str(validation_dir),
        "summary_path": str(Path(args.summary).resolve()) if args.summary else None,
        "task63_audit_path": (
            str(Path(args.task63_audit).resolve()) if args.task63_audit else None
        ),
        "environment": environment,
        "summary_category_counts": summary_category_counts,
        "derived_category_counts": dict(category_counts.most_common()),
        "blocked_model_count": len(reports),
        "blocked_model_ids": blocked_model_ids,
        "blocked_reason_model_counts": dict(reason_model_counts.most_common()),
        "blocked_release_decision_counts": dict(release_decision_counts.most_common()),
        "auxiliary_threshold_metadata_status_counts": dict(
            threshold_status_counts.most_common()
        ),
        "task63_comparison": task63_comparison,
        "needs_diagnostic_or_optimization_models": needs_models,
        "regression_failures": regression_failures,
        "models": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit BioModels validation_blocked reports for release evidence."
    )
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--task63-audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="source-tree")
    parser.add_argument("--expected-blocked-count", type=int)
    parser.add_argument("--require-site-packages", action="store_true")
    args = parser.parse_args()

    result = audit_validation_blocked(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Task 75 blocked-set audit: {output}")
    if result["regression_failures"]:
        print("Regression failures:", file=sys.stderr)
        for failure in result["regression_failures"]:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
