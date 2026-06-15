#!/usr/bin/env python3
"""Generate focused auxiliary-threshold metadata evidence for a recast pair."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import antimony

import ssys
from ssys.validator import validate_recast_pair


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _threshold_status(metadata: dict[str, Any]) -> str:
    if "operation_threshold" in metadata or "free_symbol_threshold" in metadata:
        return "canonical"
    if "max_operation_count" in metadata or "max_free_symbol_count" in metadata:
        return "legacy"
    return "missing"


def _auxiliary_entry(test) -> dict[str, Any]:
    metadata = test.metadata or {}
    entry = {
        "name": test.name,
        "result": test.result.value,
        "reason": test.reason,
        "active_subphase": metadata.get("active_subphase"),
        "candidate_index": metadata.get("candidate_index"),
        "operation_count": metadata.get("operation_count"),
        "free_symbol_count": metadata.get("free_symbol_count"),
        "operation_threshold": metadata.get("operation_threshold"),
        "free_symbol_threshold": metadata.get("free_symbol_threshold"),
        "max_operation_count": metadata.get("max_operation_count"),
        "max_free_symbol_count": metadata.get("max_free_symbol_count"),
        "risky_symbolic_power_guard": metadata.get("risky_symbolic_power_guard"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }
    entry["threshold_metadata_status"] = _threshold_status(metadata)
    entry["threshold_metadata_complete"] = bool(
        entry["active_subphase"]
        and entry["candidate_index"] is not None
        and entry["operation_count"] is not None
        and entry["free_symbol_count"] is not None
        and entry["threshold_metadata_status"] == "canonical"
    )
    return entry


def _original_antimony_from_sbml(sbml_path: Path) -> Path:
    antimony.clearPreviousLoads()
    result = antimony.loadSBMLFile(str(sbml_path))
    if result == -1:
        raise RuntimeError(f"Failed to load SBML: {antimony.getLastError()}")
    text = antimony.getAntimonyString()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ant", delete=False) as tmp:
        tmp.write(text)
        return Path(tmp.name)


def probe(args: argparse.Namespace) -> dict[str, Any]:
    original_sbml = Path(args.original_sbml).resolve()
    recast = Path(args.recast).resolve()
    original_antimony = _original_antimony_from_sbml(original_sbml)
    try:
        report = validate_recast_pair(
            str(original_antimony),
            str(recast),
            mode=args.mode,
            parser="sbml",
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
            run_auxiliaries=True,
        )
    finally:
        original_antimony.unlink(missing_ok=True)

    auxiliary_entries = [
        _auxiliary_entry(test)
        for test in report.auxiliary_tests
        if test.reason == "auxiliary_complexity"
    ]
    threshold_counts = Counter(
        entry["threshold_metadata_status"] for entry in auxiliary_entries
    )
    complete = bool(auxiliary_entries) and all(
        entry["threshold_metadata_complete"] for entry in auxiliary_entries
    )

    ssys_file = Path(getattr(ssys, "__file__", "")).resolve()
    environment = {
        "profile": args.profile,
        "python_executable": sys.executable,
        "pythonpath_present": bool(os.environ.get("PYTHONPATH")),
        "ssys_file": str(ssys_file),
        "ssys_file_is_site_packages": "site-packages" in ssys_file.parts,
    }

    regression_failures = []
    if args.require_site_packages and not environment["ssys_file_is_site_packages"]:
        regression_failures.append(f"ssys import is not installed-artifact clean: {ssys_file}")
    if not complete:
        regression_failures.append("auxiliary threshold metadata is incomplete")
    if report.overall_result.value in {"failed", "timeout"}:
        regression_failures.append(f"unexpected overall_result: {report.overall_result.value}")

    artifact_summary = None
    if args.artifact_summary:
        artifact_summary = json.loads(
            Path(args.artifact_summary).read_text(encoding="utf-8")
        ).get("artifacts")

    return {
        "created_at": _utc_now(),
        "task": "68",
        "description": "Focused auxiliary complexity threshold metadata probe.",
        "model_id": args.model_id,
        "mode": args.mode,
        "original_sbml": str(original_sbml),
        "recast": str(recast),
        "environment": environment,
        "artifacts": artifact_summary,
        "overall_result": report.overall_result.value,
        "overall_pass": report.overall_pass,
        "auxiliary_complexity_count": len(auxiliary_entries),
        "auxiliary_threshold_metadata_complete": complete,
        "auxiliary_threshold_metadata_status_counts": dict(threshold_counts.most_common()),
        "auxiliary_complexity_entries": auxiliary_entries,
        "regression_failures": regression_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe auxiliary complexity threshold metadata for one model."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--original-sbml", required=True)
    parser.add_argument("--recast", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="simplified")
    parser.add_argument("--profile", default="source-tree")
    parser.add_argument("--artifact-summary")
    parser.add_argument("--require-site-packages", action="store_true")
    args = parser.parse_args()

    result = probe(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Task 68 auxiliary metadata probe: {output}")
    if result["regression_failures"]:
        print("Regression failures:", file=sys.stderr)
        for failure in result["regression_failures"]:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
