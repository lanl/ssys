#!/usr/bin/env python3
"""Enforce local line and branch coverage thresholds for critical modules."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LINE_THRESHOLD = 90.0
DEFAULT_BRANCH_THRESHOLD = 85.0

CRITICAL_MODULES = (
    "src/ssys/_recaster/algorithms.py",
    "src/ssys/_recaster/antimony_formatting.py",
    "src/ssys/_recaster/lifting.py",
    "src/ssys/_recaster/parsing.py",
    "src/ssys/_recaster/templates.py",
    "src/ssys/_validator/core.py",
    "src/ssys/_validator/mapping.py",
    "src/ssys/_validator/numerical.py",
    "src/ssys/_validator/report.py",
    "src/ssys/_validator/serialization.py",
    "src/ssys/_validator/symbolic.py",
    "src/ssys/_validator/trajectory.py",
    "src/ssys/ode_backends/dae_backend.py",
    "src/ssys/ode_backends/ida_sundials_backend.py",
    "src/ssys/ode_backends/interface.py",
    "src/ssys/ode_backends/roadrunner_backend.py",
)


@dataclass(frozen=True)
class CoverageFailure:
    """Coverage threshold failure for one module."""

    module: str
    line_percent: float | None
    branch_percent: float | None
    message: str


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _summary_percent(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _module_summary(report: dict[str, Any], module: str) -> dict[str, Any] | None:
    file_data = report.get("files", {}).get(module)
    if not isinstance(file_data, dict):
        return None
    summary = file_data.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def evaluate_critical_coverage(
    report: dict[str, Any],
    *,
    modules: tuple[str, ...] = CRITICAL_MODULES,
    line_threshold: float = DEFAULT_LINE_THRESHOLD,
    branch_threshold: float = DEFAULT_BRANCH_THRESHOLD,
) -> list[CoverageFailure]:
    """Return all threshold failures in a coverage.py JSON report."""
    failures: list[CoverageFailure] = []
    meta = report.get("meta", {})
    if not isinstance(meta, dict) or meta.get("branch_coverage") is not True:
        failures.append(
            CoverageFailure(
                module="<coverage report>",
                line_percent=None,
                branch_percent=None,
                message="coverage JSON was not generated with branch coverage enabled",
            )
        )
        return failures

    for module in modules:
        summary = _module_summary(report, module)
        if summary is None:
            failures.append(
                CoverageFailure(
                    module=module,
                    line_percent=None,
                    branch_percent=None,
                    message="module missing from coverage report",
                )
            )
            continue

        line_percent = _summary_percent(
            summary,
            "percent_statements_covered",
            "percent_covered",
        )
        branch_percent = _summary_percent(summary, "percent_branches_covered")
        if summary.get("num_branches") == 0 and branch_percent is None:
            branch_percent = 100.0

        messages: list[str] = []
        if line_percent is None:
            messages.append("line coverage percentage missing")
        elif line_percent < line_threshold:
            messages.append(f"line coverage {_format_percent(line_percent)} < {line_threshold:.1f}%")

        if branch_percent is None:
            messages.append("branch coverage percentage missing")
        elif branch_percent < branch_threshold:
            messages.append(
                f"branch coverage {_format_percent(branch_percent)} < {branch_threshold:.1f}%"
            )

        if messages:
            failures.append(
                CoverageFailure(
                    module=module,
                    line_percent=line_percent,
                    branch_percent=branch_percent,
                    message="; ".join(messages),
                )
            )

    return failures


def load_coverage_json(path: Path) -> dict[str, Any]:
    """Load a coverage.py JSON report."""
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"{path} is not a coverage.py JSON object")
    return report


def _remove_existing_coverage_data(coverage_file: Path) -> None:
    coverage_file.unlink(missing_ok=True)
    for path in coverage_file.parent.glob(f"{coverage_file.name}.*"):
        path.unlink(missing_ok=True)


def _run_pytest_with_branch_coverage(output: Path, pytest_args: list[str]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    coverage_file = output.with_suffix(output.suffix + ".coverage")
    _remove_existing_coverage_data(coverage_file)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--strict-markers",
        "--strict-config",
        "-m",
        "not slow",
        "--cov=ssys",
        "--cov-branch",
        f"--cov-report=json:{output}",
        "--cov-report=term-missing",
        *pytest_args,
    ]
    print("+ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_file)
    return subprocess.run(command, check=False, env=env).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check critical ssys modules against local line and branch coverage thresholds."
        )
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("coverage-critical.json"),
        help="Path to read or write coverage.py JSON output.",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Run the non-slow pytest suite with branch coverage before checking thresholds.",
    )
    parser.add_argument(
        "--line-threshold",
        type=float,
        default=DEFAULT_LINE_THRESHOLD,
        help="Required statement coverage percentage for each critical module.",
    )
    parser.add_argument(
        "--branch-threshold",
        type=float,
        default=DEFAULT_BRANCH_THRESHOLD,
        help="Required branch coverage percentage for each critical module.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional pytest arguments after -- when --run-pytest is used.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    if args.run_pytest:
        pytest_status = _run_pytest_with_branch_coverage(args.coverage_json, pytest_args)
        if pytest_status != 0:
            return pytest_status

    report = load_coverage_json(args.coverage_json)
    failures = evaluate_critical_coverage(
        report,
        line_threshold=args.line_threshold,
        branch_threshold=args.branch_threshold,
    )
    if not failures:
        print(
            "critical coverage passed: "
            f"line >= {args.line_threshold:.1f}%, branch >= {args.branch_threshold:.1f}%"
        )
        return 0

    print(
        "critical coverage failed: "
        f"line >= {args.line_threshold:.1f}%, branch >= {args.branch_threshold:.1f}% required"
    )
    for failure in failures:
        print(
            f"- {failure.module}: {failure.message} "
            f"(line={_format_percent(failure.line_percent)}, "
            f"branch={_format_percent(failure.branch_percent)})"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
