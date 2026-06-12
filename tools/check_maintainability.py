#!/usr/bin/env python3
"""Check local maintainability metrics for critical source modules."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("tools/maintainability_baseline.json")


@dataclass(frozen=True)
class ModuleMetrics:
    module_lines: int
    max_function_lines: int
    max_cyclomatic_complexity: int


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.score = 1

    def visit_If(self, node: ast.If) -> Any:
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self.score += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self.score += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        self.score += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.score += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> Any:
        self.score += len(node.cases)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> Any:
        self.score += len(node.ifs)
        self.generic_visit(node)


def _function_complexity(node: ast.AST) -> int:
    visitor = _ComplexityVisitor()
    visitor.visit(node)
    return visitor.score


def measure_file(path: Path) -> ModuleMetrics:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    function_lengths: list[int] = []
    function_complexities: list[int] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end_lineno = node.end_lineno or node.lineno
            function_lengths.append(end_lineno - node.lineno + 1)
            function_complexities.append(_function_complexity(node))

    return ModuleMetrics(
        module_lines=len(text.splitlines()),
        max_function_lines=max(function_lengths, default=0),
        max_cyclomatic_complexity=max(function_complexities, default=0),
    )


def load_baseline(path: Path) -> dict[str, ModuleMetrics]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"{path} must contain a 'paths' object")

    baseline: dict[str, ModuleMetrics] = {}
    for relpath, metrics in paths.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: metrics for {relpath!r} must be an object")
        baseline[relpath] = ModuleMetrics(
            module_lines=int(metrics["module_lines"]),
            max_function_lines=int(metrics["max_function_lines"]),
            max_cyclomatic_complexity=int(metrics["max_cyclomatic_complexity"]),
        )
    return baseline


def measure_paths(root: Path, paths: list[str]) -> dict[str, ModuleMetrics]:
    return {relpath: measure_file(root / relpath) for relpath in paths}


def evaluate_against_baseline(
    current: dict[str, ModuleMetrics],
    baseline: dict[str, ModuleMetrics],
) -> list[str]:
    failures: list[str] = []
    for relpath, expected in sorted(baseline.items()):
        observed = current.get(relpath)
        if observed is None:
            failures.append(f"{relpath}: missing current metrics")
            continue
        for field in ModuleMetrics.__dataclass_fields__:
            current_value = getattr(observed, field)
            baseline_value = getattr(expected, field)
            if current_value > baseline_value:
                failures.append(
                    f"{relpath}: {field} increased from {baseline_value} to {current_value}"
                )
    return failures


def _baseline_payload(metrics: dict[str, ModuleMetrics]) -> str:
    payload = {
        "version": 1,
        "description": (
            "Local maintainability baseline for critical modules. Values are ceilings; "
            "lower numbers are allowed and should be committed after refactors."
        ),
        "paths": {relpath: asdict(value) for relpath, value in sorted(metrics.items())},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check critical-module maintainability metrics against a local baseline."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Baseline JSON file. Defaults to tools/maintainability_baseline.json.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--print-current",
        action="store_true",
        help="Print current metrics JSON instead of checking.",
    )
    args = parser.parse_args(argv)

    baseline = load_baseline(args.baseline)
    current = measure_paths(args.root, list(baseline))

    if args.print_current:
        print(_baseline_payload(current), end="")
        return 0

    failures = evaluate_against_baseline(current, baseline)
    if failures:
        print("Maintainability check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Maintainability check passed for {len(current)} critical modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
