"""Maintainability metric gate tests."""

from __future__ import annotations

from pathlib import Path

from tools.check_maintainability import (
    ModuleMetrics,
    evaluate_against_baseline,
    measure_file,
)


def test_measure_file_counts_lines_function_length_and_complexity(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "\n".join([
            "def branchy(x):",
            "    if x and x > 0:",
            "        return [item for item in range(3) if item]",
            "    return 0",
            "",
            "def small():",
            "    return 1",
            "",
        ]),
        encoding="utf-8",
    )

    metrics = measure_file(source)

    assert metrics.module_lines == 7
    assert metrics.max_function_lines == 4
    assert metrics.max_cyclomatic_complexity == 4


def test_evaluate_against_baseline_allows_equal_or_lower_metrics() -> None:
    current = {
        "module.py": ModuleMetrics(
            module_lines=10,
            max_function_lines=4,
            max_cyclomatic_complexity=3,
        )
    }
    baseline = {
        "module.py": ModuleMetrics(
            module_lines=12,
            max_function_lines=4,
            max_cyclomatic_complexity=4,
        )
    }

    assert evaluate_against_baseline(current, baseline) == []


def test_evaluate_against_baseline_reports_regressions() -> None:
    current = {
        "module.py": ModuleMetrics(
            module_lines=13,
            max_function_lines=5,
            max_cyclomatic_complexity=6,
        )
    }
    baseline = {
        "module.py": ModuleMetrics(
            module_lines=12,
            max_function_lines=4,
            max_cyclomatic_complexity=4,
        )
    }

    failures = evaluate_against_baseline(current, baseline)

    assert "module.py: module_lines increased from 12 to 13" in failures
    assert "module.py: max_function_lines increased from 4 to 5" in failures
    assert "module.py: max_cyclomatic_complexity increased from 4 to 6" in failures
