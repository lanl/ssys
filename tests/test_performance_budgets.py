"""Performance budget gate tests."""

from __future__ import annotations

import json
from pathlib import Path

from tools.check_performance_budgets import (
    PerformanceBudget,
    TaskExecution,
    evaluate_task_execution,
    load_budgets,
    run_performance_budgets,
)


def test_load_budgets_reads_task_specs(tmp_path: Path) -> None:
    budget_file = tmp_path / "budgets.json"
    budget_file.write_text(
        json.dumps({
            "tasks": {
                "sample": {
                    "max_seconds": 1.5,
                    "timeout_seconds": 3.0,
                    "description": "sample task",
                }
            }
        }),
        encoding="utf-8",
    )

    budgets = load_budgets(budget_file)

    assert budgets == {
        "sample": PerformanceBudget(
            max_seconds=1.5,
            timeout_seconds=3.0,
            description="sample task",
        )
    }


def test_evaluate_task_execution_classifies_failures() -> None:
    budget = PerformanceBudget(max_seconds=2.0, timeout_seconds=5.0)

    assert evaluate_task_execution("task", budget, 1.0, 0, False).status == "pass"
    assert evaluate_task_execution("task", budget, 3.0, 0, False).status == "budget_exceeded"
    assert evaluate_task_execution("task", budget, 1.0, 2, False).status == "error"
    assert evaluate_task_execution("task", budget, 5.0, 124, True).status == "timeout"


def test_run_performance_budgets_uses_selected_tasks(tmp_path: Path) -> None:
    budgets = {
        "fast": PerformanceBudget(max_seconds=1.0, timeout_seconds=2.0),
        "slow": PerformanceBudget(max_seconds=1.0, timeout_seconds=2.0),
    }
    called = []

    def fake_runner(name: str, budget: PerformanceBudget, evidence_dir: Path) -> TaskExecution:
        called.append((name, evidence_dir))
        return TaskExecution(
            name=name,
            status="pass",
            duration_seconds=0.1,
            max_seconds=budget.max_seconds,
            timeout_seconds=budget.timeout_seconds,
            returncode=0,
            timed_out=False,
            details={},
        )

    results = run_performance_budgets(
        budgets,
        tmp_path / "evidence",
        selected_tasks=["fast"],
        task_runner=fake_runner,
    )

    assert [result.name for result in results] == ["fast"]
    assert called == [("fast", tmp_path / "evidence")]
