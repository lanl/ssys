#!/usr/bin/env python3
"""Run local performance budget checks for representative ssys workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BUDGET_FILE = Path("tools/performance_budgets.json")
DEFAULT_EVIDENCE_DIR = Path("release-evidence/performance")

MODEL_TEXT = """model perf()
  J0: X -> ; k*X;
  X = 1.0
  k = 0.5
end
"""


@dataclass(frozen=True)
class PerformanceBudget:
    max_seconds: float
    timeout_seconds: float
    description: str = ""


@dataclass(frozen=True)
class TaskExecution:
    name: str
    status: str
    duration_seconds: float
    max_seconds: float
    timeout_seconds: float
    returncode: int
    timed_out: bool
    details: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _prepare_recast(work_dir: Path) -> tuple[Path, Path]:
    from ssys import recast_to_ssystem, ssystem_to_antimony
    from ssys.parsing import parse_antimony_via_sbml

    work_dir.mkdir(parents=True, exist_ok=True)
    original = work_dir / "original.ant"
    recast = work_dir / "recast.ant"
    original.write_text(MODEL_TEXT, encoding="utf-8")

    sym = parse_antimony_via_sbml(MODEL_TEXT)
    result = recast_to_ssystem(sym, mode="simplified")
    recast.write_text(
        ssystem_to_antimony(result, model_name="perf_recast", mode="simplified"),
        encoding="utf-8",
    )
    return original, recast


def _task_recast_small(work_dir: Path) -> dict[str, Any]:
    _, recast = _prepare_recast(work_dir)
    return {"recast_bytes": recast.stat().st_size}


def _task_validation_profile(work_dir: Path, profile: str) -> dict[str, Any]:
    from ssys.validator import validate_recast_pair

    original, recast = _prepare_recast(work_dir)
    report = validate_recast_pair(
        str(original),
        str(recast),
        mode="simplified",
        parser="sbml",
        profile=profile,
    )
    if not report.overall_pass:
        raise RuntimeError(f"{profile} validation did not pass: {report.summary}")
    return {"profile": profile, "summary": report.summary}


def _run_worker_task(task_name: str, work_dir: Path) -> dict[str, Any]:
    if task_name == "recast_small":
        return _task_recast_small(work_dir)
    if task_name == "symbolic_validation_small":
        return _task_validation_profile(work_dir, "symbolic")
    if task_name == "numerical_validation_small":
        return _task_validation_profile(work_dir, "numerical")
    if task_name == "trajectory_validation_small":
        return _task_validation_profile(work_dir, "trajectory")
    raise ValueError(f"unknown performance task {task_name!r}")


def load_budgets(path: Path) -> dict[str, PerformanceBudget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{path} must contain a 'tasks' object")
    return {
        name: PerformanceBudget(
            max_seconds=float(spec["max_seconds"]),
            timeout_seconds=float(spec["timeout_seconds"]),
            description=str(spec.get("description", "")),
        )
        for name, spec in tasks.items()
    }


def evaluate_task_execution(
    name: str,
    budget: PerformanceBudget,
    duration_seconds: float,
    returncode: int,
    timed_out: bool,
    details: dict[str, Any] | None = None,
) -> TaskExecution:
    if timed_out:
        status = "timeout"
    elif returncode != 0:
        status = "error"
    elif duration_seconds > budget.max_seconds:
        status = "budget_exceeded"
    else:
        status = "pass"
    return TaskExecution(
        name=name,
        status=status,
        duration_seconds=round(duration_seconds, 3),
        max_seconds=budget.max_seconds,
        timeout_seconds=budget.timeout_seconds,
        returncode=returncode,
        timed_out=timed_out,
        details=details or {},
    )


def _run_task_subprocess(
    task_name: str,
    budget: PerformanceBudget,
    evidence_dir: Path,
) -> TaskExecution:
    task_dir = evidence_dir / "work" / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task",
        task_name,
        "--worker-dir",
        str(task_dir),
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=budget.timeout_seconds,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"task timed out after {budget.timeout_seconds} seconds",
        )
        timed_out = True
    duration = time.perf_counter() - start

    logs_dir = evidence_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"{task_name}_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs_dir / f"{task_name}_stderr.log").write_text(completed.stderr, encoding="utf-8")

    details: dict[str, Any] = {"command": command}
    if completed.stdout.strip():
        try:
            details["worker_result"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            details["worker_stdout"] = completed.stdout.strip()
    if completed.stderr.strip():
        details["worker_stderr"] = completed.stderr.strip()

    return evaluate_task_execution(
        task_name,
        budget,
        duration,
        completed.returncode,
        timed_out,
        details,
    )


TaskRunner = Callable[[str, PerformanceBudget, Path], TaskExecution]


def run_performance_budgets(
    budgets: dict[str, PerformanceBudget],
    evidence_dir: Path,
    selected_tasks: list[str] | None = None,
    task_runner: TaskRunner = _run_task_subprocess,
) -> list[TaskExecution]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    task_names = selected_tasks or list(budgets)
    unknown = sorted(set(task_names) - set(budgets))
    if unknown:
        raise ValueError(f"unknown performance task(s): {', '.join(unknown)}")
    return [task_runner(name, budgets[name], evidence_dir) for name in task_names]


def _write_summary(evidence_dir: Path, results: list[TaskExecution]) -> None:
    payload = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "tasks": [asdict(result) for result in results],
        "overall_pass": all(result.status == "pass" for result in results),
    }
    (evidence_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check representative local performance budgets."
    )
    parser.add_argument("--budget-file", type=Path, default=DEFAULT_BUDGET_FILE)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--task", action="append", help="Run only this task; may repeat.")
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_task:
        if args.worker_dir is None:
            raise SystemExit("--worker-dir is required with --worker-task")
        with tempfile.TemporaryDirectory(dir=args.worker_dir.parent) as tmp:
            result = _run_worker_task(args.worker_task, Path(tmp))
        print(json.dumps(result, sort_keys=True))
        return 0

    budgets = load_budgets(args.budget_file)
    results = run_performance_budgets(budgets, args.evidence_dir, selected_tasks=args.task)
    _write_summary(args.evidence_dir, results)

    failures = [result for result in results if result.status != "pass"]
    if failures:
        print("Performance budget check failed:", file=sys.stderr)
        for failure in failures:
            print(
                f"- {failure.name}: {failure.status} "
                f"({failure.duration_seconds:.3f}s > {failure.max_seconds:.3f}s budget)",
                file=sys.stderr,
            )
        return 1

    print(f"Performance budget check passed for {len(results)} task(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
