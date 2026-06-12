#!/usr/bin/env python3
"""Run or summarize the local BioModels benchmark with release evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_DIR = Path("release-evidence/biomodels")
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
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _validation_files(validation_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in VALIDATION_PATTERNS:
        files.extend(validation_dir.glob(pattern))
    return sorted(set(files))


def summarize_biomodels_outputs(benchmark_dir: Path, subset_size: int = 10) -> dict[str, Any]:
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

    for path in validation_files:
        report = _read_json(path)
        if "_read_error" in report:
            validation_result_counts["unreadable"] += 1
            continue
        result = str(report.get("overall_result") or report.get("overall_pass"))
        validation_result_counts[result] += 1
        profile = report.get("validation_profile")
        if isinstance(profile, dict):
            validation_profile_counts[str(profile.get("name", "unknown"))] += 1
        elif profile:
            validation_profile_counts[str(profile)] += 1
        for test in report.get("tests", {}).values():
            if isinstance(test, dict) and test.get("reason"):
                validation_reason_counts[str(test["reason"])] += 1

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
        "validation_pass_count_from_results_csv": validation_pass_count,
        "validation_result_counts": dict(sorted(validation_result_counts.items())),
        "validation_profile_counts": dict(sorted(validation_profile_counts.items())),
        "validation_reason_counts": dict(validation_reason_counts.most_common(20)),
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
    parser.add_argument("--subset-size", type=int, default=10)
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

    if args.artifact and not args.skip_run:
        benchmark_dir, benchmark_python, artifact_metadata = _prepare_artifact_benchmark(
            source_benchmark_dir=benchmark_dir,
            artifact=args.artifact,
            evidence_dir=evidence_dir,
            python_executable=args.artifact_python,
        )
        benchmark_env = os.environ.copy()
        benchmark_env["PATH"] = f"{benchmark_python.parent}:{benchmark_env.get('PATH', '')}"
        benchmark_env.pop("PYTHONPATH", None)

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

    summary = summarize_biomodels_outputs(benchmark_dir, subset_size=args.subset_size)
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
