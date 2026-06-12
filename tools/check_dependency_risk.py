#!/usr/bin/env python3
"""Record local dependency evidence and optionally run pip-audit."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a command, capture output in ``log_path``, and fail closed on errors."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    log_path.write_text(
        "\n".join(
            [
                f"$ {shlex.join(cmd)}",
                f"cwd: {cwd}",
                f"returncode: {result.returncode}",
                "",
                "[stdout]",
                result.stdout,
                "",
                "[stderr]",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"Command failed: {shlex.join(cmd)}", file=sys.stderr)
        print(f"Log: {log_path}", file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def _environment_command() -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import json, platform, sys; "
            "print(json.dumps({"
            "'python': sys.version, "
            "'executable': sys.executable, "
            "'platform': platform.platform()"
            "}, indent=2))"
        ),
    ]


def run_dependency_risk_checks(
    *,
    root: Path,
    evidence_dir: Path,
    run_pip_audit: bool,
) -> dict[str, object]:
    """Run the local dependency-risk gate and write a summary."""
    logs = evidence_dir / "logs"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    requirements_path = evidence_dir / "requirements-all-extras.txt"
    pip_audit_report = evidence_dir / "pip-audit.json"

    lock_cmd = ["uv", "lock", "--check"]
    export_cmd = [
        "uv",
        "export",
        "--all-extras",
        "--format",
        "requirements.txt",
        "--no-hashes",
        "--output-file",
        str(requirements_path),
        "--locked",
    ]
    pip_audit_cmd = [
        "uv",
        "run",
        "--with",
        "pip-audit",
        "pip-audit",
        "-r",
        str(requirements_path),
        "--no-deps",
        "--disable-pip",
        "--format",
        "json",
        "--output",
        str(pip_audit_report),
    ]

    _run(lock_cmd, cwd=root, log_path=logs / "uv-lock-check.log")
    _run(export_cmd, cwd=root, log_path=logs / "uv-export-all-extras.log")
    environment = _run(
        _environment_command(),
        cwd=root,
        log_path=logs / "environment.log",
    ).stdout

    pip_audit: dict[str, object]
    if run_pip_audit:
        _run(pip_audit_cmd, cwd=root, log_path=logs / "pip-audit.log")
        pip_audit = {
            "status": "passed",
            "command": pip_audit_cmd,
            "report": str(pip_audit_report),
            "log": str(logs / "pip-audit.log"),
        }
    else:
        pip_audit = {
            "status": "skipped",
            "reason": "disabled by --skip-pip-audit",
            "command": pip_audit_cmd,
            "report": str(pip_audit_report),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": str(requirements_path),
        "environment": json.loads(environment),
        "checks": {
            "uv_lock_check": {
                "command": lock_cmd,
                "log": str(logs / "uv-lock-check.log"),
            },
            "uv_export_all_extras": {
                "command": export_cmd,
                "log": str(logs / "uv-export-all-extras.log"),
            },
            "pip_audit": pip_audit,
        },
    }
    summary_path = evidence_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Dependency risk evidence written: {summary_path}")
    if not run_pip_audit:
        print("pip-audit was skipped; this is not sufficient for release evidence.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check and record local dependency/supply-chain release evidence.",
    )
    parser.add_argument(
        "--evidence-dir",
        default="release-evidence/dependency-risk",
        help="Directory for exported requirements, logs, audit report, and summary JSON.",
    )
    parser.add_argument(
        "--skip-pip-audit",
        action="store_true",
        help="Skip the vulnerability audit; intended only for offline development checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dependency_risk_checks(
        root=_repo_root(),
        evidence_dir=Path(args.evidence_dir),
        run_pip_audit=not args.skip_pip_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
