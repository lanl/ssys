"""Tests for the local dependency-risk release helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ssys
from tools import check_dependency_risk


def test_dependency_risk_helper_records_lock_export_and_skipped_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert ssys.__version__
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: Path, log_path: Path):
        calls.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log", encoding="utf-8")
        if cmd[:2] == ["uv", "export"]:
            output_path = Path(cmd[cmd.index("--output-file") + 1])
            output_path.write_text("ssys==0.5.5\n", encoding="utf-8")
        stdout = "{}"
        if cmd[0].endswith("python") or cmd[0] == check_dependency_risk.sys.executable:
            stdout = json.dumps({
                "python": "3.12.0",
                "executable": cmd[0],
                "platform": "test-platform",
            })
        return check_dependency_risk.subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(check_dependency_risk, "_run", fake_run)

    summary = check_dependency_risk.run_dependency_risk_checks(
        root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        run_pip_audit=False,
    )

    assert calls[0] == ["uv", "lock", "--check"]
    assert calls[1][:4] == ["uv", "export", "--all-extras", "--format"]
    assert not any("pip-audit" in cmd for call in calls for cmd in call)
    assert summary["checks"]["pip_audit"]["status"] == "skipped"
    assert (tmp_path / "evidence" / "requirements-all-extras.txt").read_text() == (
        "ssys==0.5.5\n"
    )
    saved_summary = json.loads((tmp_path / "evidence" / "summary.json").read_text())
    assert saved_summary["environment"]["platform"] == "test-platform"


def test_dependency_risk_helper_runs_pip_audit_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: Path, log_path: Path):
        calls.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log", encoding="utf-8")
        if cmd[:2] == ["uv", "export"]:
            output_path = Path(cmd[cmd.index("--output-file") + 1])
            output_path.write_text("ssys==0.5.5\n", encoding="utf-8")
        if "pip-audit" in cmd:
            report_path = Path(cmd[cmd.index("--output") + 1])
            report_path.write_text('{"dependencies": []}\n', encoding="utf-8")
        stdout = "{}"
        if cmd[0].endswith("python") or cmd[0] == check_dependency_risk.sys.executable:
            stdout = json.dumps({
                "python": "3.12.0",
                "executable": cmd[0],
                "platform": "test-platform",
            })
        return check_dependency_risk.subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(check_dependency_risk, "_run", fake_run)

    summary = check_dependency_risk.run_dependency_risk_checks(
        root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        run_pip_audit=True,
    )

    audit_calls = [call for call in calls if "pip-audit" in call]
    assert len(audit_calls) == 1
    assert audit_calls[0][:4] == ["uv", "run", "--with", "pip-audit"]
    assert "--no-deps" in audit_calls[0]
    assert "--disable-pip" in audit_calls[0]
    assert summary["checks"]["pip_audit"]["status"] == "passed"
    assert (tmp_path / "evidence" / "pip-audit.json").exists()


def test_dependency_risk_main_supports_offline_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_checks(*, root: Path, evidence_dir: Path, run_pip_audit: bool):
        captured["root"] = root
        captured["evidence_dir"] = evidence_dir
        captured["run_pip_audit"] = run_pip_audit
        return {}

    monkeypatch.setattr(check_dependency_risk, "run_dependency_risk_checks", fake_run_checks)

    result = check_dependency_risk.main([
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--skip-pip-audit",
    ])

    assert result == 0
    assert captured["evidence_dir"] == tmp_path / "evidence"
    assert captured["run_pip_audit"] is False
