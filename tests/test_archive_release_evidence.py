"""Release evidence archive manifest tests."""

from __future__ import annotations

import json
from pathlib import Path

from tools.archive_release_evidence import (
    build_manifest,
    collect_file_records,
    main,
    missing_required_dirs,
)


def test_collect_file_records_hashes_evidence_files(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "log.txt").write_text("local evidence\n", encoding="utf-8")
    output = evidence / "evidence-manifest.json"

    records = collect_file_records(evidence, output_path=output)

    assert len(records) == 1
    assert records[0].path == "log.txt"
    assert records[0].size_bytes == len("local evidence\n")
    assert len(records[0].sha256) == 64


def test_missing_required_dirs_reports_absent_directories(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "present").mkdir(parents=True)

    assert missing_required_dirs(evidence, ["present", "missing"]) == ["missing"]


def test_build_manifest_records_artifacts_and_required_dirs(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    artifact_dir = tmp_path / "dist"
    (evidence / "performance").mkdir(parents=True)
    artifact_dir.mkdir()
    (evidence / "performance" / "summary.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "ssys.whl").write_text("wheel", encoding="utf-8")

    manifest = build_manifest(
        evidence_dir=evidence,
        output_path=evidence / "evidence-manifest.json",
        artifact_dir=artifact_dir,
        required_dirs=["performance"],
        repo_root=tmp_path,
    )

    assert manifest["overall_pass"] is True
    assert manifest["file_count"] == 1
    assert manifest["artifact_count"] == 1
    assert manifest["files"][0]["path"] == "performance/summary.json"
    assert manifest["artifacts"][0]["path"].endswith("ssys.whl")


def test_main_writes_manifest_and_fails_when_required_dir_missing(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "present").mkdir()
    (evidence / "present" / "summary.json").write_text("{}", encoding="utf-8")

    exit_code = main([
        "--evidence-dir",
        str(evidence),
        "--require",
        "present",
        "--require",
        "missing",
    ])

    assert exit_code == 1
    payload = json.loads((evidence / "evidence-manifest.json").read_text(encoding="utf-8"))
    assert payload["missing_required_dirs"] == ["missing"]
