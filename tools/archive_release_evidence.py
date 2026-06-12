#!/usr/bin/env python3
"""Create a hashed manifest for local release-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_DIR = Path("release-evidence")
DEFAULT_OUTPUT_NAME = "evidence-manifest.json"


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def collect_file_records(root: Path, output_path: Path | None = None) -> list[FileRecord]:
    records: list[FileRecord] = []
    if not root.exists():
        return records
    resolved_output = output_path.resolve() if output_path is not None else None
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if resolved_output is not None and path.resolve() == resolved_output:
            continue
        records.append(
            FileRecord(
                path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return records


def collect_artifact_records(artifact_dir: Path) -> list[FileRecord]:
    if not artifact_dir.exists():
        return []
    records = []
    for path in sorted(artifact_dir.glob("*")):
        if path.is_file():
            records.append(
                FileRecord(
                    path=path.as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=_sha256(path),
                )
            )
    return records


def missing_required_dirs(evidence_dir: Path, required: list[str]) -> list[str]:
    return [name for name in required if not (evidence_dir / name).is_dir()]


def build_manifest(
    evidence_dir: Path,
    output_path: Path,
    artifact_dir: Path,
    required_dirs: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    evidence_records = collect_file_records(evidence_dir, output_path=output_path)
    artifact_records = collect_artifact_records(artifact_dir)
    missing = missing_required_dirs(evidence_dir, required_dirs)
    return {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "repo_root": str(repo_root),
        "evidence_dir": str(evidence_dir),
        "artifact_dir": str(artifact_dir),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": _git_metadata(repo_root),
        "required_dirs": required_dirs,
        "missing_required_dirs": missing,
        "file_count": len(evidence_records),
        "files": [asdict(record) for record in evidence_records],
        "artifact_count": len(artifact_records),
        "artifacts": [asdict(record) for record in artifact_records],
        "overall_pass": not missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a hashed manifest for local release evidence."
    )
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest path. Defaults to <evidence-dir>/evidence-manifest.json.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Evidence subdirectory that must exist. May be repeated.",
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_path = (args.output or evidence_dir / DEFAULT_OUTPUT_NAME).resolve()

    manifest = build_manifest(
        evidence_dir=evidence_dir,
        output_path=output_path,
        artifact_dir=args.artifact_dir.resolve(),
        required_dirs=args.require,
        repo_root=repo_root,
    )
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if manifest["missing_required_dirs"]:
        print("Release evidence archive incomplete:", file=sys.stderr)
        for name in manifest["missing_required_dirs"]:
            print(f"- missing required evidence directory: {name}", file=sys.stderr)
        return 1

    print(
        f"Release evidence manifest written to {output_path} "
        f"({manifest['file_count']} evidence files, {manifest['artifact_count']} artifacts)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
