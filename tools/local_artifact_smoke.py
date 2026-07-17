#!/usr/bin/env python3
"""Build and smoke-test local ssys release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12")
TINY_MODEL = """model decay()
    species X;
    X' = -k*X;
    k = 0.5;
    X = 1.0;
end
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("._") or "python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(venv: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return venv / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(cmd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    log_path.write_text(
        "\n".join(
            [
                f"$ {command_text}",
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
        print(f"Command failed: {command_text}", file=sys.stderr)
        print(f"Log: {log_path}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def _run_quiet(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def _require_clean_tree(root: Path) -> None:
    if not (root / ".git").exists():
        return
    result = _run_quiet(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    if result.returncode != 0:
        raise SystemExit(f"could not inspect git status:\n{result.stderr}")
    if result.stdout.strip():
        raise SystemExit(
            "local artifact smoke requires a clean git tree; commit/stash changes or pass "
            "--allow-dirty for development-only verification"
        )


def _discover_supported_python_specs(root: Path) -> list[str]:
    discovered: list[str] = []
    uv = shutil.which("uv")
    for version in SUPPORTED_PYTHONS:
        if uv:
            result = _run_quiet(["uv", "python", "find", version], cwd=root)
            if result.returncode == 0 and result.stdout.strip():
                discovered.append(result.stdout.strip())
                continue
        for candidate in (f"python{version}", f"python{version.replace('.', '')}"):
            path = shutil.which(candidate)
            if path:
                discovered.append(path)
                break
    return _dedupe_preserving_order(discovered)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _python_minor_version(python_spec: str, root: Path) -> str:
    """Return the major.minor version for an interpreter spec."""
    executable = None
    uv = shutil.which("uv")
    if uv:
        result = _run_quiet(["uv", "python", "find", python_spec], cwd=root)
        if result.returncode == 0 and result.stdout.strip():
            executable = result.stdout.strip()
    if executable is None:
        executable = _resolve_python_executable(python_spec)
    result = _run_quiet(
        [
            executable,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        cwd=root,
    )
    if result.returncode != 0:
        raise SystemExit(f"could not inspect Python interpreter {python_spec!r}:\n{result.stderr}")
    return result.stdout.strip()


def _validate_supported_python_specs(python_specs: list[str], root: Path) -> list[str]:
    supported = []
    for python_spec in _dedupe_preserving_order(python_specs):
        minor = _python_minor_version(python_spec, root)
        if minor not in SUPPORTED_PYTHONS:
            supported_text = ", ".join(SUPPORTED_PYTHONS)
            raise SystemExit(
                f"Python interpreter {python_spec!r} is {minor}; supported versions for "
                f"this release are: {supported_text}"
            )
        supported.append(python_spec)
    return supported


def _select_python_specs(args: argparse.Namespace, root: Path) -> list[str]:
    """Select supported interpreters for the smoke run."""
    if args.all_supported_pythons:
        python_specs = _discover_supported_python_specs(root)
        if not python_specs:
            raise SystemExit("no supported Python 3.10/3.11/3.12 interpreters found locally")
        return _validate_supported_python_specs(python_specs, root)

    if args.python_specs:
        return _validate_supported_python_specs(args.python_specs, root)

    current_minor = _python_minor_version(sys.executable, root)
    if current_minor in SUPPORTED_PYTHONS:
        return [sys.executable]

    python_specs = _discover_supported_python_specs(root)
    if python_specs:
        return _validate_supported_python_specs(python_specs, root)

    supported_text = ", ".join(SUPPORTED_PYTHONS)
    raise SystemExit(
        f"current Python {current_minor} is not supported by this release and no supported "
        f"interpreter was discovered locally; pass --python with one of: {supported_text}"
    )


def _create_venv(python_spec: str, venv: Path, root: Path, log_path: Path) -> Path:
    if venv.exists():
        shutil.rmtree(venv)
    uv = shutil.which("uv")
    if uv:
        _run([uv, "venv", "--seed", "--python", python_spec, str(venv)], cwd=root, log_path=log_path)
    else:
        executable = _resolve_python_executable(python_spec)
        _run([executable, "-m", "venv", str(venv)], cwd=root, log_path=log_path)
        _run([str(_venv_python(venv)), "-m", "ensurepip", "--upgrade"], cwd=root, log_path=log_path)
    return _venv_python(venv)


def _resolve_python_executable(python_spec: str) -> str:
    candidates = [python_spec]
    if os.sep not in python_spec:
        candidates.extend([f"python{python_spec}", f"python{python_spec.replace('.', '')}"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    raise SystemExit(f"could not resolve Python interpreter: {python_spec}")


def _find_one(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {dist_dir}, found {matches}")
    return matches[0]


def _build_artifacts(root: Path, evidence_dir: Path) -> tuple[Path, Path]:
    dist_dir = evidence_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    logs = evidence_dir / "logs"
    _run(
        ["uv", "run", "--with", "build", "python", "-m", "build", "--outdir", str(dist_dir)],
        cwd=root,
        log_path=logs / "build.log",
    )
    wheel = _find_one(dist_dir, "*.whl")
    sdist = _find_one(dist_dir, "*.tar.gz")
    _run(
        ["uv", "run", "--with", "twine", "twine", "check", str(wheel), str(sdist)],
        cwd=root,
        log_path=logs / "twine-check.log",
    )
    return wheel, sdist


def _write_tiny_manifest(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    model = work_dir / "decay.ant"
    model.write_text(TINY_MODEL, encoding="utf-8")
    manifest = work_dir / "models.manifest"
    manifest.write_text(f"{model}\n", encoding="utf-8")
    return manifest


def _import_smoke_code() -> str:
    return (
        "import importlib.metadata, json, platform, sys; "
        "import ssys; "
        "assert ssys.__version__ == importlib.metadata.version('ssys'); "
        "assert ssys.__release_maturity__ == 'alpha'; "
        "assert callable(ssys.recast_to_ssystem); "
        "assert callable(ssys.ssystem_to_antimony); "
        "print(json.dumps({"
        "'python': sys.version, "
        "'platform': platform.platform(), "
        "'ssys_version': ssys.__version__, "
        "'release_date': ssys.__release_date__, "
        "'release_maturity': ssys.__release_maturity__"
        "}, indent=2))"
    )


def _smoke_installed_artifact(
    *,
    artifact: Path,
    artifact_kind: str,
    python_spec: str,
    label: str,
    root: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    logs = evidence_dir / "logs" / label
    venv = evidence_dir / f"{artifact_kind}-venv-{label}"
    work_dir = evidence_dir / f"{artifact_kind}-work-{label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    python_exe = _create_venv(python_spec, venv, root, logs / f"{artifact_kind}-venv.log")

    _run(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=root,
        log_path=logs / f"{artifact_kind}-pip-upgrade.log",
    )
    _run(
        [str(python_exe), "-m", "pip", "install", str(artifact)],
        cwd=root,
        log_path=logs / f"{artifact_kind}-install.log",
    )
    _run(
        [str(python_exe), "-m", "pip", "freeze", "--all"],
        cwd=root,
        log_path=logs / f"{artifact_kind}-freeze.log",
    )
    import_result = _run(
        [str(python_exe), "-c", _import_smoke_code()],
        cwd=work_dir,
        log_path=logs / f"{artifact_kind}-import-smoke.log",
    )
    manifest = _write_tiny_manifest(work_dir)
    recast = _venv_script(venv, "ssys-recast")
    _run([str(recast), "--version"], cwd=work_dir, log_path=logs / f"{artifact_kind}-version.log")

    out_dir = work_dir / "out"
    _run(
        [
            str(recast),
            "--manifest",
            str(manifest),
            "--outdir",
            str(out_dir),
            "--mode",
            "simplified",
        ],
        cwd=work_dir,
        log_path=logs / f"{artifact_kind}-cli-recast.log",
    )
    output = out_dir / "decay_recast.ant"
    if not output.exists() or output.stat().st_size == 0:
        raise SystemExit(f"{artifact_kind} smoke did not produce {output}")

    validated_dir = work_dir / "out_validated"
    _run(
        [
            str(recast),
            "--manifest",
            str(manifest),
            "--outdir",
            str(validated_dir),
            "--mode",
            "simplified",
            "--validate",
        ],
        cwd=work_dir,
        log_path=logs / f"{artifact_kind}-cli-validate.log",
    )
    report_path = validated_dir / "decay_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("overall_pass") is not True:
        raise SystemExit(f"{artifact_kind} validation smoke did not pass: {report}")

    return {
        "artifact_kind": artifact_kind,
        "python_spec": python_spec,
        "python_label": label,
        "venv": str(venv),
        "work_dir": str(work_dir),
        "import_smoke": json.loads(import_result.stdout),
        "recast_output": str(output),
        "validation_report": str(report_path),
    }


def _safe_extract_sdist(sdist: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(sdist) as tar:
        root_resolved = extract_root.resolve()
        for member in tar.getmembers():
            target = (extract_root / member.name).resolve()
            if root_resolved not in (target, *target.parents):
                raise SystemExit(f"refusing unsafe sdist member path: {member.name}")
        extract_kwargs = {}
        if "filter" in inspect.signature(tar.extractall).parameters:
            extract_kwargs["filter"] = "data"
        tar.extractall(extract_root, **extract_kwargs)
    roots = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one unpacked sdist root, found {roots}")
    return roots[0]


def _verify_sdist_fixtures(sdist_root: Path) -> int:
    required_paths = [
        "tests/test_integration.py",
        "test_models1/models.manifest",
        "test_models2/models.manifest",
        "test_models3/models.manifest",
        "test_models4/models.manifest",
        "tools/check_release_metadata.py",
        "tools/local_artifact_smoke.py",
        "RELEASE_NOTES.md",
    ]
    missing_paths = [relpath for relpath in required_paths if not (sdist_root / relpath).exists()]
    if missing_paths:
        raise SystemExit(f"sdist is missing required paths: {missing_paths}")

    model_count = 0
    for dirname in ["test_models1", "test_models2", "test_models3", "test_models4"]:
        manifest = sdist_root / dirname / "models.manifest"
        entries = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        missing = [entry for entry in entries if not (manifest.parent / entry).exists()]
        if missing:
            raise SystemExit(f"{dirname} manifest references missing models: {missing}")
        model_count += len(entries)
    if model_count != 117:
        raise SystemExit(f"expected 117 committed test models in sdist, found {model_count}")
    return model_count


def _verify_unpacked_sdist(
    *,
    sdist: Path,
    python_spec: str,
    label: str,
    root: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    logs = evidence_dir / "logs" / label
    extract_root = evidence_dir / f"sdist-unpacked-{label}"
    sdist_root = _safe_extract_sdist(sdist, extract_root)
    model_count = _verify_sdist_fixtures(sdist_root)

    venv = evidence_dir / f"sdist-unpacked-venv-{label}"
    python_exe = _create_venv(python_spec, venv, root, logs / "sdist-unpacked-venv.log")
    _run(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=root,
        log_path=logs / "sdist-unpacked-pip-upgrade.log",
    )
    _run(
        [str(python_exe), "-m", "pip", "install", ".", "pytest"],
        cwd=sdist_root,
        log_path=logs / "sdist-unpacked-install.log",
    )
    _run(
        [str(python_exe), "tools/check_release_metadata.py"],
        cwd=sdist_root,
        log_path=logs / "sdist-unpacked-release-metadata.log",
    )
    _run(
        [
            str(python_exe),
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "tests/test_integration.py",
            "-m",
            "integration and not slow",
            "-v",
        ],
        cwd=sdist_root,
        log_path=logs / "sdist-unpacked-fast-test.log",
    )
    _run(
        [str(python_exe), "-m", "pip", "freeze", "--all"],
        cwd=sdist_root,
        log_path=logs / "sdist-unpacked-freeze.log",
    )
    return {
        "python_spec": python_spec,
        "python_label": label,
        "sdist_root": str(sdist_root),
        "model_count": model_count,
    }


def _prepare_evidence_dir(path: Path | None, force: bool) -> Path:
    if path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(tempfile.mkdtemp(prefix=f"ssys-artifact-smoke-{timestamp}-")).resolve()
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SystemExit(f"evidence directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and smoke-test ssys wheel/sdist artifacts in clean local venvs."
    )
    parser.add_argument(
        "--python",
        action="append",
        dest="python_specs",
        help=(
            "Python interpreter or version spec for smoke venvs; repeatable. Defaults to "
            "the current interpreter when supported, otherwise a discovered supported one."
        ),
    )
    parser.add_argument(
        "--all-supported-pythons",
        action="store_true",
        help="Discover locally available Python 3.10, 3.11, and 3.12 interpreters.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Directory for built artifacts, logs, dependency freezes, and summary JSON.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running from a dirty worktree for development verification.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate a nonempty --evidence-dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    root = _repo_root()
    if not args.allow_dirty:
        _require_clean_tree(root)

    python_specs = _select_python_specs(args, root)

    evidence_dir = _prepare_evidence_dir(args.evidence_dir, args.force)
    print(f"Writing artifact smoke evidence to {evidence_dir}")
    wheel, sdist = _build_artifacts(root, evidence_dir)
    summary: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "allow_dirty": args.allow_dirty,
        "evidence_dir": str(evidence_dir),
        "artifacts": {
            "wheel": {"path": str(wheel), "sha256": _sha256(wheel)},
            "sdist": {"path": str(sdist), "sha256": _sha256(sdist)},
        },
        "python_specs": python_specs,
        "artifact_smokes": [],
        "unpacked_sdist_checks": [],
    }

    for index, python_spec in enumerate(python_specs, start=1):
        label = f"{index:02d}-{_safe_label(python_spec)}"
        print(f"Smoke-testing wheel with {python_spec}")
        summary["artifact_smokes"].append(
            _smoke_installed_artifact(
                artifact=wheel,
                artifact_kind="wheel",
                python_spec=python_spec,
                label=label,
                root=root,
                evidence_dir=evidence_dir,
            )
        )
        print(f"Smoke-testing sdist install with {python_spec}")
        summary["artifact_smokes"].append(
            _smoke_installed_artifact(
                artifact=sdist,
                artifact_kind="sdist",
                python_spec=python_spec,
                label=label,
                root=root,
                evidence_dir=evidence_dir,
            )
        )
        print(f"Verifying unpacked sdist with {python_spec}")
        summary["unpacked_sdist_checks"].append(
            _verify_unpacked_sdist(
                sdist=sdist,
                python_spec=python_spec,
                label=label,
                root=root,
                evidence_dir=evidence_dir,
            )
        )

    summary_path = evidence_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Local artifact smoke passed. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
