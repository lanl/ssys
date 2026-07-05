#!/usr/bin/env python3
"""Validate release metadata that must stay in sync across project files."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

EXPECTED_PYTHON_CLASSIFIERS = {
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
}

EXPECTED_SUPPORT_TEXT = "Python 3.10, 3.11, and 3.12"
EXPECTED_MATURITY = "alpha"
EXPECTED_TRUST_TERMS = ("trusted", "untrusted")


def _read(root: Path, relpath: str) -> str:
    return (root / relpath).read_text(encoding="utf-8")


def _match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"could not find {label}")
    return match.group(1)


def _parse_init_metadata(text: str) -> dict[str, str]:
    module = ast.parse(text)
    values: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    values[target.id] = node.value.value
    return values


def _parse_citation(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _parse_quoted_list(text: str, key: str) -> list[str]:
    match = re.search(rf"(?ms)^{re.escape(key)}\s*=\s*\[(.*?)^\]", text)
    if not match:
        raise ValueError(f"could not find pyproject list {key}")
    return re.findall(r'"([^"]+)"', match.group(1))


def _parse_tool_release(text: str) -> dict[str, str]:
    match = re.search(r"(?ms)^\[tool\.ssys\.release\]\n(.*?)(?:^\[|\Z)", text)
    if not match:
        raise ValueError("could not find [tool.ssys.release]")

    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _release_section(text: str, version: str) -> str:
    match = re.search(
        rf"(?ms)^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n(.*?)(?:^## |\Z)",
        text,
    )
    if not match:
        raise ValueError(f"could not find changelog section for {version}")
    return match.group(1)


def _check_versions_and_dates(root: Path) -> list[str]:
    errors: list[str] = []
    pyproject = _read(root, "pyproject.toml")
    init_text = _read(root, "src/ssys/__init__.py")
    citation = _parse_citation(_read(root, "CITATION.cff"))
    readme = _read(root, "README.md")
    changelog = _read(root, "CHANGELOG.md")
    release_notes = _read(root, "RELEASE_NOTES.md")

    init_metadata = _parse_init_metadata(init_text)
    tool_release = _parse_tool_release(pyproject)

    versions = {
        "pyproject.toml": _match(r'^version\s*=\s*"([^"]+)"', pyproject, "project version"),
        "src/ssys/__init__.py": init_metadata.get("__version__", ""),
        "CITATION.cff": citation.get("version", ""),
        "README.md": _match(r"\| v([^ |]+) \|", readme, "README version"),
        "CHANGELOG.md": _match(r"^## \[([^\]]+)\] - \d{4}-\d{2}-\d{2}", changelog, "changelog version"),
        "RELEASE_NOTES.md": _match(r"^# ssys ([^ ]+) Release Notes", release_notes, "release notes version"),
    }
    if len(set(versions.values())) != 1:
        errors.append(f"release version mismatch: {versions}")

    dates = {
        "pyproject.toml": tool_release.get("date", ""),
        "src/ssys/__init__.py": init_metadata.get("__release_date__", ""),
        "CITATION.cff": citation.get("date-released", ""),
        "README.md": _match(r"\| v[^ |]+ \| (\d{4}-\d{2}-\d{2})", readme, "README date"),
        "CHANGELOG.md": _match(r"^## \[[^\]]+\] - (\d{4}-\d{2}-\d{2})", changelog, "changelog date"),
        "RELEASE_NOTES.md": _match(
            r"^\*Version: [^ |]+ \| Release date: (\d{4}-\d{2}-\d{2})\*",
            release_notes,
            "release notes date",
        ),
    }
    if len(set(dates.values())) != 1:
        errors.append(f"release date mismatch: {dates}")

    return errors


def _check_maturity_support_and_trust(root: Path) -> list[str]:
    errors: list[str] = []
    pyproject = _read(root, "pyproject.toml")
    init_text = _read(root, "src/ssys/__init__.py")
    readme = _read(root, "README.md")
    changelog = _read(root, "CHANGELOG.md")
    release_notes = _read(root, "RELEASE_NOTES.md")
    init_metadata = _parse_init_metadata(init_text)
    tool_release = _parse_tool_release(pyproject)
    version = _match(r'^version\s*=\s*"([^"]+)"', pyproject, "project version")
    changelog_release = _release_section(changelog, version)

    classifiers = set(_parse_quoted_list(pyproject, "classifiers"))
    if "Development Status :: 3 - Alpha" not in classifiers:
        errors.append("pyproject.toml must advertise alpha maturity")
    if tool_release.get("maturity") != EXPECTED_MATURITY:
        errors.append("[tool.ssys.release].maturity must be alpha")
    if init_metadata.get("__release_maturity__") != EXPECTED_MATURITY:
        errors.append("src/ssys/__init__.py __release_maturity__ must be alpha")

    maturity_sources = {
        "README.md": readme,
        "CHANGELOG.md": changelog_release,
        "RELEASE_NOTES.md": release_notes,
    }
    for relpath, text in maturity_sources.items():
        if EXPECTED_MATURITY not in text.lower():
            errors.append(f"{relpath} must state alpha maturity")

    requires_python = _match(
        r'^requires-python\s*=\s*"([^"]+)"', pyproject, "requires-python"
    )
    if requires_python != ">=3.10,<3.13":
        errors.append("pyproject.toml requires-python must be >=3.10,<3.13")
    if not EXPECTED_PYTHON_CLASSIFIERS.issubset(classifiers):
        errors.append("pyproject.toml must classify Python 3.10, 3.11, and 3.12")
    if any(classifier.endswith(":: 3.13") for classifier in classifiers):
        errors.append("pyproject.toml must not advertise Python 3.13")
    if EXPECTED_SUPPORT_TEXT not in tool_release.get("support", ""):
        errors.append("[tool.ssys.release].support must name Python 3.10, 3.11, and 3.12")

    support_sources = {
        "README.md": readme,
        "CHANGELOG.md": changelog_release,
        "RELEASE_NOTES.md": release_notes,
    }
    for relpath, text in support_sources.items():
        if EXPECTED_SUPPORT_TEXT not in text:
            errors.append(f"{relpath} must state the supported Python versions")

    trust_sources = {
        "README.md": readme,
        "CHANGELOG.md": changelog_release,
        "RELEASE_NOTES.md": release_notes,
        "pyproject.toml": tool_release.get("trust-boundary", ""),
    }
    for relpath, text in trust_sources.items():
        lower = text.lower()
        if not all(term in lower for term in EXPECTED_TRUST_TERMS):
            errors.append(f"{relpath} must state the trusted/untrusted input boundary")

    return errors


def _check_project_metadata(root: Path) -> list[str]:
    errors: list[str] = []
    pyproject = _read(root, "pyproject.toml")
    readme = _read(root, "README.md")
    changelog = _read(root, "CHANGELOG.md")
    release_notes = _read(root, "RELEASE_NOTES.md")
    citation_text = _read(root, "CITATION.cff")
    citation = _parse_citation(_read(root, "CITATION.cff"))
    license_text = _read(root, "LICENSE")

    if re.search(r"(?m)^\[project\.urls\]", pyproject):
        errors.append("[project.urls] should be omitted until public project URLs exist")
    for citation_key in ("url", "repository-code"):
        if citation.get(citation_key):
            errors.append(f"CITATION.cff {citation_key!r} should be omitted until public URLs exist")
    for relpath, text in {
        "pyproject.toml": pyproject,
        "README.md": readme,
        "CHANGELOG.md": changelog,
        "RELEASE_NOTES.md": release_notes,
        "CITATION.cff": citation_text,
    }.items():
        if "github.com" in text.lower():
            errors.append(f"{relpath} must not contain GitHub URLs until public URLs exist")

    pyproject_license = _match(
        r'^license\s*=\s*\{text\s*=\s*"([^"]+)"\}', pyproject, "project license"
    )
    classifiers = set(_parse_quoted_list(pyproject, "classifiers"))
    if pyproject_license != "MIT License":
        errors.append("pyproject.toml license text must remain MIT License")
    if "License :: OSI Approved :: MIT License" not in classifiers:
        errors.append("pyproject.toml must include the MIT license classifier")
    if citation.get("license") != "MIT":
        errors.append("CITATION.cff license must be MIT")
    if "Permission is hereby granted" not in license_text:
        errors.append("LICENSE must contain the MIT permission grant text")
    if "Triad National Security" not in license_text:
        errors.append("LICENSE must retain the Triad National Security copyright notice")

    return errors


def check_release_metadata(root: Path) -> list[str]:
    """Return release metadata errors for ``root`` without printing."""
    errors: list[str] = []
    for check in (
        _check_versions_and_dates,
        _check_maturity_support_and_trust,
        _check_project_metadata,
    ):
        try:
            errors.extend(check(root))
        except Exception as exc:
            errors.append(f"{check.__name__} crashed: {exc}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check_release_metadata(root)
    if errors:
        print("Release metadata validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Release metadata validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
