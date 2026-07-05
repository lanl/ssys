"""Unit tests for the local release artifact smoke helper."""

import inspect
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import local_artifact_smoke


def _write_tarball(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in members.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def test_safe_label_stabilizes_paths_and_versions():
    assert local_artifact_smoke._safe_label("3.12") == "3.12"
    assert local_artifact_smoke._safe_label("/tmp/python 3.12/bin/python") == "tmp_python_3.12_bin_python"
    assert local_artifact_smoke._safe_label("///") == "python"


def test_select_python_specs_falls_back_when_current_python_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(local_artifact_smoke.sys, "executable", "/opt/python3.14/bin/python")
    monkeypatch.setattr(local_artifact_smoke, "_python_minor_version", lambda spec, root: "3.14")
    monkeypatch.setattr(
        local_artifact_smoke,
        "_discover_supported_python_specs",
        lambda root: ["/opt/python3.12/bin/python3.12"],
    )

    def fake_validate(specs, root):
        assert specs == ["/opt/python3.12/bin/python3.12"]
        return specs

    monkeypatch.setattr(local_artifact_smoke, "_validate_supported_python_specs", fake_validate)
    args = SimpleNamespace(all_supported_pythons=False, python_specs=None)

    assert local_artifact_smoke._select_python_specs(args, tmp_path) == [
        "/opt/python3.12/bin/python3.12"
    ]


def test_validate_supported_python_specs_rejects_unsupported_explicit_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(local_artifact_smoke, "_python_minor_version", lambda spec, root: "3.14")

    with pytest.raises(SystemExit, match="supported versions"):
        local_artifact_smoke._validate_supported_python_specs(["python3.14"], tmp_path)


def test_find_one_requires_exactly_one_match(tmp_path: Path):
    artifact = tmp_path / "ssys-0.5.5-py3-none-any.whl"
    artifact.write_text("wheel")

    assert local_artifact_smoke._find_one(tmp_path, "*.whl") == artifact

    (tmp_path / "other.whl").write_text("wheel")
    with pytest.raises(SystemExit, match="expected exactly one"):
        local_artifact_smoke._find_one(tmp_path, "*.whl")


def test_prepare_evidence_dir_resolves_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    evidence_dir = local_artifact_smoke._prepare_evidence_dir(Path("release-evidence/dev"), False)

    assert evidence_dir == (tmp_path / "release-evidence/dev").resolve()
    assert evidence_dir.is_dir()


def test_safe_extract_sdist_uses_explicit_data_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sdist = tmp_path / "ssys-0.5.5.tar.gz"
    _write_tarball(
        sdist,
        {
            "ssys-0.5.5/README.md": "# ssys\n",
            "ssys-0.5.5/src/ssys/__init__.py": "__version__ = '0.5.5'\n",
        },
    )
    captured_kwargs = {}
    original_extractall = tarfile.TarFile.extractall

    def wrapped_extractall(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_extractall(self, *args, **kwargs)

    wrapped_extractall.__signature__ = inspect.signature(original_extractall)
    monkeypatch.setattr(tarfile.TarFile, "extractall", wrapped_extractall)

    extracted = local_artifact_smoke._safe_extract_sdist(sdist, tmp_path / "extract")

    assert extracted == tmp_path / "extract" / "ssys-0.5.5"
    assert (extracted / "README.md").read_text(encoding="utf-8") == "# ssys\n"
    if "filter" in inspect.signature(tarfile.TarFile.extractall).parameters:
        assert captured_kwargs["filter"] == "data"
    else:
        assert "filter" not in captured_kwargs


def test_safe_extract_sdist_rejects_path_traversal_member(tmp_path: Path):
    sdist = tmp_path / "ssys-0.5.5.tar.gz"
    _write_tarball(
        sdist,
        {
            "ssys-0.5.5/README.md": "# ssys\n",
            "ssys-0.5.5/../../escape.txt": "escape\n",
        },
    )

    with pytest.raises(SystemExit, match="refusing unsafe sdist member path"):
        local_artifact_smoke._safe_extract_sdist(sdist, tmp_path / "extract")

    assert not (tmp_path / "escape.txt").exists()


def test_verify_sdist_fixtures_counts_manifest_entries(tmp_path: Path):
    sdist_root = tmp_path / "ssys-0.5.5"
    (sdist_root / "tests").mkdir(parents=True)
    (sdist_root / "tests/test_integration.py").write_text("")
    (sdist_root / "tools").mkdir()
    (sdist_root / "tools/check_release_metadata.py").write_text("")
    (sdist_root / "tools/local_artifact_smoke.py").write_text("")
    (sdist_root / "RELEASE_NOTES.md").write_text("")

    expected_counts = {
        "test_models1": 29,
        "test_models2": 28,
        "test_models3": 40,
        "test_models4": 20,
    }
    for dirname, count in expected_counts.items():
        directory = sdist_root / dirname
        directory.mkdir()
        entries = []
        for index in range(count):
            model = directory / f"model_{index}.ant"
            model.write_text("model m()\nend\n")
            entries.append(model.name)
        (directory / "models.manifest").write_text("\n".join(entries))

    assert local_artifact_smoke._verify_sdist_fixtures(sdist_root) == 117
