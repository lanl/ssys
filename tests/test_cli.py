"""Tests for CLI module."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ssys.cli import build_notebook, main, read_manifest, recast_file


class TestReadManifest:
    """Tests for manifest file reading."""

    def test_read_manifest_basic(self, tmp_path: Path):
        """Test reading basic manifest file."""
        # Create temp manifest
        manifest = tmp_path / "models.manifest"
        manifest.write_text("file1.ant\nfile2.ant\n")

        result = read_manifest(str(manifest))

        assert len(result) == 2
        # Paths should be resolved relative to manifest directory
        assert result[0].endswith("file1.ant")
        assert result[1].endswith("file2.ant")

    def test_read_manifest_with_comments(self, tmp_path: Path):
        """Test that comments are skipped."""
        manifest = tmp_path / "models.manifest"
        manifest.write_text("# This is a comment\nfile1.ant\n# Another comment\n")

        result = read_manifest(str(manifest))

        assert len(result) == 1
        assert result[0].endswith("file1.ant")

    def test_read_manifest_with_blank_lines(self, tmp_path: Path):
        """Test that blank lines are skipped."""
        manifest = tmp_path / "models.manifest"
        manifest.write_text("\n\nfile1.ant\n\n\nfile2.ant\n\n")

        result = read_manifest(str(manifest))

        assert len(result) == 2

    def test_read_manifest_absolute_paths(self, tmp_path: Path):
        """Test that absolute paths are preserved."""
        manifest = tmp_path / "models.manifest"
        abs_path = "/absolute/path/to/model.ant"
        manifest.write_text(f"{abs_path}\n")

        result = read_manifest(str(manifest))

        assert len(result) == 1
        assert result[0] == abs_path

    def test_read_manifest_empty(self, tmp_path: Path):
        """Test reading empty manifest."""
        manifest = tmp_path / "models.manifest"
        manifest.write_text("")

        result = read_manifest(str(manifest))

        assert len(result) == 0

    def test_read_manifest_only_comments(self, tmp_path: Path):
        """Test manifest with only comments."""
        manifest = tmp_path / "models.manifest"
        manifest.write_text("# Comment 1\n# Comment 2\n")

        result = read_manifest(str(manifest))

        assert len(result) == 0


class TestRecastFile:
    """Tests for recast_file function."""

    def test_recast_simple_model(self, tmp_path: Path):
        """Test recasting a simple model."""
        # Create input model
        input_ant = tmp_path / "simple.ant"
        input_ant.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)

        name, inp, out, val = recast_file(
            str(input_ant),
            str(tmp_path),
            mode="simplified",
            validate=False,
        )

        assert name == "simple"
        assert inp == str(input_ant)
        assert out.endswith("simple_recast.ant")
        assert val is None
        assert os.path.exists(out)

        # Check output file has content
        output_content = Path(out).read_text()
        assert "model simple_recast" in output_content
        assert "end" in output_content

    def test_recast_with_sim_metadata(self, tmp_path: Path):
        """Test that @SIM metadata is preserved."""
        input_ant = tmp_path / "sim.ant"
        input_ant.write_text("""
            // @SIM T_START=0 T_END=100 N_STEPS=500
            X' = -k*X
            k = 0.5
            X = 1.0
        """)

        _, _, out, _ = recast_file(
            str(input_ant),
            str(tmp_path),
            mode="simplified",
            validate=False,
        )

        output_content = Path(out).read_text()
        # Should preserve @SIM metadata
        assert "@SIM" in output_content

    def test_recast_canonical_mode(self, tmp_path: Path):
        """Test recasting in canonical mode."""
        input_ant = tmp_path / "canonical.ant"
        input_ant.write_text("""
            X' = a*X - b*X^2
            a = 1.0
            b = 0.1
            X = 0.5
        """)

        _, _, out, _ = recast_file(
            str(input_ant),
            str(tmp_path),
            mode="canonical",
            validate=False,
        )

        assert os.path.exists(out)
        output_content = Path(out).read_text()
        assert "model canonical_recast" in output_content


class TestValidationCliExit:
    """Tests for hard-fail CLI validation semantics."""

    def _write_manifested_model(self, tmp_path: Path, name: str = "model") -> tuple[Path, Path]:
        input_ant = tmp_path / f"{name}.ant"
        input_ant.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)
        manifest = tmp_path / "models.manifest"
        manifest.write_text(f"{input_ant}\n")
        return input_ant, manifest

    def test_validate_exits_nonzero_when_report_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _, manifest = self._write_manifested_model(tmp_path, "failed")
        outdir = tmp_path / "out"

        def fake_validate_recast_pair(*args, output_json=None, **kwargs):
            assert output_json is not None
            report = {
                "overall_pass": False,
                "summary": "forced validation failure",
            }
            Path(output_json).write_text(json.dumps(report))
            return SimpleNamespace(overall_pass=False, summary="forced validation failure")

        monkeypatch.setattr("ssys.validator.validate_recast_pair", fake_validate_recast_pair)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ssys-recast",
                "--manifest",
                str(manifest),
                "--outdir",
                str(outdir),
                "--parser",
                "legacy",
                "--validate",
            ],
        )

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
        report = json.loads((outdir / "failed_validation.json").read_text())
        assert report["overall_pass"] is False

    def test_validate_best_effort_flag_allows_failed_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _, manifest = self._write_manifested_model(tmp_path, "best_effort")
        outdir = tmp_path / "out"

        def fake_validate_recast_pair(*args, output_json=None, **kwargs):
            assert output_json is not None
            Path(output_json).write_text(json.dumps({
                "overall_pass": False,
                "summary": "forced validation failure",
            }))
            return SimpleNamespace(overall_pass=False, summary="forced validation failure")

        monkeypatch.setattr("ssys.validator.validate_recast_pair", fake_validate_recast_pair)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ssys-recast",
                "--manifest",
                str(manifest),
                "--outdir",
                str(outdir),
                "--parser",
                "legacy",
                "--validate",
                "--allow-validation-failures",
            ],
        )

        main()

        report = json.loads((outdir / "best_effort_validation.json").read_text())
        assert report["overall_pass"] is False
        assert (outdir / "recast_report.ipynb").exists()

    def test_validate_exits_nonzero_and_writes_report_for_invalid_recast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _, manifest = self._write_manifested_model(tmp_path, "invalid")
        outdir = tmp_path / "out"

        def invalid_antimony(*args, **kwargs):
            return "model invalid_recast()\nDNA := Z_1;\nend\n"

        monkeypatch.setattr("ssys.cli.ssystem_to_antimony", invalid_antimony)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ssys-recast",
                "--manifest",
                str(manifest),
                "--outdir",
                str(outdir),
                "--parser",
                "sbml",
                "--validate",
            ],
        )

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
        report = json.loads((outdir / "invalid_validation.json").read_text())
        assert report["overall_pass"] is False
        assert report["overall_result"] == "failed"
        assert report["tests"]["generated_output"]["result"] == "failed"


class TestBuildNotebook:
    """Tests for notebook generation."""

    def test_build_notebook_creates_file(self, tmp_path: Path):
        """Test that notebook file is created."""
        cases = [
            ("test", "/path/to/input.ant", str(tmp_path / "output.ant"), None),
        ]

        result = build_notebook(cases, str(tmp_path))

        assert result.endswith("recast_report.ipynb")
        assert os.path.exists(result)

    def test_build_notebook_valid_format(self, tmp_path: Path):
        """Test that generated notebook is valid."""
        import nbformat

        cases = [
            ("model1", "/path/to/model1.ant", str(tmp_path / "model1_recast.ant"), None),
            ("model2", "/path/to/model2.ant", str(tmp_path / "model2_recast.ant"), None),
        ]

        nb_path = build_notebook(cases, str(tmp_path))

        # Should be valid notebook
        nb = nbformat.read(nb_path, as_version=4)
        assert len(nb.cells) >= 3  # At least header + imports + one model

    def test_build_notebook_includes_model_sections(self, tmp_path: Path):
        """Test that notebook has sections for each model."""
        import nbformat

        cases = [
            ("alpha", "/path/alpha.ant", str(tmp_path / "alpha_recast.ant"), None),
            ("beta", "/path/beta.ant", str(tmp_path / "beta_recast.ant"), None),
        ]

        nb_path = build_notebook(cases, str(tmp_path))
        nb = nbformat.read(nb_path, as_version=4)

        # Check for model section headers
        md_cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
        all_md = "\n".join(md_cells)
        assert "## alpha" in all_md
        assert "## beta" in all_md

    def test_build_notebook_with_validation(self, tmp_path: Path):
        """Test notebook with validation paths."""
        cases = [
            (
                "validated",
                "/path/validated.ant",
                str(tmp_path / "validated_recast.ant"),
                str(tmp_path / "validated_validation.json"),
            ),
        ]

        nb_path = build_notebook(cases, str(tmp_path))

        assert os.path.exists(nb_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
