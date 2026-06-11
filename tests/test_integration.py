"""
Integration tests for ssys recasting pipeline.

These tests run the full recasting pipeline on all 117 test models across
4 test directories, validating both simplified and canonical modes.

Tests are marked as 'integration' and 'slow' - skip during rapid development:
    pytest -m "not slow"           # Skip slow tests
    pytest tests/test_integration.py -v  # Run only integration tests

Total: 8 tests (4 directories × 2 modes)
Expected runtime: ~2-5 minutes depending on hardware
"""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import nbformat
import pytest

# Test model directories (relative to project root)
TEST_DIRS = ["test_models1", "test_models2", "test_models3", "test_models4"]
MODES = ["simplified", "canonical"]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.mark.integration
def test_cli_recasts_tiny_manifest_as_subprocess(tmp_path: Path):
    """Fast integration test for the real CLI, manifest resolution, and artifacts."""
    model = tmp_path / "decay.ant"
    model.write_text("""
        X' = -k*X
        k = 0.5
        X = 1.0
    """)
    manifest = tmp_path / "models.manifest"
    manifest.write_text("decay.ant\n")
    outdir = tmp_path / "out"

    cmd = [
        sys.executable,
        "-m",
        "ssys.cli",
        "--manifest",
        str(manifest),
        "--outdir",
        str(outdir),
        "--mode",
        "simplified",
        "--parser",
        "legacy",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=get_project_root(),
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Processing 1 model(s)" in result.stdout

    output = outdir / "decay_recast.ant"
    assert output.exists()
    output_text = output.read_text()
    assert "model decay_recast" in output_text
    assert "S-SYSTEM DYNAMICS" in output_text
    assert "Z_1' =" in output_text
    assert "X :=" in output_text

    notebook = outdir / "recast_report.ipynb"
    nb = nbformat.read(notebook, as_version=4)
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "## decay" in markdown
    assert "load_and_report" in code
    assert "decay_recast.ant" in code


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_dir", TEST_DIRS)
@pytest.mark.parametrize("mode", MODES)
def test_recast_all_models(test_dir: str, mode: str, tmp_path: Path):
    """
    Integration test: recast all models in a directory.

    For each combination of test directory and mode:
    1. Verify the models.manifest exists
    2. Run ssys-recaster with --validate flag
    3. Assert exit code is 0 (all models passed validation)

    Args:
        test_dir: Name of test directory (e.g., 'test_models1')
        mode: Recasting mode ('simplified' or 'canonical')
        tmp_path: pytest fixture providing temporary directory for output
    """
    if find_spec("sksundae") is None:
        pytest.skip("full manifest validation with dae_required models needs `uv sync --extra dae`")

    project_root = get_project_root()
    manifest = project_root / test_dir / "models.manifest"

    assert manifest.exists(), f"Missing manifest: {manifest}"

    # Count expected models
    model_count = len([
        line for line in manifest.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ])

    # Use temporary output directory to avoid cluttering project
    outdir = tmp_path / f"out_{test_dir}_{mode}"

    cmd = [
        sys.executable, "-m", "ssys.cli",
        "--manifest", str(manifest),
        "--outdir", str(outdir),
        "--mode", mode,
        "--parser", "sbml",
        "--validate"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    # Provide detailed failure info
    if result.returncode != 0:
        pytest.fail(
            f"Recasting failed for {test_dir} in {mode} mode\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    # Verify output directory was created
    assert outdir.exists(), f"Output directory not created: {outdir}"

    # Verify expected number of output files
    output_files = list(outdir.glob("*.ant"))
    assert len(output_files) >= model_count, (
        f"Expected at least {model_count} output files, got {len(output_files)}"
    )
