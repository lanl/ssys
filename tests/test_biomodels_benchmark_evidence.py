"""Local BioModels benchmark evidence wrapper tests."""

from __future__ import annotations

import json
from pathlib import Path

from tools import run_biomodels_benchmark


def _write_minimal_benchmark_tree(root: Path) -> None:
    (root / "data" / "sbml_downloads").mkdir(parents=True)
    (root / "data" / "sbml_candidates").mkdir(parents=True)
    (root / "results" / "recasts").mkdir(parents=True)
    (root / "results" / "failures").mkdir(parents=True)
    (root / "results" / "validation").mkdir(parents=True)
    (root / "results" / "validated").mkdir(parents=True)
    (root / "run_benchmark.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    (root / "data" / "sbml_downloads" / "MODEL1.xml").write_text("<sbml/>", encoding="utf-8")
    (root / "data" / "sbml_candidates" / "MODEL1.xml").write_text("<sbml/>", encoding="utf-8")
    (root / "results" / "recasts" / "MODEL1_simplified.ant").write_text(
        "model MODEL1_recast()\nend\n",
        encoding="utf-8",
    )
    (root / "results" / "failures" / "MODEL2_simplified.log").write_text(
        "TimeoutError",
        encoding="utf-8",
    )
    (root / "results" / "validation" / "MODEL1_simplified_numerical.json").write_text(
        json.dumps({
            "overall_pass": True,
            "overall_result": "pass",
            "validation_profile": {"name": "numerical"},
            "tests": {"numerical": {"reason": "pass"}},
        }),
        encoding="utf-8",
    )
    (root / "results" / "validated" / "MODEL1_recast.ant").write_text(
        "model MODEL1_recast()\nend\n",
        encoding="utf-8",
    )
    (root / "results" / "validated" / "manifest.csv").write_text(
        "model_id,original_type,recast_type,max_error\nMODEL1,General,GMA,0.0\n",
        encoding="utf-8",
    )
    (root / "results" / "batch_recast_results.csv").write_text(
        "model_id,mode,status,recast_success,validation_attempted,validation_pass,error\n"
        "MODEL1,simplified,success,True,True,True,\n"
        "MODEL2,simplified,timeout,False,False,False,TimeoutError\n",
        encoding="utf-8",
    )


def test_summarize_biomodels_outputs_counts_local_artifacts(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["counts"]["sbml_downloads"] == 1
    assert summary["counts"]["candidate_models"] == 1
    assert summary["counts"]["recast_artifacts"] == 1
    assert summary["counts"]["failure_logs"] == 1
    assert summary["counts"]["validation_reports"] == 1
    assert summary["counts"]["validated_manifest_rows"] == 1
    assert summary["status_counts"] == {"success": 1, "timeout": 1}
    assert summary["validation_profile_counts"] == {"numerical": 1}
    assert summary["representative_validated_subset"][0]["model_id"] == "MODEL1"


def test_main_skip_run_writes_summary_and_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    evidence_dir = tmp_path / "evidence"
    _write_minimal_benchmark_tree(benchmark_dir)

    monkeypatch.setattr(run_biomodels_benchmark, "_record_dependency_freeze", lambda _: None)

    exit_code = run_biomodels_benchmark.main([
        "--benchmark-dir",
        str(benchmark_dir),
        "--evidence-dir",
        str(evidence_dir),
        "--skip-run",
    ])

    assert exit_code == 0
    payload = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["run"]["skipped"] is True
    assert payload["summary"]["counts"]["validated_manifest_rows"] == 1
    assert (evidence_dir / "batch_recast_results.csv").exists()
    assert (evidence_dir / "representative-validation" / "MODEL1_simplified_numerical.json").exists()


def test_copy_benchmark_tree_excludes_stale_results(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "copy"
    _write_minimal_benchmark_tree(source)

    copied = run_biomodels_benchmark._copy_benchmark_tree(source, destination)

    assert (copied / "data" / "sbml_candidates" / "MODEL1.xml").exists()
    assert not (copied / "results").exists()


def test_main_artifact_mode_records_isolated_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    evidence_dir = tmp_path / "evidence"
    artifact = tmp_path / "ssys.whl"
    python_exe = tmp_path / "venv" / "bin" / "python"
    _write_minimal_benchmark_tree(benchmark_dir)
    artifact.write_text("wheel", encoding="utf-8")

    captured = {}

    def fake_prepare_artifact_benchmark(**kwargs):
        assert kwargs["artifact"] == artifact
        return benchmark_dir, python_exe, {"artifact": str(artifact), "python": str(python_exe)}

    def fake_run_benchmark(
        command,
        benchmark_dir_arg,
        evidence_dir_arg,
        timeout,
        env=None,
        log_prefix="benchmark",
    ):
        captured.setdefault("commands", []).append((log_prefix, command))
        captured["benchmark_dir"] = benchmark_dir_arg
        captured["env"] = env
        return {"returncode": 0, "timed_out": False, "duration_seconds": 0.1}

    monkeypatch.setattr(
        run_biomodels_benchmark,
        "_prepare_artifact_benchmark",
        fake_prepare_artifact_benchmark,
    )
    monkeypatch.setattr(run_biomodels_benchmark, "_run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(run_biomodels_benchmark, "_record_dependency_freeze_for_python", lambda *_: None)

    exit_code = run_biomodels_benchmark.main([
        "--benchmark-dir",
        str(benchmark_dir),
        "--evidence-dir",
        str(evidence_dir),
        "--artifact",
        str(artifact),
    ])

    assert exit_code == 0
    assert captured["benchmark_dir"] == benchmark_dir
    assert [prefix for prefix, _ in captured["commands"]] == ["benchmark", "report"]
    assert captured["env"]["PATH"].split(":")[0] == str(python_exe.parent)
    assert "PYTHONPATH" not in captured["env"]
    payload = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["artifact_environment"] == {"artifact": str(artifact), "python": str(python_exe)}


def test_threshold_failures_make_main_fail(tmp_path: Path, monkeypatch) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    evidence_dir = tmp_path / "evidence"
    _write_minimal_benchmark_tree(benchmark_dir)

    monkeypatch.setattr(run_biomodels_benchmark, "_record_dependency_freeze", lambda _: None)

    exit_code = run_biomodels_benchmark.main([
        "--benchmark-dir",
        str(benchmark_dir),
        "--evidence-dir",
        str(evidence_dir),
        "--skip-run",
        "--min-validated",
        "2",
    ])

    assert exit_code == 1
    payload = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["threshold_failures"] == [
        "validated_manifest_rows=1 is below required minimum 2"
    ]
