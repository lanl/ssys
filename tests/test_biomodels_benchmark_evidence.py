"""Local BioModels benchmark evidence wrapper tests."""

from __future__ import annotations

import csv
import json
import os
import sys
from importlib import util as importlib_util
from pathlib import Path

from jsonschema import Draft202012Validator

from ssys.validator import load_validation_report_schema
from tools import run_biomodels_benchmark


def _load_step4_validate_module():
    module_path = Path(__file__).resolve().parents[1] / "biomodels_batch" / "step4_validate.py"
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib_util.spec_from_file_location("biomodels_step4_validate_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:
            pass


def _load_step3_recast_module():
    module_path = Path(__file__).resolve().parents[1] / "biomodels_batch" / "step3_recast.py"
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib_util.spec_from_file_location("biomodels_step3_recast_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:
            pass


def _load_step6_report_module():
    module_path = Path(__file__).resolve().parents[1] / "biomodels_batch" / "step6_report.py"
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib_util.spec_from_file_location("biomodels_step6_report_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:
            pass


def _assert_validation_report_schema(data: dict) -> None:
    validator = Draft202012Validator(load_validation_report_schema())
    errors = sorted(validator.iter_errors(data), key=lambda error: error.json_path)
    assert not errors, "\n".join(f"{error.json_path}: {error.message}" for error in errors)


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


class _CandidateFrame:
    def __init__(self, model_ids: list[str]) -> None:
        self._model_ids = model_ids

    @property
    def empty(self) -> bool:
        return not self._model_ids

    def head(self, limit: int) -> _CandidateFrame:
        return _CandidateFrame(self._model_ids[:limit])

    def iterrows(self):
        for index, model_id in enumerate(self._model_ids):
            yield index, {"model_id": model_id}

    def __len__(self) -> int:
        return len(self._model_ids)


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
    assert summary["validation_model_category_counts"] == {"validation_pass": 1}
    assert summary["validation_timeout_phase_counts"] == {}
    assert summary["validation_timeout_dominant_phase_counts"] == {}
    assert summary["near_timeout"]["recast"]["count"] == 0
    assert summary["near_timeout"]["validation"]["count"] == 0
    assert summary["representative_validated_subset"][0]["model_id"] == "MODEL1"


def test_summarize_biomodels_outputs_classifies_legacy_error_reports(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "validation" / "MODEL2_simplified_numerical.json").write_text(
        json.dumps({
            "model_id": "MODEL2",
            "mode": "simplified",
            "overall_pass": False,
            "error": "Timeout after 60s (subprocess killed)",
        }),
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["validation_model_category_counts"] == {
        "validation_pass": 1,
        "validator_timeout": 1,
    }
    assert summary["validation_timeout_phase_counts"] == {"unknown": 1}
    assert summary["validation_timeout_dominant_phase_counts"] == {"unknown": 1}


def test_summarize_biomodels_outputs_counts_timeout_phases(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "validation" / "MODEL2_simplified_numerical.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "overall_pass": False,
            "overall_result": "timeout",
            "validation_profile": {"name": "custom"},
            "tests": {
                "numerical": {
                    "name": "numerical_pointwise",
                    "result": "timeout",
                    "reason": "validator_timeout",
                    "metadata": {
                        "validation_phase": "numerical",
                        "timeout_seconds": 60,
                        "phase_history": [
                            {"phase": "validator_parser", "elapsed_seconds": 0.01},
                            {"phase": "numerical", "elapsed_seconds": 4.5},
                        ],
                    },
                }
            },
        }),
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["validation_model_category_counts"] == {
        "validation_pass": 1,
        "validator_timeout": 1,
    }
    assert summary["validation_timeout_phase_counts"] == {"numerical": 1}
    assert summary["validation_timeout_dominant_phase_counts"] == {"numerical": 1}
    assert summary["near_timeout"]["validation"]["count"] == 1
    assert summary["near_timeout"]["validation"]["models"][0]["model_id"] == "MODEL2"
    assert summary["near_timeout"]["validation"]["models"][0]["budget_fraction"] == 1.0


def test_numerical_sample_wrapper_timeout_is_structured_complexity(
    tmp_path: Path,
) -> None:
    step4_validate = _load_step4_validate_module()
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)

    report = step4_validate._validation_failure_report(
        model_id="MODEL1601080000",
        mode="simplified",
        original_file="data/sbml_candidates/MODEL1601080000.xml",
        recast_file="results/recasts/MODEL1601080000_simplified.ant",
        details="Timeout after 60s (subprocess killed)",
        symbolic_only=False,
        numerical_only=True,
        use_jax=False,
        overall_result="timeout",
        reason="validator_timeout",
        timeout_seconds=60,
        phase="numerical_sample_evaluation:original_ode_evaluation:FabA",
        phase_history=[
            {"phase": "numerical_sample_evaluation", "elapsed_seconds": 7.5},
            {
                "phase": "numerical_sample_evaluation:original_ode_evaluation:FabA",
                "elapsed_seconds": 7.6,
            },
        ],
    )

    _assert_validation_report_schema(report)
    numerical = report["tests"]["numerical"]
    assert report["overall_result"] == "not_attempted"
    assert numerical["result"] == "not_attempted"
    assert numerical["reason"] == "numerical_complexity"
    diagnostic = numerical["metadata"]["diagnostics"][0]
    assert diagnostic["active_subphase"] == "original_ode_evaluation"
    assert diagnostic["active_expression_label"] == "FabA"
    assert diagnostic["limit_seconds"] == 45.0
    assert diagnostic["wrapper_timeout_seconds"] == 60

    (benchmark_dir / "results" / "validation" / "MODEL2_simplified_numerical.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["validation_model_category_counts"] == {
        "validation_pass": 1,
        "validation_blocked": 1,
    }
    assert summary["validation_timeout_phase_counts"] == {}
    assert summary["validation_timeout_dominant_phase_counts"] == {}


def test_summarize_biomodels_outputs_reports_near_timeout_margins(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "batch_recast_results.csv").write_text(
        "model_id,mode,status,recast_success,recast_time,validation_attempted,validation_pass,error\n"
        "MODEL1,simplified,success,True,54.0,True,True,\n",
        encoding="utf-8",
    )
    (benchmark_dir / "results" / "validation" / "MODEL1_simplified_numerical.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "overall_pass": True,
            "overall_result": "pass",
            "validation_profile": {"name": "custom"},
            "tests": {
                "numerical": {
                    "name": "numerical_pointwise",
                    "result": "pass",
                    "reason": None,
                    "metadata": {
                        "timeout_seconds": 60,
                        "phase_history": [
                            {"phase": "input_load", "elapsed_seconds": 0.0},
                            {"phase": "validator_parser", "elapsed_seconds": 5.0},
                            {"phase": "numerical", "elapsed_seconds": 45.0},
                            {"phase": "completed", "elapsed_seconds": 55.0},
                        ],
                    },
                }
            },
        }),
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(
        benchmark_dir,
        validation_timeout_seconds=60,
        recast_timeout_seconds=60,
        near_timeout_fraction=0.85,
    )

    assert summary["near_timeout"]["threshold_fraction"] == 0.85
    assert summary["near_timeout"]["recast"]["count"] == 1
    assert summary["near_timeout"]["recast"]["status_counts"] == {"success": 1}
    assert summary["near_timeout"]["recast"]["models"][0] == {
        "model_id": "MODEL1",
        "elapsed_seconds": 54.0,
        "timeout_seconds": 60,
        "budget_fraction": 0.9,
        "mode": "simplified",
        "status": "success",
        "recast_success": True,
        "dominant_phase": "unknown",
        "dominant_phase_seconds": None,
        "dominant_phase_attribution": "unknown",
        "last_phase": "unknown",
        "error": None,
    }
    assert summary["near_timeout"]["validation"]["count"] == 1
    validation_entry = summary["near_timeout"]["validation"]["models"][0]
    assert validation_entry["model_id"] == "MODEL1"
    assert validation_entry["budget_fraction"] == 0.916667
    assert validation_entry["last_phase"] == "completed"
    assert validation_entry["dominant_phase"] == "validator_parser"
    assert validation_entry["dominant_phase_seconds"] == 40.0


def test_summarize_biomodels_outputs_reports_measured_successful_recast_phase(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "batch_recast_results.csv").write_text(
        "model_id,mode,status,recast_success,recast_time,recast_phase_history,"
        "recast_phase_seconds,recast_dominant_phase,recast_dominant_phase_seconds,"
        "validation_attempted,validation_pass,error\n"
        'MODEL1,simplified,success,True,54.0,,"{""parse_sbml"":0.25,'
        '""recast_to_ssystem"":52.5,""ssystem_to_antimony"":1.25}",,,True,True,\n',
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(
        benchmark_dir,
        recast_timeout_seconds=60,
        near_timeout_fraction=0.85,
    )

    assert summary["near_timeout"]["recast"]["status_counts"] == {"success": 1}
    entry = summary["near_timeout"]["recast"]["models"][0]
    assert entry["model_id"] == "MODEL1"
    assert entry["dominant_phase"] == "recast_to_ssystem"
    assert entry["dominant_phase_seconds"] == 52.5
    assert entry["dominant_phase_attribution"] == "measured"
    assert entry["last_phase"] == "unknown"


def test_summarize_biomodels_outputs_infers_open_recast_phase_for_timeout(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    history = [
        {"event": "phase_start", "phase": "parse_sbml", "elapsed_seconds": 0.0},
        {
            "event": "phase_end",
            "phase": "parse_sbml",
            "elapsed_seconds": 0.02,
            "phase_seconds": 0.02,
        },
        {
            "event": "phase_start",
            "phase": "recast_to_ssystem",
            "elapsed_seconds": 0.02,
        },
    ]
    rows_path = benchmark_dir / "results" / "batch_recast_results.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "mode",
                "status",
                "recast_success",
                "recast_time",
                "recast_phase_history",
                "recast_phase_seconds",
                "recast_dominant_phase",
                "recast_dominant_phase_seconds",
                "validation_attempted",
                "validation_pass",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "model_id": "MODEL_TIMEOUT",
            "mode": "simplified",
            "status": "timeout",
            "recast_success": "False",
            "recast_time": "60.1",
            "recast_phase_history": json.dumps(history, separators=(",", ":")),
            "recast_phase_seconds": json.dumps({"parse_sbml": 0.02}, separators=(",", ":")),
            "recast_dominant_phase": "parse_sbml",
            "recast_dominant_phase_seconds": "0.02",
            "validation_attempted": "False",
            "validation_pass": "False",
            "error": "TimeoutError",
        })

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(
        benchmark_dir,
        recast_timeout_seconds=60,
        near_timeout_fraction=0.85,
    )

    assert summary["near_timeout"]["recast"]["status_counts"] == {"timeout": 1}
    entry = summary["near_timeout"]["recast"]["models"][0]
    assert entry["model_id"] == "MODEL_TIMEOUT"
    assert entry["dominant_phase"] == "recast_to_ssystem"
    assert entry["dominant_phase_seconds"] == 60.08
    assert entry["dominant_phase_attribution"] == "inferred_open_interval"
    assert entry["last_phase"] == "recast_to_ssystem"


def test_summarize_biomodels_outputs_reports_recast_retry_policy_counts(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    rows_path = benchmark_dir / "results" / "batch_recast_results.csv"
    fieldnames = [
        "model_id",
        "mode",
        "status",
        "recast_success",
        "recast_time",
        "recast_attempt_role",
        "recast_attempt_count",
        "recast_base_timeout_seconds",
        "recast_retry_timeout_seconds",
        "recast_final_attempt_timeout_seconds",
        "recast_retry_policy",
        "recast_recovered_by_retry",
        "validation_attempted",
        "validation_pass",
        "error",
    ]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                "model_id": "MODEL_BASE",
                "mode": "simplified",
                "status": "success",
                "recast_success": "True",
                "recast_time": "14.0",
                "recast_attempt_role": "base",
                "recast_attempt_count": "1",
                "recast_base_timeout_seconds": "15",
                "recast_retry_timeout_seconds": "60",
                "recast_final_attempt_timeout_seconds": "15",
                "recast_retry_policy": "quick_then_retry_timeouts",
                "recast_recovered_by_retry": "False",
                "validation_attempted": "False",
                "validation_pass": "False",
                "error": "",
            },
            {
                "model_id": "MODEL_RETRY",
                "mode": "simplified",
                "status": "success",
                "recast_success": "True",
                "recast_time": "58.0",
                "recast_attempt_role": "retry",
                "recast_attempt_count": "2",
                "recast_base_timeout_seconds": "15",
                "recast_retry_timeout_seconds": "60",
                "recast_final_attempt_timeout_seconds": "60",
                "recast_retry_policy": "quick_then_retry_timeouts",
                "recast_recovered_by_retry": "True",
                "validation_attempted": "False",
                "validation_pass": "False",
                "error": "",
            },
            {
                "model_id": "MODEL_TIMEOUT",
                "mode": "simplified",
                "status": "timeout",
                "recast_success": "False",
                "recast_time": "60.1",
                "recast_attempt_role": "retry",
                "recast_attempt_count": "2",
                "recast_base_timeout_seconds": "15",
                "recast_retry_timeout_seconds": "60",
                "recast_final_attempt_timeout_seconds": "60",
                "recast_retry_policy": "quick_then_retry_timeouts",
                "recast_recovered_by_retry": "False",
                "validation_attempted": "False",
                "validation_pass": "False",
                "error": "TimeoutError",
            },
        ])

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(
        benchmark_dir,
        recast_timeout_seconds=60,
        near_timeout_fraction=0.85,
    )

    assert summary["recast_policy"] == {
        "policy_counts": {"quick_then_retry_timeouts": 3},
        "attempt_role_counts": {"base": 1, "retry": 2},
        "first_attempt_successes": 1,
        "retry_recovered_successes": 1,
        "unrecovered_timeouts": 1,
        "rows_with_policy_metadata": 3,
    }
    assert summary["near_timeout"]["recast"]["count"] == 3
    assert summary["near_timeout"]["recast"]["attempt_role_counts"] == {
        "base": 1,
        "retry": 2,
    }
    retry_entry = next(
        item
        for item in summary["near_timeout"]["recast"]["models"]
        if item["model_id"] == "MODEL_RETRY"
    )
    assert retry_entry["timeout_seconds"] == 60
    assert retry_entry["recast_attempt_role"] == "retry"
    assert retry_entry["recast_attempt_count"] == 2
    assert retry_entry["recast_base_timeout_seconds"] == 15
    assert retry_entry["recast_retry_timeout_seconds"] == 60
    assert retry_entry["recast_final_attempt_timeout_seconds"] == 60
    assert retry_entry["recast_retry_policy"] == "quick_then_retry_timeouts"
    assert retry_entry["recast_recovered_by_retry"] is True
    base_entry = next(
        item
        for item in summary["near_timeout"]["recast"]["models"]
        if item["model_id"] == "MODEL_BASE"
    )
    assert base_entry["timeout_seconds"] == 15
    assert base_entry["budget_fraction"] == 0.933333


def test_summarize_biomodels_outputs_classifies_auxiliary_identity_failures(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "validation" / "MODEL2_simplified_numerical.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "overall_pass": False,
            "overall_result": "failed",
            "validation_profile": {"name": "custom"},
            "tests": {
                "generated_output": {"result": "pass", "reason": None},
                "parser": {"result": "pass", "reason": None},
                "mapping": {"result": "pass", "reason": None},
                "numerical": {"result": "pass", "reason": None},
                "auxiliaries": [
                    {
                        "name": "ode_auxiliary_identity:Y_1",
                        "result": "failed",
                        "reason": "failed",
                        "details": "Y_1 ODE identity residual: X - 1",
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["validation_model_category_counts"] == {
        "validation_pass": 1,
        "auxiliary_identity_failed": 1,
    }


def test_summarize_biomodels_outputs_classifies_unsupported_parser_features(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "validation" / "MODEL2_simplified_numerical.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "overall_pass": False,
            "overall_result": "unsupported",
            "validation_profile": {"name": "numerical"},
            "tests": {
                "generated_output": {"result": "pass", "reason": None},
                "parser": {
                    "result": "unsupported",
                    "reason": "unsupported_feature",
                    "metadata": {"unsupported_features": ["gt", "piecewise"]},
                },
                "mapping": {
                    "result": "not_attempted",
                    "reason": "unsupported_feature",
                },
                "numerical": {
                    "result": "not_attempted",
                    "reason": "unsupported_feature",
                },
            },
        }),
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["validation_model_category_counts"] == {
        "validation_pass": 1,
        "unsupported_validation_feature": 1,
    }


def test_summarize_biomodels_outputs_reports_recast_complexity_failures(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    failures_dir = benchmark_dir / "results" / "failures"
    error = (
        "RecastComplexityError: recast_complexity: "
        "stage=direct_ssystem_recast; operation=sympy_expand; "
        "expression_label=Y_7; operation_count=227; max_ops=1500; "
        "free_symbol_count=65; max_free_symbol_count=64; "
        "expression_preview=Y_7 + A*B"
    )
    (failures_dir / "MODEL_COMPLEX_simplified.log").write_text(
        "\n".join(
            [
                "Model: MODEL_COMPLEX",
                "Mode: simplified",
                "Category: RECAST_COMPLEXITY",
                f"Error: {error}",
            ]
        ),
        encoding="utf-8",
    )
    (failures_dir / "MODEL_COMPLEX_simplified_recast_metadata.json").write_text(
        json.dumps({
            "recast_time": 0.853957,
            "recast_attempt_role": "base",
            "recast_attempt_count": 1,
            "recast_base_timeout_seconds": 15,
            "recast_retry_timeout_seconds": 60,
            "recast_final_attempt_timeout_seconds": 15,
            "recast_retry_policy": "quick_then_retry_timeouts",
            "recast_recovered_by_retry": False,
            "recast_last_phase": "recast_to_ssystem",
            "recast_dominant_phase": "parse_sbml",
            "recast_dominant_phase_seconds": 0.031459,
            "recast_dominant_phase_attribution": "measured",
        }),
        encoding="utf-8",
    )

    (benchmark_dir / "results" / "batch_recast_results.csv").write_text(
        "model_id,mode,status,recast_success,validation_attempted,validation_pass,error\n"
        "MODEL1,simplified,success,True,True,True,\n"
        "MODEL_COMPLEX,simplified,error,False,False,False,RecastComplexityError\n",
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["recast_failure_category_counts"]["recast_complexity"] == 1
    entry = next(
        item
        for item in summary["recast_complexity_models"]
        if item["model_id"] == "MODEL_COMPLEX"
    )
    assert entry["stage"] == "direct_ssystem_recast"
    assert entry["operation"] == "sympy_expand"
    assert entry["expression_label"] == "Y_7"
    assert entry["operation_count"] == 227
    assert entry["max_ops"] == 1500
    assert entry["free_symbol_count"] == 65
    assert entry["max_free_symbol_count"] == 64
    assert entry["elapsed_seconds"] == 0.853957
    assert entry["expression_preview"] == "Y_7 + A*B"
    assert entry["recast_attempt_role"] == "base"
    assert entry["recast_final_attempt_timeout_seconds"] == 15


def test_summarize_biomodels_outputs_reclassifies_sbml_parsing_other_logs(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    failures_dir = benchmark_dir / "results" / "failures"
    (failures_dir / "MODEL_PARSE_simplified.log").write_text(
        "\n".join(
            [
                "Model: MODEL_PARSE",
                "Mode: simplified",
                "Category: OTHER",
                "Error: ValueError: SBML parsing errors in MODEL_PARSE.xml: "
                "The contents of the <notes> element must be explicitly placed "
                "in the XHTML XML namespace.",
            ]
        ),
        encoding="utf-8",
    )
    (benchmark_dir / "results" / "batch_recast_results.csv").write_text(
        "model_id,mode,status,recast_success,validation_attempted,validation_pass,error\n"
        "MODEL1,simplified,success,True,True,True,\n"
        "MODEL_PARSE,simplified,error,False,False,False,ValueError: SBML parsing errors\n",
        encoding="utf-8",
    )

    summary = run_biomodels_benchmark.summarize_biomodels_outputs(benchmark_dir)

    assert summary["recast_failure_category_counts"]["parse_error"] == 1
    assert "other" not in summary["recast_failure_category_counts"]


def test_step6_rebuild_preserves_existing_recast_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step6_report = _load_step6_report_module()
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    (benchmark_dir / "results" / "batch_recast_results.csv").write_text(
        "model_id,mode,status,recast_success,recast_time,recast_last_phase,"
        "recast_dominant_phase_attribution,recast_attempt_role,"
        "recast_attempt_count,recast_base_timeout_seconds,recast_retry_timeout_seconds,"
        "recast_final_attempt_timeout_seconds,recast_retry_policy,"
        "recast_recovered_by_retry,validation_attempted,validation_pass,error\n"
        "MODEL1,simplified,success,True,54.2,recast_to_ssystem,measured,retry,2,15,60,60,"
        "quick_then_retry_timeouts,True,True,True,\n"
        "MODEL2,simplified,timeout,False,60.1,recast_to_ssystem,"
        "inferred_open_interval,retry,2,15,60,60,"
        "quick_then_retry_timeouts,False,False,False,TimeoutError\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        step6_report.config,
        "SBML_CANDIDATES_DIR",
        str(benchmark_dir / "data" / "sbml_candidates"),
    )
    monkeypatch.setattr(
        step6_report.config,
        "RESULTS_DIR",
        str(benchmark_dir / "results"),
    )
    monkeypatch.setattr(
        step6_report.config,
        "RECASTS_DIR",
        str(benchmark_dir / "results" / "recasts"),
    )
    monkeypatch.setattr(
        step6_report.config,
        "FAILURES_DIR",
        str(benchmark_dir / "results" / "failures"),
    )
    monkeypatch.setattr(
        step6_report.config,
        "VALIDATION_DIR",
        str(benchmark_dir / "results" / "validation"),
    )

    step6_report.rebuild_results_csv("simplified")

    rows = {
        row["model_id"]: row
        for row in run_biomodels_benchmark._read_csv_rows(
            benchmark_dir / "results" / "batch_recast_results.csv"
        )
    }
    assert rows["MODEL1"]["recast_time"] == "54.2"
    assert "recast_phase_seconds" in rows["MODEL1"]
    assert rows["MODEL1"]["recast_last_phase"] == "recast_to_ssystem"
    assert rows["MODEL1"]["recast_dominant_phase_attribution"] == "measured"
    assert rows["MODEL1"]["recast_attempt_role"] == "retry"
    assert rows["MODEL1"]["recast_attempt_count"] == "2"
    assert rows["MODEL1"]["recast_base_timeout_seconds"] == "15"
    assert rows["MODEL1"]["recast_retry_timeout_seconds"] == "60"
    assert rows["MODEL1"]["recast_final_attempt_timeout_seconds"] == "60"
    assert rows["MODEL1"]["recast_retry_policy"] == "quick_then_retry_timeouts"
    assert rows["MODEL1"]["recast_recovered_by_retry"] == "True"


def test_step3_retry_merge_records_policy_metadata_and_preserves_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step3_recast = _load_step3_recast_module()
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    results_csv = benchmark_dir / "results" / "batch_recast_results.csv"
    results_csv.write_text(
        "model_id,mode,status,recast_success,recast_time,validation_attempted,"
        "validation_pass,error\n"
        "MODEL1,simplified,success,True,1.0,False,False,\n"
        "MODEL2,simplified,timeout,False,15.0,False,False,TimeoutError\n",
        encoding="utf-8",
    )
    (benchmark_dir / "results" / "failures" / "MODEL2_simplified.log").write_text(
        "Model: MODEL2\nCategory: TIMEOUT\nError: Timeout\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(step3_recast.utils, "setup_logging", lambda *_: None)
    monkeypatch.setattr(
        step3_recast,
        "load_candidates",
        lambda _filter: _CandidateFrame(["MODEL2"]),
    )

    def fake_process_models(model_ids, **kwargs):
        assert model_ids == ["MODEL2"]
        assert kwargs["attempt_role"] == "retry"
        assert kwargs["base_timeout"] == 15
        assert kwargs["retry_timeout"] == 60
        assert kwargs["recast_policy"] == "quick_then_retry_timeouts"
        return [{
            "model_id": "MODEL2",
            "mode": "simplified",
            "recast_success": True,
            "recast_time": 58.0,
            "recast_phase_history": "",
            "recast_phase_seconds": "",
            "recast_dominant_phase": "",
            "recast_dominant_phase_seconds": "",
            **step3_recast.recast_policy_metadata(
                attempt_role="retry",
                timeout=60,
                base_timeout=15,
                retry_timeout=60,
                recast_policy="quick_then_retry_timeouts",
                recast_success=True,
            ),
            "validation_attempted": False,
            "validation_pass": False,
            "error": None,
        }]

    monkeypatch.setattr(step3_recast, "process_models", fake_process_models)
    monkeypatch.setattr(step3_recast.config, "RESULTS_DIR", str(benchmark_dir / "results"))
    monkeypatch.setattr(step3_recast.config, "RECASTS_DIR", str(benchmark_dir / "results" / "recasts"))
    monkeypatch.setattr(step3_recast.config, "FAILURES_DIR", str(benchmark_dir / "results" / "failures"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "step3_recast.py",
            "--mode",
            "simplified",
            "--timeout",
            "60",
            "--retry-timeouts",
            "--no-validate",
            "--attempt-role",
            "retry",
            "--base-timeout",
            "15",
            "--retry-timeout",
            "60",
            "--recast-policy",
            "quick_then_retry_timeouts",
        ],
    )

    step3_recast.main()

    rows = {
        row["model_id"]: row
        for row in run_biomodels_benchmark._read_csv_rows(results_csv)
    }
    assert rows["MODEL1"]["recast_time"] == "1.0"
    assert rows["MODEL2"]["recast_success"] == "True"
    assert rows["MODEL2"]["recast_attempt_role"] == "retry"
    assert rows["MODEL2"]["recast_attempt_count"] == "2"
    assert rows["MODEL2"]["recast_base_timeout_seconds"] == "15"
    assert rows["MODEL2"]["recast_retry_timeout_seconds"] == "60"
    assert rows["MODEL2"]["recast_final_attempt_timeout_seconds"] == "60"
    assert rows["MODEL2"]["recast_retry_policy"] == "quick_then_retry_timeouts"
    assert rows["MODEL2"]["recast_recovered_by_retry"] == "True"


def test_step3_empty_retry_pass_preserves_existing_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step3_recast = _load_step3_recast_module()
    benchmark_dir = tmp_path / "biomodels_batch"
    _write_minimal_benchmark_tree(benchmark_dir)
    results_csv = benchmark_dir / "results" / "batch_recast_results.csv"
    original = results_csv.read_text(encoding="utf-8")

    monkeypatch.setattr(step3_recast.utils, "setup_logging", lambda *_: None)
    monkeypatch.setattr(
        step3_recast,
        "load_candidates",
        lambda _filter: _CandidateFrame(["MODEL1"]),
    )
    def fake_empty_process_models(model_ids, **_kwargs):
        assert model_ids == []
        return []

    monkeypatch.setattr(step3_recast, "process_models", fake_empty_process_models)
    monkeypatch.setattr(step3_recast.config, "RESULTS_DIR", str(benchmark_dir / "results"))
    monkeypatch.setattr(step3_recast.config, "RECASTS_DIR", str(benchmark_dir / "results" / "recasts"))
    monkeypatch.setattr(step3_recast.config, "FAILURES_DIR", str(benchmark_dir / "results" / "failures"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "step3_recast.py",
            "--mode",
            "simplified",
            "--timeout",
            "60",
            "--retry-timeouts",
            "--no-validate",
        ],
    )

    step3_recast.main()

    assert results_csv.read_text(encoding="utf-8") == original


def test_step3_process_model_preserves_open_phase_on_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step3_recast = _load_step3_recast_module()
    failures_dir = tmp_path / "failures"
    monkeypatch.setattr(step3_recast.config, "FAILURES_DIR", str(failures_dir))

    def timeout_after_phase_start(model_id, mode, phase_recorder=None):
        assert model_id == "MODEL_TIMEOUT"
        assert mode == "simplified"
        if phase_recorder is not None:
            phase_recorder.update({
                "phase_history": [
                    {
                        "event": "phase_start",
                        "phase": "recast_to_ssystem",
                        "elapsed_seconds": 0.0,
                    }
                ],
                "phase_seconds": {},
                "last_phase": "recast_to_ssystem",
            })
        raise step3_recast.utils.TimeoutError("Operation timed out after 1s")

    monkeypatch.setattr(step3_recast, "attempt_recast", timeout_after_phase_start)

    result = step3_recast.process_model(
        "MODEL_TIMEOUT",
        "simplified",
        validate=False,
        timeout=1,
        attempt_role="retry",
        base_timeout=15,
        retry_timeout=60,
        recast_policy="quick_then_retry_timeouts",
    )

    assert result["recast_success"] is False
    assert result["error"].startswith("Timeout:")
    assert result["recast_last_phase"] == "recast_to_ssystem"
    assert result["recast_dominant_phase"] == "recast_to_ssystem"
    assert result["recast_dominant_phase_attribution"] == "inferred_open_interval"
    assert result["recast_dominant_phase_seconds"] > 0
    assert json.loads(result["recast_phase_history"]) == [
        {
            "event": "phase_start",
            "phase": "recast_to_ssystem",
            "elapsed_seconds": 0.0,
        }
    ]

    sidecar = failures_dir / "MODEL_TIMEOUT_simplified_recast_metadata.json"
    assert sidecar.exists()
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_data["recast_last_phase"] == "recast_to_ssystem"
    assert sidecar_data["recast_dominant_phase"] == "recast_to_ssystem"
    assert sidecar_data["recast_dominant_phase_attribution"] == "inferred_open_interval"
    assert sidecar_data["recast_final_attempt_timeout_seconds"] == 1
    assert sidecar_data["recast_retry_policy"] == "quick_then_retry_timeouts"


def test_step3_process_model_failure_evicts_stale_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A model that now fails closed must not leave a prior run's successful
    recast/validation artifacts on disk.

    The validate/collect/report stages count files on disk, so a lingering
    ``.ant`` (and its validation report) from an earlier run would be
    re-validated and re-counted as a pass -- inflating the totals and hiding
    the success->failure regression from the benchmark. ``process_model`` must
    evict the stale outputs on the failure path.
    """
    step3_recast = _load_step3_recast_module()
    recasts_dir = tmp_path / "recasts"
    validation_dir = tmp_path / "validation"
    failures_dir = tmp_path / "failures"
    for directory in (recasts_dir, validation_dir, failures_dir):
        directory.mkdir()
    monkeypatch.setattr(step3_recast.config, "RECASTS_DIR", str(recasts_dir))
    monkeypatch.setattr(step3_recast.config, "VALIDATION_DIR", str(validation_dir))
    monkeypatch.setattr(step3_recast.config, "FAILURES_DIR", str(failures_dir))

    # Seed artifacts from an earlier run where the model recast successfully.
    stale_recast = recasts_dir / "MODEL_NEG_simplified.ant"
    stale_validation = validation_dir / "MODEL_NEG_simplified_numerical.json"
    stale_recast.write_text("model MODEL_NEG_recast()\nend\n", encoding="utf-8")
    stale_validation.write_text(json.dumps({"overall_pass": True}), encoding="utf-8")

    # Current code fails closed on this model (e.g. a negative initial value).
    def fail_closed(model_id, mode, phase_recorder=None):
        return (
            False,
            None,
            "NegativeInitialConditionError: S-system recasting requires positive "
            "initial states, but V starts at -60",
        )

    monkeypatch.setattr(step3_recast, "attempt_recast", fail_closed)

    result = step3_recast.process_model(
        "MODEL_NEG",
        "simplified",
        validate=False,
        timeout=5,
    )

    assert result["recast_success"] is False
    assert "NegativeInitialConditionError" in result["error"]
    # The stale success is gone, so downstream stages cannot re-count it.
    assert not stale_recast.exists()
    assert not stale_validation.exists()
    # The failure itself is still recorded.
    assert (failures_dir / "MODEL_NEG_simplified.log").exists()


def test_step3_categorizes_unsupported_generated_derivative() -> None:
    step3_recast = _load_step3_recast_module()

    category, explanation = step3_recast.categorize_error(
        "UnsupportedCompositeDerivativeError: unsupported_generated_output: "
        "unsupported composite derivative for function(s) floor; "
        "derivative_expr=Subs(Derivative(floor(_xi_1), _xi_1), _xi_1, T + phase)"
    )

    assert category == "UNSUPPORTED_CONSTRUCT"
    assert "fails closed" in explanation


def test_step3_categorizes_sbml_parsing_errors() -> None:
    step3_recast = _load_step3_recast_module()

    category, explanation = step3_recast.categorize_error(
        "ValueError: SBML parsing errors in MODEL1012110001.xml: "
        "The contents of the <notes> element must be explicitly placed in the "
        "XHTML XML namespace."
    )

    assert category == "PARSE_ERROR"
    assert "Failed to parse" in explanation


def test_step3_categorizes_direct_recast_complexity() -> None:
    step3_recast = _load_step3_recast_module()

    category, explanation = step3_recast.categorize_error(
        "RecastComplexityError: recast_complexity: "
        "stage=direct_ssystem_recast; operation=sympy_expand; "
        "expression_label=Y_7; operation_count=227; max_ops=1500; "
        "free_symbol_count=65; max_free_symbol_count=64; "
        "expression_preview=Y_7 + A*B"
    )

    assert category == "RECAST_COMPLEXITY"
    assert "failed closed" in explanation


def test_step3_attempt_recast_propagates_timeout_for_safe_execute(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step3_recast = _load_step3_recast_module()
    candidates_dir = tmp_path / "sbml_candidates"
    candidates_dir.mkdir()
    (candidates_dir / "MODEL_TIMEOUT.xml").write_text("<sbml/>", encoding="utf-8")
    monkeypatch.setattr(step3_recast.config, "SBML_CANDIDATES_DIR", str(candidates_dir))
    monkeypatch.setattr(step3_recast.ssys, "parse_sbml", lambda _path: object())

    def timeout_recast(_sym, *, mode):
        assert mode == "simplified"
        raise step3_recast.utils.TimeoutError("Operation timed out after 1s")

    monkeypatch.setattr(step3_recast.ssys, "recast_to_ssystem", timeout_recast)
    phase_recorder = {}

    try:
        step3_recast.attempt_recast(
            "MODEL_TIMEOUT",
            "simplified",
            phase_recorder=phase_recorder,
        )
    except step3_recast.utils.TimeoutError:
        pass
    else:
        raise AssertionError("attempt_recast swallowed the timeout exception")

    assert phase_recorder["last_phase"] == "recast_to_ssystem"
    assert phase_recorder["phase_history"][-1]["event"] == "phase_start"
    assert phase_recorder["phase_history"][-1]["phase"] == "recast_to_ssystem"


def test_step4_subprocess_timeout_returns_schema_shaped_report(monkeypatch) -> None:
    step4_validate = _load_step4_validate_module()

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

        def get_nowait(self):
            if not self.items:
                raise step4_validate.queue.Empty
            return self.items.pop(0)

    class AlwaysAliveProcess:
        def __init__(self, *args, **kwargs):
            self.terminated = False
            phase_queue = kwargs["args"][1]
            phase_queue.put({"phase": "numerical", "elapsed_seconds": 0.25})

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    monkeypatch.setattr(step4_validate.mp, "Queue", FakeQueue)
    monkeypatch.setattr(step4_validate.mp, "Process", AlwaysAliveProcess)

    report = step4_validate._subprocess_validate((
        "MODEL2",
        "MODEL2.xml",
        "MODEL2_simplified.ant",
        "simplified",
        False,
        True,
        False,
        1,
    ))

    assert report["schema_version"] == "1.0"
    assert report["overall_result"] == "timeout"
    assert report["tests"]["numerical"]["reason"] == "validator_timeout"
    assert report["tests"]["numerical"]["metadata"]["timeout_seconds"] == 1
    assert report["tests"]["numerical"]["metadata"]["validation_phase"] == "numerical"
    assert report["tests"]["numerical"]["metadata"]["phase_history"] == [
        {"phase": "numerical", "elapsed_seconds": 0.25}
    ]
    _assert_validation_report_schema(report)


def test_step4_subprocess_returns_queued_result_before_timeout(monkeypatch) -> None:
    step4_validate = _load_step4_validate_module()
    success_report = {
        "schema_version": "1.0",
        "overall_pass": True,
        "overall_result": "pass",
        "tests": {"numerical": {"result": "pass", "reason": None}},
    }
    processes = []

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

        def get_nowait(self):
            if not self.items:
                raise step4_validate.queue.Empty
            return self.items.pop(0)

    class LingeringProcess:
        def __init__(self, *args, **kwargs):
            self.terminated = False
            result_queue, phase_queue = kwargs["args"][:2]
            result_queue.put(("success", success_report))
            phase_queue.put({"phase": "completed", "elapsed_seconds": 0.5})
            processes.append(self)

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    monkeypatch.setattr(step4_validate.mp, "Queue", FakeQueue)
    monkeypatch.setattr(step4_validate.mp, "Process", LingeringProcess)

    report = step4_validate._subprocess_validate((
        "MODEL2",
        "MODEL2.xml",
        "MODEL2_simplified.ant",
        "simplified",
        False,
        True,
        False,
        1,
    ))

    assert report["overall_result"] == "pass"
    metadata = report["tests"]["numerical"]["metadata"]
    assert metadata["timeout_seconds"] == 1
    assert metadata["validation_phase"] == "completed"
    assert metadata["phase_history"] == [
        {"phase": "completed", "elapsed_seconds": 0.5}
    ]
    assert processes[0].terminated is True


def test_step4_timing_metadata_attaches_to_auxiliary_list_entries() -> None:
    step4_validate = _load_step4_validate_module()
    report = {
        "tests": {
            "numerical": {"metadata": {}},
            "auxiliaries": [
                {
                    "name": "ode_auxiliary_identity:Y_1",
                    "result": "inconclusive",
                    "reason": "auxiliary_complexity",
                    "metadata": {"auxiliary": "Y_1"},
                }
            ],
        }
    }

    timed = step4_validate._attach_validation_timing_metadata(
        report,
        phase="auxiliaries_simplify:ode_auxiliary_identity:Y_1:candidate_0",
        phase_history=[
            {"phase": "auxiliaries", "elapsed_seconds": 1.0},
            {
                "phase": "auxiliaries_simplify:ode_auxiliary_identity:Y_1:candidate_0",
                "elapsed_seconds": 1.5,
            },
        ],
        timeout_seconds=60,
    )

    metadata = timed["tests"]["auxiliaries"][0]["metadata"]
    assert metadata["auxiliary"] == "Y_1"
    assert metadata["timeout_seconds"] == 60
    assert metadata["validation_phase"] == (
        "auxiliaries_simplify:ode_auxiliary_identity:Y_1:candidate_0"
    )
    assert metadata["phase_history"][-1]["elapsed_seconds"] == 1.5


def test_step4_subprocess_communication_failure_returns_schema_shaped_report(
    monkeypatch,
) -> None:
    step4_validate = _load_step4_validate_module()

    class FinishedProcess:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(step4_validate.mp, "Process", FinishedProcess)

    report = step4_validate._subprocess_validate((
        "MODEL3",
        "MODEL3.xml",
        "MODEL3_simplified.ant",
        "simplified",
        False,
        True,
        False,
        1,
    ))

    assert report["schema_version"] == "1.0"
    assert report["overall_result"] == "failed"
    assert report["tests"]["numerical"]["reason"] == "validator_subprocess_failed"
    _assert_validation_report_schema(report)


def test_main_skip_run_writes_summary_and_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    evidence_dir = tmp_path / "evidence"
    _write_minimal_benchmark_tree(benchmark_dir)

    monkeypatch.delenv("TIMEOUT_VALIDATION", raising=False)
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
    assert payload["benchmark_parameters"] == {
        "near_timeout_fraction": 0.85,
        "recast_quick_timeout_seconds": 15,
        "recast_quick_timeout_source": "default",
        "recast_retry_policy": "quick_then_retry_timeouts",
        "recast_timeout_seconds": 60,
        "recast_timeout_source": "default",
        "validation_timeout_seconds": 60,
        "validation_timeout_source": "default",
        "whole_run_timeout_seconds": None,
    }
    assert payload["summary"]["counts"]["validated_manifest_rows"] == 1
    assert (evidence_dir / "batch_recast_results.csv").exists()
    assert (evidence_dir / "representative-validation" / "MODEL1_simplified_numerical.json").exists()


def test_main_passes_configured_validation_timeout_and_records_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    evidence_dir = tmp_path / "evidence"
    _write_minimal_benchmark_tree(benchmark_dir)

    captured_runs = []

    def fake_run_benchmark(
        command,
        benchmark_dir_arg,
        evidence_dir_arg,
        timeout,
        env=None,
        log_prefix="benchmark",
    ):
        captured_runs.append({
            "log_prefix": log_prefix,
            "timeout": timeout,
            "env": dict(env or {}),
        })
        return {"returncode": 0, "timed_out": False, "duration_seconds": 0.1}

    monkeypatch.setattr(run_biomodels_benchmark, "_run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(run_biomodels_benchmark, "_record_dependency_freeze", lambda _: None)

    exit_code = run_biomodels_benchmark.main([
        "--benchmark-dir",
        str(benchmark_dir),
        "--evidence-dir",
        str(evidence_dir),
        "--recast-quick-timeout",
        "20",
        "--recast-timeout",
        "75",
        "--validation-timeout",
        "90",
        "--timeout",
        "123",
    ])

    assert exit_code == 0
    assert [run["log_prefix"] for run in captured_runs] == ["benchmark", "report"]
    assert [run["timeout"] for run in captured_runs] == [123, 123]
    assert [run["env"]["TIMEOUT_QUICK"] for run in captured_runs] == ["20", "20"]
    assert [run["env"]["TIMEOUT_RECAST"] for run in captured_runs] == ["75", "75"]
    assert [run["env"]["TIMEOUT_VALIDATION"] for run in captured_runs] == ["90", "90"]
    payload = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["benchmark_parameters"] == {
        "near_timeout_fraction": 0.85,
        "recast_quick_timeout_seconds": 20,
        "recast_quick_timeout_source": "cli",
        "recast_retry_policy": "quick_then_retry_timeouts",
        "recast_timeout_seconds": 75,
        "recast_timeout_source": "cli",
        "validation_timeout_seconds": 90,
        "validation_timeout_source": "cli",
        "whole_run_timeout_seconds": 123,
    }


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
    monkeypatch.delenv("TIMEOUT_VALIDATION", raising=False)

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
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(python_exe.parent)
    assert "PYTHONPATH" not in captured["env"]
    payload = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["artifact_environment"] == {"artifact": str(artifact), "python": str(python_exe)}


def test_threshold_failures_make_main_fail(tmp_path: Path, monkeypatch) -> None:
    benchmark_dir = tmp_path / "biomodels_batch"
    evidence_dir = tmp_path / "evidence"
    _write_minimal_benchmark_tree(benchmark_dir)

    monkeypatch.delenv("TIMEOUT_VALIDATION", raising=False)
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
