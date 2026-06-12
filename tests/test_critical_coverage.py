"""Tests for the local critical-module coverage gate."""

from tools.check_critical_coverage import (
    CRITICAL_MODULES,
    _remove_existing_coverage_data,
    evaluate_critical_coverage,
)


def _coverage_report(
    *,
    branch_coverage: bool = True,
    line_percent: float = 95.0,
    branch_percent: float = 90.0,
) -> dict:
    return {
        "meta": {"branch_coverage": branch_coverage},
        "files": {
            module: {
                "summary": {
                    "percent_statements_covered": line_percent,
                    "num_branches": 10,
                    "percent_branches_covered": branch_percent,
                }
            }
            for module in CRITICAL_MODULES
        },
    }


def test_critical_coverage_passes_when_all_modules_meet_thresholds():
    assert evaluate_critical_coverage(_coverage_report()) == []


def test_critical_coverage_requires_branch_measurement():
    failures = evaluate_critical_coverage(_coverage_report(branch_coverage=False))

    assert len(failures) == 1
    assert failures[0].module == "<coverage report>"
    assert "branch coverage enabled" in failures[0].message


def test_critical_coverage_reports_line_and_branch_shortfalls():
    failures = evaluate_critical_coverage(
        _coverage_report(line_percent=89.9, branch_percent=84.9),
        modules=("src/ssys/_validator/core.py",),
    )

    assert len(failures) == 1
    assert failures[0].module == "src/ssys/_validator/core.py"
    assert failures[0].line_percent == 89.9
    assert failures[0].branch_percent == 84.9
    assert "line coverage 89.9% < 90.0%" in failures[0].message
    assert "branch coverage 84.9% < 85.0%" in failures[0].message


def test_critical_coverage_reports_missing_modules():
    failures = evaluate_critical_coverage(
        {"meta": {"branch_coverage": True}, "files": {}},
        modules=("src/ssys/_validator/core.py",),
    )

    assert len(failures) == 1
    assert failures[0].module == "src/ssys/_validator/core.py"
    assert failures[0].message == "module missing from coverage report"


def test_remove_existing_coverage_data_deletes_parallel_fragments(tmp_path):
    coverage_file = tmp_path / ".coverage-critical"
    coverage_file.write_text("data")
    fragment = tmp_path / ".coverage-critical.host.pid.random"
    fragment.write_text("fragment")

    _remove_existing_coverage_data(coverage_file)

    assert not coverage_file.exists()
    assert not fragment.exists()
