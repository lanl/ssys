"""Public API contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import ssys
from ssys.cli import main

STABLE_TOP_LEVEL_NAMES = {
    "SymSystem",
    "RecastResult",
    "NegativeInitialConditionError",
    "SBMLParseError",
    "SSysEquation",
    "SolverRequirement",
    "SystemClass",
    "ValidationProfile",
    "VALIDATION_REPORT_SCHEMA_RESOURCE",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "parse_antimony_via_sbml",
    "parse_sbml",
    "recast_to_ssystem",
    "ssystem_to_antimony",
    "canonicalize_aux_names",
    "classify_result",
    "classify_solver_requirement",
    "load_validation_report_schema",
    "validation_profile_choices",
}


def test_top_level_public_api_is_intentional_and_documented():
    public_api_doc = Path("PUBLIC_API.md").read_text(encoding="utf-8")

    assert set(ssys.__all__) == STABLE_TOP_LEVEL_NAMES
    for name in STABLE_TOP_LEVEL_NAMES:
        assert hasattr(ssys, name), name
        assert f"`{name}`" in public_api_doc


@pytest.mark.parametrize(
    ("module_name", "expected_names"),
    [
        pytest.param(
            "ssys.parsing",
            {
                "expand_antimony_function_templates",
                "parse_antimony_via_sbml",
                "parse_sbml",
                "parse_sbml_from_string",
            },
            id="parsing",
        ),
        pytest.param(
            "ssys.recasting",
            {"canonicalize_aux_names", "recast_to_ssystem", "term_to_coeff_exps"},
            id="recasting",
        ),
        pytest.param(
            "ssys.formatting",
            {
                "gma_to_antimony",
                "latex_odes",
                "latex_ssys",
                "product_to_antimony",
                "ssystem_to_antimony",
            },
            id="formatting",
        ),
        pytest.param(
            "ssys.lifting",
            {
                "AutonomousLiftResult",
                "add_dummy_for_constants",
                "find_composite_functions",
                "find_rational_denominators",
                "find_sqrt_of_sums",
                "lift_composite_functions",
                "lift_rational_functions",
                "lift_squared_for_sqrt",
                "lift_time_functions_to_autonomous",
            },
            id="lifting",
        ),
    ],
)
def test_focused_public_modules_expose_documented_names(module_name, expected_names):
    module = __import__(module_name, fromlist=["__all__"])
    public_api_doc = Path("PUBLIC_API.md").read_text(encoding="utf-8")

    assert set(module.__all__) == expected_names
    assert not any(name.startswith("_") for name in module.__all__)
    for name in expected_names:
        assert hasattr(module, name), name
        assert f"`{name}`" in public_api_doc


def test_validator_public_api_and_compatibility_shim_are_documented():
    import ssys.validator as validator

    public_api_doc = Path("PUBLIC_API.md").read_text(encoding="utf-8")
    stable_validator_names = {
        "RecastValidator",
        "validate_recast_pair",
        "validate_generated_output_roundtrip",
        "EquivalenceTest",
        "ValidationReport",
        "ValidationResult",
        "ValidationProfile",
        "ValidationProfileSpec",
        "validation_profile_choices",
        "load_validation_report_schema",
        "VALIDATION_REPORT_SCHEMA_VERSION",
    }

    for name in stable_validator_names:
        assert hasattr(validator, name), name
        assert f"`{name}`" in public_api_doc
    assert "Compatibility note" in public_api_doc
    assert "_test_passed" in validator.__all__


def test_cli_help_lists_stable_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ssys-recast", "--help"])

    with pytest.raises(SystemExit) as exc:
        main()
    output = capsys.readouterr().out

    assert exc.value.code == 0
    for option in (
        "--manifest",
        "--outdir",
        "--mode",
        "--validate",
        "--validation-profile",
        "--allow-validation-failures",
        "--version",
    ):
        assert option in output
    assert "trusted scientific inputs" in output
