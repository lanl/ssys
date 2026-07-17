"""Tests for validator module."""

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import ssys._validator.common as validator_common
import ssys._validator.mapping as validator_mapping
import ssys.classification as classification_module
from ssys._validator.numerical import NumericalValidationMixin
from ssys._validator.trajectory import TrajectoryValidationMixin
from ssys.ode_backends.ida_sundials_backend import IDASundialsUnavailable
from ssys.recaster import SolverRequirement, SystemClass
from ssys.validator import (
    EquivalenceTest,
    RecastValidator,
    ValidationProfile,
    ValidationReport,
    ValidationResult,
    validate_generated_output_roundtrip,
    validate_recast_pair,
)


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_validation_result_values(self):
        """Test the complete public result vocabulary."""
        assert {result.value for result in ValidationResult} == {
            "pass",
            "failed",
            "timeout",
            "not_attempted",
            "unsupported",
            "inconclusive",
        }

    def test_validation_result_names(self):
        """Test result name access."""
        assert ValidationResult.PASS.name == "PASS"
        assert ValidationResult.FAIL.name == "FAIL"
        assert ValidationResult.FAIL.value == "failed"


class TestEquivalenceTest:
    """Tests for EquivalenceTest dataclass."""

    def test_equivalence_test_pass(self):
        """Test creating passing equivalence test."""
        test = EquivalenceTest(
            name="symbolic_equivalence",
            result=ValidationResult.PASS,
            details="All equations match",
        )

        assert test.name == "symbolic_equivalence"
        assert test.result == ValidationResult.PASS

    def test_equivalence_test_fail_with_details(self):
        """Test creating failing test with details."""
        test = EquivalenceTest(
            name="trajectory_comparison",
            result=ValidationResult.FAIL,
            details="max_error: 0.5, location: t=10",
        )

        assert test.result == ValidationResult.FAIL
        assert test.reason == "failed"
        assert "max_error" in test.details


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_validation_report_creation(self):
        """Test creating validation report."""
        test1 = EquivalenceTest(
            name="test1",
            result=ValidationResult.PASS,
            details="OK",
        )

        report = ValidationReport(
            original_file="/path/to/original.ant",
            recast_file="/path/to/recast.ant",
            original_class=SystemClass.GENERAL,
            recast_class=SystemClass.CANONICAL_SSYSTEM,
            symbolic_test=test1,
            overall_pass=True,
        )

        assert report.original_file == "/path/to/original.ant"
        assert report.symbolic_test is test1
        assert report.overall_pass is True

    def test_validation_report_to_dict(self):
        """Test serializing report to dict."""
        test1 = EquivalenceTest(
            name="test1",
            result=ValidationResult.PASS,
            details="OK",
        )

        report = ValidationReport(
            original_file="/path/to/original.ant",
            recast_file="/path/to/recast.ant",
            original_class=SystemClass.GENERAL,
            recast_class=SystemClass.CANONICAL_SSYSTEM,
            symbolic_test=test1,
            overall_pass=True,
        )

        d = report.to_dict()

        assert d["schema_version"] == "1.0"
        assert d["original_file"] == "/path/to/original.ant"
        assert d["overall_pass"] is True
        assert d["tests"]["symbolic"]["reason"] is None

    def test_validation_report_serializes_profile_metadata_and_reasons(self):
        test = EquivalenceTest(
            name="symbolic_equivalence",
            result=ValidationResult.TIMEOUT,
            details="forced timeout",
        )
        report = ValidationReport(
            original_file="/path/to/original.ant",
            recast_file="/path/to/recast.ant",
            original_class=SystemClass.GENERAL,
            recast_class=SystemClass.CANONICAL_SSYSTEM,
            symbolic_test=test,
            validation_profile="symbolic",
            validation_profile_description="symbolic proof profile",
            required_tests=["generated_output", "parser", "mapping", "symbolic"],
        )

        data = report.to_dict()

        assert data["validation_profile"]["name"] == "symbolic"
        assert data["validation_profile"]["required_tests"] == [
            "generated_output",
            "parser",
            "mapping",
            "symbolic",
        ]
        assert data["tests"]["symbolic"]["result"] == "timeout"
        assert data["tests"]["symbolic"]["reason"] == "timeout"


class TestSystemClassification:
    """Focused tests for validator system classification branches."""

    def test_non_monomial_denominator_classifies_general_before_expand(
        self, monkeypatch
    ):
        x, y, z = sp.symbols("X Y Z", positive=True)
        system = SimpleNamespace(
            vars=[x, y, z],
            params={},
            odes={x: x / (y + z)},
            assignment_rules={},
        )
        phases = []

        def fail_expand(expr):
            raise AssertionError(f"classification should not expand {expr}")

        monkeypatch.setattr(classification_module.sp, "expand", fail_expand)

        result = classification_module.classify_system(
            system,
            progress_callback=phases.append,
            progress_prefix="validator_parser_classify_systems:recast",
        )

        assert result == SystemClass.GENERAL
        assert "validator_parser_classify_systems:recast:denominator:X" in phases

    def test_assignment_rule_denominator_classifies_general_before_expand(
        self, monkeypatch
    ):
        x, y, z, a = sp.symbols("X Y Z A", positive=True)
        system = SimpleNamespace(
            vars=[x, y, z],
            params={},
            odes={x: x / a},
            assignment_rules={"A": "Y + Z"},
        )

        def fail_expand(expr):
            raise AssertionError(f"classification should not expand {expr}")

        monkeypatch.setattr(classification_module.sp, "expand", fail_expand)

        result = classification_module.classify_system(system)

        assert result == SystemClass.GENERAL

    def test_assignment_rule_substitution_uses_shared_demand_cache(
        self, monkeypatch
    ):
        x, y, rate = sp.symbols("X Y rate", positive=True)
        system = SimpleNamespace(
            vars=[x, y],
            params={"k": 1.0},
            odes={x: rate - x, y: 2 * rate - y},
            assignment_rules={
                "rate": "inner * X",
                "inner": "k * X",
                "unused": "unused_inner * X",
                "unused_inner": "k + 1",
            },
        )
        cache_misses = []
        original_expand = classification_module._expand_assignment_rule_by_name

        def counting_expand(
            rule_name,
            assignment_rules,
            all_syms,
            parse_cache,
            expanded_cache,
            visiting,
        ):
            if rule_name not in expanded_cache:
                cache_misses.append(rule_name)
            return original_expand(
                rule_name,
                assignment_rules,
                all_syms,
                parse_cache,
                expanded_cache,
                visiting,
            )

        monkeypatch.setattr(
            classification_module,
            "_expand_assignment_rule_by_name",
            counting_expand,
        )

        result = classification_module.classify_system(system)

        assert result == SystemClass.CANONICAL_SSYSTEM
        assert cache_misses == ["rate", "inner"]


class TestFailClosedValidation:
    """Tests for fail-closed report aggregation."""

    def test_validate_recast_pair_defaults_to_sbml_parser(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("model original()\nend\n")
        recast = tmp_path / "recast.ant"
        recast.write_text("model recast()\nend\n")

        captured = {}

        def fake_roundtrip(*args, **kwargs):
            return EquivalenceTest(
                name="generated_output_roundtrip",
                result=ValidationResult.PASS,
                details="ok",
            )

        class FakeValidator:
            def __init__(self, original_file, recast_file, factor_map, mode, parser):
                captured["parser"] = parser

            def validate(self, *args, **kwargs):
                return ValidationReport(
                    original_file=str(original),
                    recast_file=str(recast),
                    original_class=SystemClass.SSYSTEM,
                    recast_class=SystemClass.SSYSTEM,
                    overall_pass=True,
                    overall_result=ValidationResult.PASS,
                )

        monkeypatch.setattr("ssys._validator.core.validate_generated_output_roundtrip", fake_roundtrip)
        monkeypatch.setattr("ssys._validator.core.RecastValidator", FakeValidator)

        report = validate_recast_pair(str(original), str(recast))

        assert captured["parser"] == "sbml"
        assert report.overall_pass is True

    def test_validate_recast_pair_reports_parser_subphase_progress(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        phases = []

        report = validate_recast_pair(
            str(original),
            str(recast),
            parser="sbml",
            profile="structural",
            progress_callback=phases.append,
        )

        assert report.overall_pass is True
        expected_subsequence = [
            "generated_output_roundtrip",
            "validator_parser",
            "validator_parser_recast_read",
            "validator_parser_original_read",
            "validator_parser_original_sbml",
            "validator_parser_original_sbml_preprocess",
            "validator_parser_original_sbml_antimony_load",
            "validator_parser_original_sbml_sbml_export",
            "validator_parser_original_sbml_sym_system_parse_libsbml_read",
            "validator_parser_original_sbml_sym_system_parse_sym_system_build",
            "validator_parser_recast_sbml",
            "validator_parser_recast_sbml_preprocess",
            "validator_parser_recast_sbml_antimony_load",
            "validator_parser_recast_sbml_sbml_export",
            "validator_parser_recast_sbml_sym_system_parse_libsbml_read",
            "validator_parser_recast_sbml_sym_system_parse_sym_system_build",
            "generated_output_roundtrip",
            "mapping",
        ]
        cursor = 0
        for phase in phases:
            if phase == expected_subsequence[cursor]:
                cursor += 1
                if cursor == len(expected_subsequence):
                    break
        assert cursor == len(expected_subsequence), phases

    def test_symbolic_profile_records_excluded_checks_without_requiring_them(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        report = validator.validate(profile=ValidationProfile.SYMBOLIC)
        data = report.to_dict()

        assert report.validation_profile == "symbolic"
        assert report.overall_pass is True
        assert report.numerical_test is not None
        assert report.numerical_test.result == ValidationResult.NOT_ATTEMPTED
        assert report.numerical_test.reason == "profile_excluded"
        assert report.trajectory_test is not None
        assert report.trajectory_test.reason == "profile_excluded"
        assert "numerical" not in report.required_tests
        assert data["tests"]["numerical"]["reason"] == "profile_excluded"

    def test_validator_init_accepts_explicit_factor_map(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        X = sp.Symbol("X", positive=True)

        validator = RecastValidator(str(original), str(recast), factor_map={X: X}, parser="sbml")

        assert validator.factor_map[X] == X
        assert validator.mapping[X] == X

    def test_strict_profile_overrides_legacy_boolean_flags(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def symbolic_pass(*args, **kwargs):
            return EquivalenceTest("symbolic_equivalence", ValidationResult.PASS)

        def numerical_pass(*args, **kwargs):
            return EquivalenceTest("numerical_pointwise", ValidationResult.PASS)

        def trajectory_pass(*args, **kwargs):
            return EquivalenceTest("trajectory_comparison", ValidationResult.PASS)

        monkeypatch.setattr(validator, "check_symbolic_equivalence", symbolic_pass)
        monkeypatch.setattr(validator, "check_numerical_pointwise", numerical_pass)
        monkeypatch.setattr(validator, "check_trajectory_comparison", trajectory_pass)

        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
            run_auxiliaries=False,
            profile="strict",
        )

        assert report.validation_profile == "strict"
        assert report.symbolic_test is not None
        assert report.symbolic_test.result == ValidationResult.PASS
        assert report.numerical_test is not None
        assert report.numerical_test.result == ValidationResult.PASS
        assert report.trajectory_test is not None
        assert report.trajectory_test.result == ValidationResult.PASS
        assert report.overall_pass is True

    def test_timeout_required_check_fails_overall_with_timeout_result(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def symbolic_timeout(*args, **kwargs):
            return EquivalenceTest(
                name="symbolic_equivalence",
                result=ValidationResult.TIMEOUT,
                details="forced timeout",
            )

        monkeypatch.setattr(validator, "check_symbolic_equivalence", symbolic_timeout)
        report = validator.validate(
            profile="symbolic",
        )

        assert report.overall_pass is False
        assert report.overall_result == ValidationResult.TIMEOUT
        assert "TIMEOUT" in report.summary
        assert report.symbolic_test is not None
        assert report.symbolic_test.reason == "timeout"

    def test_not_attempted_required_symbolic_test_fails_overall(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def not_attempted(*args, **kwargs):
            return EquivalenceTest(
                name="symbolic_equivalence",
                result=ValidationResult.NOT_ATTEMPTED,
                details="forced skip",
            )

        monkeypatch.setattr(validator, "check_symbolic_equivalence", not_attempted)
        report = validator.validate(
            run_symbolic=True,
            run_numerical=False,
            run_trajectory=False,
            run_auxiliaries=False,
        )

        assert report.overall_pass is False
        assert report.overall_result == ValidationResult.NOT_ATTEMPTED

    def test_core_helpers_cover_expected_classes_and_overall_results(self):
        validator = RecastValidator.__new__(RecastValidator)

        assert validator._expected_class_for_mode("canonical") == SystemClass.CANONICAL_SSYSTEM
        assert validator._expected_class_for_mode("simplified") == SystemClass.SSYSTEM
        assert validator._expected_class_for_mode("gma") == SystemClass.GMA
        assert validator._expected_class_for_mode("unknown") is None

        assert validator._overall_result([
            EquivalenceTest("ok", ValidationResult.PASS),
        ]) == ValidationResult.PASS
        assert validator._overall_result([
            EquivalenceTest("unsupported", ValidationResult.UNSUPPORTED),
        ]) == ValidationResult.UNSUPPORTED
        assert validator._overall_result([
            EquivalenceTest("inconclusive", ValidationResult.INCONCLUSIVE),
        ]) == ValidationResult.INCONCLUSIVE

    def test_unsupported_required_check_sets_unsupported_summary(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        monkeypatch.setattr(
            "ssys._validator.core.validate_generated_output_roundtrip",
            lambda *args, **kwargs: EquivalenceTest(
                "generated_output_roundtrip", ValidationResult.PASS
            ),
        )
        monkeypatch.setattr(
            validator,
            "check_symbolic_equivalence",
            lambda *args, **kwargs: EquivalenceTest(
                "symbolic_equivalence",
                ValidationResult.UNSUPPORTED,
                details="symbolic engine unavailable",
            ),
        )

        report = validator.validate(profile="symbolic")

        assert report.overall_pass is False
        assert report.overall_result == ValidationResult.UNSUPPORTED
        assert report.summary == "Validation UNSUPPORTED: a required backend is unavailable"

    def test_strict_profile_runs_jax_and_requires_present_algebraic_residual(
        self, tmp_path, monkeypatch
    ):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        monkeypatch.setattr(
            "ssys._validator.core.validate_generated_output_roundtrip",
            lambda *args, **kwargs: EquivalenceTest(
                "generated_output_roundtrip", ValidationResult.PASS
            ),
        )
        monkeypatch.setattr(
            validator,
            "check_symbolic_equivalence",
            lambda *args, **kwargs: EquivalenceTest("symbolic_equivalence", ValidationResult.PASS),
        )
        monkeypatch.setattr(
            validator,
            "check_numerical_pointwise_jax",
            lambda *args, **kwargs: EquivalenceTest(
                "numerical_pointwise_jax",
                ValidationResult.PASS,
                metadata={"engine": "jax"},
            ),
        )
        monkeypatch.setattr(
            validator,
            "check_trajectory_comparison",
            lambda *args, **kwargs: EquivalenceTest("trajectory_comparison", ValidationResult.PASS),
        )
        monkeypatch.setattr(
            validator,
            "check_algebraic_manifold_preservation",
            lambda *args, **kwargs: EquivalenceTest(
                "algebraic_manifold_residuals", ValidationResult.PASS
            ),
        )
        monkeypatch.setattr(validator, "check_auxiliary_identities", lambda *args, **kwargs: [])

        report = validator.validate(use_jax=True)

        assert report.validation_profile == "strict"
        assert report.numerical_test is not None
        assert report.numerical_test.metadata == {"engine": "jax"}
        assert report.algebraic_residual_test is not None
        assert report.algebraic_residual_test.result == ValidationResult.PASS
        assert report.overall_pass is True
        assert report.summary == "Validation PASSED: recast roundtrips and required checks passed"

    def test_simulation_failure_fails_overall(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def simulation_failed(*args, **kwargs):
            return EquivalenceTest(
                name="trajectory_comparison",
                result=ValidationResult.INCONCLUSIVE,
                details="forced simulation failure",
            )

        monkeypatch.setattr(validator, "check_trajectory_comparison", simulation_failed)
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=True,
            run_auxiliaries=False,
        )

        assert report.overall_pass is False
        assert report.overall_result == ValidationResult.INCONCLUSIVE

    def test_auxiliary_solver_requirement_refinement_skips_non_manifold_cases(self):
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        Z = sp.Symbol("Z_1", positive=True)
        T = sp.Symbol("T", positive=True)
        W = sp.Symbol("W_1", positive=True)
        k = sp.Symbol("k", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.recast_solver_requirement = SolverRequirement.ODE_ONLY
        validator.recast_odes = {X: -X, Y: -Y, T: sp.Integer(1), W: -W}
        validator.recast_ir = SimpleNamespace(assignment_rules={"Y_1": "X + 1"})
        validator.recast_system = SimpleNamespace(solver_requirement=SolverRequirement.ODE_ONLY)
        validator.auxiliary_defs = {
            Y: X + 1,  # assignment rule owns this name
            Z: X + 2,  # not a recast state
            T: sp.Symbol("time"),  # generated clock definition
            W: k + 1,  # state variable but not a state-dependent manifold
        }

        validator._refine_recast_solver_requirement_from_auxiliaries()

        assert validator.recast_solver_requirement == SolverRequirement.ODE_ONLY
        assert validator.recast_system.solver_requirement == SolverRequirement.ODE_ONLY

        validator.auxiliary_defs = {Y: X + 1}
        validator.recast_ir.assignment_rules = {}

        validator._refine_recast_solver_requirement_from_auxiliaries()

        assert validator.recast_solver_requirement == SolverRequirement.DAE_REQUIRED
        assert validator.recast_system.solver_requirement == SolverRequirement.DAE_REQUIRED

    def test_missing_mapping_fails_overall(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species Z_1;
                Z_1' = -k*Z_1;
                k = 0.5;
                Z_1 = 1.0;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
            run_auxiliaries=False,
        )

        assert report.overall_pass is False
        assert report.mapping_test is not None
        assert report.mapping_test.result == ValidationResult.FAIL
        assert report.overall_result == ValidationResult.FAIL

    def test_invalid_generated_output_roundtrip_is_reported(self, tmp_path):
        invalid = tmp_path / "invalid.ant"
        invalid.write_text("model invalid()\n    DNA := Z_1;\nend\n")

        result = validate_generated_output_roundtrip(str(invalid))

        assert result.result == ValidationResult.FAIL
        assert result.metadata["antimony_parse_success"] is False
        assert result.metadata["parser_diagnostics"]

    def test_validate_recast_pair_writes_parser_failure_report(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "bad_recast.ant"
        recast.write_text("model bad_recast()\n    DNA := Z_1;\nend\n")
        output_json = tmp_path / "validation.json"

        report = validate_recast_pair(
            str(original),
            str(recast),
            output_json=str(output_json),
            parser="sbml",
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
            run_auxiliaries=False,
        )

        assert report.overall_pass is False
        data = json.loads(output_json.read_text())
        assert data["tests"]["generated_output"]["result"] == "failed"
        assert data["tests"]["parser"]["result"] == "failed"
        assert data["tests"]["mapping"]["result"] == "not_attempted"
        assert data["tests"]["mapping"]["reason"] == "parser_failed"
        assert data["tests"]["symbolic"]["reason"] == "profile_excluded"
        assert data["overall_result"] == "failed"

    def test_validate_recast_pair_parser_failure_defaults_to_strict_profile(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "bad_recast.ant"
        recast.write_text("model bad_recast()\n    DNA := Z_1;\nend\n")

        report = validate_recast_pair(str(original), str(recast), parser="sbml")

        assert report.validation_profile == "strict"
        assert report.required_tests == [
            "generated_output",
            "parser",
            "mapping",
            "symbolic",
            "numerical",
            "trajectory",
            "algebraic_residuals",
            "auxiliaries",
        ]
        assert report.symbolic_test is not None
        assert report.symbolic_test.reason == "parser_failed"
        assert report.numerical_test is not None
        assert report.numerical_test.reason == "parser_failed"
        assert report.trajectory_test is not None
        assert report.trajectory_test.reason == "parser_failed"
        assert report.auxiliary_tests[0].reason == "parser_failed"

    def test_validate_recast_pair_parser_failure_preserves_named_profile(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "bad_recast.ant"
        recast.write_text("model bad_recast()\n    DNA := Z_1;\nend\n")

        report = validate_recast_pair(str(original), str(recast), parser="sbml", profile="symbolic")

        assert report.validation_profile == "symbolic"
        assert "symbolic" in report.required_tests
        assert "numerical" not in report.required_tests
        assert report.symbolic_test is not None
        assert report.symbolic_test.reason == "parser_failed"
        assert report.numerical_test is not None
        assert report.numerical_test.reason == "profile_excluded"
        assert report.trajectory_test is not None
        assert report.trajectory_test.reason == "profile_excluded"

    def test_validate_recast_pair_classifies_unsupported_parser_features(
        self,
        tmp_path,
    ):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1.0;
            end
        """)
        recast = tmp_path / "piecewise_recast.ant"
        recast.write_text("""
            model piecewise_recast()
                species Z_1;
                Z_1' = piecewise(-Z_1, T > threshold, 0);
                Z_1 = 1.0;
                T = 1.0;
                threshold = 0.5;
            end
        """)
        output_json = tmp_path / "validation.json"

        report = validate_recast_pair(
            str(original),
            str(recast),
            output_json=str(output_json),
            parser="sbml",
            profile="numerical",
        )
        data = json.loads(output_json.read_text())

        assert report.generated_output_test is not None
        assert report.generated_output_test.result == ValidationResult.PASS
        assert data["tests"]["parser"]["result"] == "unsupported"
        assert data["tests"]["parser"]["reason"] == "unsupported_feature"
        assert data["tests"]["parser"]["metadata"]["unsupported_features"] == [
            "gt",
            "piecewise",
        ]
        assert "unsupported function(s): gt" in data["tests"]["parser"]["details"]
        assert data["tests"]["mapping"]["reason"] == "unsupported_feature"
        assert data["tests"]["numerical"]["reason"] == "unsupported_feature"
        assert data["overall_result"] == "unsupported"


class TestSolverAwareValidation:
    """Tests for solver requirement reporting and algebraic residual checks."""

    def test_trajectory_docstring_matches_available_solver_backends(self):
        doc = TrajectoryValidationMixin.check_trajectory_comparison.__doc__

        assert doc is not None
        assert "libRoadRunner/CVODE" in doc
        assert "IDA/SUNDIALS" in doc
        assert "RK4" not in doc
        assert "LSODA" not in doc

    def _write_identity_pair(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                // @SSYS SOLVER_REQUIREMENT=ode_only
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1;
            end
        """)
        return original, recast

    def test_report_records_solver_requirements(self, tmp_path):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
            run_auxiliaries=False,
        )
        data = report.to_dict()

        assert data["solver"]["recast_requirement"] == SolverRequirement.ODE_ONLY.value
        assert data["tests"]["parser"]["metadata"]["recast_solver_requirement"] == (
            SolverRequirement.ODE_ONLY.value
        )

    def test_trajectory_report_includes_backend_selection(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def fake_simulate_model(*args, **kwargs):
            model_name = args[7]
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0], [0.5]]),
                "message": "",
                "backend": f"{model_name}_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
                "step_diagnostics": {
                    "requested_output_points": 100,
                    "actual_output_points": 2,
                },
            }

        monkeypatch.setattr(validator, "_simulate_model", fake_simulate_model)

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.PASS
        assert result.metadata["original_backend"] == "original_backend"
        assert result.metadata["recast_backend"] == "recast_backend"
        assert result.metadata["threshold"] == 3.0e-2
        assert result.metadata["scaling_method"] == "peak_scaled_absolute"
        assert result.metadata["solver_tolerances"] == {
            "relative_tolerance": 1.0e-10,
            "absolute_tolerance": 1.0e-12,
            "maximum_num_steps": 200000,
        }
        assert result.metadata["original_step_diagnostics"]["actual_output_points"] == 2
        assert result.metadata["recast_step_diagnostics"]["actual_output_points"] == 2
        assert result.metadata["error_metrics"]["max_absolute_error"] == 0.0
        assert result.metadata["error_metrics"]["max_scaled_error"] == 0.0
        assert result.metadata["worst_point"]["variable"] == "X"
        assert result.metadata["worst_point"]["scaled_error"] == 0.0

    def test_trajectory_uses_simulation_metadata_time_grid(
        self, tmp_path, monkeypatch
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        validator.orig_ir.sim_t_start = 0.25
        validator.orig_ir.sim_t_end = 2.5
        validator.orig_ir.sim_n_steps = 4
        calls = []

        def fake_simulate_model(*args, **kwargs):
            calls.append(
                {
                    "model_name": args[7],
                    "t_end": args[3],
                    "n_points": args[4],
                    "t_start": kwargs["t_start"],
                }
            )
            return {
                "success": True,
                "t": np.linspace(0.25, 2.5, 5),
                "y": np.array([[1.0], [0.8], [0.6], [0.4], [0.2]]),
                "message": "",
                "backend": f"{args[7]}_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
                "step_diagnostics": {
                    "requested_t_start": 0.25,
                    "requested_t_end": 2.5,
                    "requested_output_points": 5,
                },
            }

        monkeypatch.setattr(validator, "_simulate_model", fake_simulate_model)

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.PASS
        assert calls == [
            {"model_name": "original", "t_end": 2.5, "n_points": 5, "t_start": 0.25},
            {"model_name": "recast", "t_end": 2.5, "n_points": 5, "t_start": 0.25},
        ]
        assert result.metadata["time_grid"] == {
            "t_start": 0.25,
            "t_end": 2.5,
            "n_output_points": 5,
            "source": "simulation_metadata",
            "recast_interpolated_to_original": False,
        }

    def test_unsupported_recast_solver_requirement_is_unsupported(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def fake_simulate_model(*args, **kwargs):
            model_name = args[7]
            if model_name == "original":
                return {
                    "success": True,
                    "t": np.array([0.0, 1.0]),
                    "y": np.array([[1.0], [0.5]]),
                    "message": "",
                    "backend": "original_backend",
                    "solver_requirement": SolverRequirement.ODE_ONLY.value,
                    "unsupported_solver_requirement": False,
                    "algebraic_residuals": {},
                }
            return {
                "success": False,
                "t": np.array([]),
                "y": np.array([]),
                "message": "unsupported dae_required",
                "backend": "dae_projection",
                "solver_requirement": SolverRequirement.DAE_REQUIRED.value,
                "unsupported_solver_requirement": True,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", fake_simulate_model)

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.UNSUPPORTED
        assert "unsupported dae_required" in result.details
        assert result.metadata["recast_backend"] == "dae_projection"

    @pytest.mark.parametrize(
        ("unsupported_solver_requirement", "expected_result"),
        [
            pytest.param(False, ValidationResult.NOT_ATTEMPTED, id="solver-failed"),
            pytest.param(True, ValidationResult.UNSUPPORTED, id="unsupported-solver"),
        ],
    )
    def test_original_simulation_failure_is_fail_closed(
        self, tmp_path, monkeypatch, unsupported_solver_requirement, expected_result
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def failed_original(*args, **kwargs):
            assert args[7] == "original"
            return {
                "success": False,
                "t": np.array([]),
                "y": np.array([]),
                "message": "original solver failed",
                "backend": "original_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": unsupported_solver_requirement,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", failed_original)

        result = validator.check_trajectory_comparison()

        assert result.result == expected_result
        assert "Original simulation failed: original solver failed" in result.details
        assert result.metadata["original_backend"] == "original_backend"
        assert result.metadata["recast_backend"] is None

    def test_recast_simulation_failure_without_solver_gap_is_not_attempted(
        self, tmp_path, monkeypatch
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def recast_fails(*args, **kwargs):
            model_name = args[7]
            if model_name == "original":
                return {
                    "success": True,
                    "t": np.array([0.0, 1.0]),
                    "y": np.array([[1.0], [0.5]]),
                    "message": "",
                    "backend": "original_backend",
                    "solver_requirement": SolverRequirement.ODE_ONLY.value,
                    "unsupported_solver_requirement": False,
                    "algebraic_residuals": {},
                }
            return {
                "success": False,
                "t": np.array([]),
                "y": np.array([]),
                "message": "integration did not converge",
                "backend": "recast_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {"Y_1": {"max_abs": 2.0}},
            }

        monkeypatch.setattr(validator, "_simulate_model", recast_fails)

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert "Recast simulation failed: integration did not converge" in result.details
        assert result.metadata["recast_backend"] == "recast_backend"
        assert result.metadata["algebraic_residuals"] == {"Y_1": {"max_abs": 2.0}}

    def test_trajectory_passes_time_symbol_to_backend(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = next(iter(validator.orig_odes))
        t = sp.Symbol("t")
        validator.orig_odes = {X: -t * X}
        validator.recast_odes = {X: -t * X}
        seen_time_symbols = []

        def successful_simulation(*args, **kwargs):
            seen_time_symbols.append(args[6])
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0], [0.5]]),
                "message": "",
                "backend": f"{args[7]}_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", successful_simulation)

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.PASS
        assert [str(sym) for sym in seen_time_symbols] == ["t", "t"]

    def test_trajectory_exception_is_not_attempted(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def successful_simulation(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0], [0.5]]),
                "message": "",
                "backend": f"{args[7]}_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        def bad_initial_conditions(*args, **kwargs):
            raise RuntimeError("bad auxiliary initial condition")

        monkeypatch.setattr(validator, "_simulate_model", successful_simulation)
        monkeypatch.setattr(
            validator,
            "_compute_recast_initial_conditions",
            bad_initial_conditions,
        )

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert "bad auxiliary initial condition" in result.details
        assert "Traceback" in result.details

    def test_dae_required_missing_ida_dependency_is_unsupported(self, tmp_path, monkeypatch):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                // @SSYS SOLVER_REQUIREMENT=dae_required
                species X;
                Y_1 := X + 1;
                X' = -k*X;
                k = 0.5;
                X = 1;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def fake_simulate_ode(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0], [0.5]]),
                "state_names": ["X"],
                "message": "",
                "integrator_stats": {},
            }

        def missing_ida():
            raise IDASundialsUnavailable("scikit-SUNDAE is not installed. uv sync --extra dae")

        monkeypatch.setattr("ssys.ode_backends.interface.simulate_ode", fake_simulate_ode)
        monkeypatch.setattr(
            "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
            missing_ida,
        )

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.UNSUPPORTED
        assert "uv sync --extra dae" in result.details
        assert result.metadata["recast_backend"] == "ida_sundials"

    def test_algebraic_residual_detects_ode_mode_drift(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -X/(K + X);
                K = 1;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X, Y_1;
                // ========================================================================
                // AUXILIARY DEFINITIONS (for lifted variables)
                // ========================================================================
                // Y_1 := K + X
                // ========================================================================
                X' = -X/Y_1;
                Y_1' = -X/Y_1;
                K = 1;
                X = 1;
                Y_1 = 2;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        residuals, errors = validator._compute_algebraic_residual_norms(
            np.array([[1.0, 2.2], [0.5, 1.8]]),
            [X, Y],
            {"K": 1.0},
            np.array([0.0, 1.0]),
        )

        assert errors == []
        assert residuals["Y_1"]["max_abs"] > 0.1

    def test_assignment_rule_auxiliary_residual_is_enforced(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -X/(K + X);
                K = 1;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                // ========================================================================
                // AUXILIARY DEFINITIONS (for lifted variables)
                // ========================================================================
                // Y_1 := K + X
                // ========================================================================
                Y_1 := K + X;
                X' = -X/Y_1;
                K = 1;
                X = 1;
            end
        """)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        residuals, errors = validator._compute_algebraic_residual_norms(
            np.array([[1.0], [0.5]]),
            [X],
            {"K": 1.0},
            np.array([0.0, 1.0]),
        )

        assert errors == []
        assert residuals["Y_1"]["max_abs"] == 0.0
        assert residuals["Y_1"]["enforced_by_assignment_rule"] is True

    def test_trajectory_comparison_fails_on_divergent_recast(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def fake_simulate_model(*args, **kwargs):
            model_name = args[7]
            y = np.array([[1.0], [0.5]]) if model_name == "original" else np.array([[1.0], [2.0]])
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": y,
                "message": "",
                "backend": f"{model_name}_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", fake_simulate_model)

        result = validator.check_trajectory_comparison(threshold=1.0e-6)

        assert result.result == ValidationResult.FAIL
        assert result.max_error > 0.0
        assert result.counterexamples[0]["variable"] == "X"
        assert result.counterexamples[0]["absolute_error"] > 0.0
        assert result.counterexamples[0]["relative_error"] > 0.0
        assert result.counterexamples[0]["scaled_error"] > 0.0
        assert result.metadata["worst_point"]["absolute_error"] > 0.0

    def test_trajectory_comparison_interpolates_recast_time_grid(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def fake_simulate_model(*args, **kwargs):
            model_name = args[7]
            if model_name == "original":
                return {
                    "success": True,
                    "t": np.array([0.0, 0.5, 1.0]),
                    "y": np.array([[1.0], [0.5], [0.0]]),
                    "message": "",
                    "backend": "original_backend",
                    "solver_requirement": SolverRequirement.ODE_ONLY.value,
                    "unsupported_solver_requirement": False,
                    "algebraic_residuals": {},
                }
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0], [0.0]]),
                "message": "",
                "backend": "recast_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", fake_simulate_model)

        result = validator.check_trajectory_comparison(threshold=1.0e-12)

        assert result.result == ValidationResult.PASS
        assert result.metadata["recast_backend"] == "recast_backend"
        assert result.metadata["time_grid"]["recast_interpolated_to_original"] is True

    def test_simulate_model_reorders_backend_columns(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)

        def fake_backend(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[20.0, 10.0], [40.0, 30.0]]),
                "state_names": ["Y", "X"],
                "message": "",
                "backend": "fake_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
                "algebraic_residuals": {"Y": {"max_abs": 0.0}},
            }

        monkeypatch.setattr("ssys.ode_backends.simulate_model", fake_backend)

        result = validator._simulate_model(
            validator.recast_ir,
            validator.recast_odes,
            [X, Y],
            1.0,
            2,
            {},
            None,
            "recast",
        )

        assert result["success"] is True
        np.testing.assert_allclose(result["y"], np.array([[10.0, 20.0], [30.0, 40.0]]))
        assert result["algebraic_residuals"] == {"Y": {"max_abs": 0.0}}

    def test_simulate_model_fills_missing_backend_states_with_zero(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)

        def backend_missing_y(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[10.0], [30.0]]),
                "state_names": ["X"],
                "message": "",
                "backend": "fake_backend",
                "solver_requirement": SolverRequirement.ODE_ONLY.value,
            }

        monkeypatch.setattr("ssys.ode_backends.simulate_model", backend_missing_y)

        result = validator._simulate_model(
            validator.recast_ir,
            validator.recast_odes,
            [X, Y],
            1.0,
            2,
            {},
            None,
            "recast",
        )

        assert result["success"] is True
        np.testing.assert_allclose(result["y"], np.array([[10.0, 0.0], [30.0, 0.0]]))

    def test_simulate_model_failure_preserves_backend_metadata(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")

        def fake_backend(*args, **kwargs):
            return {
                "success": False,
                "t": np.array([]),
                "y": np.array([]),
                "state_names": [],
                "message": "solver refused dae",
                "backend": "fake_backend",
                "solver_requirement": SolverRequirement.DAE_REQUIRED.value,
                "unsupported_solver_requirement": True,
                "algebraic_residuals": {"Y": {"max_abs": 1.0}},
            }

        monkeypatch.setattr("ssys.ode_backends.simulate_model", fake_backend)

        result = validator._simulate_model(
            validator.recast_ir,
            validator.recast_odes,
            validator.recast_state_vars,
            1.0,
            2,
            {},
            None,
            "recast",
        )

        assert result["success"] is False
        assert result["message"] == "solver refused dae"
        assert result["unsupported_solver_requirement"] is True
        assert result["algebraic_residuals"] == {"Y": {"max_abs": 1.0}}

    def test_roadrunner_wrapper_reorders_success_and_reports_failure(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)

        def success_backend(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0]),
                "y": np.array([[2.0, 1.0]]),
                "state_names": ["Y", "X"],
            }

        monkeypatch.setattr(
            "ssys.ode_backends.roadrunner_backend.simulate_with_roadrunner",
            success_backend,
        )
        result = validator._simulate_with_roadrunner(
            validator.recast_ir,
            validator.recast_odes,
            [X, Y],
            1.0,
            1,
            {},
            None,
            "recast",
        )

        assert result["success"] is True
        np.testing.assert_allclose(result["y"], np.array([[1.0, 2.0]]))

        def missing_state_backend(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0]),
                "y": np.array([[3.0]]),
                "state_names": ["X"],
            }

        monkeypatch.setattr(
            "ssys.ode_backends.roadrunner_backend.simulate_with_roadrunner",
            missing_state_backend,
        )
        missing = validator._simulate_with_roadrunner(
            validator.recast_ir,
            validator.recast_odes,
            [X, Y],
            1.0,
            1,
            {},
            None,
            "recast",
        )

        assert missing["success"] is True
        np.testing.assert_allclose(missing["y"], np.array([[3.0, 0.0]]))

        def failure_backend(*args, **kwargs):
            return {"success": False, "message": "bad antimony"}

        monkeypatch.setattr(
            "ssys.ode_backends.roadrunner_backend.simulate_with_roadrunner",
            failure_backend,
        )
        failed = validator._simulate_with_roadrunner(
            validator.recast_ir,
            validator.recast_odes,
            [X],
            1.0,
            1,
            {},
            None,
            "recast",
        )

        assert failed["success"] is False
        assert "bad antimony" in failed["message"]

        def raising_backend(*args, **kwargs):
            raise RuntimeError("roadrunner import failed")

        monkeypatch.setattr(
            "ssys.ode_backends.roadrunner_backend.simulate_with_roadrunner",
            raising_backend,
        )
        errored = validator._simulate_with_roadrunner(
            validator.recast_ir,
            validator.recast_odes,
            [X],
            1.0,
            1,
            {},
            None,
            "recast",
        )

        assert errored["success"] is False
        assert "roadrunner import failed" in errored["message"]

    def test_recast_initial_conditions_cover_priority_and_fallbacks(self, tmp_path):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)
        Z = sp.Symbol("Z", positive=True)
        W = sp.Symbol("W", positive=True)
        Q = sp.Symbol("Q", positive=True)
        k = sp.Symbol("k", positive=True)

        validator.orig_ir.initials = {}
        validator.recast_ir.initials = {Y: 7.0}
        validator.auxiliary_defs = {Z: X + k, W: sp.Symbol("missing")}

        y0 = validator._compute_recast_initial_conditions([X, Y, Z, W, Q], [X], {"k": 4.0})

        assert y0 == {"X": 1.0, "Y": 7.0, "Z": 5.0, "W": 1.0, "Q": 1.0}

    def test_reconstruct_from_recast_covers_mapping_forms(self, tmp_path):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        A, B, C, D, Z1, Z2, k = sp.symbols("A B C D Z1 Z2 k", positive=True)
        validator.mapping = {
            A: A,
            B: Z1 * Z2**2,
            C: Z2,
            D: k * Z1 + 1,
        }
        z_values = np.array([[2.0, 3.0, 4.0], [5.0, 7.0, 11.0]])

        reconstructed = validator._reconstruct_from_recast(
            z_values,
            [A, Z1, Z2],
            [A, B, C, D],
            {"k": 3.0},
            np.array([0.0, 1.0]),
            None,
        )

        expected = np.array([
            [2.0, 3.0 * 4.0**2, 4.0, 3.0 * 3.0 + 1.0],
            [5.0, 7.0 * 11.0**2, 11.0, 3.0 * 7.0 + 1.0],
        ])
        np.testing.assert_allclose(reconstructed, expected)

    def test_reconstruct_from_recast_covers_fallback_mapping_forms(self, tmp_path):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        A, B, C, D, E, F, G, H = sp.symbols("A B C D E F G H")
        Z1, Z2, m = sp.symbols("Z1 Z2 m")
        alt_z1 = sp.Symbol("Z1", real=True)
        validator.mapping = {
            A: A,
            B: Z1 * m,
            C: Z1 * m**2,
            D: 2 * Z1 * Z2,
            E: sp.sin(Z1) * Z2,
            F: m,
            G: alt_z1 + 1,
            H: Z1 + 1,
        }
        z_values = np.array([[2.0, 3.0], [5.0, 7.0]])

        reconstructed = validator._reconstruct_from_recast(
            z_values,
            [Z1, Z2],
            [A, B, C, D, E, F, G, H],
            {"m": 4.0},
            np.array([0.0, 1.0]),
            None,
        )

        expected = np.array([
            [0.0, 8.0, 32.0, 12.0, np.sin(2.0) * 3.0, 4.0, 3.0, 3.0],
            [0.0, 20.0, 80.0, 70.0, np.sin(5.0) * 7.0, 4.0, 6.0, 6.0],
        ])
        np.testing.assert_allclose(reconstructed, expected)

    def test_evaluate_recast_expression_covers_time_broadcast_and_missing_symbol(
        self, tmp_path
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        t_array = np.array([0.0, 1.0, 2.0])
        z_values = np.array([[2.0], [3.0], [4.0]])

        constant = validator._evaluate_expr_on_recast_trajectory(
            sp.Integer(5),
            z_values,
            [X],
            {},
            t_array,
        )
        param_only = validator._evaluate_expr_on_recast_trajectory(
            sp.Symbol("k"),
            z_values,
            [X],
            {"k": 7.0},
            t_array,
        )
        time_values = validator._evaluate_expr_on_recast_trajectory(
            sp.Symbol("time") + X,
            z_values,
            [X],
            {},
            t_array,
        )
        t_values = validator._evaluate_expr_on_recast_trajectory(
            sp.Symbol("t") + 1,
            z_values,
            [X],
            {},
            t_array,
        )

        np.testing.assert_allclose(constant, np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(param_only, np.array([7.0, 7.0, 7.0]))
        np.testing.assert_allclose(time_values, np.array([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(t_values, np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="missing value for symbol 'missing'"):
            validator._evaluate_expr_on_recast_trajectory(
                sp.Symbol("missing"),
                z_values,
                [X],
                {},
                t_array,
            )

    def test_algebraic_definition_collection_skips_clock_and_bad_assignment(
        self, tmp_path
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        T = sp.Symbol("T", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        validator.auxiliary_defs = {T: sp.Symbol("time"), Y: X + 1}
        validator.recast_ir.assignment_rules = {"bad_rule": object()}

        definitions = validator._algebraic_definitions_for_residuals()

        assert "T" not in definitions
        assert definitions == {"Y_1": X + 1}

    def test_algebraic_residuals_report_evaluation_and_constraint_errors(
        self, tmp_path
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        validator.recast_ir.assignment_rules = {}
        validator.recast_ir.algebraic_constraints = ["Y_1 - X - 1", object()]
        validator.auxiliary_defs = {
            Y: X + sp.Symbol("missing"),
            sp.Symbol("AUX_ONLY", positive=True): X + 2,
        }

        residuals, errors = validator._compute_algebraic_residual_norms(
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            [X, Y],
            {},
            np.array([0.0, 1.0]),
        )

        assert residuals["AUX_ONLY"]["max_abs"] == 0.0
        assert residuals["algebraic_constraint:1"]["max_abs"] == 0.0
        assert {error["constraint"] for error in errors} == {
            "Y_1",
            "algebraic_constraint:2",
        }

    def test_algebraic_manifold_check_reports_pass_and_failure(self, tmp_path, monkeypatch):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        validator.recast_state_vars = [X, Y]
        validator.auxiliary_defs = {Y: X + 1}
        validator.recast_ir.assignment_rules = {}
        validator.recast_ir.params = {}

        def passing_simulation(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0, 2.0], [2.0, 3.0]]),
                "backend": "fake_backend",
                "message": "",
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", passing_simulation)
        passed = validator.check_algebraic_manifold_preservation(threshold=1.0e-12)

        assert passed is not None
        assert passed.result == ValidationResult.PASS
        assert passed.metadata["backend"] == "fake_backend"

        def drifting_simulation(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0, 2.2], [2.0, 3.5]]),
                "backend": "fake_backend",
                "message": "",
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", drifting_simulation)
        failed = validator.check_algebraic_manifold_preservation(threshold=1.0e-3)

        assert failed is not None
        assert failed.result == ValidationResult.FAIL
        assert failed.counterexamples[0]["constraint"] == "Y_1"

    @pytest.mark.parametrize(
        ("unsupported_solver_requirement", "expected_result"),
        [
            pytest.param(False, ValidationResult.NOT_ATTEMPTED, id="solver-failed"),
            pytest.param(True, ValidationResult.UNSUPPORTED, id="unsupported-solver"),
        ],
    )
    def test_algebraic_manifold_simulation_failure_is_fail_closed(
        self, tmp_path, monkeypatch, unsupported_solver_requirement, expected_result
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        validator.recast_state_vars = [X, Y]
        validator.auxiliary_defs = {Y: X + 1}
        validator.recast_ir.assignment_rules = {}

        def failed_simulation(*args, **kwargs):
            return {
                "success": False,
                "t": np.array([]),
                "y": np.array([]),
                "backend": "fake_backend",
                "message": "residual solver failed",
                "unsupported_solver_requirement": unsupported_solver_requirement,
                "algebraic_residuals": {"Y_1": {"max_abs": 9.0}},
            }

        monkeypatch.setattr(validator, "_simulate_model", failed_simulation)

        result = validator.check_algebraic_manifold_preservation()

        assert result is not None
        assert result.result == expected_result
        assert "residual solver failed" in result.details
        assert result.metadata["backend"] == "fake_backend"
        assert result.metadata["residual_norms"] == {"Y_1": {"max_abs": 9.0}}

    def test_algebraic_manifold_evaluation_errors_are_inconclusive(
        self, tmp_path, monkeypatch
    ):
        original, recast = self._write_identity_pair(tmp_path)
        validator = RecastValidator(str(original), str(recast), parser="sbml")
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        validator.recast_state_vars = [X, Y]
        validator.auxiliary_defs = {Y: X + sp.Symbol("missing")}
        validator.recast_ir.assignment_rules = {}

        def successful_simulation(*args, **kwargs):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0, 2.0], [2.0, 3.0]]),
                "backend": "fake_backend",
                "message": "",
                "unsupported_solver_requirement": False,
                "algebraic_residuals": {},
            }

        monkeypatch.setattr(validator, "_simulate_model", successful_simulation)

        result = validator.check_algebraic_manifold_preservation()

        assert result is not None
        assert result.result == ValidationResult.INCONCLUSIVE
        assert result.metadata["errors"][0]["constraint"] == "Y_1"

    def test_numerical_pointwise_fails_on_mismatched_rate(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                X' = -2*k*X;
                k = 0.5;
                X = 1.0;
            end
        """)

        validator = RecastValidator(str(original), str(recast))
        result = validator.check_numerical_pointwise(n_samples=8, threshold=1.0e-9)

        assert result.result == ValidationResult.FAIL
        assert result.max_error > 1.0e-9
        assert result.counterexamples


class _NumericalHarness(NumericalValidationMixin):
    pass


def _make_numerical_harness(*, recast_multiplier: float = 1.0, k_value: float = 0.5):
    X = sp.Symbol("X", positive=True)
    Z = sp.Symbol("Z", positive=True)
    k = sp.Symbol("k", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -k * X}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {Z: -recast_multiplier * k * Z}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [Z]
    harness.mapping = {X: Z}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={"k": k_value},
        initials={Z: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {}
    harness.canonical_symbols = {"X": X, "Z": Z, "k": k}
    return harness


def _install_fake_jax(monkeypatch):
    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    for name in ("abs", "array", "concatenate", "max"):
        setattr(fake_jnp, name, getattr(np, name))

    def jacfwd(func):
        def jacobian(z_values):
            z_values = np.asarray(z_values, dtype=float)
            base = np.asarray(func(z_values), dtype=float)
            jac = np.zeros((base.size, z_values.size))
            step = 1.0e-6
            for idx in range(z_values.size):
                delta = np.zeros_like(z_values)
                delta[idx] = step
                plus = np.asarray(func(z_values + delta), dtype=float)
                minus = np.asarray(func(z_values - delta), dtype=float)
                jac[:, idx] = (plus - minus) / (2 * step)
            return jac

        return jacobian

    fake_jax.numpy = fake_jnp
    fake_jax.jacfwd = jacfwd
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)


def _make_time_clock_numerical_harness():
    X = sp.Symbol("X", positive=True)
    T = sp.Symbol("T", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    k = sp.Symbol("k", positive=True)
    time = sp.Symbol("time", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -X / (time + k)}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -X / Y, T: sp.Integer(1), Y: sp.Integer(1)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, T, Y]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={"k": 1.0},
        initials={X: 1.0, T: 1.0, Y: 2.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {T: time, Y: T + k}
    harness.canonical_symbols = {"X": X, "T": T, "Y_1": Y, "k": k}
    return harness


def _make_time_auxiliary_numerical_harness():
    X = sp.Symbol("X", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    k = sp.Symbol("k", positive=True)
    time = sp.Symbol("time", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -(time + k)}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -Y, Y: sp.Integer(0)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, Y]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={"k": 1.0},
        initials={X: 1.0, Y: 2.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {Y: time + k}
    harness.canonical_symbols = {"X": X, "Y_1": Y, "k": k}
    return harness


def _make_unresolved_auxiliary_numerical_harness():
    X = sp.Symbol("X", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    u = sp.Symbol("u", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -X}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -X, Y: sp.Integer(0)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, Y]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={},
        initials={X: 1.0, Y: 2.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {Y: X + u}
    harness.canonical_symbols = {"X": X, "Y_1": Y, "u": u}
    return harness


def _make_nested_auxiliary_numerical_harness():
    X = sp.Symbol("X", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    Z = sp.Symbol("Z_1", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -(X + 2)}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -Y, Y: sp.Integer(0), Z: sp.Integer(0)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, Y, Z]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={},
        initials={X: 1.0, Y: 3.0, Z: 2.0},
        assignment_rules={},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {Y: X + Z, Z: sp.Integer(2)}
    harness.canonical_symbols = {"X": X, "Y_1": Y, "Z_1": Z}
    return harness


def _make_assignment_rule_state_numerical_harness():
    X = sp.Symbol("X", positive=True)
    A = sp.Symbol("A", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -(X + 1)}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -A, A: sp.Integer(0)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, A]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={},
        initials={X: 1.0, A: 2.0},
        assignment_rules={"A": "X + 1"},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {}
    harness.canonical_symbols = {"X": X, "A": A}
    return harness


def _make_assignment_rule_auxiliary_numerical_harness():
    X = sp.Symbol("X", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    ell = sp.Symbol("l", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -(X + 2)}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -Y, Y: sp.Integer(0)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, Y]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={"l": 0.0},
        initials={X: 1.0, Y: 3.0},
        assignment_rules={"l": "2"},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {Y: X + ell}
    harness.canonical_symbols = {"X": X, "Y_1": Y, "l": ell}
    return harness


def _make_jax_auxiliary_harness():
    X = sp.Symbol("X", positive=True)
    T = sp.Symbol("T", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    k = sp.Symbol("k", positive=True)
    u = sp.Symbol("u", positive=True)
    time = sp.Symbol("time", positive=True)

    harness = _NumericalHarness()
    harness.orig_odes = {X: -X}
    harness.orig_odes_expanded = dict(harness.orig_odes)
    harness.recast_odes = {X: -X, T: sp.Integer(1), Y: sp.Integer(0)}
    harness.recast_odes_expanded = dict(harness.recast_odes)
    harness.recast_state_vars = [X, T, Y]
    harness.mapping = {X: X}
    harness.orig_ir = SimpleNamespace(
        initials={X: 1.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.recast_ir = SimpleNamespace(
        params={"k": 0.5, "u": 0.25},
        initials={X: 1.0, T: 1.0, Y: 2.0},
        sim_t_start=None,
        sim_t_end=None,
    )
    harness.auxiliary_defs = {T: time, Y: X + k + u}
    harness.canonical_symbols = {"X": X, "T": T, "Y_1": Y, "k": k, "u": u}
    return harness


class TestNumericalValidationMixinDirect:
    def test_numerical_pointwise_passes_identity_mapping(self):
        harness = _make_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=6, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert "6 samples" in result.details
        assert result.metadata["sample_seed"] == 42
        assert result.metadata["n_samples"] == 6
        assert result.metadata["parameter_values"] == {"k": 0.5}
        assert result.metadata["sampling"]["state_variables"]["Z"] == {
            "min": 0.01,
            "max": 10.0,
            "source": "positive_initial",
            "initial": 1.0,
        }

    def test_numerical_pointwise_reports_subphase_progress(self):
        harness = _make_numerical_harness()
        phases = []

        result = harness.check_numerical_pointwise(
            n_samples=2,
            threshold=1.0e-10,
            progress_callback=phases.append,
        )

        assert result.result == ValidationResult.PASS
        for expected in [
            "numerical_setup",
            "numerical_collect_symbols",
            "numerical_parameters",
            "numerical_sampling_metadata",
            "numerical_jacobian",
            "numerical_lambdify_jacobian",
            "numerical_lambdify_recast_odes",
            "numerical_lambdify_original_odes",
            "numerical_computed_definitions",
            "numerical_preflight",
            "numerical_sample_evaluation",
            "numerical_result_aggregation",
        ]:
            assert expected in phases
        assert any(
            phase.startswith("numerical_sample_evaluation:") for phase in phases
        )

    def test_numerical_pointwise_sample_budget_reports_complexity(self):
        harness = _make_numerical_harness()

        result = harness.check_numerical_pointwise(
            n_samples=2,
            threshold=1.0e-10,
            sample_evaluation_timeout=0.0,
        )

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "numerical_complexity"
        assert "complexity budget" in result.details
        diagnostic = result.metadata["diagnostics"][0]
        assert diagnostic["reason"] == "numerical_complexity"
        assert diagnostic["phase"] == "numerical_sample_evaluation"
        assert diagnostic["active_subphase"] == "sample_generation"
        assert diagnostic["sample_index"] == 0
        assert diagnostic["samples_completed"] == 0
        assert diagnostic["n_samples"] == 2
        assert diagnostic["limit_seconds"] == 0.0

    def test_numerical_pointwise_expression_complexity_fails_closed(self):
        harness = _make_numerical_harness()

        result = harness.check_numerical_pointwise(
            n_samples=2,
            threshold=1.0e-10,
            expression_complexity_limit=0,
        )

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "numerical_complexity"
        assert "complexity budget" in result.details
        diagnostic = result.metadata["diagnostics"][0]
        assert diagnostic["reason"] == "numerical_complexity"
        assert diagnostic["phase"] == "numerical_preflight"
        assert diagnostic["active_subphase"] == "expression_complexity"
        assert diagnostic["side"] == "recast"
        assert diagnostic["expression_label"] == "Z"
        assert diagnostic["expression_ops"] > diagnostic["max_expression_ops"]
        assert diagnostic["free_symbol_count"] == 2

    def test_numerical_pointwise_skips_zero_jacobian_lambdify(self, monkeypatch):
        import ssys._validator.numerical as numerical_module

        real_lambdify = numerical_module.lambdify

        def reject_zero_jacobian_lambdify(args, expr, modules):
            if expr == 0:
                raise AssertionError("zero Jacobian entries should not be lambdified")
            return real_lambdify(args, expr, modules)

        X = sp.Symbol("X", positive=True)
        Z = sp.Symbol("Z", positive=True)
        Y = sp.Symbol("Y", positive=True)
        k = sp.Symbol("k", positive=True)
        harness = _NumericalHarness()
        harness.orig_odes = {X: -k * X}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {Z: -k * Z, Y: -Y}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_state_vars = [Z, Y]
        harness.mapping = {X: Z}
        harness.orig_ir = SimpleNamespace(
            initials={X: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.recast_ir = SimpleNamespace(
            params={"k": 0.5},
            initials={Z: 1.0, Y: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.auxiliary_defs = {}
        harness.canonical_symbols = {"X": X, "Z": Z, "Y": Y, "k": k}
        monkeypatch.setattr(numerical_module, "lambdify", reject_zero_jacobian_lambdify)

        result = harness.check_numerical_pointwise(n_samples=2, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0

    def test_numerical_pointwise_expands_domain_from_model_initials(self):
        harness = _make_numerical_harness()
        Z = harness.recast_state_vars[0]
        harness.recast_ir.initials[Z] = 100.0

        result = harness.check_numerical_pointwise(n_samples=2, threshold=1.0e-10)

        z_domain = result.metadata["sampling"]["state_variables"]["Z"]
        assert result.result == ValidationResult.PASS
        assert z_domain["source"] == "positive_initial"
        assert z_domain["initial"] == 100.0
        assert z_domain["max"] == 1000.0

    def test_numerical_pointwise_records_simulation_time_domain(self):
        harness = _make_time_auxiliary_numerical_harness()
        harness.orig_ir.sim_t_start = 0.0
        harness.orig_ir.sim_t_end = 2.0

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.metadata["sampling"]["time"] == {
            "min": 1.0e-12,
            "max": 2.0,
            "source": "simulation_metadata",
        }

    def test_numerical_pointwise_invalid_domain_is_reported(self):
        harness = _make_numerical_harness()

        result = harness.check_numerical_pointwise(domain_min=0.0, domain_max=10.0)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "invalid_sampling_domain"
        assert "Invalid numerical sampling domain" in result.details

    def test_numerical_pointwise_failure_reports_counterexample(self):
        harness = _make_numerical_harness(recast_multiplier=2.0)

        result = harness.check_numerical_pointwise(n_samples=4, threshold=1.0e-12)

        assert result.result == ValidationResult.FAIL
        assert result.max_error > 0.0
        assert result.counterexamples
        assert {"Z", "lhs", "rhs", "diff"} <= set(result.counterexamples[0])

    def test_numerical_pointwise_time_clock_substitution_passes(self):
        harness = _make_time_clock_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error < 1.0e-10
        assert "time" in harness.canonical_symbols

    def test_numerical_pointwise_ignores_exported_time_parameter_binding(self):
        harness = _make_time_clock_numerical_harness()
        harness.recast_ir.params["time"] = 0.0

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error < 1.0e-10
        assert "time" not in result.metadata["parameter_values"]

    def test_numerical_pointwise_canonicalizes_time_symbol_case_variants(self):
        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        lower_time = sp.Symbol("time", positive=True)
        upper_time = sp.Symbol("Time", positive=True)

        harness = _NumericalHarness()
        harness.orig_odes = {X: -X / (lower_time + k)}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {X: -X / (upper_time + k)}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_state_vars = [X]
        harness.mapping = {X: X}
        harness.orig_ir = SimpleNamespace(
            initials={X: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.recast_ir = SimpleNamespace(
            params={"k": 1.0, "Time": 0.0},
            initials={X: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.auxiliary_defs = {}
        harness.canonical_symbols = {"X": X, "k": k, "time": lower_time, "Time": upper_time}

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert "Time" not in result.metadata["parameter_values"]

    def test_numerical_pointwise_canonicalizes_recast_lower_t_clock(self):
        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        time = sp.Symbol("time", positive=True)
        t = sp.Symbol("t", positive=True)

        harness = _NumericalHarness()
        harness.orig_odes = {X: -X / (time + k)}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {X: -X / (t + k)}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_state_vars = [X]
        harness.mapping = {X: X}
        harness.orig_ir = SimpleNamespace(
            initials={X: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.recast_ir = SimpleNamespace(
            params={"k": 1.0, "t": 0.0},
            initials={X: 1.0},
            assignment_rules={"X0": "sin(t)"},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.auxiliary_defs = {}
        harness.canonical_symbols = {"X": X, "k": k, "time": time, "t": t}

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert "t" not in result.metadata["parameter_values"]
        assert "time" in result.metadata["sampling"]

    def test_numerical_pointwise_keeps_declared_t_parameter(self):
        X = sp.Symbol("X", positive=True)
        t = sp.Symbol("t", positive=True)

        harness = _NumericalHarness()
        harness.orig_odes = {X: -t * X}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {X: -t * X}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_state_vars = [X]
        harness.mapping = {X: X}
        harness.orig_ir = SimpleNamespace(
            params={"t": 2.0},
            initials={X: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.recast_ir = SimpleNamespace(
            params={"t": 2.0},
            initials={X: 1.0},
            assignment_rules={},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.auxiliary_defs = {}
        harness.canonical_symbols = {"X": X, "t": t}

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert result.metadata["parameter_values"]["t"] == 2.0
        assert "time" not in result.metadata["sampling"]

    def test_numerical_pointwise_keeps_t_state_variable(self):
        t = sp.Symbol("t", positive=True)
        k = sp.Symbol("k", positive=True)

        harness = _NumericalHarness()
        harness.orig_odes = {t: -k * t}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {t: -k * t}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_state_vars = [t]
        harness.mapping = {t: t}
        harness.orig_ir = SimpleNamespace(
            params={"k": 1.0},
            initials={t: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.recast_ir = SimpleNamespace(
            params={"k": 1.0, "t": 0.0},
            initials={t: 1.0},
            assignment_rules={},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.auxiliary_defs = {}
        harness.canonical_symbols = {"t": t, "k": k}

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert "t" not in result.metadata["parameter_values"]
        assert "time" not in result.metadata["sampling"]

    def test_numerical_pointwise_samples_time_for_auxiliary_definition(self):
        harness = _make_time_auxiliary_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error < 1.0e-10

    def test_numerical_pointwise_auxiliary_unresolved_symbol_is_diagnostic(self):
        harness = _make_unresolved_auxiliary_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=2, threshold=1.0e-10)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "unresolved_parameter"
        assert result.metadata["diagnostics"][0]["unresolved_symbols"] == ["u"]

    def test_numerical_pointwise_evaluates_nested_auxiliaries_in_dependency_order(self):
        harness = _make_nested_auxiliary_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0

    def test_numerical_pointwise_computes_recast_assignment_rule_state(self):
        harness = _make_assignment_rule_state_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0

    def test_numerical_pointwise_expands_assignment_rules_in_auxiliary_defs(self):
        harness = _make_assignment_rule_auxiliary_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=3, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0

    def test_numerical_pointwise_uses_safe_scoped_parameter_alias(self):
        harness = _make_numerical_harness()
        original_param = sp.Symbol("reaction1_vi", positive=True)
        recast_param = sp.Symbol("reaction1__vi", positive=True)
        x = next(iter(harness.orig_odes))
        z = harness.recast_state_vars[0]

        harness.orig_odes = {x: -original_param * x}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {z: -recast_param * z}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.orig_ir.params = {}
        harness.recast_ir.params = {"reaction1__vi": 0.5}
        harness.canonical_symbols.update({
            "reaction1_vi": original_param,
            "reaction1__vi": recast_param,
        })

        result = harness.check_numerical_pointwise(n_samples=4, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert result.metadata["parameter_aliases"] == {
            "reaction1_vi": "reaction1__vi"
        }
        assert result.metadata["parameter_values"]["reaction1_vi"] == 0.5

    def test_numerical_pointwise_uses_sanitized_parameter_alias(self):
        harness = _make_numerical_harness()
        original_param = sp.Symbol("Pi", positive=True)
        recast_param = sp.Symbol("Pi_var", positive=True)
        x = next(iter(harness.orig_odes))
        z = harness.recast_state_vars[0]

        harness.orig_odes = {x: -original_param * x}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {z: -recast_param * z}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_ir.params = {"Pi_var": 0.5}
        harness.canonical_symbols.update({"Pi": original_param, "Pi_var": recast_param})

        result = harness.check_numerical_pointwise(n_samples=4, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0
        assert result.metadata["parameter_aliases"] == {"Pi": "Pi_var"}
        assert result.metadata["parameter_values"]["Pi"] == 0.5

    def test_numerical_pointwise_treats_lowercase_pi_as_constant(self):
        X = sp.Symbol("X", positive=True)
        T = sp.Symbol("T", positive=True)
        A = sp.Symbol("A", positive=True)
        period = sp.Symbol("period", positive=True)

        harness = _NumericalHarness()
        forcing = 1 + sp.sin(2 * sp.pi * T / period)
        harness.orig_odes = {T: sp.Integer(1), X: -forcing}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes = {T: sp.Integer(1), X: -A, A: sp.Integer(0)}
        harness.recast_odes_expanded = dict(harness.recast_odes)
        harness.recast_state_vars = [T, X, A]
        harness.mapping = {T: T, X: X}
        harness.orig_ir = SimpleNamespace(
            params={"period": 24.0},
            initials={T: 1.0, X: 1.0},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.recast_ir = SimpleNamespace(
            params={"period": 24.0},
            initials={T: 1.0, X: 1.0, A: 1.25},
            assignment_rules={"A": "1 + sin(2 * pi * T / period)"},
            sim_t_start=None,
            sim_t_end=None,
        )
        harness.auxiliary_defs = {}
        harness.canonical_symbols = {"T": T, "X": X, "A": A, "period": period}

        result = harness.check_numerical_pointwise(n_samples=4, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error < 1.0e-10
        assert "pi" not in result.metadata["parameter_values"]

    def test_numerical_pointwise_relational_function_is_structured_unsupported(self):
        harness = _make_numerical_harness()
        z = harness.recast_state_vars[0]
        lt = sp.Function("lt")
        relation_expr = lt(z, sp.Integer(1))
        harness.orig_odes = {next(iter(harness.orig_odes)): -relation_expr}
        harness.orig_odes_expanded = dict(harness.orig_odes)
        harness.recast_odes[z] = -relation_expr
        harness.recast_odes_expanded[z] = -relation_expr

        result = harness.check_numerical_pointwise(n_samples=1)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "unsupported_feature"
        assert result.metadata["diagnostics"][0]["unsupported_functions"] == ["lt"]

    def test_numerical_pointwise_floor_auxiliary_is_structured_unsupported(self):
        harness = _make_time_auxiliary_numerical_harness()
        y = sp.Symbol("Y_1", positive=True)
        time = sp.Symbol("time", positive=True)
        harness.auxiliary_defs = {y: sp.floor(time)}
        harness.canonical_symbols["time"] = time

        result = harness.check_numerical_pointwise(n_samples=1)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "unsupported_feature"
        diagnostic = result.metadata["diagnostics"][0]
        assert diagnostic["expression_label"] == "Y_1"
        assert diagnostic["unsupported_functions"] == ["floor"]

    def test_numerical_pointwise_unparsed_auxiliary_fails_closed(self):
        harness = _make_numerical_harness()
        harness.auxiliary_definition_parse_errors = [
            {
                "auxiliary": "Bad",
                "expression": "X +",
                "exception": "invalid syntax",
            }
        ]

        result = harness.check_numerical_pointwise(n_samples=1)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "unsupported_feature"
        diagnostic = result.metadata["diagnostics"][0]
        assert diagnostic["reason"] == "auxiliary_definition_parse_failed"
        assert diagnostic["unparsed_auxiliary_definitions"][0]["auxiliary"] == "Bad"

    def test_numerical_pointwise_evaluates_symbolic_jacobian_result(self, monkeypatch):
        import ssys._validator.numerical as numerical_module

        real_lambdify = numerical_module.lambdify

        def lambdify_symbolic_unit_jacobian(args, expr, modules):
            if expr == 1:
                return lambda *values: sp.Integer(1)
            return real_lambdify(args, expr, modules)

        monkeypatch.setattr(
            numerical_module,
            "lambdify",
            lambdify_symbolic_unit_jacobian,
        )
        harness = _make_numerical_harness()

        result = harness.check_numerical_pointwise(n_samples=2, threshold=1.0e-10)

        assert result.result == ValidationResult.PASS
        assert result.max_error == 0.0

    def test_numerical_pointwise_nonfinite_values_are_diagnostic(self):
        harness = _make_numerical_harness(k_value=float("inf"))

        result = harness.check_numerical_pointwise(n_samples=1)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "nonfinite_sample"
        assert "Non-finite value" in result.details

    def test_numerical_pointwise_singular_surface_produces_nonfinite_diagnostic(self):
        harness = _make_numerical_harness()
        z = harness.recast_state_vars[0]
        singular_rhs = sp.Pow(sp.Add(z, -z, evaluate=False), -1, evaluate=False)
        harness.recast_odes[z] = singular_rhs
        harness.recast_odes_expanded[z] = singular_rhs

        result = harness.check_numerical_pointwise(n_samples=1)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "nonfinite_sample"
        assert "Non-finite value" in result.details

    def test_numerical_pointwise_jax_passes_with_fake_backend(self, monkeypatch):
        _install_fake_jax(monkeypatch)

        harness = _make_numerical_harness()
        result = harness.check_numerical_pointwise_jax(n_samples=4, threshold=1.0e-9)

        assert result.result == ValidationResult.PASS
        assert result.max_error < 1.0e-9
        assert "JAX autodiff" in result.details

    def test_numerical_pointwise_jax_without_jax_is_not_attempted(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jax", None)
        monkeypatch.setitem(sys.modules, "jax.numpy", None)
        harness = _make_numerical_harness()

        result = harness.check_numerical_pointwise_jax()

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert "JAX not available" in result.details

    def test_numerical_pointwise_jax_failure_reports_counterexample(self, monkeypatch):
        _install_fake_jax(monkeypatch)
        harness = _make_numerical_harness(recast_multiplier=2.0)

        result = harness.check_numerical_pointwise_jax(n_samples=3, threshold=1.0e-12)

        assert result.result == ValidationResult.FAIL
        assert result.max_error > 1.0e-12
        assert result.counterexamples
        assert {"Z", "lhs", "rhs", "diff"} <= set(result.counterexamples[0])

    def test_numerical_pointwise_jax_nonfinite_values_are_diagnostic(self, monkeypatch):
        _install_fake_jax(monkeypatch)
        harness = _make_numerical_harness(k_value=float("inf"))

        result = harness.check_numerical_pointwise_jax(n_samples=1)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert "Non-finite value" in result.details

    def test_numerical_pointwise_jax_classifies_clock_and_auxiliary_variables(
        self, monkeypatch
    ):
        _install_fake_jax(monkeypatch)
        harness = _make_jax_auxiliary_harness()

        result = harness.check_numerical_pointwise_jax(n_samples=2, threshold=1.0e-9)

        assert result.result == ValidationResult.PASS
        assert result.max_error < 1.0e-9


class TestRecastValidator:
    """Tests for RecastValidator class."""

    @pytest.fixture
    def simple_model_paths(self, tmp_path):
        """Create simple original and recast model files."""
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                // ========================================================================
                // VARIABLE MAPPING
                // ========================================================================
                // X = Z_1
                // ========================================================================
                species Z_1;
                Z_1' = -k * Z_1;
                k = 0.5;
                Z_1 = 1.0;
            end
        """)

        return str(original), str(recast)

    def test_validator_init(self, simple_model_paths):
        """Test validator initialization."""
        orig, recast = simple_model_paths

        validator = RecastValidator(orig, recast)

        assert validator.original_file == orig
        assert validator.recast_file == recast
        mapping_by_name = {orig.name: mapped for orig, mapped in validator.factor_map.items()}
        assert str(mapping_by_name["X"]) == "Z_1"


    def test_validator_symbolic_check(self, simple_model_paths):
        """Test symbolic equivalence check."""
        orig, recast = simple_model_paths

        validator = RecastValidator(orig, recast)
        result = validator.check_symbolic_equivalence(timeout=5.0)

        assert result.name == "symbolic_equivalence"
        assert result.result == ValidationResult.PASS
        assert "exact equivalence" in result.details


class TestRecastValidatorAuxiliaryExtraction:
    """Tests for auxiliary definition extraction."""

    def test_identity_simplifier_handles_float_roundoff_before_slow_simplify(
        self, monkeypatch
    ):
        """Avoid expensive general simplification for tiny float residuals."""
        x, y = sp.symbols("x y", positive=True)
        residual = (
            sp.Float("0.1") * x / y
            + sp.Float("0.2") * x / y
            - sp.Float("0.3") * x / y
        )

        def fail_slow_simplifier(*args, **kwargs):
            raise AssertionError("slow simplification should not be needed")

        monkeypatch.setattr(validator_common.sp, "nsimplify", fail_slow_simplifier)
        monkeypatch.setattr(validator_common.sp, "simplify", fail_slow_simplifier)

        assert validator_common._simplify_identity_difference(residual) == 0

    def test_auxiliary_identity_reports_subphase_progress(self):
        x = sp.Symbol("X", positive=True)
        aux = sp.Symbol("Y_1", positive=True)
        phases = []
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = phases.append
        validator.orig_odes = {x: -x}
        validator.orig_odes_expanded = {x: -x}
        validator.recast_odes = {x: -x, aux: -x}
        validator.recast_state_vars = [x, aux]
        validator.mapping = {}
        validator.auxiliary_defs = {aux: x + 1}
        validator.canonical_symbols = {"X": x, "Y_1": aux}
        validator.recast_ir = SimpleNamespace(assignment_rules={}, params={})

        tests = validator.check_auxiliary_identities()

        assert tests[0].result == ValidationResult.PASS
        assert "auxiliaries_ode_identity:Y_1" in phases
        assert "auxiliaries_ode_identity_build_expected:Y_1" in phases
        assert "auxiliaries_equivalence:ode_auxiliary_identity:Y_1" in phases
        assert any(
            phase.startswith("auxiliaries_simplify:ode_auxiliary_identity:Y_1")
            for phase in phases
        )

    def test_auxiliary_identity_checks_later_cheap_candidates_before_slow_simplify(
        self, monkeypatch
    ):
        x = sp.Symbol("X", positive=True)
        aux = sp.Symbol("Y_1", positive=True)
        phases = []
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = phases.append
        validator.auxiliary_defs = {aux: x + 1}
        validator.recast_state_vars = [x, aux]
        validator.canonical_symbols = {"X": x, "Y_1": aux}
        validator.recast_ir = SimpleNamespace(params={})

        def fail_slow_simplifier(*args, **kwargs):
            raise AssertionError("later cheap candidate should be checked first")

        monkeypatch.setattr(
            validator_mapping,
            "_simplify_identity_difference",
            fail_slow_simplifier,
        )

        equivalent, residual = validator._expressions_equivalent(
            aux,
            x + 1,
            context="ode_auxiliary_identity:Y_1",
        )

        assert equivalent is True
        assert residual == 0
        assert "auxiliaries_simplify:ode_auxiliary_identity:Y_1:candidate_0" in phases
        assert "auxiliaries_simplify:ode_auxiliary_identity:Y_1:candidate_1" in phases

    def test_auxiliary_identity_complexity_reports_structured_reason(self):
        x = sp.Symbol("X", positive=True)
        exponent = sp.Symbol("a", positive=True)
        aux = sp.Symbol("Y_1", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = None
        validator.orig_odes = {x: x}
        validator.orig_odes_expanded = {x: x}
        validator.recast_odes = {x: x, aux: sp.Float("2.0") * x**exponent}
        validator.recast_state_vars = [x, aux]
        validator.mapping = {}
        validator.auxiliary_defs = {aux: x**exponent + 1}
        validator.canonical_symbols = {"X": x, "Y_1": aux, "a": exponent}
        validator.recast_ir = SimpleNamespace(assignment_rules={}, params={"a": "a"})

        tests = validator.check_auxiliary_identities()

        assert tests[0].result == ValidationResult.INCONCLUSIVE
        assert tests[0].reason == "auxiliary_complexity"
        assert tests[0].metadata["auxiliary"] == "Y_1"
        assert tests[0].metadata["operation_count"] > 0

    def test_auxiliary_identity_complexity_reports_candidate_index(self, monkeypatch):
        symbols = sp.symbols("x0:260", positive=True)
        aux = sp.Symbol("Y_1", positive=True)
        definition = sum(symbols, sp.Integer(0))
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = None
        validator.orig_odes_expanded = {}
        validator.recast_odes = {symbol: symbol for symbol in symbols}
        validator.recast_odes[aux] = 2 * definition
        validator.recast_state_vars = [*symbols, aux]
        validator.mapping = {}
        validator.auxiliary_defs = {aux: definition}
        validator.canonical_symbols = {str(symbol): symbol for symbol in [*symbols, aux]}
        validator.recast_ir = SimpleNamespace(assignment_rules={}, params={})

        def fail_slow_simplifier(*args, **kwargs):
            raise AssertionError("high-complexity residual should be bounded")

        monkeypatch.setattr(
            validator_mapping,
            "_simplify_identity_difference",
            fail_slow_simplifier,
        )

        tests = validator._ode_auxiliary_identity_tests()

        assert tests[0].result == ValidationResult.INCONCLUSIVE
        assert tests[0].reason == "auxiliary_complexity"
        assert tests[0].metadata["auxiliary"] == "Y_1"
        assert tests[0].metadata["candidate_index"] == 0
        assert tests[0].metadata["active_subphase"] == "cheap_simplification"
        assert tests[0].metadata["operation_count"] > 250
        assert tests[0].metadata["operation_threshold"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_OPS
        )
        assert tests[0].metadata["free_symbol_threshold"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
        )
        assert tests[0].metadata["max_operation_count"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_OPS
        )
        assert tests[0].metadata["max_free_symbol_count"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
        )
        assert tests[0].metadata["risky_symbolic_power_guard"] is False
        assert tests[0].metadata["elapsed_seconds"] is not None

    def test_assignment_auxiliary_complexity_reports_threshold_metadata(
        self, monkeypatch
    ):
        symbols = sp.symbols("x0:260", positive=True)
        aux = sp.Symbol("Y_1", positive=True)
        definition = sum(symbols, sp.Integer(0))
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = None
        validator.orig_odes = {}
        validator.mapping = {}
        validator.auxiliary_defs = {aux: definition}
        validator.recast_state_vars = [*symbols]
        validator.canonical_symbols = {str(symbol): symbol for symbol in [*symbols, aux]}
        validator.recast_ir = SimpleNamespace(
            assignment_rules={"Y_1": str(2 * definition)},
            params={},
        )

        def fail_slow_simplifier(*args, **kwargs):
            raise AssertionError("assignment auxiliary complexity should be bounded")

        monkeypatch.setattr(
            validator_mapping,
            "_simplify_identity_difference",
            fail_slow_simplifier,
        )

        tests = validator._assignment_identity_tests()

        assert len(tests) == 1
        assert tests[0].result == ValidationResult.INCONCLUSIVE
        assert tests[0].reason == "auxiliary_complexity"
        assert tests[0].metadata["rule"] == "Y_1"
        assert tests[0].metadata["kind"] == "assignment_auxiliary"
        assert tests[0].metadata["active_subphase"] == "cheap_simplification"
        assert tests[0].metadata["candidate_index"] == 0
        assert tests[0].metadata["operation_count"] > 250
        assert tests[0].metadata["free_symbol_count"] == 260
        assert tests[0].metadata["operation_threshold"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_OPS
        )
        assert tests[0].metadata["free_symbol_threshold"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
        )
        assert tests[0].metadata["max_operation_count"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_OPS
        )
        assert tests[0].metadata["max_free_symbol_count"] == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
        )

    def test_auxiliary_equivalence_bounds_large_candidate_generation(self):
        symbols = sp.symbols("x0:760", positive=True)
        residual = sum(symbols, sp.Integer(0))
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = None
        validator.auxiliary_defs = {}
        validator.recast_state_vars = list(symbols)
        validator.canonical_symbols = {str(symbol): symbol for symbol in symbols}
        validator.recast_ir = SimpleNamespace(params={})

        with pytest.raises(validator_mapping.AuxiliaryIdentityComplexityError) as exc:
            validator._expressions_equivalent(
                residual,
                sp.Integer(0),
                context="ode_auxiliary_identity:Y_1",
            )

        assert exc.value.candidate_index == 0
        assert exc.value.active_subphase == "candidate_generation"
        assert exc.value.operation_count > 750
        assert exc.value.operation_threshold == (
            validator_mapping.AUXILIARY_CANDIDATE_GENERATION_MAX_OPS
        )
        assert exc.value.free_symbol_threshold is None

    def test_auxiliary_equivalence_bounds_float_risky_preflight(self, monkeypatch):
        x, exponent = sp.symbols("X a", positive=True)
        residual = sp.Float("2.0") * x**exponent + sp.Float("1.0") * x
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = None
        validator.auxiliary_defs = {}
        validator.recast_state_vars = [x]
        validator.canonical_symbols = {"X": x, "a": exponent}
        validator.recast_ir = SimpleNamespace(params={})

        def fail_cheap_simplification(*args, **kwargs):
            raise AssertionError("float/risky preflight should run before simplification")

        monkeypatch.setattr(
            validator_mapping,
            "_cheap_zero_simplification",
            fail_cheap_simplification,
        )

        with pytest.raises(validator_mapping.AuxiliaryIdentityComplexityError) as exc:
            validator._expressions_equivalent(
                residual,
                sp.Integer(0),
                context="ode_auxiliary_identity:Y_1",
            )

        assert exc.value.candidate_index == 0
        assert exc.value.active_subphase == "float_simplification_preflight"
        assert exc.value.operation_count > 0
        assert exc.value.max_operation_count == validator_mapping.AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS
        assert exc.value.max_free_symbol_count == validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
        assert exc.value.operation_threshold == validator_mapping.AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS
        assert exc.value.free_symbol_threshold == (
            validator_mapping.AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
        )
        assert exc.value.risky_symbolic_power_guard is True
        assert exc.value.elapsed_seconds is not None

    def test_auxiliary_equivalence_allows_small_numeric_fractional_power(self):
        y_1, z2, z_1, u_1 = sp.symbols("Y_1 Z2 Z_1 u_1", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.progress_callback = None
        validator.auxiliary_defs = {
            z_1: sp.sqrt(z2**2 + 1),
            u_1: z2**2 + 1,
        }
        validator.recast_state_vars = [y_1, z2, z_1, u_1]
        validator.canonical_symbols = {
            "Y_1": y_1,
            "Z2": z2,
            "Z_1": z_1,
            "u_1": u_1,
        }
        validator.recast_ir = SimpleNamespace(params={})

        lhs = z2 * u_1 ** sp.Float("0.5") / (y_1 * sp.sqrt(z2**2 + 1))
        rhs = z2 * u_1 ** sp.Float("0.5") / (y_1 * z_1)

        equivalent, residual = validator._expressions_equivalent(
            lhs,
            rhs,
            context="ode_auxiliary_identity:Z_1",
        )

        assert equivalent is True
        assert residual == 0

    def test_auxiliary_inference_skips_name_matched_existing_definition(self):
        """Avoid expensive equality inference for auxiliaries already defined by name."""

        class ExplodingExpr:
            free_symbols = set()

            def equals(self, other):
                raise AssertionError("name-matched auxiliary should have been skipped")

        validator = RecastValidator.__new__(RecastValidator)
        original_x = sp.Symbol("X", positive=True)
        recast_aux = sp.Symbol("Y_1", positive=True)
        comment_aux = sp.Symbol("Y_1")
        validator.factor_map = {comment_aux: sp.Symbol("X") + 1}
        validator.orig_odes = {original_x: original_x}
        validator.recast_odes = {
            original_x: original_x,
            recast_aux: ExplodingExpr(),
        }

        validator._infer_auxiliary_definitions()


class TestAuxiliaryIdentityValidation:
    """Tests for auxiliary and observable identity validation."""

    def test_assignment_rule_auxiliary_matches_definition(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -X/(K + X);
                K = 1;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X;
                // ========================================================================
                // AUXILIARY DEFINITIONS (for lifted variables)
                // ========================================================================
                // Y_1 := K + X
                // ========================================================================
                Y_1 := K + X;
                X' = -X/Y_1;
                K = 1;
                X = 1;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
        )

        assignment_tests = [
            test for test in report.auxiliary_tests if test.name == "assignment_auxiliary:Y_1"
        ]
        assert assignment_tests
        assert assignment_tests[0].result == ValidationResult.PASS
        assert report.overall_pass is True

    def test_ode_auxiliary_lifted_denominator_matches_definition(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -X/(K + X);
                K = 1;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X, Y_1;
                // ========================================================================
                // AUXILIARY DEFINITIONS (for lifted variables)
                // ========================================================================
                // Y_1 := K + X
                // ========================================================================
                X' = -X/Y_1;
                Y_1' = -X/Y_1;
                K = 1;
                X = 1;
                Y_1 = 2;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
        )

        ode_tests = [
            test for test in report.auxiliary_tests if test.name == "ode_auxiliary_identity:Y_1"
        ]
        assert ode_tests
        assert ode_tests[0].result == ValidationResult.PASS
        assert report.overall_pass is True

    def test_ode_auxiliary_composite_function_matches_definition(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = exp(X);
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species X, Z_1;
                // ========================================================================
                // AUXILIARY DEFINITIONS (for lifted variables)
                // ========================================================================
                // Z_1 := exp(X)
                // ========================================================================
                X' = Z_1;
                Z_1' = Z_1^2;
                X = 1;
                Z_1 = 2.718281828;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
        )

        ode_tests = [
            test for test in report.auxiliary_tests if test.name == "ode_auxiliary_identity:Z_1"
        ]
        assert ode_tests
        assert ode_tests[0].result == ValidationResult.PASS
        assert report.overall_pass is True

    def test_clock_auxiliary_matches_definition(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -X / (time + 1)
            X = 1.0
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
            // ============================================================
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ============================================================
            // T := time
            // Y_1 := T + 1
            // ============================================================

            T' = 1
            Y_1' = 1
            X' = -X / Y_1

            T = 0
            Y_1 = 1
            X = 1.0
            end
        """)

        validator = RecastValidator(str(original), str(recast))
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
        )

        ode_results = {
            test.name: test.result
            for test in report.auxiliary_tests
            if test.name.startswith("ode_auxiliary_identity:")
        }
        assert ode_results["ode_auxiliary_identity:T"] == ValidationResult.PASS
        assert ode_results["ode_auxiliary_identity:Y_1"] == ValidationResult.PASS
        assert report.overall_pass is True

    def test_observable_assignment_matches_variable_mapping(self, tmp_path):
        original = tmp_path / "original.ant"
        original.write_text("""
            model original()
                species X;
                X' = -k*X;
                k = 0.5;
                X = 1;
            end
        """)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
                species Z_1;
                // ========================================================================
                // VARIABLE MAPPING
                // ========================================================================
                // X = Z_1
                // ========================================================================
                X := Z_1;
                Z_1' = -k*Z_1;
                k = 0.5;
                Z_1 = 1;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        report = validator.validate(
            run_symbolic=False,
            run_numerical=False,
            run_trajectory=False,
        )

        observable_tests = [
            test for test in report.auxiliary_tests if test.name == "observable_mapping:X"
        ]
        assert observable_tests
        assert observable_tests[0].result == ValidationResult.PASS
        assert report.overall_pass is True


class TestRecastValidatorEdgeCases:
    """Tests for edge cases in validation."""

    def test_validator_with_refusal(self, tmp_path):
        """Test validator when recast is a refusal."""
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            // NOTE: Canonical S-system recast was not attempted because:
            //   Model contains unsupported features
            model refused
            end
        """)

        validator = RecastValidator(str(original), str(recast))

        reason = validator._extract_refusal_reason(recast.read_text())
        assert reason == "Model contains unsupported features"

    def test_validator_missing_mapping(self, tmp_path):
        """Test validator when mapping comment is missing."""
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast
                species Z_1;
                Z_1' = -k * Z_1
                k = 0.5
                Z_1 = 1.0
            end
        """)

        validator = RecastValidator(str(original), str(recast))
        result = validator.check_mapping_complete()

        assert result.result == ValidationResult.FAIL
        assert result.metadata["missing_variables"] == ["X"]


class TestVariableICollisionWithSpI:
    """Tests for variable 'I' collision with SymPy's imaginary unit sp.I.

    Regression test for bug where sympify('I') returns sp.I (imaginary unit),
    which has empty free_symbols. When a model uses variable 'I' (e.g., for
    infected population in epidemic models), the validator's _canonicalize_symbols()
    failed because:
    1. sp.I.free_symbols is empty, so substitution loop had nothing to substitute
    2. Jacobian computation then saw sp.I instead of a proper Symbol
    """

    def test_variable_I_not_confused_with_imaginary(self, tmp_path):
        """Test that variable I is handled correctly, not as imaginary unit."""
        original = tmp_path / "original.ant"
        original.write_text("""
            // Simple SIR-like model with variable I
            model sir()
                species S, I, R;
                S' = -beta*S*I;
                I' = beta*S*I - gamma_rate*I;
                R' = gamma_rate*I;

                beta = 0.3;
                gamma_rate = 0.1;
                S = 0.99;
                I = 0.01;
                R = 0.0;
            end
        """)

        # Simplified recast (2 terms per ODE)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
            species Z_1, Z_2, Z_3, Z_4, Z_5;
            // ============================================================
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ============================================================
            // S -> [Z_1, Z_2]
            // I -> [Z_3, Z_4]
            // R -> [Z_5]
            // ============================================================

            Z_1' = -beta * Z_1 * Z_2^-1 * Z_3 * Z_4;
            Z_2' = -beta * Z_2 * Z_1^-1 * Z_3 * Z_4;
            Z_3' = beta * Z_1 * Z_2 * Z_3^-1;
            Z_4' = -gamma_rate * Z_4;
            Z_5' = gamma_rate * Z_3 * Z_4 * Z_5^-1;

            beta = 0.3;
            gamma_rate = 0.1;
            Z_1 = 0.99;
            Z_2 = 1.0;
            Z_3 = 0.01;
            Z_4 = 1.0;
            Z_5 = 1e-06;
            end
        """)

        validator = RecastValidator(str(original), str(recast))
        result = validator.check_symbolic_equivalence(timeout=10.0)

        assert result is not None
        assert result.name == "symbolic_equivalence"
        # The key test: should NOT fail due to sp.I confusion
        # (may still fail for other reasons, but not the I/sp.I issue)
        if result.result == ValidationResult.FAIL:
            # If it fails, make sure it's NOT due to I as sp.I
            details_lower = result.details.lower()
            assert "imaginary" not in details_lower, \
                f"Variable I confused with imaginary: {result.details}"
            assert "sp.I" not in result.details, \
                f"sp.I appeared in error: {result.details}"


class TestVariableSCollisionWithSympyRegistry:
    """Tests for variable 'S' collision with SymPy's singleton registry sp.S."""

    def test_identity_mapping_s_not_confused_with_sympy_registry(self, tmp_path):
        """Mapping comments must parse model variable S as a Symbol, not sympy.S."""
        original = tmp_path / "original.ant"
        original.write_text("""
            S' = -k*S
            k = 0.5
            S = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
            // ========================================================================
            // VARIABLE MAPPING
            // ========================================================================
            // S = S
            // ========================================================================

            species S;
            S' = -k*S;
            k = 0.5;
            S = 1.0;
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="sbml")
        s_symbol = sp.Symbol("S", positive=True)

        assert validator.mapping[s_symbol] == s_symbol
        assert validator.check_mapping_complete().result == ValidationResult.PASS


class TestTimeDependentValidation:
    """Tests for time-dependent model validation with clock variables."""

    def test_time_dependent_symbolic_validation(self, tmp_path):
        """Test symbolic validation of time-dependent model with clock.

        This tests the fix for S1987_E1_bessel where:
        - Original model uses 'time' in ODEs
        - Recast model uses clock variable T with T' = 1
        - Lifted auxiliary Y_1 := T + 1 represents (time + 1)

        The validator must:
        1. Substitute time -> T in the original ODEs
        2. Substitute Y_1 -> T + 1 for lifted auxiliaries
        3. Skip clock variable T := time (avoid circular substitution)
        """
        original = tmp_path / "original.ant"
        original.write_text("""
            // Simple time-dependent decay
            X' = -X / (time + 1)
            X = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
            // ============================================================
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ============================================================
            // T := time
            // Y_1 := T + 1
            // ============================================================

            T' = 1
            Y_1' = 1
            X' = -X / Y_1

            T = 0
            Y_1 = 1
            X = 1.0
            end
        """)

        validator = RecastValidator(str(original), str(recast))
        result = validator.check_symbolic_equivalence(timeout=10.0)

        assert result is not None
        assert result.name == "symbolic_equivalence"
        assert result.result == ValidationResult.PASS, \
            f"Symbolic test failed: {result.details}"

    def test_time_dependent_numerical_validation(self, tmp_path):
        """Test numerical validation passes for time-dependent models."""
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -X / (time + 1)
            X = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
            // ============================================================
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ============================================================
            // T := time
            // Y_1 := T + 1
            // ============================================================

            T' = 1
            Y_1' = 1
            X' = -X / Y_1

            T = 0
            Y_1 = 1
            X = 1.0
            end
        """)

        validator = RecastValidator(str(original), str(recast))
        result = validator.check_numerical_pointwise(n_samples=100)

        assert result is not None
        assert result.name == "numerical_pointwise"
        assert result.result == ValidationResult.PASS, \
            f"Numerical test failed: {result.details}"


class TestSymbolicValidationBranches:
    """Focused tests for symbolic validation branch behavior."""

    def test_symbolic_validation_handles_clock_without_time_and_aux_substitutions(self):
        X = sp.Symbol("X", positive=True)
        T = sp.Symbol("T", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        A = sp.Symbol("A_1", positive=True)
        k = sp.Symbol("k", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: -X}
        validator.mapping = {X: X}
        validator.recast_state_vars = [X, T, Y, A]
        validator.recast_odes = {
            X: -X,
            T: sp.Integer(1),
            Y: k * Y,
            A: sp.Integer(0),
        }
        validator.recast_ir = SimpleNamespace(params={"k": 2.0})
        validator.auxiliary_defs = {
            X: X + 1,  # original variable, not a lifted auxiliary substitution
            T: sp.Symbol("time"),  # clock definition is skipped after clock handling
            Y: k + X,  # state and parameter symbols are canonicalized by name
            A: sp.Integer(2),  # constant auxiliary has no symbol substitutions
        }

        result = validator.check_symbolic_equivalence()

        assert result.result == ValidationResult.PASS
        assert "exact equivalence" in result.details

    def test_symbolic_validation_does_not_substitute_nonfinite_parameter_values(self):
        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: sp.Integer(0)}
        validator.mapping = {X: X}
        validator.recast_state_vars = [X]
        validator.recast_odes = {X: 1 / k}
        validator.recast_ir = SimpleNamespace(params={"k": 0.0})
        validator.auxiliary_defs = {}

        result = validator.check_symbolic_equivalence()

        assert result.result == ValidationResult.FAIL
        assert "Non-zero components" in result.details
        assert "1/k" in result.details

    def test_symbolic_validation_reports_simplification_errors_as_timeout(self, monkeypatch):
        X = sp.Symbol("X", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: sp.Integer(0)}
        validator.mapping = {X: X}
        validator.recast_state_vars = [X]
        validator.recast_odes = {X: X}
        validator.recast_ir = SimpleNamespace(params={})
        validator.auxiliary_defs = {}

        def fail_nsimplify(*args, **kwargs):
            raise RuntimeError("forced simplification failure")

        monkeypatch.setattr(sp, "nsimplify", fail_nsimplify)

        result = validator.check_symbolic_equivalence()

        assert result.result == ValidationResult.TIMEOUT
        assert result.reason == "timeout"
        assert "forced simplification failure" in result.details

    def test_symbolic_validation_continues_when_factor_strategy_fails(self, monkeypatch):
        import inspect

        X = sp.Symbol("X", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: sp.Integer(0)}
        validator.mapping = {X: X}
        validator.recast_state_vars = [X]
        validator.recast_odes = {X: X + 1}
        validator.recast_ir = SimpleNamespace(params={})
        validator.auxiliary_defs = {}
        real_factor = sp.factor

        def factor_fails_only_in_symbolic_strategy(expr, *args, **kwargs):
            for frame in inspect.stack():
                if frame.filename.endswith("_validator/symbolic.py") and frame.lineno == 185:
                    raise TypeError("factor strategy unavailable")
            return real_factor(expr, *args, **kwargs)

        monkeypatch.setattr(sp, "factor", factor_fails_only_in_symbolic_strategy)

        result = validator.check_symbolic_equivalence()

        assert result.result == ValidationResult.FAIL
        assert "Non-zero components" in result.details
        assert "X + 1" in result.details

    def test_symbolic_validation_outer_exception_is_not_attempted(self):
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {sp.Symbol("X", positive=True): sp.Integer(0)}
        validator.mapping = {}

        result = validator.check_symbolic_equivalence()

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "not_attempted"
        assert "Exception during symbolic test" in result.details


class TestMappingValidationBranches:
    """Focused tests for mapping and auxiliary identity helper branches."""

    def test_build_mapping_uses_sanitized_identity_for_reserved_state(self):
        pi_orig = sp.Symbol("Pi", positive=True)
        pi_recast = sp.Symbol("Pi_var", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {pi_orig: -pi_orig}
        validator.recast_odes = {pi_recast: -pi_recast}
        validator.factor_map = {}
        validator._infer_auxiliary_definitions = lambda: None

        validator._build_mapping()

        assert validator.mapping[pi_orig] == pi_recast

    def test_build_mapping_uses_sanitized_identity_for_antimony_suffix_state(self):
        gamma_orig = sp.Symbol("gamma_", positive=True)
        gamma_recast = sp.Symbol("gamma_var", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {gamma_orig: -gamma_orig}
        validator.recast_odes = {gamma_recast: -gamma_recast}
        validator.factor_map = {}
        validator._infer_auxiliary_definitions = lambda: None

        validator._build_mapping()

        assert validator.mapping[gamma_orig] == gamma_recast

    def test_mapping_comment_parser_supports_old_format_and_bad_expression_fallback(self):
        validator = RecastValidator.__new__(RecastValidator)
        mapping = validator._extract_mapping_from_comments("""
            // Mapping from original variables
            // this line is descriptive, not a mapping
            // X = Z_1*Z_2
            // Bad = X +
            // --- end mapping ---
            // Y = should_not_be_seen
        """)

        assert mapping[sp.Symbol("X")] == sp.Symbol(
            "Z_1", positive=True
        ) * sp.Symbol("Z_2", positive=True)
        assert mapping[sp.Symbol("Bad")] == sp.Symbol("X +")
        assert sp.Symbol("Y") not in mapping

    def test_refusal_reason_requires_following_comment_line(self):
        validator = RecastValidator.__new__(RecastValidator)

        assert validator._extract_refusal_reason(
            "// NOTE: Canonical S-system recast was not attempted because:"
        ) is None
        assert validator._extract_refusal_reason(
            """
            // NOTE: Canonical S-system recast was not attempted because:
            not a comment
            """
        ) is None

    def test_assignment_rule_auxiliary_merge_skips_owned_state_and_bad_rules(self):
        X = sp.Symbol("X", positive=True)
        existing = sp.Symbol("Y_1", positive=True)
        added = sp.Symbol("Z_1", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_ir = SimpleNamespace(assignment_rules={"obs": "X + 1"})
        validator.recast_ir = SimpleNamespace(
            assignment_rules={
                "obs": "X + 1",  # original rule, not a lifted auxiliary
                "X": "X + 2",  # state variable, not an auxiliary
                "Y_1": "X + 3",  # already present
                "Z_1": "X + k",  # lifted auxiliary to merge
                "W_1": "X +",  # malformed rule is skipped
            },
            params={"k": 2.0},
        )
        validator.recast_odes = {X: -X}
        validator.auxiliary_defs = {existing: X + 3}

        validator._merge_assignment_rules_as_auxiliaries()

        assert validator.auxiliary_defs[existing] == X + 3
        assert sp.simplify(validator.auxiliary_defs[added] - (X + sp.Symbol("k", positive=True))) == 0
        assert sp.Symbol("W_1", positive=True) not in validator.auxiliary_defs

    def test_auxiliary_definition_parser_handles_old_end_and_malformed_lines(self):
        validator = RecastValidator.__new__(RecastValidator)
        definitions = validator._extract_auxiliary_definitions("""
            // Auxiliary variable definitions
            // ========================================================================
            // Z_1 := X + 1
            // Bad := X +
            // --- end auxiliary definitions ---
            // Z_2 := should_not_be_seen
        """)

        assert definitions[sp.Symbol("Z_1")] == sp.Symbol("X") + 1
        assert sp.Symbol("Bad") not in definitions
        assert sp.Symbol("Z_2") not in definitions
        assert len(validator.auxiliary_definition_parse_errors) == 1
        parse_error = validator.auxiliary_definition_parse_errors[0]
        assert parse_error["auxiliary"] == "Bad"
        assert parse_error["expression"] == "X +"
        assert "invalid syntax" in parse_error["exception"]

    def test_auxiliary_definition_parser_handles_keyword_identifier(self):
        validator = RecastValidator.__new__(RecastValidator)

        definitions = validator._extract_auxiliary_definitions("""
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ========================================================================
            // Y_3 := lambda^2 + ba2^2/ba1^2
            // ========================================================================
        """)

        lam = sp.Symbol("lambda")
        ba1 = sp.Symbol("ba1")
        ba2 = sp.Symbol("ba2")
        assert sp.simplify(
            definitions[sp.Symbol("Y_3")] - (lam**2 + ba2**2 / ba1**2)
        ) == 0
        assert validator.auxiliary_definition_parse_errors == []

    def test_auxiliary_definition_parser_preserves_floor_function(self):
        validator = RecastValidator.__new__(RecastValidator)

        definitions = validator._extract_auxiliary_definitions("""
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ========================================================================
            // Z_1 := floor(floor(phase + t)/cyclePeriod)
            // ========================================================================
        """)

        expr = definitions[sp.Symbol("Z_1")]
        assert expr.has(sp.floor)
        assert sp.Symbol("floor") not in expr.free_symbols
        assert validator.auxiliary_definition_parse_errors == []

    def test_infer_auxiliary_definitions_uses_denominator_matching(self):
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        M = sp.Symbol("M_1", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.factor_map = {M: X + 1}
        validator.orig_odes = {X: 1 / (X + 2)}
        validator.recast_odes = {
            X: sp.Integer(1),
            Y: sp.Integer(1),
            M: sp.Integer(0),
        }

        validator._infer_auxiliary_definitions()

        assert validator.factor_map[Y] == X + 2
        assert validator.factor_map[M] == X + 1

    def test_denominator_helpers_classify_negative_nontrivial_powers(self):
        X = sp.Symbol("X", positive=True)
        validator = RecastValidator.__new__(RecastValidator)

        assert validator._has_negative_exponent(1 / X, X) is True
        assert validator._has_negative_exponent(X**2, X) is False
        assert validator._find_denominators((X + 1) ** -1 + X**-1 + X**2) == [X + 1]

    def test_mapping_builder_handles_products_substitutions_and_auxiliary_identities(self):
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)
        Q = sp.Symbol("Q", positive=True)
        Z_1 = sp.Symbol("Z_1", positive=True)
        Z_2 = sp.Symbol("Z_2", positive=True)
        A_1 = sp.Symbol("A_1", positive=True)
        stale_z = sp.Symbol("Z_1")
        validator = RecastValidator.__new__(RecastValidator)
        validator._infer_auxiliary_definitions = lambda: None
        validator.orig_odes = {X: -X, Y: -Y, Q: -Q}
        validator.recast_odes = {Z_1: -Z_1, Z_2: -Z_2, A_1: -A_1}
        validator.factor_map = {
            X: [Z_1, Z_2],
            Y: stale_z + 1,
            A_1: stale_z + 2,
        }

        validator._build_mapping()

        assert validator.mapping[X] == Z_1 * Z_2
        assert validator.mapping[Y] == Z_1 + 1
        assert validator.mapping[Q] == Q
        assert validator.mapping[A_1] == Z_1 + 2
        assert validator.recast_state_vars == [Z_1, Z_2, A_1]

    def test_assignment_rule_expansion_handles_empty_nested_and_malformed_rules(self):
        X = sp.Symbol("X", positive=True)
        J_1 = sp.Symbol("J_1", positive=True)
        empty_validator = RecastValidator.__new__(RecastValidator)
        odes = {X: -J_1}

        assert empty_validator._expand_assignment_rules_in_odes(
            odes,
            SimpleNamespace(assignment_rules={}, params={}),
        ) is odes

        validator = RecastValidator.__new__(RecastValidator)
        expanded = validator._expand_assignment_rules_in_odes(
            odes,
            SimpleNamespace(
                assignment_rules={
                    "A": "X + K",
                    "J_1": "A + 1",
                    "Bad": "X +",
                },
                params={"K": 2.0},
            ),
        )

        K = sp.Symbol("K", positive=True)
        assert sp.simplify(expanded[X] - (-(X + K + 1))) == 0

    def test_assignment_rule_expansion_does_not_simplify_expanded_odes(self, monkeypatch):
        X = sp.Symbol("X", positive=True)
        J_1 = sp.Symbol("J_1", positive=True)
        validator = RecastValidator.__new__(RecastValidator)

        real_simplify = sp.simplify

        def fail_simplify(expr):
            raise AssertionError(f"assignment-rule expansion should not simplify {expr}")

        monkeypatch.setattr(sp, "simplify", fail_simplify)
        expanded = validator._expand_assignment_rules_in_odes(
            {X: -J_1},
            SimpleNamespace(
                assignment_rules={
                    "A": "X + K",
                    "B": "A + 1",
                    "J_1": "B + 2",
                },
                params={"K": 2.0},
            ),
        )
        monkeypatch.setattr(sp, "simplify", real_simplify)

        K = sp.Symbol("K", positive=True)
        assert real_simplify(expanded[X] - (-(X + K + 3))) == 0

    def test_computed_auxiliary_definition_expands_abs_assignment_rules(self):
        fbp, fback, kg, r, rgpdh, vg, y, Z_1 = sp.symbols(
            "fbp fback kg r rgpdh vg y Z_1",
            positive=True,
        )
        validator = RecastValidator.__new__(RecastValidator)
        validator.canonical_symbols = {
            str(symbol): symbol
            for symbol in [fbp, fback, kg, r, rgpdh, vg, y, Z_1]
        }
        validator.auxiliary_defs = {Z_1: sp.exp(fback)}
        validator.recast_ir = SimpleNamespace(
            assignment_rules={
                "rgpdh": "0.2 * pow(abs(fbp * 1 / pow(1, 2)), 1 / 2)",
                "y": "vg * (rgpdh / (kg + rgpdh))",
                "fback": "r + y",
            },
            params={"kg": 10.0, "r": 1.0, "vg": 2.2},
        )

        definitions = validator._computed_recast_variable_definitions([fbp, Z_1])

        expected = sp.exp(r + 0.2 * sp.sqrt(fbp) * vg / (kg + 0.2 * sp.sqrt(fbp)))
        assert sp.simplify(definitions[Z_1] - expected) == 0

    def test_auxiliary_identity_keeps_uppercase_t_as_state_variable(self):
        T, Y_1, h, k = sp.symbols("T Y_1 h k", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.canonical_symbols = {str(sym): sym for sym in [T, Y_1, h, k]}
        validator.auxiliary_defs = {Y_1: T**2 + h}
        validator.recast_odes = {
            T: -k * T,
            Y_1: -2 * k * T**2,
        }
        validator.recast_state_vars = [T, Y_1]
        validator.orig_odes_expanded = {}
        validator.recast_ir = SimpleNamespace(params={"h": 1.0, "k": 0.5}, assignment_rules={})

        assert validator._is_clock_definition(T) is False
        assert validator._is_clock_definition(sp.Symbol("t", positive=True)) is True

        tests = validator._ode_auxiliary_identity_tests()

        assert len(tests) == 1
        assert tests[0].result == ValidationResult.PASS

    def test_auxiliary_identity_substitutes_finite_recast_parameters(self):
        X, Y, K, P = sp.symbols("X Y K P", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.canonical_symbols = {str(sym): sym for sym in [X, Y, K, P]}
        validator.auxiliary_defs = {Y: X / (1 + P / K)}
        validator.recast_odes = {
            X: -X,
            Y: -X,
        }
        validator.recast_state_vars = [X, Y]
        validator.orig_odes_expanded = {}
        validator.recast_ir = SimpleNamespace(params={"K": 1.0, "P": 0.0}, assignment_rules={})

        tests = validator._ode_auxiliary_identity_tests()

        assert len(tests) == 1
        assert tests[0].result == ValidationResult.PASS

    def test_auxiliary_identity_preserves_nonzero_parameter_residual(self):
        X, Y, K, P = sp.symbols("X Y K P", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.canonical_symbols = {str(sym): sym for sym in [X, Y, K, P]}
        validator.auxiliary_defs = {Y: X / (1 + P / K)}
        validator.recast_odes = {
            X: -X,
            Y: -X,
        }
        validator.recast_state_vars = [X, Y]
        validator.orig_odes_expanded = {}
        validator.recast_ir = SimpleNamespace(params={"K": 1.0, "P": 2.0}, assignment_rules={})

        tests = validator._ode_auxiliary_identity_tests()

        assert len(tests) == 1
        assert tests[0].result == ValidationResult.FAIL

    def test_mapping_completeness_allows_assignment_rule_identity(self):
        X = sp.Symbol("X", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: -X}
        validator.mapping = {X: X}
        validator.recast_state_vars = []
        validator.recast_ir = SimpleNamespace(assignment_rules={"X": "Z_1 + Z_2"})

        result = validator.check_mapping_complete()

        assert result.result == ValidationResult.PASS


class TestNumericalValidationBranches:
    """Focused tests for numerical validation branch behavior."""

    def test_jax_numerical_validation_reports_missing_jax(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def missing_jax(name, *args, **kwargs):
            if name == "jax" or name == "jax.numpy":
                raise ImportError("jax missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing_jax)
        validator = RecastValidator.__new__(RecastValidator)

        result = validator.check_numerical_pointwise_jax()

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "not_attempted"
        assert "JAX not available" in result.details

    def test_numerical_validation_passes_time_dependent_clock_auxiliary(self):
        X = sp.Symbol("X", positive=True)
        T = sp.Symbol("T", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        time = sp.Symbol("time")
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: -X / (time + 1)}
        validator.orig_odes_expanded = dict(validator.orig_odes)
        validator.mapping = {X: X}
        validator.recast_state_vars = [X, T, Y]
        validator.recast_odes_expanded = {
            X: -X / Y,
            T: sp.Integer(1),
            Y: sp.Integer(1),
        }
        validator.recast_ir = SimpleNamespace(params={})
        validator.auxiliary_defs = {T: time, Y: T + 1}
        validator.canonical_symbols = {"X": X, "T": T, "Y_1": Y, "time": time}

        result = validator.check_numerical_pointwise(n_samples=3)

        assert result.result == ValidationResult.PASS
        assert result.max_error is not None
        assert result.max_error < 1.0e-12

    def test_numerical_validation_reports_partially_unresolved_auxiliary(self):
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        U = sp.Symbol("U", positive=True)
        validator = RecastValidator.__new__(RecastValidator)
        validator.orig_odes = {X: -X}
        validator.orig_odes_expanded = dict(validator.orig_odes)
        validator.mapping = {X: X}
        validator.recast_state_vars = [X, Y]
        validator.recast_odes_expanded = {X: -X, Y: sp.Integer(0)}
        validator.recast_ir = SimpleNamespace(params={})
        validator.auxiliary_defs = {Y: X + U}
        validator.canonical_symbols = {"X": X, "Y_1": Y, "U": U}

        result = validator.check_numerical_pointwise(n_samples=3)

        assert result.result == ValidationResult.NOT_ATTEMPTED
        assert result.reason == "unresolved_parameter"
        assert result.metadata["diagnostics"][0]["unresolved_symbols"] == ["U"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
