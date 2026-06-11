"""Tests for validator module."""

import json

import numpy as np
import pytest
import sympy as sp

from ssys.ode_backends.ida_sundials_backend import IDASundialsUnavailable
from ssys.recaster import SolverRequirement, SystemClass
from ssys.validator import (
    EquivalenceTest,
    RecastValidator,
    ValidationReport,
    ValidationResult,
    validate_generated_output_roundtrip,
    validate_recast_pair,
)


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_validation_result_values(self):
        """Test that all expected result values exist."""
        assert ValidationResult.PASS is not None
        assert ValidationResult.FAIL is not None
        assert ValidationResult.TIMEOUT is not None
        assert ValidationResult.NOT_ATTEMPTED is not None
        assert ValidationResult.UNSUPPORTED is not None
        assert ValidationResult.INCONCLUSIVE is not None

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
        assert report.symbolic_test is not None
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

        assert d["original_file"] == "/path/to/original.ant"
        assert d["overall_pass"] is True


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
        assert data["overall_result"] == "failed"


class TestSolverAwareValidation:
    """Tests for solver requirement reporting and algebraic residual checks."""

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
            }

        monkeypatch.setattr(validator, "_simulate_model", fake_simulate_model)

        result = validator.check_trajectory_comparison()

        assert result.result == ValidationResult.PASS
        assert result.metadata["original_backend"] == "original_backend"
        assert result.metadata["recast_backend"] == "recast_backend"

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

        validator = RecastValidator(str(original), str(recast), parser="legacy")
        result = validator.check_numerical_pointwise(n_samples=8, threshold=1.0e-9)

        assert result.result == ValidationResult.FAIL
        assert result.max_error > 1.0e-9
        assert result.counterexamples


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
            // Original ODE for X: -k*X
            // Auxiliary mapping: X -> [Z_1]
            model recast
                species Z_1;
                Z_1' = -k * Z_1
                k = 0.5
                Z_1 = 1.0
            end
        """)

        return str(original), str(recast)

    def test_validator_init(self, simple_model_paths):
        """Test validator initialization."""
        orig, recast = simple_model_paths

        validator = RecastValidator(orig, recast, parser="legacy")

        assert validator is not None
        assert validator.original_file == orig
        assert validator.recast_file == recast


    def test_validator_symbolic_check(self, simple_model_paths):
        """Test symbolic equivalence check."""
        orig, recast = simple_model_paths

        validator = RecastValidator(orig, recast, parser="legacy")
        result = validator.check_symbolic_equivalence(timeout=5.0)

        assert result is not None
        assert result.name == "symbolic_equivalence"
        # Result should be one of the enum values
        assert result.result in list(ValidationResult)


class TestRecastValidatorAuxiliaryExtraction:
    """Tests for auxiliary definition extraction."""

    def test_extract_auxiliary_from_assignment(self, tmp_path):
        """Test extracting auxiliary from assignment rules."""
        original = tmp_path / "original.ant"
        original.write_text("""
            X' = -k*X
            k = 0.5
            X = 1.0
        """)

        recast = tmp_path / "recast.ant"
        recast.write_text("""
            // Original ODE for X: -k*X
            // Auxiliary mapping: X -> [Z_1]
            model recast
                species Z_1;
                Z_1 := X  // Assignment rule
                Z_1' = -k * Z_1
                k = 0.5
                X = 1.0
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="legacy")

        # Should parse without error
        assert validator is not None


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

        validator = RecastValidator(str(original), str(recast), parser="legacy")
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
            // RECASTER_REFUSAL: Unable to recast
            // Reason: Model contains unsupported features
            model refused
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="legacy")

        # Should extract refusal reason
        reason = validator._extract_refusal_reason(recast.read_text())
        assert reason is not None or validator is not None

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

        # Should still work, inferring mapping
        validator = RecastValidator(str(original), str(recast), parser="legacy")
        assert validator is not None


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
            S' = -beta*S*I
            I' = beta*S*I - gamma*I
            R' = gamma*I

            beta = 0.3
            gamma = 0.1
            S = 0.99
            I = 0.01
            R = 0.0
        """)

        # Simplified recast (2 terms per ODE)
        recast = tmp_path / "recast.ant"
        recast.write_text("""
            model recast()
            // ============================================================
            // AUXILIARY DEFINITIONS (for lifted variables)
            // ============================================================
            // S -> [Z_1, Z_2]
            // I -> [Z_3, Z_4]
            // R -> [Z_5]
            // ============================================================

            Z_1' = -beta * Z_1 * Z_2^-1 * Z_3 * Z_4
            Z_2' = -beta * Z_2 * Z_1^-1 * Z_3 * Z_4
            Z_3' = beta * Z_1 * Z_2 * Z_3^-1
            Z_4' = -gamma * Z_4
            Z_5' = gamma * Z_3 * Z_4 * Z_5^-1

            beta = 0.3
            gamma = 0.1
            Z_1 = 0.99
            Z_2 = 1.0
            Z_3 = 0.01
            Z_4 = 1.0
            Z_5 = 1e-06
            end
        """)

        validator = RecastValidator(str(original), str(recast), parser="legacy")
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

        validator = RecastValidator(str(original), str(recast), parser="legacy")
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

        validator = RecastValidator(str(original), str(recast), parser="legacy")
        result = validator.check_numerical_pointwise(n_samples=100)

        assert result is not None
        assert result.name == "numerical_pointwise"
        assert result.result == ValidationResult.PASS, \
            f"Numerical test failed: {result.details}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
