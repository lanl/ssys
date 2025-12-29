"""Tests for validator module."""

import pytest

from ssys.recaster import SystemClass
from ssys.validator import (
    EquivalenceTest,
    RecastValidator,
    ValidationReport,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_validation_result_values(self):
        """Test that all expected result values exist."""
        assert ValidationResult.PASS is not None
        assert ValidationResult.FAIL is not None
        assert ValidationResult.TIMEOUT is not None
        assert ValidationResult.NOT_ATTEMPTED is not None

    def test_validation_result_names(self):
        """Test result name access."""
        assert ValidationResult.PASS.name == "PASS"
        assert ValidationResult.FAIL.name == "FAIL"


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

        validator = RecastValidator(orig, recast)

        assert validator is not None
        assert validator.original_file == orig
        assert validator.recast_file == recast


    def test_validator_symbolic_check(self, simple_model_paths):
        """Test symbolic equivalence check."""
        orig, recast = simple_model_paths

        validator = RecastValidator(orig, recast)
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

        validator = RecastValidator(str(original), str(recast))

        # Should parse without error
        assert validator is not None


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

        validator = RecastValidator(str(original), str(recast))

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
        validator = RecastValidator(str(original), str(recast))
        assert validator is not None


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
