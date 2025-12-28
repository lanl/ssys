"""Tests for output formatting and classification functions."""

import pytest
import sympy as sp

from ssys.recaster import (
    GMAEquation,
    RecastResult,
    RecastStatus,
    SSysEquation,
    SymSystem,
    SystemClass,
    _antimony_to_sympy_syntax,
    _sympy_to_antimony_syntax,
    build_sym_system,
    classify_result,
    classify_system,
    gma_to_antimony,
    latex_ssys,
    parse_antimony,
    product_to_antimony,
    recast_to_ssystem,
    ssystem_to_antimony,
)


class TestSyntaxConversion:
    """Tests for Antimony/SymPy syntax conversion."""

    def test_antimony_to_sympy_caret_to_double_star(self):
        """Test converting ^ to ** for exponentiation."""
        result = _antimony_to_sympy_syntax("x^2 + y^3")
        assert "**" in result
        assert "^" not in result
        assert result == "x**2 + y**3"

    def test_sympy_to_antimony_double_star_to_caret(self):
        """Test converting ** to ^ for Antimony."""
        result = _sympy_to_antimony_syntax("x**2 + y**3")
        assert "^" in result
        assert "**" not in result
        assert result == "x^2 + y^3"

    def test_sympy_to_antimony_abs(self):
        """Test converting Abs() to abs()."""
        result = _sympy_to_antimony_syntax("Abs(x)")
        assert "abs" in result
        assert "Abs" not in result


class TestProductToAntimony:
    """Tests for product_to_antimony formatting."""

    def test_simple_monomial(self):
        """Test formatting simple monomial."""
        x = sp.Symbol("x", positive=True)
        coeff = 2.0
        exps = {x: 3.0}
        result = product_to_antimony(coeff, exps)
        assert "2" in result
        assert "x" in result
        # Should have exponent
        assert "3" in result or "^" in result

    def test_unity_coefficient_omitted(self):
        """Test that coefficient of 1 is omitted."""
        x = sp.Symbol("x", positive=True)
        coeff = 1.0
        exps = {x: 2.0}
        result = product_to_antimony(coeff, exps)
        # Should start with x, not 1*x
        assert result.startswith("x") or result == "x^2"

    def test_unity_exponent_omitted(self):
        """Test that exponent of 1 is omitted."""
        x = sp.Symbol("x", positive=True)
        coeff = 2.0
        exps = {x: 1.0}
        result = product_to_antimony(coeff, exps)
        assert "2*x" in result or result == "2*x"
        # Should not have ^1
        assert "^1" not in result

    def test_zero_coefficient(self):
        """Test that zero coefficient returns '0'."""
        x = sp.Symbol("x", positive=True)
        coeff = 0.0
        exps = {x: 2.0}
        result = product_to_antimony(coeff, exps)
        assert result == "0"

    def test_empty_exponents(self):
        """Test pure coefficient with no exponents."""
        coeff = 5.0
        exps = {}
        result = product_to_antimony(coeff, exps)
        assert result == "5"

    def test_symbolic_coefficient(self):
        """Test symbolic coefficient formatting."""
        x = sp.Symbol("x", positive=True)
        k = sp.Symbol("k", positive=True)
        coeff = k
        exps = {x: 2.0}
        result = product_to_antimony(coeff, exps)
        assert "k" in result
        assert "x" in result

    def test_negative_exponent(self):
        """Test negative exponent formatting."""
        x = sp.Symbol("x", positive=True)
        coeff = 1.0
        exps = {x: -1.0}
        result = product_to_antimony(coeff, exps)
        assert "-1" in result


class TestSystemClassification:
    """Tests for system classification."""

    def test_classify_canonical_ssystem(self):
        """Test classifying canonical S-system (1 growth + 1 decay)."""
        X = sp.Symbol("X", positive=True)
        k1 = sp.Symbol("k1", positive=True)
        k2 = sp.Symbol("k2", positive=True)

        # X' = k1*X - k2*X^2 (canonical form)
        sym = SymSystem(
            vars=[X],
            params={"k1": 1.0, "k2": 0.1},
            odes={X: k1 * X - k2 * X**2},
            initials={X: 1.0},
        )

        result = classify_system(sym)
        # Should be canonical S-system or S-system
        assert result in [SystemClass.CANONICAL_SSYSTEM, SystemClass.SSYSTEM]

    def test_classify_gma(self):
        """Test classifying GMA system (multiple terms)."""
        X = sp.Symbol("X", positive=True)

        # X' = X + X^2 + X^3 (multiple growth terms, different exponents)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: X + X**2 + X**3},
            initials={X: 1.0},
        )

        result = classify_system(sym)
        # Multiple incompatible terms should be GMA or general
        assert result in [SystemClass.GMA, SystemClass.GENERAL]

    def test_classify_general_non_monomial(self):
        """Test classifying system with non-monomial terms."""
        X = sp.Symbol("X", positive=True)

        # X' = exp(X) - non-monomial term
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.exp(X)},
            initials={X: 0.0},
        )

        result = classify_system(sym)
        assert result == SystemClass.GENERAL


class TestClassifyResult:
    """Tests for classifying RecastResult."""

    def test_classify_result_canonical(self):
        """Test classifying canonical S-system result."""
        Z = sp.Symbol("Z_1", positive=True)

        eq = SSysEquation(
            var=Z,
            growth=(sp.Float(1.0), {Z: 1.0}),
            decay=(sp.Float(0.5), {Z: 2.0}),
        )

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[eq],
            initials={Z: 1.0},
            variables=[Z],
        )

        classification = classify_result(result)
        assert classification == SystemClass.CANONICAL_SSYSTEM

    def test_classify_result_gma(self):
        """Test classifying GMA result."""
        Z = sp.Symbol("Z_1", positive=True)

        gma_eq = GMAEquation(
            var=Z,
            production=[
                (sp.Float(1.0), {Z: 1.0}),
                (sp.Float(2.0), {Z: 2.0}),  # Different exponent
            ],
            degradation=[(sp.Float(0.5), {Z: 3.0})],
        )

        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            gma_equations=[gma_eq],
        )

        classification = classify_result(result)
        assert classification == SystemClass.GMA


class TestAntimonyExport:
    """Tests for exporting to Antimony format."""

    def test_ssystem_to_antimony_basic(self):
        """Test basic S-system to Antimony export."""
        text = """
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)
        output = ssystem_to_antimony(rec, model_name="test")

        # Check structure
        assert "model test" in output
        assert "end" in output
        # Should have species declarations
        assert "species" in output
        # Should have Z_1 (auxiliary)
        assert "Z_1" in output

    def test_ssystem_to_antimony_model_name_prefix(self):
        """Test that numeric model names get prefixed."""
        text = """
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)
        output = ssystem_to_antimony(rec, model_name="123model")

        # Numeric prefix should become m_123model
        assert "model m_123model" in output

    def test_gma_to_antimony_basic(self):
        """Test GMA to Antimony export."""
        Z = sp.Symbol("Z_1", positive=True)

        gma_eq = GMAEquation(
            var=Z,
            production=[(sp.Float(2.0), {Z: 1.0})],
            degradation=[(sp.Float(0.5), {Z: 2.0})],
        )

        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            gma_equations=[gma_eq],
            params={},
        )

        output = gma_to_antimony(result, model_name="gma_test")

        assert "model gma_test" in output
        assert "end" in output
        assert "GMA" in output  # Should have GMA comment


class TestLatexExport:
    """Tests for LaTeX export."""

    def test_latex_ssys_basic(self):
        """Test basic LaTeX export."""
        Z = sp.Symbol("Z_1", positive=True)

        eq = SSysEquation(
            var=Z,
            growth=(sp.Float(1.0), {Z: 1.0}),
            decay=(sp.Float(0.5), {Z: 2.0}),
        )

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[eq],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
        )

        latex = latex_ssys(result)

        assert "\\dot" in latex  # Time derivative
        assert "Z" in latex  # Variable name
        assert "aligned" in latex  # LaTeX environment


class TestRecastResultDataclass:
    """Tests for RecastResult dataclass."""

    def test_recast_result_creation(self):
        """Test basic RecastResult creation."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
        )

        assert result.status == RecastStatus.CANONICAL_SSYSTEM
        assert Z in result.variables

    def test_recast_result_with_params(self):
        """Test RecastResult with parameters."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            params={"k": 0.5, "V": 2.0},
        )

        assert result.params["k"] == 0.5
        assert result.params["V"] == 2.0


class TestSSysEquationDataclass:
    """Tests for SSysEquation dataclass."""

    def test_ssys_equation_creation(self):
        """Test SSysEquation creation."""
        Z = sp.Symbol("Z_1", positive=True)

        eq = SSysEquation(
            var=Z,
            growth=(sp.Float(1.0), {Z: 1.0}),
            decay=(sp.Float(0.5), {Z: 2.0}),
        )

        assert eq.var == Z
        assert float(eq.growth[0]) == 1.0
        assert float(eq.decay[0]) == 0.5


class TestGMAEquationDataclass:
    """Tests for GMAEquation dataclass."""

    def test_gma_equation_creation(self):
        """Test GMAEquation creation."""
        Z = sp.Symbol("Z_1", positive=True)

        eq = GMAEquation(
            var=Z,
            production=[(sp.Float(1.0), {Z: 1.0})],
            degradation=[(sp.Float(0.5), {Z: 2.0})],
        )

        assert eq.var == Z
        assert len(eq.production) == 1
        assert len(eq.degradation) == 1

    def test_gma_equation_multiple_terms(self):
        """Test GMAEquation with multiple terms."""
        Z = sp.Symbol("Z_1", positive=True)

        eq = GMAEquation(
            var=Z,
            production=[
                (sp.Float(1.0), {Z: 1.0}),
                (sp.Float(2.0), {Z: 2.0}),
            ],
            degradation=[
                (sp.Float(0.5), {Z: 3.0}),
                (sp.Float(0.3), {Z: 4.0}),
            ],
        )

        assert len(eq.production) == 2
        assert len(eq.degradation) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
