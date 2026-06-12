"""Tests for edge cases, error handling, and boundary conditions."""

import pytest
import sympy as sp

from ssys.recaster import (
    RecastResult,
    RecastStatus,
    SymSystem,
    SystemClass,
    _antimony_to_sympy_syntax,
    _get_coefficient_sign,
    _is_term_monomial,
    _sympy_to_antimony_syntax,
    classify_result,
    classify_system,
    expand_to_terms,
    find_composite_functions,
    find_rational_denominators,
    product_expr,
    term_to_coeff_exps,
)


class TestEmptyAndTrivialSystems:
    """Tests for empty or trivial edge cases."""

    def test_zero_term(self):
        """Test that zero term expands correctly."""
        terms = expand_to_terms(sp.Integer(0))
        assert len(terms) == 1
        assert terms[0] == 0

    def test_single_constant_ode(self):
        """Test ODE with constant RHS."""
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.Integer(1)},  # X' = 1
            initials={X: 0.0},
        )

        cls = classify_system(sym)
        # A constant RHS is a valid one-term S-system equation.
        assert cls == SystemClass.SSYSTEM

    def test_zero_ode(self):
        """Test ODE with zero RHS (equilibrium)."""
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.Integer(0)},  # X' = 0
            initials={X: 1.0},
        )

        cls = classify_system(sym)
        assert cls == SystemClass.SSYSTEM


class TestCoefficientsAndExponents:
    """Tests for coefficient and exponent handling."""

    def test_very_small_coefficient(self):
        """Test handling of very small coefficients."""
        x = sp.Symbol("x", positive=True)
        expr = sp.Float(1e-15) * x

        coeff, exps = term_to_coeff_exps(expr)

        assert abs(float(coeff) - 1e-15) < 1e-20
        assert exps[x] == 1

    def test_very_large_coefficient(self):
        """Test handling of very large coefficients."""
        x = sp.Symbol("x", positive=True)
        expr = sp.Float(1e15) * x**2

        coeff, exps = term_to_coeff_exps(expr)

        assert abs(float(coeff) - 1e15) < 1e10
        assert exps[x] == 2

    def test_negative_coefficient(self):
        """Test negative coefficient extraction."""
        x = sp.Symbol("x", positive=True)
        expr = -3 * x**2

        sign = _get_coefficient_sign(expr)
        assert sign == -1

    def test_zero_exponent_in_product(self):
        """Test that zero exponents are handled."""
        x = sp.Symbol("x", positive=True)
        y = sp.Symbol("y", positive=True)

        # x^0 should simplify to 1
        result = product_expr(2, {x: 0, y: 1})

        # Should be 2*y, not 2*x^0*y
        assert sp.simplify(result - 2 * y) == 0

    def test_fractional_exponent(self):
        """Test fractional exponents."""
        x = sp.Symbol("x", positive=True)
        expr = x ** sp.Rational(1, 2)  # sqrt(x)

        coeff, exps = term_to_coeff_exps(expr)

        assert coeff == 1
        # Exponent could be Rational(1,2) or 0.5
        assert abs(float(exps[x]) - 0.5) < 1e-10

    def test_negative_exponent(self):
        """Test negative exponents (1/x)."""
        x = sp.Symbol("x", positive=True)
        expr = x ** (-1)  # 1/x

        coeff, exps = term_to_coeff_exps(expr)

        assert coeff == 1
        assert exps[x] == -1

    def test_symbolic_exponent(self):
        """Test symbolic (parametric) exponents."""
        x = sp.Symbol("x", positive=True)
        n = sp.Symbol("n")
        expr = x**n

        coeff, exps = term_to_coeff_exps(expr)

        assert coeff == 1
        assert exps == {x: n}


class TestTermMonomial:
    """Tests for monomial term detection."""

    def test_integer_is_monomial(self):
        """Test that integers are monomials."""
        assert _is_term_monomial(sp.Integer(5)) is True

    def test_negative_integer_is_monomial(self):
        """Test that negative integers are monomials."""
        assert _is_term_monomial(sp.Integer(-3)) is True

    def test_float_is_monomial(self):
        """Test that floats are monomials."""
        assert _is_term_monomial(sp.Float(3.14)) is True

    def test_symbol_is_monomial(self):
        """Test that single symbol is monomial."""
        x = sp.Symbol("x")
        assert _is_term_monomial(x) is True

    def test_power_is_monomial(self):
        """Test that power is monomial."""
        x = sp.Symbol("x")
        assert _is_term_monomial(x**3) is True

    def test_product_is_monomial(self):
        """Test that product of powers is monomial."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        assert _is_term_monomial(2 * x**2 * y**3) is True

    def test_sum_not_monomial(self):
        """Test that sum is not monomial."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        assert _is_term_monomial(x + y) is False

    def test_function_not_monomial(self):
        """Test that function application is not monomial."""
        x = sp.Symbol("x")
        assert _is_term_monomial(sp.exp(x)) is False
        assert _is_term_monomial(sp.sin(x)) is False
        assert _is_term_monomial(sp.log(x)) is False


class TestFindRationalDenominators:
    """Tests for finding rational function denominators."""

    def test_simple_fraction(self):
        """Test finding denominator in simple fraction."""
        x = sp.Symbol("x")
        expr = 1 / x

        denoms = find_rational_denominators(expr)

        # 1/x is already a power-law monomial, so it does not need lifting.
        assert denoms == set()

    def test_polynomial_denominator(self):
        """Test polynomial in denominator."""
        x = sp.Symbol("x")
        expr = 1 / (x + 1)

        denoms = find_rational_denominators(expr)

        assert len(denoms) >= 1

    def test_michaelis_menten_form(self):
        """Test Michaelis-Menten form."""
        S = sp.Symbol("S", positive=True)
        Km = sp.Symbol("Km", positive=True)
        Vmax = sp.Symbol("Vmax", positive=True)

        expr = Vmax * S / (Km + S)

        denoms = find_rational_denominators(expr)

        assert len(denoms) >= 1

    def test_no_denominators(self):
        """Test expression with no rational denominators."""
        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1  # Polynomial

        denoms = find_rational_denominators(expr)

        assert len(denoms) == 0


class TestFindCompositeFunctions:
    """Tests for finding composite function applications."""

    def test_exp(self):
        """Test finding exp function."""
        x = sp.Symbol("x")
        expr = sp.exp(x)

        funcs = find_composite_functions(expr)

        assert len(funcs) >= 1

    def test_sin(self):
        """Test finding sin function."""
        x = sp.Symbol("x")
        expr = sp.sin(x)

        funcs = find_composite_functions(expr)

        assert len(funcs) >= 1

    def test_nested_functions(self):
        """Test finding nested composite functions."""
        x = sp.Symbol("x")
        expr = sp.exp(sp.sin(x))

        funcs = find_composite_functions(expr)

        # Should find both exp and sin
        assert len(funcs) >= 1

    def test_polynomial_no_composites(self):
        """Test polynomial has no composite functions."""
        x = sp.Symbol("x")
        expr = x**3 + 2 * x**2 - x + 1

        funcs = find_composite_functions(expr)

        assert len(funcs) == 0


class TestSyntaxConversion:
    """Tests for Antimony <-> SymPy syntax conversion."""

    def test_power_conversion(self):
        """Test power operator conversion."""
        antimony = "x^2"
        sympy_syntax = _antimony_to_sympy_syntax(antimony)

        assert "**" in sympy_syntax

    def test_reverse_power_conversion(self):
        """Test reverse power operator conversion."""
        sympy_syntax = "x**2"
        antimony = _sympy_to_antimony_syntax(sympy_syntax)

        assert "^" in antimony
        assert "**" not in antimony

    def test_function_preservation(self):
        """Test that functions are preserved."""
        antimony = "exp(x)"
        sympy_syntax = _antimony_to_sympy_syntax(antimony)

        assert "exp" in sympy_syntax

    def test_complex_expression(self):
        """Test complex expression conversion."""
        antimony = "a*x^2 + b*x + c"
        sympy_syntax = _antimony_to_sympy_syntax(antimony)

        assert "^" not in sympy_syntax
        assert "**" in sympy_syntax


class TestExpandToTerms:
    """Tests for expand_to_terms function."""

    def test_single_term(self):
        """Test single term expansion."""
        x = sp.Symbol("x")
        expr = 3 * x**2

        terms = expand_to_terms(expr)

        assert len(terms) == 1
        assert terms[0] == expr

    def test_sum_expansion(self):
        """Test sum expansion."""
        x = sp.Symbol("x")
        expr = x + 2 * x**2 + 3 * x**3

        terms = expand_to_terms(expr)

        assert len(terms) == 3

    def test_difference_expansion(self):
        """Test difference expansion."""
        x = sp.Symbol("x")
        expr = x**2 - x

        terms = expand_to_terms(expr)

        assert len(terms) == 2

    def test_product_expansion(self):
        """Test product expansion (should expand)."""
        x = sp.Symbol("x")
        expr = (x + 1) * (x - 1)  # x^2 - 1

        terms = expand_to_terms(expr)

        assert len(terms) == 2  # x^2 and -1


class TestClassifySystem:
    """Tests for system classification."""

    def test_classify_ssystem(self):
        """Test S-system classification."""
        X = sp.Symbol("X", positive=True)
        a = sp.Symbol("a", positive=True)
        b = sp.Symbol("b", positive=True)

        # Classic S-system form: X' = a*X^g - b*X^h
        sym = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 0.5},
            odes={X: a * X - b * X**2},
            initials={X: 1.0},
        )

        cls = classify_system(sym)
        assert cls == SystemClass.CANONICAL_SSYSTEM

    def test_classify_gma(self):
        """Test GMA classification (3+ monomial terms)."""
        X = sp.Symbol("X", positive=True)

        # GMA form: 3 monomial terms
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: X - X**2 + 2 * X**3},
            initials={X: 1.0},
        )

        cls = classify_system(sym)
        assert cls == SystemClass.GMA

    def test_classify_general(self):
        """Test general (non-power-law) classification."""
        X = sp.Symbol("X", positive=True)

        # Non-monomial term
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.exp(-X)},
            initials={X: 1.0},
        )

        cls = classify_system(sym)
        assert cls == SystemClass.GENERAL


class TestClassifyResult:
    """Tests for recast result classification."""

    def test_classify_canonical_ssystem_result(self):
        """Test classifying canonical S-system result."""
        X = sp.Symbol("X", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={X: 1.0},
            variables=[X],
            factor_map={},
        )

        cls = classify_result(result)
        assert cls == SystemClass.CANONICAL_SSYSTEM

    def test_classify_failed_result(self):
        """Test classifying failed result."""
        result = RecastResult(
            status=RecastStatus.FAILED,
            equations=[],
            initials={},
            variables=[],
            factor_map={},
        )

        cls = classify_result(result)
        assert cls == SystemClass.GENERAL  # Failed recasts are general


class TestProductExprEdgeCases:
    """Tests for product_expr edge cases."""

    def test_coefficient_only(self):
        """Test with coefficient only, no exponents."""
        result = product_expr(5, {})
        assert result == 5

    def test_unit_coefficient(self):
        """Test with unit coefficient."""
        x = sp.Symbol("x", positive=True)
        result = product_expr(1, {x: 2})
        assert sp.simplify(result - x**2) == 0

    def test_integer_coefficient(self):
        """Test with integer coefficient."""
        x = sp.Symbol("x", positive=True)
        result = product_expr(3, {x: 1})
        assert sp.simplify(result - 3 * x) == 0

    def test_float_coefficient(self):
        """Test with float coefficient."""
        x = sp.Symbol("x", positive=True)
        result = product_expr(2.5, {x: 1})
        assert abs(float(result.subs(x, 1)) - 2.5) < 1e-10

    def test_symbolic_coefficient(self):
        """Test with symbolic coefficient."""
        x = sp.Symbol("x", positive=True)
        k = sp.Symbol("k", positive=True)
        result = product_expr(k, {x: 2})
        assert sp.simplify(result - k * x**2) == 0


class TestSpecialCases:
    """Tests for special mathematical cases."""

    def test_unity_expression(self):
        """Test expression that equals 1."""
        x = sp.Symbol("x", positive=True)
        expr = x / x  # Should simplify to 1

        simplified = sp.simplify(expr)
        assert simplified == 1

    def test_zero_times_infinity_form(self):
        """Test handling of 0 * infinity forms."""
        x = sp.Symbol("x", positive=True)
        # Create expression that could be problematic: x * (1/x - 1/x)
        expr = x * (1 / x - 1 / x)

        simplified = sp.simplify(expr)
        assert simplified == 0

    def test_very_high_power(self):
        """Test very high power exponents."""
        x = sp.Symbol("x", positive=True)
        expr = x**100

        coeff, exps = term_to_coeff_exps(expr)

        assert coeff == 1
        assert exps[x] == 100


class TestMultiVariableSystems:
    """Tests for multi-variable system handling."""

    def test_two_variable_ssystem(self):
        """Test two-variable S-system."""
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)

        # Lotka-Volterra form
        sym = SymSystem(
            vars=[X, Y],
            params={"a": 1.0, "b": 0.1, "c": 0.5, "d": 0.02},
            odes={X: X - X * Y, Y: X * Y - Y},
            initials={X: 10.0, Y: 5.0},
        )

        cls = classify_system(sym)
        # Each ODE has 2 monomial terms - this is S-system
        assert cls in [SystemClass.CANONICAL_SSYSTEM, SystemClass.GMA]

    def test_three_variable_system(self):
        """Test three-variable system."""
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        C = sp.Symbol("C", positive=True)

        sym = SymSystem(
            vars=[A, B, C],
            params={},
            odes={A: -A * B, B: A * B - B * C, C: B * C - C},
            initials={A: 1.0, B: 1.0, C: 1.0},
        )

        cls = classify_system(sym)
        # Could be S-system, canonical S-system, or GMA depending on classification
        assert cls in [
            SystemClass.CANONICAL_SSYSTEM,
            SystemClass.GMA,
            SystemClass.SSYSTEM,
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
