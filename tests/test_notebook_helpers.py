"""Tests for notebook_helpers module."""

import numpy as np
import pytest
import sympy as sp

import ssys
import ssys.notebook_helpers as nh
from ssys import math_utils
from ssys.classification import classify_system as core_classify_system
from ssys.notebook_helpers import (
    _antimony_to_latex_direct,
    _beautify_latex,
    _expand_exps_through_factors,
    _get_term_sign,
    _is_already_gma,
    _is_already_ssystem,
    _is_monomial,
    _simplify_exponent_content,
    build_rhs_from_sympy,
    find_clock_variable,
    get_autonomy_label,
    is_nonautonomous,
    latex_factor_map,
    latex_odes_from_sym,
    latex_ssys_from_antimony,
    parse_antimony_odes,
    product_expr,
    was_nonautonomous,
)
from ssys.recaster import RecastResult, RecastStatus, SymSystem
from ssys.types import SystemClass


class TestBeautifyLatex:
    """Tests for LaTeX beautification."""

    def test_greek_letters_converted(self):
        """Test that Greek letter names become LaTeX symbols."""
        result = _beautify_latex("alpha + beta")
        assert r"\alpha" in result
        assert r"\beta" in result

    def test_greek_with_subscript(self):
        """Test Greek letters with subscripts."""
        result = _beautify_latex("gamma_rate")
        assert r"\gamma_{rate}" in result

    def test_greek_with_digit(self):
        """Test Greek letters followed by digits."""
        result = _beautify_latex("alpha1")
        assert r"\alpha_{1}" in result

    def test_compound_subscripts(self):
        """Test compound subscript handling."""
        result = _beautify_latex("S_in")
        assert "S_{in}" in result

    def test_single_letter_digit_subscript(self):
        """Test single letter + digit subscripting."""
        result = _beautify_latex("k1")
        assert "k_{1}" in result

    def test_eps_to_epsilon(self):
        """Test eps → epsilon conversion."""
        result = _beautify_latex("eps")
        assert r"\epsilon" in result

    def test_decimal_to_fraction(self):
        """Test decimal exponent to fraction conversion."""
        result = _beautify_latex("x^{0.5}")
        assert "1/2" in result

    def test_negative_fraction(self):
        """Test negative decimal exponent."""
        result = _beautify_latex("x^{-0.5}")
        assert "-1/2" in result

    def test_already_escaped_not_doubled(self):
        """Test that already escaped Greek letters aren't doubled."""
        result = _beautify_latex(r"\alpha")
        # Should not become \\alpha or have doubled backslash
        assert result.count(r"\alpha") == 1


class TestAntimonyToLatexDirect:
    """Tests for direct Antimony to LaTeX conversion."""

    def test_simple_term(self):
        """Test simple term conversion."""
        result = _antimony_to_latex_direct("k*X")
        assert "k" in result
        assert "X" in result
        # * should become space (implicit multiplication)
        assert "*" not in result

    def test_exponent_conversion(self):
        """Test caret exponent to braced form."""
        result = _antimony_to_latex_direct("X^2")
        assert "X^{2}" in result

    def test_negative_exponent(self):
        """Test negative exponents."""
        result = _antimony_to_latex_direct("X^-1")
        assert "X^{-1}" in result

    def test_decimal_exponent(self):
        """Test decimal exponents."""
        result = _antimony_to_latex_direct("X^0.5")
        assert "^{" in result
        assert "0.5" in result or "1/2" in result

    def test_parenthesized_exponent(self):
        """Test symbolic exponent in parentheses."""
        result = _antimony_to_latex_direct("X^(a - 1)")
        assert "^{a - 1}" in result

    def test_multiplication_spacing(self):
        """Test that * becomes space."""
        result = _antimony_to_latex_direct("2*x*y")
        assert "*" not in result

    def test_preserves_structure(self):
        """Test that expression structure is preserved."""
        result = _antimony_to_latex_direct("a + b - c")
        # Should have both + and -
        assert "+" in result
        assert "-" in result


class TestNumericalRhsBuilder:
    """Tests for notebook numerical RHS construction."""

    def test_assignment_rules_and_time_are_substituted(self):
        X = sp.Symbol("X")
        time = sp.Symbol("time")
        rhs = build_rhs_from_sympy(
            [X],
            [sp.Symbol("A") + time],
            {"k": 2.0},
            assignment_rules={"A": "k*X"},
        )

        result = rhs(3.0, [4.0])

        assert result.shape == (1,)
        assert result[0] == pytest.approx(11.0)

    def test_missing_parameter_fails_closed(self):
        X = sp.Symbol("X")
        k = sp.Symbol("k")

        with pytest.raises(ValueError, match="Missing numeric values"):
            build_rhs_from_sympy([X], [k * X], {})


class TestNotebookLatexGeneration:
    """Tests for notebook-level LaTeX helpers."""

    def test_latex_odes_from_sym_orders_and_formats_equations(self):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(vars=[X], params={}, odes={X: -X}, initials={X: 1.0})

        result = latex_odes_from_sym(sym)

        assert "\\dot{X}" in result
        assert "- X" in result or "-X" in result

    def test_latex_ssys_from_antimony_uses_antimony_structure(self):
        result = latex_ssys_from_antimony("Z_1' = epsilon*Z_1 - k*Z_1^2")

        assert "\\dot{Z_{1}}" in result
        assert "\\epsilon" in result
        assert "Z_{1}^{2}" in result

    def test_latex_ssys_from_antimony_reports_missing_odes(self):
        result = latex_ssys_from_antimony("k = 1")

        assert "No ODEs found" in result


class TestSimplifyExponentContent:
    """Tests for exponent content simplification."""

    def test_integer_decimal_simplified(self):
        """Test 1.0 → 1 simplification."""
        result = _simplify_exponent_content("1.0")
        assert result == "1"

    def test_negative_integer_simplified(self):
        """Test -2.0 → -2 simplification."""
        result = _simplify_exponent_content("-2.0")
        assert result == "-2"

    def test_non_integer_preserved(self):
        """Test that non-integers are preserved."""
        result = _simplify_exponent_content("0.5")
        assert "0.5" in result

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        result = _simplify_exponent_content("  a + b  ")
        assert result == "a + b"


class TestIsMonomial:
    """Tests for monomial detection."""

    def test_notebook_monomial_detection_uses_core_semantics(self):
        """Notebook monomial detection should match core symbolic-exponent semantics."""
        x = sp.Symbol("x")
        a = sp.Symbol("a")

        assert _is_monomial(x**a) is True
        assert _is_monomial(x**a) == math_utils._is_term_monomial(x**a)

    def test_number_is_monomial(self):
        """Test that a number is a monomial."""
        assert _is_monomial(sp.Float(5.0)) is True

    def test_symbol_is_monomial(self):
        """Test that a symbol is a monomial."""
        x = sp.Symbol("x")
        assert _is_monomial(x) is True

    def test_power_is_monomial(self):
        """Test that x^2 is a monomial."""
        x = sp.Symbol("x")
        assert _is_monomial(x**2) is True

    def test_product_is_monomial(self):
        """Test that 3*x^2 is a monomial."""
        x = sp.Symbol("x")
        assert _is_monomial(3 * x**2) is True

    def test_multi_var_product_is_monomial(self):
        """Test that 2*x*y^3 is a monomial."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        assert _is_monomial(2 * x * y**3) is True

    def test_sum_not_monomial(self):
        """Test that x + y is not a monomial."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        assert _is_monomial(x + y) is False

    def test_exp_not_monomial(self):
        """Test that exp(x) is not a monomial."""
        x = sp.Symbol("x")
        assert _is_monomial(sp.exp(x)) is False


class TestGetTermSign:
    """Tests for term sign detection."""

    def test_positive_number(self):
        """Test positive number sign."""
        assert _get_term_sign(sp.Float(5.0)) == 1

    def test_negative_number(self):
        """Test negative number sign."""
        assert _get_term_sign(sp.Float(-3.0)) == -1

    def test_positive_monomial(self):
        """Test positive monomial."""
        x = sp.Symbol("x")
        assert _get_term_sign(2 * x) == 1

    def test_negative_monomial(self):
        """Test negative monomial."""
        x = sp.Symbol("x")
        assert _get_term_sign(-3 * x**2) == -1

    def test_zero_is_positive(self):
        """Test that zero is considered positive."""
        assert _get_term_sign(sp.Float(0.0)) == 1


class TestIsAlreadySSsystem:
    """Tests for S-system detection."""

    def test_notebook_ssystem_detection_delegates_to_core_classifier(self):
        """Notebook helper classification should match classify_system()."""
        X = sp.Symbol("X", positive=True)
        a = sp.Symbol("a", positive=True)
        b = sp.Symbol("b", positive=True)

        sym = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 0.1},
            odes={X: a * X**2 - b * X},
            initials={X: 1.0},
        )

        assert core_classify_system(sym) == SystemClass.CANONICAL_SSYSTEM
        assert _is_already_ssystem(sym) is True

    def test_canonical_ssystem(self):
        """Test canonical S-system detection."""
        X = sp.Symbol("X", positive=True)
        a = sp.Symbol("a", positive=True)
        b = sp.Symbol("b", positive=True)

        # X' = a*X - b*X^2 (one pos, one neg monomial)
        sym = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 0.1},
            odes={X: a * X - b * X**2},
            initials={X: 1.0},
        )

        assert _is_already_ssystem(sym) is True

    def test_multi_term_not_ssystem(self):
        """Test that 3+ terms is not S-system."""
        X = sp.Symbol("X", positive=True)

        # X' = X - X^2 - X^3 (two negative terms)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: X - X**2 - X**3},
            initials={X: 1.0},
        )

        assert _is_already_ssystem(sym) is False


class TestIsAlreadyGMA:
    """Tests for GMA detection."""

    def test_all_monomials_is_gma(self):
        """Test that system with all monomial terms is GMA."""
        X = sp.Symbol("X", positive=True)

        # X' = X + X^2 - X^3 (all monomials)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: X + X**2 - X**3},
            initials={X: 1.0},
        )

        assert _is_already_gma(sym) is True

    def test_non_monomial_not_gma(self):
        """Test that non-monomial term disqualifies GMA."""
        X = sp.Symbol("X", positive=True)

        # X' = exp(X) (not a monomial)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.exp(X)},
            initials={X: 0.0},
        )

        assert _is_already_gma(sym) is False


class TestParseAntimonyOdes:
    """Tests for ODE extraction from Antimony text."""

    def test_simple_ode(self):
        """Test parsing simple ODE."""
        text = "X' = -k*X"
        odes = parse_antimony_odes(text)
        assert len(odes) == 1
        assert odes[0][0] == "X"
        assert "-k*X" in odes[0][1]

    def test_multiple_odes(self):
        """Test parsing multiple ODEs."""
        text = """
        X' = -k*X
        Y' = k*X - d*Y
        """
        odes = parse_antimony_odes(text)
        assert len(odes) == 2
        var_names = [o[0] for o in odes]
        assert "X" in var_names
        assert "Y" in var_names

    def test_ode_with_semicolon(self):
        """Test that semicolons are stripped."""
        text = "X' = -k*X;"
        odes = parse_antimony_odes(text)
        assert len(odes) == 1
        assert odes[0][1].strip() == "-k*X"

    def test_comments_ignored(self):
        """Test that comments are ignored."""
        text = """
        // This is a comment
        X' = -k*X  // inline comment
        """
        odes = parse_antimony_odes(text)
        assert len(odes) == 1

    def test_empty_returns_empty(self):
        """Test that empty text returns empty list."""
        odes = parse_antimony_odes("")
        assert len(odes) == 0

    def test_no_odes_returns_empty(self):
        """Test that text without ODEs returns empty list."""
        text = "k = 0.5\nX = 1.0"
        odes = parse_antimony_odes(text)
        assert len(odes) == 0


class TestLatexFactorMap:
    """Tests for factor map LaTeX generation."""

    def test_empty_factor_map(self):
        """Test empty factor map output."""
        X = sp.Symbol("X", positive=True)
        rec = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={X: 1.0},
            variables=[X],
            factor_map={},  # Empty
        )

        result = latex_factor_map(rec)
        assert "No factorization" in result or "direct form" in result

    def test_single_factor(self):
        """Test single factor mapping."""
        X = sp.Symbol("X", positive=True)
        Z = sp.Symbol("Z_1", positive=True)

        rec = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={X: [Z]},
        )

        result = latex_factor_map(rec)
        assert "aligned" in result
        assert "X" in result
        assert "Z" in result

    def test_multiple_factors(self):
        """Test multiple factor mapping (product)."""
        X = sp.Symbol("X", positive=True)
        Z1 = sp.Symbol("Z_1", positive=True)
        Z2 = sp.Symbol("Z_2", positive=True)

        rec = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z1: 1.0, Z2: 1.0},
            variables=[Z1, Z2],
            factor_map={X: [Z1, Z2]},
        )

        result = latex_factor_map(rec)
        assert "cdot" in result  # Product notation


class TestExpandExpsThroughFactors:
    """Tests for expanding exponents through factor map."""

    def test_notebook_imports_core_exponent_expansion(self):
        """Notebook helpers should import the core exponent expansion utility."""
        assert _expand_exps_through_factors is math_utils._expand_exps_through_factors

    def test_simple_expansion(self):
        """Test simple expansion through factor map."""
        X = sp.Symbol("X")
        Z = sp.Symbol("Z_1")

        exps = {X: 2.0}
        factor_map = {X: [Z]}

        result = _expand_exps_through_factors(exps, factor_map)

        assert Z in result
        assert result[Z] == 2.0
        assert X not in result

    def test_no_mapping_passes_through(self):
        """Test that unmapped variables pass through."""
        Y = sp.Symbol("Y")

        exps = {Y: 3.0}
        factor_map = {}

        result = _expand_exps_through_factors(exps, factor_map)

        assert Y in result
        assert result[Y] == 3.0

    def test_multiple_factors_split(self):
        """Test that exponent is distributed to multiple factors."""
        X = sp.Symbol("X")
        Z1 = sp.Symbol("Z_1")
        Z2 = sp.Symbol("Z_2")

        exps = {X: 2.0}
        factor_map = {X: [Z1, Z2]}

        result = _expand_exps_through_factors(exps, factor_map)

        # X^2 → Z1^2 * Z2^2 (each factor gets the exponent)
        assert Z1 in result
        assert Z2 in result
        assert result[Z1] == 2.0
        assert result[Z2] == 2.0


class TestProductExpr:
    """Tests for building product expressions."""

    def test_notebook_imports_core_product_expr(self):
        """Notebook helpers should import the core product builder."""
        assert product_expr is math_utils.product_expr

    def test_symbolic_exponent_matches_core_output(self):
        """Notebook rendering helpers and core output use the same symbolic powers."""
        x = sp.Symbol("x", positive=True)
        a = sp.Symbol("a")

        assert product_expr(2, {x: a}) == math_utils.product_expr(2, {x: a})

    def test_simple_product(self):
        """Test simple product."""
        x = sp.Symbol("x", positive=True)
        result = product_expr(2, {x: 3})
        expected = 2 * x**3
        assert sp.simplify(result - expected) == 0

    def test_symbolic_coefficient(self):
        """Test symbolic coefficient."""
        x = sp.Symbol("x", positive=True)
        k = sp.Symbol("k", positive=True)
        result = product_expr(k, {x: 2})
        expected = k * x**2
        assert sp.simplify(result - expected) == 0

    def test_zero_exponent_skipped(self):
        """Test that zero exponents are skipped."""
        x = sp.Symbol("x", positive=True)
        result = product_expr(5, {x: 0})
        assert result == 5

    def test_empty_exponents(self):
        """Test with no exponents."""
        result = product_expr(3, {})
        assert result == 3


class TestLoadAndReport:
    """Smoke tests for the rendered notebook report workflow."""

    def test_load_and_report_renders_successful_report(self, tmp_path, monkeypatch):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(vars=[X], params={}, odes={X: -X}, initials={X: 1.0})
        rec = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={X: 1.0},
            variables=[X],
            factor_map={},
        )
        ant_path = tmp_path / "original.ant"
        ant_path.write_text("X' = -X\nX = 1")
        recast_path = tmp_path / "recast.ant"
        recast_path.write_text("X' = -X\nX = 1")

        rendered = []
        monkeypatch.setattr(nh, "display", lambda obj: rendered.append(obj))
        monkeypatch.setattr(nh.plt, "show", lambda: None)
        monkeypatch.setattr(nh, "parse_antimony_via_sbml", lambda text: sym)
        monkeypatch.setattr(nh.ssys, "recast_to_ssystem", lambda parsed, mode="simplified": rec)

        def fake_simulate_ode(model_ir, t_start, t_end, n_points):
            return {
                "success": True,
                "t": np.array([0.0, 1.0]),
                "y": np.array([[1.0], [0.5]]),
                "state_names": ["X"],
                "message": "",
            }

        monkeypatch.setattr("ssys.ode_backends.simulate_ode", fake_simulate_ode)

        nh.load_and_report(str(ant_path), str(recast_path), T=1.0, steps=1)

        assert rendered
        assert any("Trajectory Comparison" in getattr(item, "data", str(item)) for item in rendered)

    def test_load_and_report_stops_when_original_simulation_fails(self, tmp_path, monkeypatch):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(vars=[X], params={}, odes={X: -X}, initials={X: 1.0})
        rec = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={X: 1.0},
            variables=[X],
            factor_map={},
        )
        ant_path = tmp_path / "original.ant"
        ant_path.write_text("X' = -X\nX = 1")
        recast_path = tmp_path / "recast.ant"
        recast_path.write_text("X' = -X\nX = 1")

        rendered = []
        monkeypatch.setattr(nh, "display", lambda obj: rendered.append(obj))
        monkeypatch.setattr(nh, "parse_antimony_via_sbml", lambda text: sym)
        monkeypatch.setattr(nh.ssys, "recast_to_ssystem", lambda parsed, mode="simplified": rec)
        monkeypatch.setattr(
            "ssys.ode_backends.simulate_ode",
            lambda *args, **kwargs: {
                "success": False,
                "message": "forced failure",
                "t": np.array([]),
                "y": np.empty((0, 0)),
                "state_names": [],
            },
        )

        nh.load_and_report(str(ant_path), str(recast_path), T=1.0, steps=1)

        assert any("forced failure" in getattr(item, "data", str(item)) for item in rendered)

    def test_load_and_report_simulates_non_unit_compartment(
        self, tmp_path, monkeypatch
    ):
        """A non-unit-compartment model still renders a full simulation section,
        because the report drives RoadRunner from the SBML SymSystem's cached
        Antimony text."""
        ant_text = (
            "model nonunit()\n"
            "  compartment cell = 2;\n"
            "  species A in cell, B in cell;\n"
            "  A = 1; B = 0; k = 0.5;\n"
            "  J0: A -> B; cell*k*A;\n"
            "end\n"
        )
        sym = nh.parse_antimony_via_sbml(ant_text)
        rec = ssys.recast_to_ssystem(sym, mode="simplified")
        rec_text = ssys.ssystem_to_antimony(rec, model_name="nonunit_recast", mode="simplified")

        ant_path = tmp_path / "nonunit.ant"
        ant_path.write_text(ant_text)
        recast_path = tmp_path / "nonunit_recast.ant"
        recast_path.write_text(rec_text)

        rendered = []
        monkeypatch.setattr(nh, "display", lambda obj: rendered.append(obj))
        monkeypatch.setattr(nh.plt, "show", lambda: None)

        nh.load_and_report(str(ant_path), str(recast_path), T=2.0, steps=5)

        assert any(
            "Trajectory Comparison" in getattr(item, "data", str(item)) for item in rendered
        )


class TestIsNonautonomous:
    """Tests for explicit time dependence detection."""

    def test_time_symbol_detected(self):
        """Test that 'time' symbol is detected."""
        X = sp.Symbol("X", positive=True)
        time_sym = sp.Symbol("time")

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: -X + time_sym},  # X' = -X + time
            initials={X: 1.0},
        )

        assert is_nonautonomous(sym) is True

    def test_no_time_is_autonomous(self):
        """Test autonomous system detection."""
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: -X},  # X' = -X
            initials={X: 1.0},
        )

        assert is_nonautonomous(sym) is False

    def test_time_in_complex_expression(self):
        """Test time in complex expression."""
        X = sp.Symbol("X", positive=True)
        time_sym = sp.Symbol("time")
        k = sp.Symbol("k", positive=True)

        sym = SymSystem(
            vars=[X],
            params={"k": 0.1},
            odes={X: k * sp.sin(time_sym) * X},
            initials={X: 1.0},
        )

        assert is_nonautonomous(sym) is True

    def test_multiple_odes_one_with_time(self):
        """Test multiple ODEs, one with time."""
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)
        time_sym = sp.Symbol("time")

        sym = SymSystem(
            vars=[X, Y],
            params={},
            odes={X: -X, Y: time_sym * Y},
            initials={X: 1.0, Y: 1.0},
        )

        assert is_nonautonomous(sym) is True


class TestFindClockVariable:
    """Tests for clock variable detection."""

    def test_clock_detected(self):
        """Test ODE = 1 detected as clock."""
        t = sp.Symbol("t", positive=True)
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[t, X],
            params={},
            odes={t: sp.Integer(1), X: -X * t},  # t' = 1, X' = -X*t
            initials={t: 0.0, X: 1.0},
        )

        result = find_clock_variable(sym)
        assert result == "t"

    def test_no_clock_returns_none(self):
        """Test no clock variable returns None."""
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)

        sym = SymSystem(
            vars=[X, Y],
            params={},
            odes={X: -X, Y: X - Y},
            initials={X: 1.0, Y: 0.0},
        )

        result = find_clock_variable(sym)
        assert result is None

    def test_ode_equals_constant_not_one(self):
        """Test that ODE = 2 (not 1) is not clock."""
        Z = sp.Symbol("Z", positive=True)
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[Z, X],
            params={},
            odes={Z: sp.Integer(2), X: -X},  # Z' = 2 (not clock)
            initials={Z: 0.0, X: 1.0},
        )

        result = find_clock_variable(sym)
        assert result is None

    def test_clock_with_float_one(self):
        """Test that ODE = 1.0 is also detected."""
        t = sp.Symbol("t", positive=True)

        sym = SymSystem(
            vars=[t],
            params={},
            odes={t: sp.Float(1.0)},
            initials={t: 0.0},
        )

        result = find_clock_variable(sym)
        assert result == "t"


class TestWasNonautonomous:
    """Tests for detecting systems with time dependence (explicit or lifted)."""

    def test_explicit_time_detected(self):
        """Test explicit time detected."""
        X = sp.Symbol("X", positive=True)
        time_sym = sp.Symbol("time")

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: time_sym * X},
            initials={X: 1.0},
        )

        assert was_nonautonomous(sym) is True

    def test_clock_variable_detected(self):
        """Test clock variable detected as lifted nonautonomous."""
        t = sp.Symbol("t", positive=True)
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[t, X],
            params={},
            odes={t: sp.Integer(1), X: -X * t},
            initials={t: 0.0, X: 1.0},
        )

        assert was_nonautonomous(sym) is True

    def test_pure_autonomous_not_detected(self):
        """Test pure autonomous system."""
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)

        sym = SymSystem(
            vars=[X, Y],
            params={},
            odes={X: -X * Y, Y: X * Y - Y},
            initials={X: 1.0, Y: 0.5},
        )

        assert was_nonautonomous(sym) is False


class TestGetAutonomyLabel:
    """Tests for autonomy label generation."""

    def test_pure_autonomous(self):
        """Test pure autonomous system label."""
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: -X},
            initials={X: 1.0},
        )

        label, clock = get_autonomy_label(sym)
        assert label == "autonomous"
        assert clock is None

    def test_nonautonomous(self):
        """Test nonautonomous system label."""
        X = sp.Symbol("X", positive=True)
        time_sym = sp.Symbol("time")

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: time_sym * X},
            initials={X: 1.0},
        )

        label, clock = get_autonomy_label(sym)
        assert label == "nonautonomous"
        assert clock is None

    def test_lifted_as_recast(self):
        """Test lifted system when is_recast=True."""
        t = sp.Symbol("t", positive=True)
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[t, X],
            params={},
            odes={t: sp.Integer(1), X: -X},
            initials={t: 0.0, X: 1.0},
        )

        label, clock = get_autonomy_label(sym, is_recast=True)
        assert label == "autonomous, lifted"
        assert clock == "t"

    def test_lifted_when_orig_was_nonautonomous(self):
        """Test lifted when original was nonautonomous."""
        t = sp.Symbol("t", positive=True)
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[t, X],
            params={},
            odes={t: sp.Integer(1), X: -X},
            initials={t: 0.0, X: 1.0},
        )

        label, clock = get_autonomy_label(
            sym, is_recast=False, orig_was_nonautonomous=True
        )
        assert label == "autonomous, lifted"
        assert clock == "t"

    def test_clock_not_lifted_if_pure_autonomous_context(self):
        """Test clock in pure autonomous context isn't marked lifted."""
        t = sp.Symbol("t", positive=True)

        sym = SymSystem(
            vars=[t],
            params={},
            odes={t: sp.Integer(1)},
            initials={t: 0.0},
        )

        # Without recast or orig_nonautonomous flag, just autonomous
        label, clock = get_autonomy_label(
            sym, is_recast=False, orig_was_nonautonomous=False
        )
        assert label == "autonomous"
        assert clock is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
