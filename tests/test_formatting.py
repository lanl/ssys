"""Tests for output formatting and classification functions."""

import re

import antimony
import pytest
import sympy as sp

from ssys.recaster import (
    GMAEquation,
    RecastResult,
    RecastStatus,
    SolverRequirement,
    SSysEquation,
    SymSystem,
    SystemClass,
    _antimony_to_sympy_syntax,
    _format_sim_metadata_lines,
    _sympy_to_antimony_syntax,
    classify_result,
    classify_solver_requirement,
    classify_sym_system_solver_requirement,
    classify_system,
    gma_to_antimony,
    latex_ssys,
    parse_antimony_via_sbml,
    product_to_antimony,
    recast_to_ssystem,
    ssystem_to_antimony,
)


def test_public_parsing_and_formatting_shims_hide_private_helpers():
    """Focused public modules should not re-export underscore-prefixed internals."""
    import ssys.formatting as formatting
    import ssys.parsing as parsing

    assert "_antimony_to_sympy_syntax" not in parsing.__all__
    assert "_sympy_to_antimony_syntax" not in parsing.__all__
    assert "_format_antimony_token" not in formatting.__all__
    assert "_sanitize_antimony_name" not in formatting.__all__
    assert not hasattr(parsing, "_antimony_to_sympy_syntax")
    assert not hasattr(formatting, "_format_antimony_token")


def _assert_antimony_roundtrips(output: str) -> None:
    antimony.clearPreviousLoads()
    result = antimony.loadAntimonyString(output)
    assert result >= 0, f"Antimony parse failed:\n{antimony.getLastError()}\n\n{output}"


def _executable_lines(output: str) -> list[str]:
    lines = []
    for line in output.splitlines():
        stripped = line.split("//", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return lines


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

    def test_symbolic_add_exponent_parenthesized(self):
        """Test that symbolic Add exponents are parenthesized to avoid ambiguity.

        Bug regression test for d741354:
        Without parentheses, x^-C - 1 parses as (x^-C) - 1 (subtraction)
        instead of x^(-C-1) (exponent).
        """
        x = sp.Symbol("x", positive=True)
        C = sp.Symbol("C", positive=True)
        coeff = sp.Float(1.0)
        # Symbolic exponent: -C - 1 (an Add expression)
        exps = {x: -C - 1}
        result = product_to_antimony(coeff, exps)
        # Exponent must be parenthesized: x^(-C - 1), not x^-C - 1
        assert "^(-C - 1)" in result or "^(-1 - C)" in result

    def test_symbolic_simple_exponent_not_over_parenthesized(self):
        """Test that simple symbolic exponents don't get unnecessary parentheses.

        For exponents that are just a symbol or symbol*constant,
        no parentheses are needed: x^C is fine, doesn't need x^(C).
        """
        x = sp.Symbol("x", positive=True)
        C = sp.Symbol("C", positive=True)
        coeff = sp.Float(1.0)
        # Symbolic exponent: just C (not an Add expression)
        exps = {x: C}
        result = product_to_antimony(coeff, exps)
        # Should NOT have extra parentheses around C
        assert "^C" in result
        assert "^(C)" not in result

    def test_symbolic_negative_exponent_not_over_parenthesized(self):
        """Test that negated symbolic exponents without sums don't get parentheses.

        For -C (just a negated symbol, represented as Mul(-1, C), not Add),
        parentheses are not needed: x^-C is unambiguous.
        """
        x = sp.Symbol("x", positive=True)
        C = sp.Symbol("C", positive=True)
        coeff = sp.Float(1.0)
        # Symbolic exponent: -C (Mul, not Add)
        exps = {x: -C}
        result = product_to_antimony(coeff, exps)
        # Should NOT have parentheses around -C (it's not an Add)
        assert "^-C" in result
        # But should not have spurious parentheses if it's just -C
        # (sympy may format as -C or -1*C)

    def test_symbolic_mul_exponent_parenthesized(self):
        """Test multiplicative symbolic exponents are parenthesized."""
        x = sp.Symbol("x", positive=True)
        C = sp.Symbol("C", positive=True)

        result = product_to_antimony(1.0, {x: 2 * C})

        assert "x^(2*C)" in result or "x^(2 * C)" in result
        assert "x^2*C" not in result

    def test_rational_coefficient_preserved_as_fraction(self):
        """Test that rational coefficients are preserved as fractions, not decimals."""
        x = sp.Symbol("x", positive=True)
        coeff = sp.Rational(1, 6)  # 1/6
        exps = {x: 3.0}
        result = product_to_antimony(coeff, exps)
        # Should contain (1/6), not 0.166667
        assert "(1/6)" in result
        assert "0.16" not in result

    def test_rational_coefficient_various_fractions(self):
        """Test various rational coefficients are preserved as fractions."""
        x = sp.Symbol("x", positive=True)

        # Test 2/3
        coeff = sp.Rational(2, 3)
        exps = {x: 1.0}
        result = product_to_antimony(coeff, exps)
        assert "(2/3)" in result

        # Test 3/7
        coeff = sp.Rational(3, 7)
        result = product_to_antimony(coeff, exps)
        assert "(3/7)" in result

    def test_rational_coefficient_integer_stays_integer(self):
        """Test that rationals with denominator 1 stay as integers."""
        x = sp.Symbol("x", positive=True)
        coeff = sp.Rational(5, 1)  # 5/1 = 5
        exps = {x: 2.0}
        result = product_to_antimony(coeff, exps)
        # Should be "5*x^2", not "(5/1)*x^2"
        assert "5*x" in result or result.startswith("5")
        assert "/1" not in result

    def test_symbolic_coefficients_dummy_constants_and_fractional_exponents(self):
        """Product formatting preserves symbolic and constant-term edge cases."""
        x = sp.Symbol("x", positive=True)
        a = sp.Symbol("a", positive=True)
        b = sp.Symbol("b", positive=True)
        dummy = sp.Symbol("dummy_const", positive=True)

        assert product_to_antimony(sp.Integer(0), {x: 1.0}) == "0"
        assert product_to_antimony(a + b, {x: 1.0}) == "(a + b)*x"
        assert product_to_antimony(sp.Integer(1), {dummy: sp.Integer(0)}) == (
            "1*dummy_const^0"
        )
        assert product_to_antimony(1.0, {x: 0.0}) == "1"
        assert product_to_antimony(1.0, {x: sp.Float(0.5)}) == "x^0.500000000000000"

    def test_private_factor_formatting_preserves_precedence(self):
        """Coefficient factor formatting keeps powers, sums, and exact rationals parseable."""
        from ssys._recaster.antimony_formatting import _format_factor, _format_symbolic_coeff

        x = sp.Symbol("x", positive=True)
        a = sp.Symbol("a", positive=True)
        b = sp.Symbol("b", positive=True)
        c = sp.Symbol("c", positive=True)

        assert _format_factor(sp.Rational(2, 5)) == "(2/5)"
        assert _format_factor(sp.Float(2.5)) == "2.5"
        assert _format_factor(sp.Pow(x, sp.Float(1.0), evaluate=False)) == "x"
        assert _format_factor(sp.Pow(sp.Integer(2), sp.Integer(3), evaluate=False)) == "2^3"
        assert _format_factor((a + b) ** (-c - 1)) == "((a + b))^(-c - 1)"
        assert _format_factor(sp.sin(x)) == "sin(x)"
        assert _format_symbolic_coeff(2 * a * (b + c)) == ["2", "a", "(b + c)"]


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


class TestSolverRequirementClassification:
    """Tests for recast solver requirement metadata."""

    def test_ode_only_without_assignment_or_constraints(self):
        Z = sp.Symbol("Z_1", positive=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(Z, (sp.Float(1.0), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))
            ],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
        )

        assert classify_solver_requirement(result) == SolverRequirement.ODE_ONLY

    def test_assignment_rule_only_output_does_not_force_dae(self):
        Z = sp.Symbol("Z_1", positive=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(Z, (sp.Float(1.0), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))
            ],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={},
            assignment_rules={"observable": "Z_1 + 1"},
        )

        assert (
            classify_solver_requirement(result)
            == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
        )

    def test_coupled_algebraic_constraint_requires_dae(self):
        Z = sp.Symbol("Z_1", positive=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(Z, (sp.Float(1.0), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))
            ],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            algebraic_constraints=["Z_1 - 1"],
        )

        assert classify_solver_requirement(result) == SolverRequirement.DAE_REQUIRED

    def test_lifted_assignment_mode_is_assignment_rule_solver(self):
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        K = sp.Symbol("K", positive=True)
        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={X: 1.0, Y: 2.0},
            variables=[X, Y],
            gma_equations=[
                GMAEquation(X, [(sp.Float(1.0), {X: 1.0})], []),
                GMAEquation(Y, [(sp.Float(1.0), {X: 1.0})], []),
            ],
            params={"K": 1.0},
            auxiliary_defs={Y: X + K},
        )

        assert (
            classify_solver_requirement(result, lifted_mode="assignment")
            == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
        )
        output = gma_to_antimony(result, model_name="solver_meta", lifted_mode="assignment")
        _assert_antimony_roundtrips(output)
        assert "@SSYS SOLVER_REQUIREMENT=ode_with_assignment_rules" in output

    def test_clock_auxiliary_definition_does_not_force_assignment_solver(self):
        T = sp.Symbol("T", positive=True)
        time = sp.Symbol("time")
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(T, (sp.Float(1.0), {}), (sp.Float(0.0), {}))],
            initials={T: 0.0},
            variables=[T],
            factor_map={T: [T]},
            auxiliary_defs={T: time},
        )

        assert classify_solver_requirement(result) == SolverRequirement.ODE_ONLY

    def test_uncoupled_algebraic_constraint_does_not_force_dae(self):
        Z = sp.Symbol("Z_1", positive=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(Z, (sp.Float(1.0), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))
            ],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            algebraic_constraints=["observable - 1"],
        )

        assert classify_solver_requirement(result) == SolverRequirement.ODE_ONLY

    def test_assignment_rule_targeting_state_requires_dae(self):
        Z = sp.Symbol("Z_1", positive=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            assignment_rules={"Z_1": "observable + 1"},
        )

        assert classify_solver_requirement(result) == SolverRequirement.DAE_REQUIRED

    def test_lifted_ode_mode_state_dependent_auxiliary_requires_dae(self):
        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        K = sp.Symbol("K", positive=True)
        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={X: 1.0, Y: 2.0},
            variables=[X, Y],
            gma_equations=[GMAEquation(X, [(sp.Float(1.0), {X: 1.0, Y: -1.0})], [])],
            params={"K": 1.0},
            auxiliary_defs={Y: X + K},
        )

        assert classify_solver_requirement(result) == SolverRequirement.DAE_REQUIRED

    def test_sym_system_non_state_assignment_rule_uses_assignment_solver(self):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: sp.Symbol("rate")},
            initials={X: 1.0},
            assignment_rules={"rate": "k * X"},
        )

        assert classify_sym_system_solver_requirement(sym) == (
            SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
        )

    def test_sym_system_state_assignment_rule_requires_dae(self):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: -X},
            initials={X: 1.0},
            assignment_rules={"X": "1"},
        )

        assert classify_sym_system_solver_requirement(sym) == SolverRequirement.DAE_REQUIRED


class TestAntimonyExport:
    """Tests for exporting to Antimony format."""

    def test_ssystem_to_antimony_basic(self):
        """Test basic S-system to Antimony export."""
        text = """
        species X
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        sym = parse_antimony_via_sbml(text)
        rec = recast_to_ssystem(sym)
        output = ssystem_to_antimony(rec, model_name="test")
        _assert_antimony_roundtrips(output)

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
        species X
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        sym = parse_antimony_via_sbml(text)
        rec = recast_to_ssystem(sym)
        output = ssystem_to_antimony(rec, model_name="123model")
        _assert_antimony_roundtrips(output)

        # Numeric prefix should become m_123model
        assert "model m_123model" in output

    def test_ssystem_to_antimony_preserves_float_precision_for_initials(self):
        """Test that sensitive ICs are emitted without shortening significant digits."""
        Z4 = sp.Symbol("Z4", positive=True)
        initial = 4.358898943540674

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z4, (sp.Float(1), {Z4: 1.0}), (sp.Float(0.5), {Z4: 1.0}))],
            initials={Z4: initial},
            variables=[Z4],
            factor_map={Z4: [Z4]},
            params={},
        )

        simplified = ssystem_to_antimony(result, model_name="precision_test", mode="simplified")
        canonical = ssystem_to_antimony(result, model_name="precision_test", mode="canonical")
        _assert_antimony_roundtrips(simplified)
        _assert_antimony_roundtrips(canonical)

        assert "Z4 = 4.358898943540674;" in simplified
        assert "Z4 = 4.358898943540674;" in canonical
        assert "Z4 = 4.3589;" not in simplified
        assert "Z4 = 4.3589;" not in canonical

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
        _assert_antimony_roundtrips(output)

        assert "model gma_test" in output
        assert "end" in output
        assert "GMA" in output  # Should have GMA comment

    def test_gma_assignment_mode_outputs_lifted_rules_without_clock_or_dummy(self):
        """Assignment-mode GMA keeps lifted manifolds exact and leaves clocks as states."""
        X = sp.Symbol("X", positive=True)
        T = sp.Symbol("T", positive=True)
        Y = sp.Symbol("Y_1", positive=True)
        dummy = sp.Symbol("dummy_const", positive=True)

        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            gma_equations=[
                GMAEquation(
                    var=X,
                    production=[(sp.Integer(1), {X: 1.0})],
                    degradation=[],
                ),
                GMAEquation(
                    var=T,
                    production=[(sp.Integer(1), {})],
                    degradation=[],
                ),
                GMAEquation(
                    var=Y,
                    production=[(sp.Integer(1), {X: 1.0})],
                    degradation=[],
                ),
            ],
            variables=[X, T, Y],
            initials={X: 1.0, T: 0.0, Y: 2.0, ("compartment", "cell"): 1.0},
            auxiliary_defs={T: sp.Symbol("time"), Y: X + 1, dummy: sp.Integer(1)},
            assignment_rules={"rate": "X + 1"},
            params={"k": 0.5, "rate": 9.0},
            factor_map={X: [X]},
            canonical_refusal_reason="multiple incompatible terms",
        )

        output = gma_to_antimony(result, model_name="gma_assignment", lifted_mode="assignment")

        assert "Y_1 := X + 1;" in output
        assert "T := time;" not in output
        assert "dummy_const" not in output
        assert "rate = 9" not in output
        assert "rate := X + 1;" in output
        assert "Canonical S-system recast was not attempted" in output

    def test_failed_recast_output_includes_unknown_reason_and_initials(self):
        """Failed recasts produce an auditable Antimony stub."""
        X = sp.Symbol("X", positive=True)
        result = RecastResult(
            status=RecastStatus.FAILED,
            equations=[],
            variables=[],
            initials={X: 1.25},
        )

        output = ssystem_to_antimony(result, model_name="failed.model")

        assert "model failed_model()" in output
        assert "Recasting failed for unknown reason." in output
        assert "// X = 1.25" in output
        assert "// No recast equations generated." in output

    def test_simplified_ssystem_formats_symbolic_constants_and_ic_fallbacks(self):
        """Simplified output handles constant terms, symbolic ICs, and observables."""
        A, B, C, D, Obs, Q, k = sp.symbols("A B C D Obs Q k", positive=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(A, (k, {}), (sp.Integer(2), {A: 1.0})),
                SSysEquation(B, (sp.Integer(3), {B: 1.0}), (k + 1, {})),
                SSysEquation(C, (k + 2, {}), (sp.Rational(1, 2), {})),
                SSysEquation(D, (sp.Integer(1), {D: 1.0}), (sp.Integer(0), {})),
            ],
            variables=[A, B, C, D],
            initials={A: 0.0, B: 2.0, C: 3.0, ("compartment", "cell"): 1.0},
            initial_exprs={A: "k + 1"},
            params={"k": 0.5, "D": 4.0, "Obs": 99.0},
            factor_map={Obs: [A, B], Q: [C], D: [D]},
            assignment_rules={"obs": "A + k"},
        )

        output = ssystem_to_antimony(result, model_name="constant_terms", mode="simplified")

        assert "A = k + 1;" in output
        assert "D = 4;" in output
        assert "Obs := A * B;" in output
        assert "Q := C;" in output
        assert "D := D;" not in output
        assert "A' = k - 2*A;" in output
        assert "B' = 3*B - k + 1;" in output
        assert "C' = k + 2 - 0.5;" in output


class TestAntimonyReservedNameRoundTrip:
    """Round-trip tests for generated Antimony with reserved identifiers."""

    def test_simplified_roundtrip_sanitizes_reserved_names_everywhere(self):
        """Reserved parameter, state, compartment, and observable names parse."""
        dna = sp.Symbol("DNA", positive=True)
        rna = sp.Symbol("RNA", positive=True)

        sym = SymSystem(
            vars=[dna],
            params={"RNA": 0.5},
            odes={dna: -rna * dna},
            initials={dna: 1.0},
            compartments={"compartment": 1.0},
        )

        result = recast_to_ssystem(sym, mode="simplified")
        output = ssystem_to_antimony(result, model_name="reserved_simple", mode="simplified")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        ode_text = "\n".join(line for line in _executable_lines(output) if "'" in line)

        assert "compartment compartment_var = 1" in executable
        assert "RNA_var = 0.5" in executable
        assert "DNA_var :=" in executable
        assert "DNA :=" not in executable
        assert "RNA_var" in ode_text
        assert not re.search(r"\bRNA\b", ode_text)

    def test_canonical_roundtrip_sanitizes_reserved_observable_and_assignment_rule(self):
        dna = sp.Symbol("DNA", positive=True)
        k = sp.Symbol("k", positive=True)

        sym = SymSystem(
            vars=[dna],
            params={"k": 0.5, "RNA": 0.0},
            odes={dna: -k * dna},
            initials={dna: 1.0},
            assignment_rules={"RNA": "DNA + k"},
        )

        result = recast_to_ssystem(sym, mode="canonical")
        output = ssystem_to_antimony(result, model_name="reserved_canonical", mode="canonical")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        assert "DNA_var := Z_1" in executable
        assert "RNA_var := DNA_var + k" in executable
        assert "DNA :=" not in executable
        assert "RNA :=" not in executable

    def test_gma_roundtrip_sanitizes_reserved_names(self):
        dna = sp.Symbol("DNA", positive=True)
        rna = sp.Symbol("RNA", positive=True)

        gma_eq = GMAEquation(
            var=dna,
            production=[(rna, {dna: 1.0})],
            degradation=[(sp.Float(0.5), {dna: 2.0})],
        )
        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={dna: 1.0},
            variables=[dna],
            gma_equations=[gma_eq],
            params={"RNA": 2.0},
            compartments={"compartment": 1.0},
        )

        output = gma_to_antimony(result, model_name="reserved_gma")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        ode_text = "\n".join(line for line in _executable_lines(output) if "'" in line)
        assert "species DNA_var in compartment_var" in executable
        assert "RNA_var = 2" in executable
        assert "DNA_var' = RNA_var*DNA_var" in ode_text
        assert not re.search(r"\bDNA\b", ode_text)
        assert not re.search(r"\bRNA\b", ode_text)

    def test_gma_piecewise_coefficient_uses_antimony_syntax(self):
        z = sp.Symbol("Z_1", positive=True)
        t = sp.Symbol("T", real=True)
        threshold = sp.Symbol("parameter_5", real=True)
        model_value = sp.Symbol("ModelValue_61", positive=True)
        piecewise = sp.Piecewise((-model_value * z, t > threshold), (0, True))
        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={z: 1.0, t: 0.0},
            variables=[z, t],
            gma_equations=[
                GMAEquation(
                    z,
                    production=[(piecewise, {})],
                    degradation=[],
                ),
                GMAEquation(
                    t,
                    production=[(sp.Float(1.0), {})],
                    degradation=[],
                ),
            ],
            params={"ModelValue_61": 100.0, "parameter_5": 0.0},
        )

        output = gma_to_antimony(result, model_name="piecewise_gma")

        assert "Piecewise((" not in output
        assert "piecewise(-ModelValue_61*Z_1, T > parameter_5, 0)" in output
        _assert_antimony_roundtrips(output)

    def test_roundtrip_sanitizes_antimony_function_collision_names(self):
        at = sp.Symbol("at", positive=True)
        gamma = sp.Symbol("gamma", positive=True)

        sym = SymSystem(
            vars=[at],
            params={"gamma": 0.5},
            odes={at: -gamma * at},
            initials={at: 1.0},
        )

        result = recast_to_ssystem(sym, mode="simplified")
        output = ssystem_to_antimony(result, model_name="function_collision")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        assert "gamma_var = 0.5" in executable
        assert "at_var :=" in executable
        assert "gamma =" not in executable
        assert not re.search(r"\bat\b", executable)

    def test_auxiliary_definition_comments_sanitize_reserved_names(self):
        x = sp.Symbol("x", positive=True)
        y = sp.Symbol("Y_1", positive=True)
        gamma = sp.Symbol("gamma", positive=True)

        result = RecastResult(
            status=RecastStatus.GMA,
            equations=[],
            initials={x: 0.3, y: 2.32},
            variables=[x, y],
            gma_equations=[
                GMAEquation(
                    x,
                    production=[(sp.Float(1.0), {x: 1.0})],
                    degradation=[],
                ),
                GMAEquation(
                    y,
                    production=[(sp.Float(1.0), {x: 1.0})],
                    degradation=[],
                ),
            ],
            params={"gamma": 2.02},
            auxiliary_defs={y: gamma + x},
        )

        output = gma_to_antimony(result, model_name="aux_comment_sanitized")
        _assert_antimony_roundtrips(output)

        assert "// Y_1 := gamma_var + x" in output
        assert "// Y_1 := gamma + x" not in output
        executable = "\n".join(_executable_lines(output))
        assert "gamma_var = 2.02" in executable
        assert "gamma =" not in executable

    def test_roundtrip_sanitizes_ext_keyword_identifier(self):
        x = sp.Symbol("X", positive=True)
        ext = sp.Symbol("ext", positive=True)

        sym = SymSystem(
            vars=[x],
            params={"ext": 0.0, "const": 1.0},
            odes={x: -ext * x},
            initials={x: 1.0},
            assignment_rules={"ext": "const + X"},
        )

        result = recast_to_ssystem(sym, mode="simplified")
        output = ssystem_to_antimony(result, model_name="ext_keyword")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        assert "ext_var = 0" in executable
        assert "ext_var := const_var + X" in executable
        assert "const_var = 1" in executable
        assert "ext =" not in executable
        assert "const =" not in executable

    def test_roundtrip_preserves_expression_only_pi_constant(self):
        x = sp.Symbol("X", positive=True)
        rate = sp.Symbol("rate", positive=True)

        sym = SymSystem(
            vars=[x],
            params={"k": 0.5},
            odes={x: -rate * x},
            initials={x: 1.0},
            assignment_rules={"rate": "k + cos(2*pi*time/30)"},
        )

        result = recast_to_ssystem(sym, mode="simplified")
        output = ssystem_to_antimony(result, model_name="pi_constant")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        assert "pi_var" not in executable
        assert re.search(r"\bpi\b", executable)

    def test_roundtrip_sanitizes_declared_pi_identifier(self):
        x = sp.Symbol("X", positive=True)
        pi_symbol = sp.Symbol("pi", positive=True)

        sym = SymSystem(
            vars=[x],
            params={"pi": 0.5},
            odes={x: -pi_symbol * x},
            initials={x: 1.0},
        )

        result = recast_to_ssystem(sym, mode="simplified")
        output = ssystem_to_antimony(result, model_name="pi_identifier")
        _assert_antimony_roundtrips(output)

        executable = "\n".join(_executable_lines(output))
        assert "pi_var = 0.5" in executable
        assert "pi =" not in executable


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


class TestClassifySystemWithAssignmentRules:
    """Tests for classify_system expanding assignment rules before classification."""

    def test_classify_system_with_rational_assignment_rule(self):
        """Test that rational functions in assignment rules trigger GENERAL classification."""
        X = sp.Symbol("X", positive=True)

        # X' = J_pump where J_pump := X^2 / (X^2 + K^2) (Michaelis-Menten)
        # Without expanding rules, J_pump looks like a simple symbol (monomial)
        # After expanding, it contains a rational function → GENERAL
        sym = SymSystem(
            vars=[X],
            params={"K": 1.0},
            odes={X: sp.Symbol("J_pump")},
            initials={X: 1.0},
            assignment_rules={"J_pump": "X**2 / (X**2 + K**2)"},
        )

        result = classify_system(sym)
        assert result == SystemClass.GENERAL

    def test_classify_system_with_time_dependent_rule(self):
        """Test that time-dependence in assignment rules is detected."""
        X = sp.Symbol("X", positive=True)

        # X' = J_prod where J_prod := exp(-k*time)
        # The 'time' symbol makes this nonautonomous and non-monomial
        sym = SymSystem(
            vars=[X],
            params={"k": 0.1},
            odes={X: sp.Symbol("J_prod")},
            initials={X: 1.0},
            assignment_rules={"J_prod": "exp(-k*time)"},
        )

        result = classify_system(sym)
        assert result == SystemClass.GENERAL

    def test_classify_system_with_nested_assignment_rules(self):
        """Test that nested assignment rules are properly expanded."""
        X = sp.Symbol("X", positive=True)

        # X' = outer where outer := inner * X, inner := X / (X + 1)
        # After expanding: X' = X^2 / (X + 1) → rational → GENERAL
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.Symbol("outer")},
            initials={X: 1.0},
            assignment_rules={"outer": "inner * X", "inner": "X / (X + 1)"},
        )

        result = classify_system(sym)
        assert result == SystemClass.GENERAL

    def test_classify_system_monomial_rule_stays_ssystem(self):
        """Test that monomial assignment rules preserve S-system classification."""
        X = sp.Symbol("X", positive=True)

        # X' = rate where rate := k * X^2 (pure monomial)
        # After expanding, still a monomial → stays S-system compatible
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: sp.Symbol("rate") - sp.Symbol("X")},  # k*X^2 - X
            initials={X: 1.0},
            assignment_rules={"rate": "k * X**2"},
        )

        result = classify_system(sym)
        # Should be S-system or canonical (not GENERAL)
        assert result in [SystemClass.SSYSTEM, SystemClass.CANONICAL_SSYSTEM]

    def test_classify_system_without_assignment_rules(self):
        """Test that systems without assignment rules still work correctly."""
        X = sp.Symbol("X", positive=True)

        # X' = X - X^2 (logistic, canonical S-system)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: X - X**2},
            initials={X: 0.1},
        )

        result = classify_system(sym)
        assert result == SystemClass.CANONICAL_SSYSTEM


class TestAssignmentRuleNoIC:
    """Tests for skipping ICs when variable has assignment rule."""

    def test_assignment_rule_variable_no_ic_in_output(self):
        """Test that variables with assignment rules don't get initial conditions."""
        Z_1 = sp.Symbol("Z_1", positive=True)
        Z_2 = sp.Symbol("Z_2", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(Z_1, (sp.Float(0), {}), (sp.Float(2), {Z_1: 1.0})),
                SSysEquation(Z_2, (sp.Float(1), {Z_1: 1.0, Z_2: 1.0}), (sp.Float(0), {})),
            ],
            initials={Z_1: 1.0, Z_2: 1.0},
            variables=[Z_1, Z_2],
            factor_map={},
            params={},
            # Z_1 has an assignment rule
            assignment_rules={"Z_1": "cos(time) + 2.0"},
        )

        output = ssystem_to_antimony(result, model_name="test", mode="simplified")
        _assert_antimony_roundtrips(output)

        # Z_1 should NOT have an IC line because it has an assignment rule
        assert "Z_1 = 1" not in output
        # Z_2 should have an IC line (no assignment rule)
        assert "Z_2 = 1" in output
        # Z_1 assignment rule should be present
        assert "Z_1 := cos(time) + 2.0" in output


class TestModelNameSanitization:
    """Tests for model name sanitization (periods, hyphens, leading digits)."""

    def test_model_name_period_replaced(self):
        """Test that periods in model names are replaced with underscores."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z, (sp.Float(1), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            params={},
        )

        output = ssystem_to_antimony(result, model_name="orbit_e0.1", mode="simplified")
        _assert_antimony_roundtrips(output)

        # Period should be replaced with underscore
        assert "model orbit_e0_1()" in output
        # Should NOT contain period in model name
        assert "model orbit_e0.1" not in output

    def test_model_name_hyphen_replaced(self):
        """Test that hyphens in model names are replaced with underscores."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z, (sp.Float(1), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            params={},
        )

        output = ssystem_to_antimony(result, model_name="my-model", mode="simplified")
        _assert_antimony_roundtrips(output)

        # Hyphen should be replaced with underscore
        assert "model my_model()" in output
        # Should NOT contain hyphen in model name
        assert "model my-model" not in output

    def test_model_name_leading_digit_prefixed(self):
        """Test that model names starting with digits get 'm_' prefix."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z, (sp.Float(1), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            params={},
        )

        output = ssystem_to_antimony(result, model_name="123model", mode="simplified")
        _assert_antimony_roundtrips(output)

        # Should have m_ prefix
        assert "model m_123model()" in output

    def test_model_name_combined_sanitization(self):
        """Test combined sanitization: period + hyphen + leading digit."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z, (sp.Float(1), {Z: 1.0}), (sp.Float(0.5), {Z: 2.0}))],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={Z: [Z]},
            params={},
        )

        output = ssystem_to_antimony(result, model_name="1987_D1.orbit-test", mode="simplified")
        _assert_antimony_roundtrips(output)

        # All sanitizations applied: 1987_D1_orbit_test with m_ prefix
        assert "model m_1987_D1_orbit_test()" in output


class TestSIMMetadataEmission:
    """Tests for @SIM metadata emission in Antimony output."""

    def test_format_sim_metadata_lines_all_fields(self):
        """Test formatting @SIM metadata with all fields present."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            sim_t_start=0.0,
            sim_t_end=100.0,
            sim_n_steps=500,
            eps_init=1e-6,
        )

        lines = _format_sim_metadata_lines(result)

        assert len(lines) >= 1
        # First line should be the @SIM comment
        assert "@SIM" in lines[0]
        assert "T_START=0" in lines[0]
        assert "T_END=100" in lines[0]
        assert "N_STEPS=500" in lines[0]
        assert "EPS_INIT=1e-06" in lines[0]
        # Second line should be the note about EPS_INIT
        assert any("EPS_INIT" in line and "Note" in line for line in lines)

    def test_format_sim_metadata_lines_partial(self):
        """Test formatting @SIM metadata with only some fields."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
            sim_t_start=0.0,
            sim_t_end=50.0,
            # n_steps and eps_init not set
        )

        lines = _format_sim_metadata_lines(result)

        assert len(lines) >= 1
        assert "@SIM" in lines[0]
        assert "T_START=0" in lines[0]
        assert "T_END=50" in lines[0]
        # n_steps and eps_init should not appear
        assert "N_STEPS" not in lines[0]
        assert "EPS_INIT" not in lines[0]

    def test_format_sim_metadata_lines_no_metadata(self):
        """Test formatting with no metadata (should return empty list)."""
        Z = sp.Symbol("Z_1", positive=True)

        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={Z: 1.0},
            variables=[Z],
        )

        lines = _format_sim_metadata_lines(result)

        # Should return empty list when no metadata
        assert lines == []

    def test_gma_to_antimony_with_sim_metadata(self):
        """Test that @SIM appears in GMA output when metadata present."""
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
            sim_t_start=0.0,
            sim_t_end=100.0,
            sim_n_steps=500,
        )

        output = gma_to_antimony(result, model_name="gma_sim_test")
        _assert_antimony_roundtrips(output)

        assert "@SIM" in output
        assert "T_START=0" in output
        assert "T_END=100" in output
        assert "N_STEPS=500" in output

    def test_gma_to_antimony_without_sim_metadata(self):
        """Test that @SIM does not appear when no metadata."""
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

        output = gma_to_antimony(result, model_name="gma_no_sim")
        _assert_antimony_roundtrips(output)

        assert "@SIM" not in output

    def test_ssystem_to_antimony_simplified_with_sim(self):
        """Test that @SIM appears in simplified S-system output."""
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
            params={},
            sim_t_start=0.0,
            sim_t_end=200.0,
            sim_n_steps=1000,
        )

        output = ssystem_to_antimony(result, model_name="simplified_test", mode="simplified")
        _assert_antimony_roundtrips(output)

        assert "@SIM" in output
        assert "T_START=0" in output
        assert "T_END=200" in output
        assert "N_STEPS=1000" in output

    def test_ssystem_to_antimony_simplified_without_sim(self):
        """Test that @SIM does not appear in simplified mode without metadata."""
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
            params={},
        )

        output = ssystem_to_antimony(result, model_name="simplified_no_sim", mode="simplified")
        _assert_antimony_roundtrips(output)

        assert "@SIM" not in output

    def test_ssystem_to_antimony_canonical_with_sim(self):
        """Test that @SIM appears in canonical S-system output."""
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
            params={},
            sim_t_start=5.0,
            sim_t_end=150.0,
            eps_init=1e-8,
        )

        output = ssystem_to_antimony(result, model_name="canonical_test", mode="canonical")
        _assert_antimony_roundtrips(output)

        assert "@SIM" in output
        assert "T_START=5" in output
        assert "T_END=150" in output
        assert "EPS_INIT=1e-08" in output

    def test_ssystem_to_antimony_canonical_without_sim(self):
        """Test that @SIM does not appear in canonical mode without metadata."""
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
            params={},
        )

        output = ssystem_to_antimony(result, model_name="canonical_no_sim", mode="canonical")
        _assert_antimony_roundtrips(output)

        assert "@SIM" not in output

    def test_sim_metadata_position_after_end(self):
        """Test that @SIM appears after 'end' statement (file-level metadata)."""
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
            params={},
            sim_t_end=100.0,
        )

        output = ssystem_to_antimony(result, model_name="position_test", mode="simplified")
        _assert_antimony_roundtrips(output)

        # Find positions
        sim_pos = output.find("@SIM")
        end_pos = output.rfind("end")

        assert sim_pos > 0
        assert sim_pos > end_pos  # @SIM should come AFTER end (file-level metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
