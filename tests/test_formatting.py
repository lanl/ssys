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
    _format_sim_metadata_lines,
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

        # Find positions
        sim_pos = output.find("@SIM")
        end_pos = output.rfind("end")
        
        assert sim_pos > 0
        assert sim_pos > end_pos  # @SIM should come AFTER end (file-level metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
