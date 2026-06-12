"""Tests for lifting functions and core algorithms."""

import pytest
import sympy as sp

from ssys._recaster.algorithms import (
    _analyze_ode_terms,
    _direct_ssystem_recast,
    _pool_ssystem_recast,
    _requires_gma,
    _should_attempt_pool_construction,
    _validate_pool_result,
    canonicalize_aux_names,
)
from ssys._recaster.lifting import (
    AutonomousLiftResult,
    _build_composite_inverse_mappings,
    _detect_exp_decay_pattern,
    _detect_harmonic_pattern,
    _detect_sqrt_of_squared_pattern,
    _detect_tanh_sigmoid_pattern,
    _is_composite_function_expr,
    _requires_positivity_transform,
    add_dummy_for_constants,
    create_auxiliary_for_denominator,
    lift_exp_decay,
    lift_harmonic,
    lift_squared_for_sqrt,
    lift_tanh_sigmoid,
)
from ssys.recaster import (
    RecastResult,
    RecastStatus,
    SSysEquation,
    SymSystem,
    _exponents_match,
    _is_coefficient_positive,
    expand_to_terms,
    find_composite_functions,
    find_rational_denominators,
    find_sqrt_of_sums,
    lift_composite_functions,
    lift_rational_functions,
    lift_time_functions_to_autonomous,
    product_expr,
    term_to_coeff_exps,
)


class TestTermDecomposition:
    """Tests for term decomposition utilities."""

    def test_expand_to_terms_simple_add(self):
        """Test expanding simple addition."""
        x = sp.Symbol("x")
        expr = x + 2 * x**2
        terms = expand_to_terms(expr)
        assert len(terms) == 2
        assert x in terms
        assert 2 * x**2 in terms

    def test_expand_to_terms_single(self):
        """Test that single term returns list of one."""
        x = sp.Symbol("x")
        expr = x**2
        terms = expand_to_terms(expr)
        assert len(terms) == 1
        assert terms[0] == x**2

    def test_expand_to_terms_product(self):
        """Test that product of sums expands."""
        x, y = sp.symbols("x y")
        expr = (x + y) * (x - y)
        terms = expand_to_terms(expr)
        # Should expand to x^2 - y^2
        assert len(terms) == 2

    def test_term_to_coeff_exps_simple(self):
        """Test extracting coefficient and exponents from simple term."""
        x = sp.Symbol("x", positive=True)
        term = 3 * x**2
        coeff, exps = term_to_coeff_exps(term)
        assert float(coeff) == 3.0
        assert x in exps
        assert abs(exps[x] - 2.0) < 1e-10

    def test_term_to_coeff_exps_multi_var(self):
        """Test extracting from multi-variable term."""
        x = sp.Symbol("x", positive=True)
        y = sp.Symbol("y", positive=True)
        term = 2 * x**3 * y**(-1)
        coeff, exps = term_to_coeff_exps(term)
        assert float(coeff) == 2.0
        assert abs(exps[x] - 3.0) < 1e-10
        assert abs(exps[y] - (-1.0)) < 1e-10

    def test_term_to_coeff_exps_negative_coeff(self):
        """Test extracting from term with negative coefficient."""
        x = sp.Symbol("x", positive=True)
        term = -5 * x
        coeff, exps = term_to_coeff_exps(term)
        assert float(coeff) == -5.0
        assert abs(exps[x] - 1.0) < 1e-10

    def test_term_to_coeff_exps_with_state_vars(self):
        """Test extracting with state variable filtering."""
        x = sp.Symbol("x", positive=True)
        k = sp.Symbol("k", positive=True)
        term = k * x**2
        state_vars = {x}
        coeff, exps = term_to_coeff_exps(term, state_vars)
        # k should be in coefficient, not exponents
        assert k in coeff.free_symbols or coeff == k
        assert x in exps
        assert abs(exps[x] - 2.0) < 1e-10

    def test_term_to_coeff_exps_pure_number(self):
        """Test extracting from pure number."""
        term = sp.Float(5.0)
        coeff, exps = term_to_coeff_exps(term)
        assert float(coeff) == 5.0
        assert len(exps) == 0

    def test_term_to_coeff_exps_covers_parameters_dummy_and_symbolic_powers(self):
        """Term decomposition distinguishes state variables from coefficient factors."""
        x = sp.Symbol("x", positive=True)
        h = sp.Symbol("h", positive=True)
        k = sp.Symbol("k", positive=True)
        dummy = sp.Symbol("dummy_const", positive=True)

        coeff, exps = term_to_coeff_exps(sp.Integer(5), {dummy})
        assert coeff == 5
        assert exps == {dummy: 0.0}

        coeff, exps = term_to_coeff_exps(k, {x})
        assert coeff == k
        assert exps == {}

        coeff, exps = term_to_coeff_exps(x**h, {x})
        assert coeff == 1
        assert exps == {x: h}

        coeff, exps = term_to_coeff_exps(k**2 * x, {x})
        assert coeff == k**2
        assert exps == {x: 1.0}

        coeff, exps = term_to_coeff_exps(sp.sin(x) * x, {x})
        assert coeff == sp.sin(x)
        assert exps == {x: 1.0}

        coeff, exps = term_to_coeff_exps(k**2, {x})
        assert coeff == k**2
        assert exps == {}

        coeff, exps = term_to_coeff_exps((x + 1) ** 2, {x})
        assert coeff == (x + 1) ** 2
        assert exps == {}

        coeff, exps = term_to_coeff_exps(sp.sin(x), {x})
        assert coeff == sp.sin(x)
        assert exps == {}

    def test_product_expr_simple(self):
        """Test building product expression."""
        x = sp.Symbol("x", positive=True)
        coeff = 2.0
        exps = {x: 3.0}
        expr = product_expr(coeff, exps)
        expected = 2 * x**3
        assert sp.simplify(expr - expected) == 0

    def test_product_expr_symbolic_coeff(self):
        """Test building product with symbolic coefficient."""
        x = sp.Symbol("x", positive=True)
        k = sp.Symbol("k", positive=True)
        coeff = k
        exps = {x: 2.0}
        expr = product_expr(coeff, exps)
        expected = k * x**2
        assert sp.simplify(expr - expected) == 0


class TestCoefficientAnalysis:
    """Tests for coefficient sign analysis."""

    def test_is_coefficient_positive_numeric(self):
        """Test sign detection for numeric coefficients."""
        assert _is_coefficient_positive(sp.Float(5.0)) is True
        assert _is_coefficient_positive(sp.Float(-3.0)) is False
        assert _is_coefficient_positive(sp.Float(0.0)) is True

    def test_is_coefficient_positive_symbolic(self):
        """Test sign detection for symbolic coefficients."""
        k = sp.Symbol("k", positive=True)
        # -k is negative
        assert _is_coefficient_positive(-k) is False
        # k is positive (assumed)
        assert _is_coefficient_positive(k) is True

    def test_exponents_match_same(self):
        """Test exponent matching for identical patterns."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        exps1 = {x: 2.0, y: -1.0}
        exps2 = {x: 2.0, y: -1.0}
        assert _exponents_match(exps1, exps2) is True

    def test_exponents_match_different(self):
        """Test exponent matching for different patterns."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        exps1 = {x: 2.0, y: -1.0}
        exps2 = {x: 3.0, y: -1.0}
        assert _exponents_match(exps1, exps2) is False

    def test_exponents_match_missing_var(self):
        """Test exponent matching when one dict has extra variable."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        exps1 = {x: 2.0}
        exps2 = {x: 2.0, y: 1.0}
        assert _exponents_match(exps1, exps2) is False


class TestRecastingAlgorithmBranches:
    """Focused tests for recasting safety checks and canonicalization helpers."""

    def test_canonicalize_aux_names_remaps_coefficients_by_name_and_factor_map(self):
        X = sp.Symbol("X", positive=True)
        empty = sp.Symbol("empty", positive=True)
        old_a = sp.Symbol("X_t1", positive=True)
        old_b = sp.Symbol("X_t2", positive=True)
        detached_x = sp.Symbol("X", real=True)
        detached_b = sp.Symbol("X_t2", real=True)
        result = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[
                SSysEquation(
                    old_a,
                    (sp.sqrt(detached_b**2 + 1), {old_a: 1.0}),
                    (detached_x + 1, {old_b: 1.0}),
                ),
                SSysEquation(old_b, (sp.Integer(1), {old_b: 1.0}), (sp.Integer(0), {})),
            ],
            initials={old_a: 2.0, old_b: 1.0},
            variables=[old_a, old_b],
            factor_map={X: [old_a, old_b], empty: []},
            params={"k": 1.0},
            compartments={"cell": 1.0},
        )

        canonical = canonicalize_aux_names(result)

        Z1 = sp.Symbol("Z_1")
        Z2 = sp.Symbol("Z_2")
        assert canonical.variables == [Z1, Z2]
        assert canonical.factor_map[X] == [Z1, Z2]
        assert canonical.factor_map[empty] == []
        assert sp.Symbol("X_t2", real=True) not in canonical.equations[0].growth[0].free_symbols
        assert Z2 in canonical.equations[0].growth[0].free_symbols
        assert sp.simplify(canonical.equations[0].decay[0] - (Z1 * Z2 + 1)) == 0
        assert canonical.params == {"k": 1.0}
        assert canonical.compartments == {"cell": 1.0}

    def test_pool_construction_safety_checks_report_refusal_reasons(self):
        X, Y = sp.symbols("X Y", positive=True)
        k = sp.Symbol("k", positive=True)
        too_many_terms = SymSystem(
            vars=[X],
            params={"k": 1.0},
            odes={X: sum(k * X ** n for n in range(1, 8))},
            initials={X: 1.0},
        )

        should_attempt, reason = _should_attempt_pool_construction(too_many_terms)

        assert should_attempt is False
        assert "equation has 7 terms" in reason

        too_much_expansion = SymSystem(
            vars=[X, Y],
            params={"k": 1.0},
            odes={
                X: sum(k * X ** n for n in range(1, 6)),
                Y: sum(k * Y ** n for n in range(1, 5)),
            },
            initials={X: 1.0, Y: 1.0},
        )

        should_attempt, reason = _should_attempt_pool_construction(too_much_expansion)

        assert should_attempt is False
        assert "would create 9 auxiliaries" in reason

    def test_pool_result_validation_rejects_long_products_and_extreme_exponents(self):
        X = sp.Symbol("X", positive=True)
        factors = sp.symbols("Z1:7", positive=True)
        long_mapping = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[],
            initials={},
            variables=list(factors),
            factor_map={X: list(factors)},
        )

        valid, reason = _validate_pool_result(long_mapping)

        assert valid is False
        assert "mapped to product of 6 factors" in reason

        Z = sp.Symbol("Z", positive=True)
        extreme_exponent = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z, (sp.Integer(1), {Z: -3.0}), (sp.Integer(0), {}))],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={X: [Z]},
        )

        valid, reason = _validate_pool_result(extreme_exponent)

        assert valid is False
        assert "exponent -3.0" in reason

        valid = RecastResult(
            status=RecastStatus.CANONICAL_SSYSTEM,
            equations=[SSysEquation(Z, (sp.Integer(1), {Z: -1.0}), (sp.Integer(0), {}))],
            initials={Z: 1.0},
            variables=[Z],
            factor_map={X: [Z]},
        )

        assert _validate_pool_result(valid) == (True, None)

    def test_requires_gma_detects_incompatible_growth_and_decay_terms(self):
        X, a, b = sp.symbols("X a b", positive=True)

        growth_mismatch = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 2.0},
            odes={X: a * X + b * X**2},
            initials={X: 1.0},
        )
        decay_mismatch = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 2.0},
            odes={X: -(a * X + b * X**2)},
            initials={X: 1.0},
        )
        compatible = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 2.0},
            odes={X: a * X + b * X},
            initials={X: 1.0},
        )

        assert _requires_gma(growth_mismatch) is True
        assert _requires_gma(decay_mismatch) is True
        assert _requires_gma(compatible) is False

    def test_analyze_ode_terms_skips_zero_and_unparseable_terms(self, monkeypatch):
        import ssys._recaster.algorithms as algorithms

        X = sp.Symbol("X", positive=True)
        real_term_to_coeff_exps = algorithms.term_to_coeff_exps

        def sometimes_raises(term, state_vars=None):
            if term == X**2:
                raise ValueError("bad term")
            return real_term_to_coeff_exps(term, state_vars)

        monkeypatch.setattr(algorithms, "term_to_coeff_exps", sometimes_raises)

        growth, decay = _analyze_ode_terms([sp.Integer(0), X, -X, X**2], {X})

        assert growth == [(sp.Integer(1), {X: 1.0})]
        assert decay == [(sp.Integer(1), {X: 1.0})]

    def test_direct_recast_uses_gma_for_incompatible_growth_terms(self):
        X, a, b = sp.symbols("X a b", positive=True)
        sym = SymSystem(
            vars=[X, X],
            params={"a": 1.0, "b": 2.0},
            odes={X: a * X + b * X**2},
            initials={X: 1.0},
            assignment_rules={"obs": "X + 1"},
        )

        result = _direct_ssystem_recast(sym, {X})

        assert result.status == RecastStatus.GMA
        assert result.factor_map[X] == [X]
        assert result.assignment_rules == {"obs": "X + 1"}
        assert len(result.gma_equations[0].production) == 2

    def test_pool_recast_uses_species_ic_from_params_when_initials_are_empty(self):
        X, a, b = sp.symbols("X a b", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"X": 2.5, "a": 1.0, "b": 0.5},
            odes={X: a * X - b * X**2},
            initials={},
        )

        result = _pool_ssystem_recast(sym)

        first_factor = result.factor_map[X][0]
        assert result.initials[first_factor] == 2.5
        assert result.params["X"] == 2.5

    def test_recast_to_ssystem_records_preflight_and_validation_refusals(self, monkeypatch):
        import ssys._recaster.algorithms as algorithms

        X, a, b = sp.symbols("X a b", positive=True)
        too_many_terms = SymSystem(
            vars=[X],
            params={"a": 1.0},
            odes={X: sum(a * X ** n for n in range(1, 8))},
            initials={X: 1.0},
        )

        preflight = algorithms.recast_to_ssystem(too_many_terms)

        assert preflight.status == RecastStatus.GMA
        assert "equation has 7 terms" in preflight.canonical_refusal_reason

        normal = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 0.5},
            odes={X: a * X - b * X**2},
            initials={X: 1.0},
        )
        monkeypatch.setattr(
            algorithms,
            "_validate_pool_result",
            lambda result: (False, "forced validation refusal"),
        )

        refused = algorithms.recast_to_ssystem(normal)

        assert refused.status == RecastStatus.GMA
        assert refused.canonical_refusal_reason == "forced validation refusal"


class TestFindingFunctions:
    """Tests for functions that find expressions needing lifting."""

    def test_find_rational_denominators_simple(self):
        """Test finding simple rational denominator."""
        x = sp.Symbol("x")
        expr = 1 / (x + 1)
        denoms = find_rational_denominators(expr)
        assert len(denoms) >= 1
        assert (x + 1) in denoms or any(
            sp.simplify(d - (x + 1)) == 0 for d in denoms
        )

    def test_find_rational_denominators_none(self):
        """Test that polynomial has no denominators."""
        x = sp.Symbol("x")
        expr = x**2 + 3 * x + 1
        denoms = find_rational_denominators(expr)
        assert len(denoms) == 0

    def test_find_composite_functions_exp(self):
        """Test finding exp function."""
        x = sp.Symbol("x")
        expr = sp.exp(x) + x
        funcs = find_composite_functions(expr)
        assert len(funcs) == 1
        assert sp.exp(x) in funcs

    def test_find_composite_functions_trig(self):
        """Test finding trig functions."""
        x = sp.Symbol("x")
        expr = sp.sin(x) + sp.cos(2 * x)
        funcs = find_composite_functions(expr)
        assert len(funcs) == 2
        assert sp.sin(x) in funcs
        assert sp.cos(2 * x) in funcs

    def test_find_composite_functions_none(self):
        """Test that polynomial has no composite functions."""
        x = sp.Symbol("x")
        expr = x**2 + 3 * x + 1
        funcs = find_composite_functions(expr)
        assert len(funcs) == 0

    def test_find_sqrt_of_sums(self):
        """Test finding sqrt of sum patterns."""
        x = sp.Symbol("x")
        expr = sp.sqrt(x**2 + 1)
        sqrt_sums = find_sqrt_of_sums(expr)
        assert len(sqrt_sums) == 1

    def test_find_sqrt_of_sums_not_sum(self):
        """Test that sqrt of single variable is not flagged."""
        x = sp.Symbol("x")
        expr = sp.sqrt(x)
        sqrt_sums = find_sqrt_of_sums(expr)
        assert len(sqrt_sums) == 0


class TestLiftingInternalBranches:
    """Focused branch tests for lifting helpers used by recasting."""

    def test_create_auxiliary_for_denominator_uses_chain_rule(self):
        X = sp.Symbol("X", positive=True)

        aux, aux_ode = create_auxiliary_for_denominator(X + 1, {X: -X}, 3)

        assert aux == sp.Symbol("W_3", positive=True)
        assert sp.simplify(aux_ode - X * aux**2) == 0

    def test_composite_inverse_mappings_cover_exp_log_and_nested_auxiliaries(self):
        X = sp.Symbol("X", positive=True)
        Z1 = sp.Symbol("Z_1", positive=True)
        Z2 = sp.Symbol("Z_2", positive=True)

        inverse = _build_composite_inverse_mappings(
            {
                sp.exp(Z2**2): Z1,
                sp.log(X): Z2,
            },
            {
                sp.exp(Z2**2): 0.0,
                sp.log(X): 0.0,
            },
            [X],
        )

        assert inverse[sp.exp(Z2**2)] == Z1
        assert inverse[sp.log(Z1)] == Z2**2
        assert inverse[X] == sp.exp(Z2)
        assert inverse[X**-3] == sp.exp(-3 * Z2)

    def test_positivity_transform_and_time_pattern_detectors(self):
        time = sp.Symbol("time")
        omega = sp.Symbol("omega")

        assert _is_composite_function_expr(sp.exp(time)) is True
        assert _is_composite_function_expr(time + sp.sin(time)) is True
        assert _is_composite_function_expr(sp.log(time) ** 2) is True
        assert _is_composite_function_expr(time**2) is False

        assert _requires_positivity_transform(sp.sin(time)) == (True, 2.0)
        assert _requires_positivity_transform(sp.cos(time)) == (True, 2.0)
        assert _requires_positivity_transform(sp.exp(time)) == (False, 0.0)

        assert _detect_exp_decay_pattern(sp.exp(time)) == (1, 1)
        assert _detect_exp_decay_pattern(sp.exp(-time)) == (1, -1)
        assert _detect_exp_decay_pattern(sp.exp(omega * time)) == (1, omega)
        assert _detect_exp_decay_pattern(sp.exp(time**2)) is None

        assert _detect_harmonic_pattern(sp.sin(omega * time + sp.pi / 4)) == (
            "sin",
            omega,
            sp.pi / 4,
        )
        assert _detect_harmonic_pattern(sp.cos(1)) is None
        assert _detect_harmonic_pattern(sp.cos(time**2)) is None

        assert _detect_tanh_sigmoid_pattern(sp.tanh(omega * (time - 5))) == (omega, 5)
        assert _detect_tanh_sigmoid_pattern(sp.tanh(time**2)) is None
        assert _detect_tanh_sigmoid_pattern(sp.tanh(sp.Add(time, -time, evaluate=False))) is None

    def test_sqrt_squared_pattern_rejects_multiple_or_missing_square_terms(self):
        X, Y, k = sp.symbols("X Y k", positive=True)

        assert _detect_sqrt_of_squared_pattern(sp.sqrt(X**2 + 1)) == (X, 1)
        assert _detect_sqrt_of_squared_pattern(sp.sqrt(X + 1)) is None
        assert _detect_sqrt_of_squared_pattern(sp.sqrt(X)) is None
        assert _detect_sqrt_of_squared_pattern((X**2 + 1) ** sp.Rational(1, 3)) is None
        assert _detect_sqrt_of_squared_pattern(sp.sqrt(2 * X**2 + 1)) is None
        assert _detect_sqrt_of_squared_pattern(sp.sqrt(X**2 + k)) is None
        assert _detect_sqrt_of_squared_pattern(sp.sqrt(X**2 + Y**2 + 1)) is None

    def test_lift_rational_functions_handles_constant_and_dynamic_denominators(self):
        X, k = sp.symbols("X k", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 2.0},
            odes={X: X / (k + 1) + X / (X + k)},
            initials={X: 1.0},
        )

        lifted, aux_defs = lift_rational_functions(sym)

        assert len(aux_defs) == 1
        aux, definition = next(iter(aux_defs.items()))
        assert sp.simplify(definition - (X + k)) == 0
        assert aux in lifted.vars
        assert lifted.initials[aux] == 3.0
        assert aux in lifted.odes[X].free_symbols
        assert sp.Float(1 / 3) in lifted.odes[X].atoms(sp.Float)

    def test_lift_rational_functions_covers_skip_and_fallback_paths(self):
        X, k, missing = sp.symbols("X k missing", positive=True)
        composite_inverse = sp.Pow(sp.exp(X), -1, evaluate=False)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: X**-1 + X * composite_inverse + X / missing + X / (X + k)},
            initials={X: 1.0},
        )

        lifted, aux_defs = lift_rational_functions(sym)

        aux, definition = next(iter(aux_defs.items()))
        assert sp.simplify(definition - (X + k)) == 0
        assert lifted.initials[aux] == 1.0
        assert X**-1 in lifted.odes[X].atoms(sp.Pow)
        assert sp.exp(-X) in lifted.odes[X].atoms(sp.exp)
        assert missing**-1 in lifted.odes[X].atoms(sp.Pow)

    def test_lift_rational_functions_substitutes_higher_constant_powers(self):
        X, k = sp.symbols("X k", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 2.0},
            odes={X: X / (k + 1) ** 3},
            initials={X: 1.0},
        )

        lifted, aux_defs = lift_rational_functions(sym)

        assert aux_defs == {}
        assert float(lifted.odes[X] / X) == pytest.approx(1 / 27)

    def test_add_dummy_for_constants_adds_parameter_not_state(self):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(vars=[X], params={}, odes={X: X + 2}, initials={X: 1.0})

        lifted, aux_defs = add_dummy_for_constants(sym)

        dummy = sp.Symbol("dummy_const", positive=True)
        assert lifted.vars == [X]
        assert lifted.params["dummy_const"] == 1.0
        assert aux_defs == {dummy: sp.Integer(1)}
        assert any(
            isinstance(atom, sp.Pow) and atom.base == dummy for atom in lifted.odes[X].atoms(sp.Pow)
        )

    def test_add_dummy_for_constants_noops_without_constant_terms(self):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(vars=[X], params={}, odes={X: X**2}, initials={X: 1.0})

        lifted, aux_defs = add_dummy_for_constants(sym)

        assert lifted is sym
        assert aux_defs == {}

    def test_add_dummy_for_constants_handles_constant_only_equation_and_unchanged_var(self):
        X, Y = sp.symbols("X Y", positive=True)
        sym = SymSystem(
            vars=[X, Y],
            params={},
            odes={X: sp.Integer(2), Y: Y**2},
            initials={X: 1.0, Y: 3.0},
        )

        lifted, aux_defs = add_dummy_for_constants(sym)

        dummy = sp.Symbol("dummy_const", positive=True)
        assert aux_defs == {dummy: sp.Integer(1)}
        assert lifted.odes[Y] == Y**2
        assert any(atom.base == dummy for atom in lifted.odes[X].atoms(sp.Pow))

    def test_autonomous_lift_helpers_cover_exp_harmonic_and_tanh(self):
        time = sp.Symbol("time")
        k = sp.Symbol("k", positive=True)

        exp_result = lift_exp_decay(sp.exp(-k * time), 1, {})
        assert exp_result is not None
        E = sp.Symbol("E_1", positive=True)
        assert exp_result.new_odes[E] == -k * E
        assert exp_result.new_initials[E] == 1

        harmonic = lift_harmonic(sp.cos(2 * time), 2, {})
        assert harmonic is not None
        c2 = sp.Symbol("c_2", positive=True)
        s2 = sp.Symbol("s_2", positive=True)
        assert harmonic.new_odes[c2] == -2 * s2
        assert harmonic.new_odes[s2] == 2 * c2

        reused = lift_harmonic(sp.sin(2 * time + sp.pi / 2), 3, {}, {sp.Integer(2): (c2, s2)})
        assert reused is not None
        assert reused.new_vars == []
        assert sp.simplify(reused.substitution - c2) == 0

        sigmoid = lift_tanh_sigmoid(sp.tanh(k * (time - 5)), 4, {})
        assert sigmoid is not None
        h4 = sp.Symbol("h_4", positive=True)
        assert sp.simplify(sigmoid.new_odes[h4] - (2 * k * h4 - 2 * k * h4**2)) == 0

    def test_autonomous_lift_helpers_cover_reuse_and_rejection_branches(self):
        time = sp.Symbol("time")
        phi = sp.Symbol("phi")
        X = sp.Symbol("X", positive=True)
        c2 = sp.Symbol("c_2", positive=True)
        s2 = sp.Symbol("s_2", positive=True)

        assert lift_exp_decay(X, 1, {}) is None
        assert lift_harmonic(sp.exp(time), 1, {}) is None
        assert lift_tanh_sigmoid(X, 1, {}) is None
        assert lift_squared_for_sqrt(X, 1, SymSystem(vars=[X], params={}, odes={X: X}, initials={})) is None
        assert _detect_harmonic_pattern(X) is None
        assert _detect_tanh_sigmoid_pattern(1) is None
        assert _detect_tanh_sigmoid_pattern(sp.tanh(1)) is None

        reused_cos = lift_harmonic(
            sp.cos(2 * time + phi),
            3,
            {},
            {sp.Integer(2): (c2, s2)},
        )
        assert reused_cos is not None
        assert reused_cos.new_vars == []
        assert sp.simplify(reused_cos.substitution - (c2 * sp.cos(phi) - s2 * sp.sin(phi))) == 0

        reused_sin = lift_harmonic(
            sp.sin(2 * time),
            4,
            {},
            {sp.Integer(2): (c2, s2)},
        )
        assert reused_sin is not None
        assert reused_sin.substitution == s2

        reused_shifted_sin = lift_harmonic(
            sp.sin(2 * time + phi),
            4,
            {},
            {sp.Integer(2): (c2, s2)},
        )
        assert reused_shifted_sin is not None
        assert sp.simplify(
            reused_shifted_sin.substitution - (s2 * sp.cos(phi) + c2 * sp.sin(phi))
        ) == 0

        new_sin = lift_harmonic(sp.sin(3 * time), 5, {})
        assert new_sin is not None
        assert new_sin.substitution == sp.Symbol("s_5", positive=True)

    def test_lift_squared_for_sqrt_computes_state_initial_condition_by_name(self):
        X = sp.Symbol("X", positive=True)
        expr = sp.sqrt(X**2 + 1)
        sym = SymSystem(
            vars=[sp.Symbol("X", positive=True)],
            params={},
            odes={X: -X},
            initials={X: 3.0},
        )

        result = lift_squared_for_sqrt(expr, 5, sym)

        assert result is not None
        u5 = sp.Symbol("u_5", positive=True)
        assert result.new_vars == [u5]
        assert result.new_initials[u5] == 10.0
        assert result.substitution == sp.sqrt(u5)

    def test_lift_squared_for_sqrt_covers_expression_chain_rule_and_init_fallback(
        self, monkeypatch
    ):
        import ssys._recaster.lifting as lifting

        X, Y, q = sp.symbols("X Y q", positive=True)
        expr = sp.Symbol("placeholder", positive=True)
        sym = SymSystem(
            vars=[X, Y],
            params={},
            odes={X: X, Y: -Y},
            initials={},
        )
        monkeypatch.setattr(
            lifting,
            "_detect_sqrt_of_squared_pattern",
            lambda candidate: (X + Y, q),
        )

        result = lift_squared_for_sqrt(expr, 6, sym)

        u6 = sp.Symbol("u_6", positive=True)
        assert result is not None
        assert sp.simplify(result.new_odes[u6] - 2 * (X + Y) * (X - Y)) == 0
        assert result.new_initials[u6] == 1.0

    def test_lift_time_functions_adds_clock_and_state_sqrt_auxiliary(self):
        X = sp.Symbol("X", positive=True)
        time = sp.Symbol("time", positive=True)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: time + sp.sqrt(X**2 + 1)},
            initials={X: 2.0},
            sim_t_end=4.0,
        )

        lifted, aux_defs, next_counter = lift_time_functions_to_autonomous(sym, 7)

        T = sp.Symbol("T", positive=True)
        assert T in lifted.vars
        assert lifted.odes[T] == 1
        assert lifted.initials[T] == 0.0
        assert aux_defs[T] == time
        assert any(var.name.startswith("u_") for var in lifted.vars)
        assert next_counter > 7
        assert lifted.sim_t_end == 4.0

    def test_lift_time_functions_rewrites_assignment_rules(self):
        X = sp.Symbol("X", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 2.0},
            odes={X: -X},
            initials={X: 1.0},
            assignment_rules={"gate": "time + k"},
        )

        lifted, aux_defs, _ = lift_time_functions_to_autonomous(sym)

        T = sp.Symbol("T", positive=True)
        assert T in lifted.vars
        assert aux_defs[T] == sp.Symbol("time", positive=True)
        assert lifted.assignment_rules == {"gate": "T + k"}

    def test_lift_time_functions_uses_sqrt_initial_fallback(self, monkeypatch):
        import ssys._recaster.lifting as lifting

        X, q = sp.symbols("X q", positive=True)
        u9 = sp.Symbol("u_9", positive=True)
        sqrt_expr = sp.sqrt(X**2 + 1)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sqrt_expr},
            initials={X: 1.0},
        )

        def fake_lift_squared_for_sqrt(expr, aux_counter, sym):
            return AutonomousLiftResult(
                new_vars=[u9],
                new_odes={u9: X},
                new_initials={u9: q},
                substitution=sp.sqrt(u9),
                aux_defs={u9: expr},
            )

        monkeypatch.setattr(lifting, "lift_squared_for_sqrt", fake_lift_squared_for_sqrt)

        lifted, aux_defs, next_counter = lift_time_functions_to_autonomous(sym, 9)

        assert lifted.initials[u9] == 1.0
        assert aux_defs[u9] == sqrt_expr
        assert next_counter == 10

    def test_lift_time_functions_substitutes_parameterized_sqrt_initial(self, monkeypatch):
        import ssys._recaster.lifting as lifting

        X, q = sp.symbols("X q", positive=True)
        u10 = sp.Symbol("u_10", positive=True)
        sqrt_expr = sp.sqrt(X**2 + 1)
        sym = SymSystem(
            vars=[X],
            params={"q": 4.0},
            odes={X: sqrt_expr},
            initials={X: 1.0},
        )

        def fake_lift_squared_for_sqrt(expr, aux_counter, sym):
            return AutonomousLiftResult(
                new_vars=[u10],
                new_odes={u10: X},
                new_initials={u10: q},
                substitution=sp.sqrt(u10),
                aux_defs={u10: expr},
            )

        monkeypatch.setattr(lifting, "lift_squared_for_sqrt", fake_lift_squared_for_sqrt)

        lifted, _, _ = lift_time_functions_to_autonomous(sym, 10)

        assert lifted.initials[u10] == 4.0

    def test_lift_composite_functions_handles_time_sqrt_and_time_derivatives(self):
        X = sp.Symbol("X", positive=True)
        time, c = sp.symbols("time c", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"c": 2.0},
            odes={X: sp.sqrt(time + 1) + sp.sqrt(X + time + c) + sp.sin(time)},
            initials={X: 3.0},
        )

        lifted, aux_defs = lift_composite_functions(sym)

        assert any("sqrt(time + 1)" in rule for rule in lifted.assignment_rules.values())
        assert any(sp.simplify(defn - sp.sqrt(X + time + c)) == 0 for defn in aux_defs.values())
        assert any(
            time not in ode.free_symbols and any(var.name.startswith("Z_") for var in ode.free_symbols)
            for ode in lifted.odes.values()
        )

    def test_lift_composite_functions_handles_other_function_time_derivative_fallback(self):
        X = sp.Symbol("X", positive=True)
        time = sp.Symbol("time", positive=True)
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.exp(X + time)},
            initials={X: 0.0},
        )

        lifted, aux_defs = lift_composite_functions(sym)

        aux = next(iter(aux_defs))
        assert lifted.initials[aux] == 1.0
        assert aux in lifted.odes[aux].free_symbols


class TestLiftRationalFunctions:
    """Tests for lift_rational_functions."""

    def test_lift_simple_rational(self):
        """Test lifting simple rational function."""
        X = sp.Symbol("X", positive=True)
        K = sp.Symbol("K", positive=True)

        # X' = X / (X + K)  =>  needs lifting for (X + K)
        ode = X / (X + K)

        sym = SymSystem(
            vars=[X],
            params={"K": 1.0},
            odes={X: ode},
            initials={X: 1.0},
        )

        lifted, aux_defs = lift_rational_functions(sym)

        # Should have at least one auxiliary
        assert len(lifted.vars) > 1
        # Auxiliary definitions should contain the denominator
        assert len(aux_defs) >= 1

    def test_lift_constant_denom_substitutes(self):
        """Test that constant denominators are substituted numerically."""
        X = sp.Symbol("X", positive=True)
        K = sp.Symbol("K", positive=True)

        # X' = X / K  where K is a constant parameter
        ode = X / K

        sym = SymSystem(
            vars=[X],
            params={"K": 2.0},
            odes={X: ode},
            initials={X: 1.0},
        )

        lifted, aux_defs = lift_rational_functions(sym)

        # No new variables needed for constant denominator
        assert len(lifted.vars) == 1
        assert len(aux_defs) == 0


class TestLiftCompositeFunctions:
    """Tests for lift_composite_functions."""

    def test_lift_exp_function(self):
        """Test lifting exp function."""
        X = sp.Symbol("X", positive=True)

        # X' = exp(X)
        ode = sp.exp(X)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: ode},
            initials={X: 0.0},
        )

        lifted, aux_defs = lift_composite_functions(sym)

        # Should add auxiliary for exp(X)
        assert len(lifted.vars) > 1
        assert len(aux_defs) >= 1

    def test_lift_sin_cos_pair(self):
        """Test lifting sin/cos creates coupled pair."""
        t = sp.Symbol("t", positive=True)
        omega = sp.Symbol("omega", positive=True)

        # X' = sin(t)  =>  needs both sin and cos auxiliaries
        ode = sp.sin(omega * t)

        sym = SymSystem(
            vars=[t],
            params={"omega": 1.0},
            odes={t: ode},
            initials={t: 0.0},
        )

        lifted, aux_defs = lift_composite_functions(sym)

        # Should add auxiliaries
        assert len(lifted.vars) > 1
        assert len(aux_defs) >= 1

    def test_lift_log_function(self):
        """Test lifting log function."""
        X = sp.Symbol("X", positive=True)

        # X' = log(X)
        ode = sp.log(X)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: ode},
            initials={X: 1.0},
        )

        lifted, aux_defs = lift_composite_functions(sym)

        # Should add auxiliary for log(X)
        assert len(lifted.vars) > 1
        assert len(aux_defs) >= 1


class TestLiftTimeFunctions:
    """Tests for lift_time_functions_to_autonomous."""

    def test_lift_clock_state(self):
        """Test that time-dependent system gets clock state."""
        X = sp.Symbol("X", positive=True)
        time = sp.Symbol("time")

        # X' = X * time  =>  needs clock state T
        ode = X * time

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: ode},
            initials={X: 1.0},
        )

        lifted, aux_defs, _ = lift_time_functions_to_autonomous(sym)

        # Should add clock state T
        var_names = {v.name for v in lifted.vars}
        assert "T" in var_names

        # Clock should have T' = 1 and T(0) = 0
        T = sp.Symbol("T", positive=True)
        assert T in lifted.odes
        assert lifted.odes[T] == 1
        assert lifted.initials[T] == 0.0

    def test_no_time_unchanged(self):
        """Test that time-independent system is unchanged."""
        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        # X' = -k*X  (no time dependence)
        ode = -k * X

        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: ode},
            initials={X: 1.0},
        )

        lifted, aux_defs, _ = lift_time_functions_to_autonomous(sym)

        # Should not add any auxiliaries
        assert len(lifted.vars) == 1
        assert len(aux_defs) == 0


class TestSymSystem:
    """Tests for SymSystem dataclass."""

    def test_sym_system_creation(self):
        """Test basic SymSystem creation."""
        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
        )

        assert X in sym.vars
        assert sym.params["k"] == 0.5
        assert sym.initials[X] == 1.0

    def test_sym_system_optional_attrs(self):
        """Test SymSystem optional attributes."""
        X = sp.Symbol("X", positive=True)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.Integer(0)},
            initials={X: 1.0},
            sim_t_start=0.0,
            sim_t_end=10.0,
            sim_n_steps=100,
        )

        assert sym.sim_t_start == 0.0
        assert sym.sim_t_end == 10.0
        assert sym.sim_n_steps == 100


class TestIntegrationLiftingChain:
    """Integration tests for chained lifting operations."""

    def test_michaelis_menten_lifting(self):
        """Test full lifting pipeline for Michaelis-Menten kinetics."""
        S = sp.Symbol("S", positive=True)
        K = sp.Symbol("K", positive=True)
        V = sp.Symbol("V", positive=True)

        # Michaelis-Menten: S' = -V*S/(K + S)
        ode = -V * S / (K + S)

        sym = SymSystem(
            vars=[S],
            params={"K": 1.0, "V": 2.0},
            odes={S: ode},
            initials={S: 10.0},
        )

        # Apply rational lifting
        lifted, aux_defs = lift_rational_functions(sym)

        # Verify structure: should have auxiliary for (K + S)
        assert len(lifted.vars) >= 1
        # ODE should be in power-law form after lifting
        for var, ode in lifted.odes.items():
            # Check that no sums appear in denominators
            denoms = find_rational_denominators(ode)
            # Dynamic denominators should be simple symbols now
            for d in denoms:
                assert isinstance(d, sp.Symbol) or d.is_number

    def test_exp_log_composition(self):
        """Test lifting exp(log(X)) composition."""
        X = sp.Symbol("X", positive=True)

        # X' = exp(log(X)^2)  =>  nested composition
        ode = sp.exp(sp.log(X) ** 2)

        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: ode},
            initials={X: 1.0},
        )

        # Apply composite lifting
        lifted, aux_defs = lift_composite_functions(sym)

        # Should add auxiliaries for log(X) and exp(...)
        assert len(lifted.vars) >= 2
        assert len(aux_defs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
