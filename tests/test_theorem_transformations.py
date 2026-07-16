"""Theorem-style tests for core recasting transformations."""

import sympy as sp

from ssys import SymSystem, parse_antimony_via_sbml, recast_to_ssystem
from ssys._recaster.algorithms import _pool_ssystem_recast
from ssys.formatting import ssystem_to_antimony
from ssys.types import RecastResult, RecastStatus, SolverRequirement


def _assert_zero(expr: sp.Expr) -> None:
    expr = expr.xreplace(
        {symbol: sp.Symbol(symbol.name, positive=True) for symbol in expr.free_symbols}
    )
    simplified = sp.simplify(sp.nsimplify(expr, rational=True, tolerance=1e-10))
    assert simplified == 0, simplified


def _prod(symbols: list[sp.Symbol]) -> sp.Expr:
    return sp.prod(symbols) if symbols else sp.Integer(1)


def _phi(rec: RecastResult) -> dict[sp.Symbol, sp.Expr]:
    return {orig: _prod(factors) for orig, factors in rec.factor_map.items()}


def _term_expr(
    coeff: sp.Expr,
    exponents: dict[sp.Symbol, float],
    phi: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    expr = sp.sympify(coeff)
    for base, exponent in exponents.items():
        expr *= phi.get(base, base) ** exponent
    return sp.simplify(expr)


def _recast_rhs(rec: RecastResult) -> dict[sp.Symbol, sp.Expr]:
    phi = _phi(rec)
    rhs: dict[sp.Symbol, sp.Expr] = {}

    for eq in rec.equations:
        expr = _term_expr(*eq.growth, phi) - _term_expr(*eq.decay, phi)
        if eq.var in rhs:
            _assert_zero(rhs[eq.var] - expr)
        else:
            rhs[eq.var] = sp.simplify(expr)

    for eq in rec.gma_equations:
        expr = sum((_term_expr(*term, phi) for term in eq.production), sp.Integer(0))
        expr -= sum((_term_expr(*term, phi) for term in eq.degradation), sp.Integer(0))
        if eq.var in rhs:
            _assert_zero(rhs[eq.var] - expr)
        else:
            rhs[eq.var] = sp.simplify(expr)

    return rhs


def _clock_substitutions(rec: RecastResult) -> dict[sp.Symbol, sp.Symbol]:
    substitutions: dict[sp.Symbol, sp.Symbol] = {}
    for aux, definition in rec.auxiliary_defs.items():
        if isinstance(definition, sp.Symbol) and definition.name.lower() in {"time", "t"}:
            substitutions[definition] = aux
    return substitutions


def _non_clock_auxiliary_substitutions(rec: RecastResult) -> dict[sp.Symbol, sp.Expr]:
    return {
        aux: definition
        for aux, definition in rec.auxiliary_defs.items()
        if not (isinstance(definition, sp.Symbol) and definition.name.lower() in {"time", "t"})
    }


def _assert_observable_equivalence(
    sym: SymSystem,
    rec: RecastResult,
    *,
    assignment_subs: dict[sp.Symbol, sp.Expr] | None = None,
) -> None:
    phi = _phi(rec)
    rhs = _recast_rhs(rec)
    aux_subs = _non_clock_auxiliary_substitutions(rec)
    clock_subs = _clock_substitutions(rec)
    assignment_subs = assignment_subs or {}

    recast_vars = sorted(rhs.keys(), key=str)
    for original in sorted(sym.odes.keys(), key=str):
        phi_expr = phi[original]
        dphi_dt = sum(
            sp.diff(phi_expr, recast_var) * rhs[recast_var] for recast_var in recast_vars
        )
        original_rhs = sym.odes[original].subs(phi).subs(clock_subs)
        residual = (dphi_dt - original_rhs).subs(assignment_subs).subs(aux_subs)
        _assert_zero(residual)


def _assert_auxiliary_chain_rule(
    sym: SymSystem,
    rec: RecastResult,
    *,
    assignment_subs: dict[sp.Symbol, sp.Expr] | None = None,
) -> None:
    rhs = _recast_rhs(rec)
    phi = _phi(rec)
    aux_subs = _non_clock_auxiliary_substitutions(rec)
    clock_subs = _clock_substitutions(rec)
    assignment_subs = assignment_subs or {}

    for aux, definition in sorted(rec.auxiliary_defs.items(), key=lambda item: str(item[0])):
        if aux not in rhs:
            continue
        if isinstance(definition, sp.Symbol) and definition.name.lower() in {"time", "t"}:
            expected = sp.Integer(1)
        else:
            definition = definition.subs(clock_subs)
            expected = sp.Integer(0)
            for symbol in sorted(definition.free_symbols, key=str):
                if symbol in rhs:
                    expected += sp.diff(definition, symbol) * rhs[symbol]
                elif symbol in sym.odes:
                    expected += sp.diff(definition, symbol) * sym.odes[symbol].subs(phi)
                elif symbol.name.lower() in {"time", "t"}:
                    expected += sp.diff(definition, symbol)

        residual = (rhs[aux] - expected).subs(assignment_subs).subs(aux_subs)
        _assert_zero(residual)


def _generated_system(rec: RecastResult, *, model_name: str, mode: str = "simplified") -> SymSystem:
    antimony = ssystem_to_antimony(rec, model_name=model_name, mode=mode)
    return parse_antimony_via_sbml(antimony)


def _ode_by_name(sym: SymSystem, name: str) -> sp.Expr:
    for var, ode in sym.odes.items():
        if var.name == name:
            return ode
    raise AssertionError(f"missing ODE for {name!r}; found {[var.name for var in sym.odes]}")


def test_pool_construction_logistic_phi_equations_and_antimony_semantics():
    X, a, b = sp.symbols("X a b", positive=True)
    sym = SymSystem(
        vars=[X],
        params={"a": 1.0, "b": 0.1},
        odes={X: a * X - b * X**2},
        initials={X: 0.5},
    )

    rec = _pool_ssystem_recast(sym)

    assert rec.status == RecastStatus.CANONICAL_SSYSTEM
    assert [var.name for var in rec.factor_map[X]] == ["Z_1", "Z_2"]
    assert rec.initials[rec.factor_map[X][0]] == 0.5
    _assert_observable_equivalence(sym, rec)

    generated = _generated_system(rec, model_name="pool_logistic")
    Z1, Z2 = rec.factor_map[X]
    assert generated.assignment_rules["X"] == "Z_1 * Z_2"
    _assert_zero(_ode_by_name(generated, Z1.name) - a * Z1)
    _assert_zero(_ode_by_name(generated, Z2.name) + b * Z1 * Z2**2)


def test_rational_lifting_preserves_denominator_manifold_and_generated_semantics():
    X, K = sp.symbols("X K", positive=True)
    sym = SymSystem(
        vars=[X],
        params={"K": 2.0},
        odes={X: -X / (K + X)},
        initials={X: 3.0},
    )

    rec = recast_to_ssystem(sym)

    Y = sp.Symbol("Y_1", positive=True)
    assert rec.solver_requirement == SolverRequirement.DAE_REQUIRED
    assert rec.auxiliary_defs == {Y: K + X}
    assert rec.initials[Y] == 5.0
    _assert_observable_equivalence(sym, rec)
    _assert_auxiliary_chain_rule(sym, rec)

    generated = _generated_system(rec, model_name="rational")
    _assert_zero(generated.odes[X] + X / Y)
    _assert_zero(generated.odes[Y] + X / Y)


def test_composite_exp_lifting_proves_auxiliary_identity_and_output_semantics():
    X = sp.Symbol("X", positive=True)
    sym = SymSystem(vars=[X], params={}, odes={X: sp.exp(X)}, initials={X: 0.0})

    rec = recast_to_ssystem(sym)

    Z = sp.Symbol("Z_1", positive=True)
    assert rec.solver_requirement == SolverRequirement.DAE_REQUIRED
    assert rec.auxiliary_defs == {Z: sp.exp(X)}
    _assert_observable_equivalence(sym, rec)
    _assert_auxiliary_chain_rule(sym, rec)

    generated = _generated_system(rec, model_name="composite_exp")
    _assert_zero(generated.odes[X] - Z)
    _assert_zero(generated.odes[Z] - Z**2)


def test_trigonometric_lifting_gma_fallback_remains_exact_on_auxiliary_manifold():
    X = sp.Symbol("X", positive=True)
    sym = SymSystem(vars=[X], params={}, odes={X: sp.sin(X)}, initials={X: 0.25})

    rec = recast_to_ssystem(sym)

    Z1 = sp.Symbol("Z_1", positive=True)
    Z2 = sp.Symbol("Z_2", positive=True)
    assert rec.status == RecastStatus.GMA
    assert rec.solver_requirement == SolverRequirement.DAE_REQUIRED
    assert rec.auxiliary_defs == {Z1: sp.sin(X) + 2.0, Z2: sp.cos(X) + 2.0}
    assert rec.gma_equations
    _assert_observable_equivalence(sym, rec)
    _assert_auxiliary_chain_rule(sym, rec)

    generated = _generated_system(rec, model_name="trig_gma")
    _assert_zero(generated.odes[X] - (Z1 - 2))
    assert {"Z_1", "Z_2"} <= {var.name for var in generated.vars}


def test_time_lifting_introduces_clock_and_proves_autonomous_semantics():
    X = sp.Symbol("X", positive=True)
    time = sp.Symbol("time", positive=True)
    sym = SymSystem(vars=[X], params={}, odes={X: sp.sin(time)}, initials={X: 1.0})

    rec = recast_to_ssystem(sym)

    T = sp.Symbol("T", positive=True)
    Z1 = sp.Symbol("Z_1", positive=True)
    Z2 = sp.Symbol("Z_2", positive=True)
    assert rec.auxiliary_defs == {T: time, Z1: sp.sin(T) + 2.0, Z2: sp.cos(T) + 2.0}
    assert rec.initials[T] == 0.0
    _assert_observable_equivalence(sym, rec)
    _assert_auxiliary_chain_rule(sym, rec)

    generated = _generated_system(rec, model_name="time_lift")
    _assert_zero(_ode_by_name(generated, "T") - 1)
    _assert_zero(_ode_by_name(generated, "X") - (Z1 - 2))
    _assert_zero(_ode_by_name(generated, "Z_1") - (Z2 - 2))
    _assert_zero(_ode_by_name(generated, "Z_2") - (2 - Z1))


def test_zero_negative_exponent_uses_eps_init_without_changing_symbolic_dynamics():
    X, k = sp.symbols("X k", positive=True)
    sym = SymSystem(
        vars=[X],
        params={"k": 1.0},
        odes={X: -k / X},
        initials={X: 0.0},
        eps_init=1e-6,
    )

    rec = recast_to_ssystem(sym)

    Z = rec.factor_map[X][0]
    assert rec.initials[Z] == 1e-6
    assert rec.eps_init == 1e-6
    _assert_observable_equivalence(sym, rec)

    generated = _generated_system(rec, model_name="zero_negative")
    _assert_zero(_ode_by_name(generated, Z.name) + k * Z**-1)


def test_symbolic_exponents_and_constant_terms_are_preserved_in_power_laws():
    X, k, g = sp.symbols("X k g", positive=True)

    symbolic = SymSystem(
        vars=[X],
        params={"k": 1.0, "g": 2.5},
        odes={X: k * X**g},
        initials={X: 2.0},
    )
    symbolic_rec = recast_to_ssystem(symbolic)
    symbolic_Z = symbolic_rec.factor_map[X][0]
    _assert_observable_equivalence(symbolic, symbolic_rec)
    assert symbolic_rec.equations[0].growth[1] == {X: g}
    symbolic_generated = _generated_system(symbolic_rec, model_name="symbolic_exponent")
    _assert_zero(_ode_by_name(symbolic_generated, symbolic_Z.name) - k * symbolic_Z**g)

    constant = SymSystem(vars=[X], params={"k": 1.0}, odes={X: k}, initials={X: 1.5})
    constant_rec = recast_to_ssystem(constant)
    constant_Z = constant_rec.factor_map[X][0]
    _assert_observable_equivalence(constant, constant_rec)
    assert constant_rec.equations[0].growth[1] == {}
    constant_generated = _generated_system(constant_rec, model_name="constant_term")
    _assert_zero(_ode_by_name(constant_generated, constant_Z.name) - k)


def test_assignment_rules_and_canonical_slack_have_auditable_antimony_semantics():
    X, A, k = sp.symbols("X A k", positive=True)
    assignment = SymSystem(
        vars=[X],
        params={"k": 1.0},
        odes={X: A * X},
        initials={X: 1.0},
        assignment_rules={"A": "k + X"},
    )

    assignment_rec = recast_to_ssystem(assignment)
    Z = assignment_rec.factor_map[X][0]
    assignment_subs = {A: k + Z}
    _assert_observable_equivalence(assignment, assignment_rec, assignment_subs=assignment_subs)

    assignment_generated = _generated_system(assignment_rec, model_name="assignment_rule")
    assert assignment_generated.assignment_rules == {"A": "k + X", "X": "Z_1"}
    _assert_zero(_ode_by_name(assignment_generated, Z.name) - A * Z)

    slack_rec = recast_to_ssystem(
        SymSystem(
            vars=[X],
            params={"k": 1.0},
            odes={X: -k * X},
            initials={X: 1.0},
            eps_slack=1e-5,
        ),
        mode="canonical",
    )
    slack_generated = _generated_system(
        slack_rec, model_name="canonical_slack", mode="canonical"
    )
    epsilon = sp.Symbol("epsilon", positive=True)
    slack_Z = slack_rec.factor_map[X][0]
    assert slack_generated.params["epsilon"] == 1e-5
    _assert_zero(_ode_by_name(slack_generated, slack_Z.name) + k * slack_Z)
    _assert_zero(epsilon * slack_Z - (epsilon + k) * slack_Z + k * slack_Z)
