"""Adversarial and literature-style local model corpus tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import sympy as sp

from ssys import SymSystem, recast_to_ssystem
from ssys.types import RecastResult, RecastStatus, SolverRequirement


@dataclass(frozen=True)
class AdversarialCase:
    name: str
    failure_mode: str
    system: SymSystem
    assert_recast: Callable[[SymSystem, RecastResult], None]


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
        rhs[eq.var] = sp.simplify(
            _term_expr(*eq.growth, phi) - _term_expr(*eq.decay, phi)
        )
    for eq in rec.gma_equations:
        production = sum((_term_expr(*term, phi) for term in eq.production), sp.Integer(0))
        degradation = sum((_term_expr(*term, phi) for term in eq.degradation), sp.Integer(0))
        rhs[eq.var] = sp.simplify(production - degradation)
    return rhs


def _assert_observable_equivalence(sym: SymSystem, rec: RecastResult) -> None:
    phi = _phi(rec)
    rhs = _recast_rhs(rec)
    recast_vars = sorted(rhs.keys(), key=str)
    auxiliary_subs = dict(rec.auxiliary_defs)
    parameter_subs = {
        sp.Symbol(name, positive=True): value for name, value in rec.params.items()
    }
    for original in sorted(sym.odes.keys(), key=str):
        phi_expr = phi[original]
        dphi_dt = sum(
            sp.diff(phi_expr, recast_var) * rhs[recast_var] for recast_var in recast_vars
        )
        residual = (dphi_dt - sym.odes[original].subs(phi)).subs(auxiliary_subs)
        residual = residual.subs(parameter_subs)
        _assert_zero(residual)


def _assert_recasts_exactly(sym: SymSystem, rec: RecastResult) -> None:
    assert rec.status in {RecastStatus.CANONICAL_SSYSTEM, RecastStatus.GMA}
    assert set(rec.factor_map) == set(sym.vars)
    _assert_observable_equivalence(sym, rec)


def _adversarial_cases() -> list[AdversarialCase]:
    x, y, a, b, k, g, k_small = sp.symbols("X Y a b k g K", positive=True)

    def stiff(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.params["k"] == 1000.0

    def near_singular(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.solver_requirement == SolverRequirement.DAE_REQUIRED
        assert any(sp.simplify(defn - (k_small + x)) == 0 for defn in rec.auxiliary_defs.values())

    def extreme_scale(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.params == {"a": 1.0e-12, "b": 1.0e12}

    def symbolic_order(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.equations[0].growth[1] == {x: g}

    def nonpositive_domain(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.eps_init == 1.0e-7
        assert any(value == pytest.approx(1.0e-7) for value in rec.initials.values())

    def conserved_quantity(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        total = sum(sym.odes.values(), sp.Integer(0))
        _assert_zero(total)

    def multiple_compartments(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.compartments == {"cytosol": 1.0, "nucleus": 0.25}

    def denominator_preservation(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.solver_requirement == SolverRequirement.DAE_REQUIRED
        assert any(sp.simplify(defn - (x**4 + 1)) == 0 for defn in rec.auxiliary_defs.values())
        assert not any(
            sp.simplify(defn - ((x**4 + 1) * (a**2 + 1))) == 0
            for defn in rec.auxiliary_defs.values()
        )

    def dae_required_lift(sym: SymSystem, rec: RecastResult) -> None:
        _assert_recasts_exactly(sym, rec)
        assert rec.solver_requirement == SolverRequirement.DAE_REQUIRED
        assert any(sp.simplify(defn - sp.exp(x)) == 0 for defn in rec.auxiliary_defs.values())

    return [
        AdversarialCase(
            name="stiff_linear_decay",
            failure_mode="prevents solver-scale assumptions from changing symbolic recast",
            system=SymSystem(
                vars=[x],
                params={"k": 1000.0},
                odes={x: -k * x},
                initials={x: 1.0},
            ),
            assert_recast=stiff,
        ),
        AdversarialCase(
            name="near_singular_denominator",
            failure_mode="prevents rational lifting from dropping small denominator offsets",
            system=SymSystem(
                vars=[x],
                params={"K": 1.0e-9},
                odes={x: x / (k_small + x)},
                initials={x: 1.0e-8},
            ),
            assert_recast=near_singular,
        ),
        AdversarialCase(
            name="extreme_scale_logistic",
            failure_mode="prevents coefficient underflow/overflow assumptions in recast metadata",
            system=SymSystem(
                vars=[x],
                params={"a": 1.0e-12, "b": 1.0e12},
                odes={x: a * x - b * x**2},
                initials={x: 1.0e-6},
            ),
            assert_recast=extreme_scale,
        ),
        AdversarialCase(
            name="symbolic_kinetic_order",
            failure_mode="prevents symbolic power-law exponents from being coerced to floats",
            system=SymSystem(
                vars=[x],
                params={"k": 1.0, "g": 1.5},
                odes={x: k * x**g},
                initials={x: 1.0},
            ),
            assert_recast=symbolic_order,
        ),
        AdversarialCase(
            name="nonpositive_domain_eps_init",
            failure_mode="prevents zero initial values with negative powers from producing singular ICs",
            system=SymSystem(
                vars=[x],
                params={"k": 1.0},
                odes={x: -k / x},
                initials={x: 0.0},
                eps_init=1.0e-7,
            ),
            assert_recast=nonpositive_domain,
        ),
        AdversarialCase(
            name="conserved_quantity",
            failure_mode="prevents pool construction from breaking a conserved total",
            system=SymSystem(
                vars=[x, y],
                params={"k": 0.5},
                odes={x: -k * x, y: k * x},
                initials={x: 1.0, y: 0.0},
            ),
            assert_recast=conserved_quantity,
        ),
        AdversarialCase(
            name="multiple_compartments",
            failure_mode="prevents compartment metadata from being dropped during recast",
            system=SymSystem(
                vars=[x, y],
                params={"k": 0.2},
                odes={x: -k * x, y: k * x},
                initials={x: 1.0, y: 0.5},
                compartments={"cytosol": 1.0, "nucleus": 0.25},
            ),
            assert_recast=multiple_compartments,
        ),
        AdversarialCase(
            name="denominator_preservation",
            failure_mode="prevents separate rational denominators from being merged before lifting",
            system=SymSystem(
                vars=[x],
                params={"a": 3.0, "b": 2.0},
                odes={x: x / (1 + x**4) + b / (1 + a**2)},
                initials={x: 1.0},
            ),
            assert_recast=denominator_preservation,
        ),
        AdversarialCase(
            name="dae_required_composite_lift",
            failure_mode="prevents lifted state-dependent auxiliaries from being mislabeled ODE-only",
            system=SymSystem(
                vars=[x],
                params={},
                odes={x: sp.exp(x)},
                initials={x: 0.0},
            ),
            assert_recast=dae_required_lift,
        ),
    ]


@pytest.mark.parametrize("case", _adversarial_cases(), ids=lambda case: case.name)
def test_adversarial_model_corpus_cases_state_failure_mode_and_recast_exactly(
    case: AdversarialCase,
) -> None:
    assert case.failure_mode

    rec = recast_to_ssystem(case.system)

    case.assert_recast(case.system, rec)
