"""Deterministic generated small-model recasting tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import sympy as sp

from ssys import SymSystem, recast_to_ssystem
from ssys.types import RecastResult, RecastStatus


@dataclass(frozen=True)
class TermSpec:
    coefficient: str
    sign: int
    powers: dict[str, int]


@dataclass(frozen=True)
class GeneratedSmallModel:
    name: str
    sym: SymSystem
    expected_status: RecastStatus
    regression_reason: str | None = None
    expected_refusal_fragment: str | None = None


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
        rhs[eq.var] = sp.simplify(expr)

    for eq in rec.gma_equations:
        expr = sum((_term_expr(*term, phi) for term in eq.production), sp.Integer(0))
        expr -= sum((_term_expr(*term, phi) for term in eq.degradation), sp.Integer(0))
        rhs[eq.var] = sp.simplify(expr)

    return rhs


def _assert_observable_equivalence(sym: SymSystem, rec: RecastResult) -> None:
    phi = _phi(rec)
    rhs = _recast_rhs(rec)
    recast_vars = sorted(rhs.keys(), key=str)

    for original in sorted(sym.odes.keys(), key=str):
        phi_expr = phi[original]
        dphi_dt = sum(
            sp.diff(phi_expr, recast_var) * rhs[recast_var] for recast_var in recast_vars
        )
        residual = dphi_dt - sym.odes[original].subs(phi)
        _assert_zero(residual)


def _generated_system(
    name: str,
    variable_names: tuple[str, ...],
    terms_by_variable: dict[str, tuple[TermSpec, ...]],
    initials: dict[str, float],
    *,
    expected_status: RecastStatus,
    regression_reason: str | None = None,
    expected_refusal_fragment: str | None = None,
) -> GeneratedSmallModel:
    variable_symbols = dict(
        zip(variable_names, sp.symbols(" ".join(variable_names), positive=True, seq=True))
    )
    coefficient_names = sorted(
        {term.coefficient for terms in terms_by_variable.values() for term in terms}
    )
    coefficient_symbols = dict(
        zip(
            coefficient_names,
            sp.symbols(" ".join(coefficient_names), positive=True, seq=True),
        )
    )
    odes: dict[sp.Symbol, sp.Expr] = {}
    for variable_name, terms in terms_by_variable.items():
        expr = sp.Integer(0)
        for term in terms:
            monomial = coefficient_symbols[term.coefficient]
            for base_name, power in term.powers.items():
                monomial *= variable_symbols[base_name] ** power
            expr += term.sign * monomial
        odes[variable_symbols[variable_name]] = sp.simplify(expr)

    sym = SymSystem(
        vars=[variable_symbols[name] for name in variable_names],
        params={name: float(idx + 1) for idx, name in enumerate(coefficient_names)},
        odes=odes,
        initials={variable_symbols[name]: value for name, value in initials.items()},
    )
    return GeneratedSmallModel(
        name=name,
        sym=sym,
        expected_status=expected_status,
        regression_reason=regression_reason,
        expected_refusal_fragment=expected_refusal_fragment,
    )


def _generated_cases() -> list[GeneratedSmallModel]:
    return [
        _generated_system(
            "single_state_logistic_pool",
            ("X",),
            {
                "X": (
                    TermSpec("a", 1, {"X": 1}),
                    TermSpec("b", -1, {"X": 2}),
                )
            },
            {"X": 1.2},
            expected_status=RecastStatus.CANONICAL_SSYSTEM,
        ),
        _generated_system(
            "two_state_cross_coupled_pool",
            ("X", "Y"),
            {
                "X": (
                    TermSpec("a", 1, {"X": 1, "Y": 1}),
                    TermSpec("b", -1, {"X": 1}),
                ),
                "Y": (
                    TermSpec("c", 1, {"X": 1}),
                    TermSpec("d", -1, {"Y": 2}),
                ),
            },
            {"X": 1.1, "Y": 0.7},
            expected_status=RecastStatus.CANONICAL_SSYSTEM,
        ),
        _generated_system(
            "three_state_cycle_pool",
            ("X", "Y", "Z"),
            {
                "X": (
                    TermSpec("a", 1, {"X": 1, "Y": 1}),
                    TermSpec("b", -1, {"Z": 1}),
                ),
                "Y": (
                    TermSpec("c", 1, {"Y": 1, "Z": 1}),
                    TermSpec("d", -1, {"X": 1}),
                ),
                "Z": (
                    TermSpec("e", 1, {"X": 1}),
                    TermSpec("f", -1, {"Y": 1, "Z": 1}),
                ),
            },
            {"X": 1.1, "Y": 0.7, "Z": 2.0},
            expected_status=RecastStatus.CANONICAL_SSYSTEM,
        ),
        _generated_system(
            "gma_preflight_seven_terms",
            ("X",),
            {
                "X": (
                    TermSpec("a", 1, {"X": 1}),
                    TermSpec("b", 1, {"X": 2}),
                    TermSpec("c", 1, {"X": 3}),
                    TermSpec("d", 1, {"X": 4}),
                    TermSpec("e", -1, {"X": 5}),
                    TermSpec("f", -1, {"X": 6}),
                    TermSpec("g", -1, {"X": 7}),
                )
            },
            {"X": 1.3},
            expected_status=RecastStatus.GMA,
            regression_reason=(
                "Minimized fallback fixture: one ODE with seven monomial terms must "
                "refuse pool construction without losing GMA equivalence."
            ),
            expected_refusal_fragment="equation has 7 terms",
        ),
    ]


def _nonzero_term_count(sym: SymSystem) -> int:
    return sum(
        len([term for term in sp.Add.make_args(sp.expand(ode)) if term != 0])
        for ode in sym.odes.values()
    )


@pytest.mark.parametrize("case", _generated_cases(), ids=lambda case: case.name)
def test_generated_small_models_preserve_observable_dynamics(
    case: GeneratedSmallModel,
) -> None:
    assert len(case.sym.odes) <= 3
    assert _nonzero_term_count(case.sym) <= 7

    rec = recast_to_ssystem(case.sym)

    assert rec.status == case.expected_status
    assert set(rec.factor_map) == set(case.sym.vars)
    _assert_observable_equivalence(case.sym, rec)


def test_generated_regression_fixtures_record_minimized_counterexamples() -> None:
    regression_cases = [case for case in _generated_cases() if case.regression_reason]

    assert regression_cases
    for case in regression_cases:
        assert case.expected_refusal_fragment is not None
        assert case.regression_reason

        rec = recast_to_ssystem(case.sym)

        assert rec.status == RecastStatus.GMA
        assert case.expected_refusal_fragment in (rec.canonical_refusal_reason or "")
        _assert_observable_equivalence(case.sym, rec)
