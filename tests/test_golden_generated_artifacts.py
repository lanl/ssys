"""Semantic golden tests for generated Antimony artifacts."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
    gma_to_antimony,
    parse_antimony_via_sbml,
    ssystem_to_antimony,
)


def _normalize_expr(expr: sp.Expr) -> sp.Expr:
    """Normalize symbol assumptions away so generated and reloaded expressions compare by name."""
    return sp.simplify(expr.xreplace({sym: sp.Symbol(sym.name) for sym in expr.free_symbols}))


def _assert_expr_equal(actual: sp.Expr, expected: sp.Expr) -> None:
    assert sp.simplify(_normalize_expr(actual) - _normalize_expr(expected)) == 0


def _ode_for(sym: SymSystem, name: str) -> sp.Expr:
    matches = [var for var in sym.vars if var.name == name]
    assert matches, f"reloaded artifact has no state variable named {name}"
    return sym.odes[matches[0]]


def _load_generated_artifact(artifact_path: Path, text: str) -> SymSystem:
    artifact_path.write_text(text, encoding="utf-8")
    assert artifact_path.stat().st_size > 0

    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(text) >= 0, antimony.getLastError()
    module_name = antimony.getMainModuleName()
    assert module_name
    sbml = antimony.getSBMLString(module_name)
    assert "<sbml" in sbml

    return parse_antimony_via_sbml(text)


@dataclass(frozen=True)
class GoldenArtifactCase:
    """A generated artifact and semantic checks expected after SBML reload."""

    name: str
    antimony_text: str
    assert_semantics: Callable[[SymSystem, str], None]


def _identity_ssystem_result() -> RecastResult:
    X, k, d = sp.symbols("X k d", positive=True)
    return RecastResult(
        status=RecastStatus.CANONICAL_SSYSTEM,
        equations=[SSysEquation(X, (k, {X: 1.0}), (d, {X: 2.0}))],
        initials={X: 1.0},
        variables=[X],
        factor_map={X: [X]},
        params={"k": 2.0, "d": 0.5},
    )


def _assert_identity_ssystem(sym: SymSystem, output: str) -> None:
    X, k, d = sp.symbols("X k d")
    assert {var.name for var in sym.vars} == {"X"}
    assert sym.solver_requirement == SolverRequirement.ODE_ONLY
    assert sym.assignment_rules == {}
    assert sym.params["k"] == pytest.approx(2.0)
    assert sym.params["d"] == pytest.approx(0.5)
    _assert_expr_equal(_ode_for(sym, "X"), k * X - d * X**2)
    assert "SOLVER_REQUIREMENT=dae_required" not in output


def _gma_result() -> RecastResult:
    X, a, b, d = sp.symbols("X a b d", positive=True)
    return RecastResult(
        status=RecastStatus.GMA,
        equations=[],
        gma_equations=[
            GMAEquation(
                var=X,
                production=[(a, {X: 1.0}), (b, {X: 2.0})],
                degradation=[(d, {X: 1.0})],
            )
        ],
        initials={X: 1.0},
        variables=[X],
        factor_map={X: [X]},
        params={"a": 1.0, "b": 0.25, "d": 0.5},
    )


def _assert_gma(sym: SymSystem, output: str) -> None:
    X, a, b, d = sp.symbols("X a b d")
    assert {var.name for var in sym.vars} == {"X"}
    assert sym.solver_requirement == SolverRequirement.ODE_ONLY
    _assert_expr_equal(_ode_for(sym, "X"), a * X + b * X**2 - d * X)
    assert "GMA (Generalized Mass Action) format" in output


def _time_varying_gma_result() -> RecastResult:
    X, T, d = sp.symbols("X T d", positive=True)
    return RecastResult(
        status=RecastStatus.GMA,
        equations=[],
        gma_equations=[
            GMAEquation(var=X, production=[(sp.Integer(1), {T: 1.0, X: 1.0})], degradation=[(d, {X: 1.0})]),
            GMAEquation(var=T, production=[(sp.Integer(1), {})], degradation=[]),
        ],
        initials={X: 1.0, T: 0.0},
        variables=[X, T],
        factor_map={X: [X]},
        auxiliary_defs={T: sp.Symbol("time")},
        params={"d": 0.5},
    )


def _assert_time_varying_gma(sym: SymSystem, output: str) -> None:
    X, T, d = sp.symbols("X T d")
    assert {var.name for var in sym.vars} == {"T", "X"}
    assert sym.solver_requirement == SolverRequirement.ODE_ONLY
    _assert_expr_equal(_ode_for(sym, "T"), sp.Integer(1))
    _assert_expr_equal(_ode_for(sym, "X"), T * X - d * X)
    assert "// T := time" in output


def _assignment_rule_gma_result() -> RecastResult:
    X = sp.Symbol("X", positive=True)
    Y = sp.Symbol("Y_1", positive=True)
    return RecastResult(
        status=RecastStatus.GMA,
        equations=[],
        gma_equations=[
            GMAEquation(var=X, production=[(sp.Integer(1), {X: 1.0, Y: -1.0})], degradation=[])
        ],
        initials={X: 1.0, Y: 2.0},
        variables=[X, Y],
        factor_map={X: [X]},
        auxiliary_defs={Y: X + 1},
    )


def _assert_assignment_rule_gma(sym: SymSystem, output: str) -> None:
    X, Y_1 = sp.symbols("X Y_1")
    assert {var.name for var in sym.vars} == {"X"}
    assert sym.solver_requirement == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
    assert sym.assignment_rules == {"Y_1": "X + 1"}
    _assert_expr_equal(_ode_for(sym, "X"), X / Y_1)
    assert "Y_1 := X + 1;" in output


def _dae_required_result() -> RecastResult:
    X = sp.Symbol("X", positive=True)
    return RecastResult(
        status=RecastStatus.CANONICAL_SSYSTEM,
        equations=[SSysEquation(X, (sp.Integer(1), {X: 1.0}), (sp.Integer(0), {}))],
        initials={X: 1.0},
        variables=[X],
        factor_map={X: [X]},
        algebraic_constraints=["X - 1"],
    )


def _assert_dae_required(sym: SymSystem, output: str) -> None:
    X = sp.Symbol("X")
    assert {var.name for var in sym.vars} == {"X"}
    assert sym.solver_requirement == SolverRequirement.DAE_REQUIRED
    _assert_expr_equal(_ode_for(sym, "X"), X)
    assert "SOLVER_REQUIREMENT=dae_required" in output


GOLDEN_ARTIFACTS = [
    GoldenArtifactCase(
        name="simplified_ssystem",
        antimony_text=ssystem_to_antimony(
            _identity_ssystem_result(), model_name="gold_simplified", mode="simplified"
        ),
        assert_semantics=_assert_identity_ssystem,
    ),
    GoldenArtifactCase(
        name="canonical_ssystem",
        antimony_text=ssystem_to_antimony(
            _identity_ssystem_result(), model_name="gold_canonical", mode="canonical"
        ),
        assert_semantics=_assert_identity_ssystem,
    ),
    GoldenArtifactCase(
        name="gma",
        antimony_text=gma_to_antimony(_gma_result(), model_name="gold_gma"),
        assert_semantics=_assert_gma,
    ),
    GoldenArtifactCase(
        name="gma_time_varying",
        antimony_text=gma_to_antimony(_time_varying_gma_result(), model_name="gold_time_gma"),
        assert_semantics=_assert_time_varying_gma,
    ),
    GoldenArtifactCase(
        name="ode_with_assignment_rules",
        antimony_text=gma_to_antimony(
            _assignment_rule_gma_result(),
            model_name="gold_assignment",
            lifted_mode="assignment",
        ),
        assert_semantics=_assert_assignment_rule_gma,
    ),
    GoldenArtifactCase(
        name="dae_required",
        antimony_text=ssystem_to_antimony(
            _dae_required_result(),
            model_name="gold_dae",
            mode="canonical",
        ),
        assert_semantics=_assert_dae_required,
    ),
]


@pytest.mark.parametrize("case", GOLDEN_ARTIFACTS, ids=[case.name for case in GOLDEN_ARTIFACTS])
def test_generated_antimony_artifacts_roundtrip_semantically(case: GoldenArtifactCase, tmp_path):
    """Generated artifacts reload through Antimony/SBML and preserve semantic contracts."""
    artifact_path = tmp_path / f"{case.name}.ant"

    reloaded = _load_generated_artifact(artifact_path, case.antimony_text)

    case.assert_semantics(reloaded, case.antimony_text)
