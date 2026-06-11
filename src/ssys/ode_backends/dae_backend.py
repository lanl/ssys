"""
Projection backend for explicit algebraic-manifold recast outputs.

This is intentionally narrow: it supports constraints that are explicit
assignment rules or auxiliary definitions, where algebraic variables can be
recomputed from differential states, parameters, and time. Implicit algebraic
systems are reported as unsupported instead of being treated as successful ODE
simulations.
"""

import re
from typing import Any

import numpy as np
import sympy as sp

from ..recaster import ModelIR, SolverRequirement
from .roadrunner_backend import simulate_with_roadrunner


def _failure(message: str, *, unsupported: bool = False) -> dict[str, Any]:
    return {
        "t": np.array([]),
        "y": np.array([]),
        "state_names": [],
        "success": False,
        "message": message,
        "backend": "dae_projection",
        "solver_requirement": SolverRequirement.DAE_REQUIRED.value,
        "unsupported_solver_requirement": unsupported,
        "integrator_stats": {},
        "algebraic_residuals": {},
    }


def _sympify_expr(expr: str | sp.Expr, known_names: set[str] | None = None) -> sp.Expr:
    if isinstance(expr, sp.Expr):
        return expr
    expr_text = str(expr).replace("^", "**")
    sympy_functions = {
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "sqrt",
        "pow",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
    }
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_text))
    names = (known_names or set()) | identifiers
    locals_by_name = {
        name: sp.Symbol(name, positive=True)
        for name in names
        if name not in sympy_functions
    }
    return sp.sympify(expr_text, locals=locals_by_name)


def _evaluate_expr_over_trajectory(
    expr: str | sp.Expr,
    *,
    t: np.ndarray,
    y: np.ndarray,
    state_names: list[str],
    params: dict[str, float],
) -> np.ndarray:
    known_names = set(state_names) | set(params) | {"time", "t"}
    expr_obj = _sympify_expr(expr, known_names)
    symbols = sorted(expr_obj.free_symbols, key=lambda sym: sym.name)
    state_index = {name: idx for idx, name in enumerate(state_names)}

    args: list[np.ndarray | float] = []
    for sym in symbols:
        name = sym.name
        if name in state_index:
            args.append(y[:, state_index[name]])
        elif name in params:
            args.append(float(params[name]))
        elif name.lower() == "time":
            args.append(t)
        elif name == "t":
            args.append(t)
        else:
            raise ValueError(f"Cannot evaluate algebraic expression; missing symbol {name!r}")

    if not symbols:
        return np.full_like(t, float(expr_obj), dtype=float)

    func = sp.lambdify(symbols, expr_obj, modules="numpy")
    values = func(*args)
    values_array = np.asarray(values, dtype=float)
    if values_array.shape == ():
        return np.full_like(t, float(values_array), dtype=float)
    return values_array


def _project_variable(
    *,
    name: str,
    expr: str | sp.Expr,
    t: np.ndarray,
    y: np.ndarray,
    state_names: list[str],
    params: dict[str, float],
) -> tuple[np.ndarray, float]:
    values = _evaluate_expr_over_trajectory(
        expr, t=t, y=y, state_names=state_names, params=params
    )
    if name in state_names:
        idx = state_names.index(name)
        residual = np.asarray(y[:, idx] - values, dtype=float)
        y[:, idx] = values
    else:
        residual = np.zeros_like(values, dtype=float)
        y = np.column_stack([y, values])
        state_names.append(name)
    return y, float(np.max(np.abs(residual))) if residual.size else 0.0


def simulate_with_dae_projection(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Integrate differential states and project explicit algebraic variables.

    Options:
        auxiliary_defs: optional dict of auxiliary symbol -> definition expression
        residual_threshold: currently recorded in output; enforcement is handled
            by the validator's residual test.
    """
    options = options or {}
    implicit_constraints = list(getattr(model_ir, "algebraic_constraints", []) or [])
    assignment_rules = dict(getattr(model_ir, "assignment_rules", {}) or {})
    auxiliary_defs = dict(options.get("auxiliary_defs", {}) or {})

    if implicit_constraints and not assignment_rules and not auxiliary_defs:
        return _failure(
            "DAE projection backend supports explicit assignment rules or auxiliary "
            "definitions, not unsupported implicit algebraic constraints.",
            unsupported=True,
        )

    base = simulate_with_roadrunner(model_ir, t0, t_end, n_points, y0_override, options)
    if not base.get("success", False):
        base["backend"] = "dae_projection"
        base["solver_requirement"] = SolverRequirement.DAE_REQUIRED.value
        base["unsupported_solver_requirement"] = base.get(
            "unsupported_solver_requirement", False
        )
        base.setdefault("algebraic_residuals", {})
        return base

    t = np.asarray(base["t"], dtype=float)
    y = np.asarray(base["y"], dtype=float).copy()
    state_names = list(base["state_names"])
    params = dict(getattr(model_ir, "params", {}) or {})
    residuals: dict[str, float] = {}

    try:
        for rule_name, rule_expr in assignment_rules.items():
            y, max_residual = _project_variable(
                name=str(rule_name),
                expr=rule_expr,
                t=t,
                y=y,
                state_names=state_names,
                params=params,
            )
            residuals[str(rule_name)] = max_residual

        for aux, defn in auxiliary_defs.items():
            aux_name = aux.name if hasattr(aux, "name") else str(aux)
            if aux_name in assignment_rules:
                continue
            y, max_residual = _project_variable(
                name=aux_name,
                expr=defn,
                t=t,
                y=y,
                state_names=state_names,
                params=params,
            )
            residuals[aux_name] = max_residual
    except Exception as exc:
        return _failure(f"DAE projection failed: {exc}", unsupported=True)

    base["y"] = y
    base["state_names"] = state_names
    base["backend"] = "dae_projection"
    base["solver_requirement"] = SolverRequirement.DAE_REQUIRED.value
    base["unsupported_solver_requirement"] = False
    base["algebraic_residuals"] = residuals
    return base
