"""IDA/SUNDIALS backend for DAE-capable validation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np
import sympy as sp

from ..recaster import ModelIR, SolverRequirement

_INSTALL_HINT = (
    "Install optional DAE dependencies with `uv sync --extra dae` "
    "or `uv pip install -e '.[dae]'`."
)
_DEFAULT_INITIAL_RESIDUAL_THRESHOLD = 1.0e-8
_SYMPY_FUNCTIONS = {
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


class IDASundialsUnavailable(ImportError):
    """Raised when no supported Python binding for SUNDIALS IDA is installed."""


class UnsupportedDAESystem(ValueError):
    """Raised when the model cannot be represented as a square IDA residual."""


class InconsistentInitialCondition(ValueError):
    """Raised when algebraic initial conditions fail the configured threshold."""

    def __init__(self, message: str, residuals: dict[str, float]) -> None:
        self.residuals = residuals
        super().__init__(message)


@dataclass
class _IDABinding:
    package: str
    version: str
    solver_class: type


@dataclass
class _CompiledExpression:
    expr: sp.Expr
    symbols: list[sp.Symbol]
    func: Callable[..., Any]

    def evaluate(self, values_by_name: dict[str, float]) -> float:
        args = []
        for sym in self.symbols:
            name = sym.name
            if name not in values_by_name:
                raise ValueError(f"missing value for symbol {name!r}")
            args.append(values_by_name[name])
        return float(np.asarray(self.func(*args), dtype=float))


@dataclass
class _ResidualEquation:
    name: str
    kind: str
    variable_index: int | None
    expr: _CompiledExpression | None = None
    constraint_index: int | None = None


@dataclass
class _ResidualSystem:
    variable_names: list[str]
    algebraic_indices: list[int]
    equations: list[_ResidualEquation]
    params: dict[str, float]
    y0: np.ndarray
    ydot0: np.ndarray
    initial_residual_norms: dict[str, float]
    initial_condition_repaired: bool


def _failure(
    message: str,
    *,
    unsupported: bool = False,
    integrator_stats: dict[str, Any] | None = None,
    initial_residual_norms: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "t": np.array([]),
        "y": np.array([]),
        "state_names": [],
        "success": False,
        "message": message,
        "backend": "ida_sundials",
        "solver_requirement": SolverRequirement.DAE_REQUIRED.value,
        "unsupported_solver_requirement": unsupported,
        "integrator_stats": integrator_stats or {},
        "algebraic_residuals": {},
        "initial_residual_norms": initial_residual_norms or {},
    }


def _load_ida_binding() -> _IDABinding:
    try:
        from sksundae.ida import IDA
    except ImportError as exc:
        raise IDASundialsUnavailable(
            f"scikit-SUNDAE is not installed. {_INSTALL_HINT}"
        ) from exc

    try:
        version = metadata.version("scikit-sundae")
    except metadata.PackageNotFoundError:
        version = "unknown"

    return _IDABinding(package="scikit-sundae", version=version, solver_class=IDA)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _expression_text(expr: str | sp.Expr) -> str:
    return str(expr).replace("^", "**")


def _identifier_names(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))


def _is_zero_expr(expr: str | sp.Expr) -> bool:
    try:
        return sp.simplify(_sympify(expr)) == 0
    except Exception:
        return False


def _sympify(expr: str | sp.Expr, known_names: set[str] | None = None) -> sp.Expr:
    if isinstance(expr, sp.Expr):
        return expr

    text = _expression_text(expr)
    names = set(known_names or set()) | _identifier_names(text)
    locals_by_name = {
        name: sp.Symbol(name)
        for name in names
        if name not in _SYMPY_FUNCTIONS
    }
    return sp.sympify(text, locals=locals_by_name)


def _compile_expr(expr: str | sp.Expr, known_names: set[str]) -> _CompiledExpression:
    expr_obj = _sympify(expr, known_names)
    symbols = sorted(expr_obj.free_symbols, key=lambda sym: sym.name)
    func = sp.lambdify(symbols, expr_obj, modules="numpy")
    return _CompiledExpression(expr=expr_obj, symbols=symbols, func=func)


def _append_unique(names: list[str], additions: list[str] | set[str]) -> None:
    seen = set(names)
    for name in additions:
        name_str = str(name)
        if name_str not in seen:
            names.append(name_str)
            seen.add(name_str)


def _model_variable_names(
    model_ir: ModelIR,
    ode_exprs: dict[str, str | sp.Expr],
    assignment_rules: dict[str, str | sp.Expr],
    auxiliary_defs: dict[str, str | sp.Expr],
    constraints: list[str | sp.Expr],
    params: dict[str, float],
) -> list[str]:
    names: list[str] = []

    vars_attr = getattr(model_ir, "vars", None)
    if vars_attr:
        _append_unique(names, [sym.name if hasattr(sym, "name") else str(sym) for sym in vars_attr])

    species_attr = getattr(model_ir, "species", None)
    if species_attr:
        _append_unique(names, sorted(str(name) for name in species_attr))

    _append_unique(names, list(ode_exprs.keys()))
    _append_unique(names, list(assignment_rules.keys()))
    _append_unique(names, list(auxiliary_defs.keys()))

    param_names = set(params)
    for constraint in constraints:
        for name in sorted(_identifier_names(_expression_text(constraint))):
            if name in _SYMPY_FUNCTIONS or name in param_names or name.lower() in {"time", "t"}:
                continue
            _append_unique(names, [name])

    return names


def _ode_expressions(model_ir: ModelIR) -> dict[str, str | sp.Expr]:
    odes = getattr(model_ir, "odes", None)
    if odes:
        return {
            key.name if hasattr(key, "name") else str(key): expr
            for key, expr in dict(odes).items()
        }

    explicit_rates = getattr(model_ir, "explicit_rates", None)
    if explicit_rates:
        return {str(key): expr for key, expr in dict(explicit_rates).items()}

    reactions = getattr(model_ir, "reactions", None)
    if reactions:
        from ..recaster import build_sym_system

        sym = build_sym_system(model_ir)
        return _ode_expressions(sym)

    return {}


def _initial_values(model_ir: ModelIR) -> dict[str, float]:
    raw_initials = getattr(model_ir, "initial", None) or getattr(model_ir, "initials", None) or {}
    return {
        key.name if hasattr(key, "name") else str(key): _as_float(value)
        for key, value in dict(raw_initials).items()
    }


def _parameter_values(model_ir: ModelIR) -> dict[str, float]:
    return {
        str(key): _as_float(value)
        for key, value in dict(getattr(model_ir, "params", {}) or {}).items()
    }


def _expanded_auxiliary_defs(options: dict[str, Any]) -> dict[str, str | sp.Expr]:
    aux_defs = {}
    for key, value in dict(options.get("auxiliary_defs", {}) or {}).items():
        aux_defs[key.name if hasattr(key, "name") else str(key)] = value
    return aux_defs


def _definition_mentions_state(expr: str | sp.Expr, state_names: set[str]) -> bool:
    try:
        return any(sym.name in state_names for sym in _sympify(expr, state_names).free_symbols)
    except Exception:
        return True


def _select_implicit_slots(
    variable_names: list[str],
    algebraic_names: set[str],
    ode_exprs: dict[str, str | sp.Expr],
    constraints: list[str | sp.Expr],
    params: dict[str, float],
) -> list[str]:
    if not constraints:
        return []

    candidates: list[str] = []
    for name in variable_names:
        if name in algebraic_names:
            continue
        expr = ode_exprs.get(name)
        if expr is None or _is_zero_expr(expr):
            if any(name in _identifier_names(_expression_text(constraint)) for constraint in constraints):
                candidates.append(name)

    if len(candidates) < len(constraints):
        raise UnsupportedDAESystem(
            "IDA/SUNDIALS backend needs one algebraic variable with no differential "
            "equation for each implicit algebraic constraint."
        )

    if len(candidates) > len(constraints):
        param_names = set(params)
        constrained_names = []
        for constraint in constraints:
            ids = _identifier_names(_expression_text(constraint))
            constrained_names.extend(
                name
                for name in sorted(ids)
                if name not in param_names and name not in _SYMPY_FUNCTIONS
            )
        preferred = [name for name in candidates if name in constrained_names]
        if len(preferred) >= len(constraints):
            return preferred[: len(constraints)]

    return candidates[: len(constraints)]


def _context_from_state(
    variable_names: list[str],
    params: dict[str, float],
    t: float,
    y: np.ndarray,
) -> dict[str, float]:
    values = {name: float(y[idx]) for idx, name in enumerate(variable_names)}
    values.update(params)
    values["time"] = float(t)
    values["t"] = float(t)
    return values


def _evaluate_definition_initials(
    *,
    y0: np.ndarray,
    variable_names: list[str],
    params: dict[str, float],
    definitions: dict[str, _CompiledExpression],
) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(variable_names)}
    for _ in range(max(1, len(definitions) + 1)):
        changed = False
        values = _context_from_state(variable_names, params, 0.0, y0)
        for name, expr in definitions.items():
            if name not in name_to_idx:
                continue
            expected = expr.evaluate(values)
            idx = name_to_idx[name]
            if not np.isclose(y0[idx], expected, rtol=0.0, atol=1.0e-14):
                y0[idx] = expected
                changed = True
                values[name] = expected
        if not changed:
            break


def _initial_residuals(
    residual_func: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    equations: list[_ResidualEquation],
    y0: np.ndarray,
    ydot0: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(residual_func(0.0, y0, ydot0), dtype=float)
    return {
        equation.name: float(abs(values[idx]))
        for idx, equation in enumerate(equations)
        if equation.kind != "differential"
    }


def _build_residual_system(
    model_ir: ModelIR,
    y0_override: dict[str, float] | None,
    options: dict[str, Any],
) -> _ResidualSystem:
    params = _parameter_values(model_ir)
    ode_exprs = _ode_expressions(model_ir)
    assignment_rules = {
        str(key): value
        for key, value in dict(getattr(model_ir, "assignment_rules", {}) or {}).items()
    }
    auxiliary_defs = _expanded_auxiliary_defs(options)
    constraints = list(getattr(model_ir, "algebraic_constraints", []) or [])

    variable_names = _model_variable_names(
        model_ir, ode_exprs, assignment_rules, auxiliary_defs, constraints, params
    )
    if not variable_names:
        raise UnsupportedDAESystem("IDA/SUNDIALS backend found no model variables to solve.")

    state_names = set(variable_names)
    explicit_algebraic_defs: dict[str, str | sp.Expr] = {}
    for name, expr in assignment_rules.items():
        explicit_algebraic_defs[name] = expr
    for name, expr in auxiliary_defs.items():
        if name in state_names and _definition_mentions_state(expr, state_names):
            explicit_algebraic_defs.setdefault(name, expr)

    algebraic_names = set(explicit_algebraic_defs)
    implicit_slots = _select_implicit_slots(
        variable_names, algebraic_names, ode_exprs, constraints, params
    )
    algebraic_names.update(implicit_slots)
    name_to_idx = {name: idx for idx, name in enumerate(variable_names)}

    known_names = set(variable_names) | set(params) | {"time", "t"}
    for name in assignment_rules:
        known_names.add(name)

    compiled_odes = {
        name: _compile_expr(expr, known_names)
        for name, expr in ode_exprs.items()
        if name in name_to_idx
    }
    compiled_defs = {
        name: _compile_expr(expr, known_names)
        for name, expr in explicit_algebraic_defs.items()
        if name in name_to_idx
    }
    compiled_constraints = [
        _compile_expr(constraint, known_names)
        for constraint in constraints
    ]

    equations: list[_ResidualEquation] = []
    constraint_by_slot = dict(zip(implicit_slots, compiled_constraints, strict=False))
    for name in variable_names:
        idx = name_to_idx[name]
        if name in compiled_defs:
            equations.append(
                _ResidualEquation(
                    name=name,
                    kind="explicit_algebraic",
                    variable_index=idx,
                    expr=compiled_defs[name],
                )
            )
        elif name in constraint_by_slot:
            constraint_idx = implicit_slots.index(name)
            equations.append(
                _ResidualEquation(
                    name=f"algebraic_constraint:{constraint_idx + 1}",
                    kind="implicit_algebraic",
                    variable_index=None,
                    expr=constraint_by_slot[name],
                    constraint_index=constraint_idx,
                )
            )
        else:
            equations.append(
                _ResidualEquation(
                    name=name,
                    kind="differential",
                    variable_index=idx,
                    expr=compiled_odes.get(name),
                )
            )

    initial_map = _initial_values(model_ir)
    y0 = np.array([initial_map.get(name, 0.0) for name in variable_names], dtype=float)
    override_names = set()
    if y0_override:
        for name, value in y0_override.items():
            name_str = name.name if hasattr(name, "name") else str(name)
            if name_str in name_to_idx:
                y0[name_to_idx[name_str]] = _as_float(value)
                override_names.add(name_str)

    initial_condition_repaired = False
    repair_user_initials = bool(options.get("repair_consistent_initial_conditions", False))
    if compiled_defs:
        before = y0.copy()
        _evaluate_definition_initials(
            y0=y0,
            variable_names=variable_names,
            params=params,
            definitions={
                name: expr
                for name, expr in compiled_defs.items()
                if name not in override_names or repair_user_initials
            },
        )
        initial_condition_repaired = not np.allclose(before, y0, rtol=0.0, atol=1.0e-14)

    algebraic_indices = [name_to_idx[name] for name in variable_names if name in algebraic_names]

    def residual_values(t: float, y: np.ndarray, ydot: np.ndarray) -> np.ndarray:
        values = _context_from_state(variable_names, params, t, y)
        residual = np.zeros(len(equations), dtype=float)
        for eq_idx, equation in enumerate(equations):
            if equation.kind == "differential":
                if equation.variable_index is None:
                    raise ValueError(f"differential equation {equation.name!r} has no state")
                rhs = 0.0 if equation.expr is None else equation.expr.evaluate(values)
                residual[eq_idx] = float(ydot[equation.variable_index]) - rhs
            elif equation.kind == "explicit_algebraic":
                if equation.variable_index is None or equation.expr is None:
                    raise ValueError(f"algebraic equation {equation.name!r} is incomplete")
                residual[eq_idx] = float(y[equation.variable_index]) - equation.expr.evaluate(values)
            else:
                if equation.expr is None:
                    raise ValueError(f"implicit equation {equation.name!r} is incomplete")
                residual[eq_idx] = equation.expr.evaluate(values)
        return residual

    ydot0 = np.zeros_like(y0)
    values = _context_from_state(variable_names, params, 0.0, y0)
    for equation in equations:
        if equation.kind == "differential" and equation.variable_index is not None:
            ydot0[equation.variable_index] = (
                0.0 if equation.expr is None else equation.expr.evaluate(values)
            )

    initial_residual_norms = _initial_residuals(residual_values, equations, y0, ydot0)
    threshold = options.get("initial_residual_threshold", _DEFAULT_INITIAL_RESIDUAL_THRESHOLD)
    inconsistent = {
        name: residual
        for name, residual in initial_residual_norms.items()
        if residual > threshold
    }
    if inconsistent:
        user_algebraic = sorted(override_names & set(compiled_defs))
        user_context = (
            " Inconsistent user-provided algebraic initial conditions are repaired "
            "only when repair_consistent_initial_conditions=True."
            if user_algebraic
            else ""
        )
        raise InconsistentInitialCondition(
            "Inconsistent algebraic initial conditions before integration: "
            + ", ".join(f"{name}={value:.3e}" for name, value in inconsistent.items())
            + user_context,
            initial_residual_norms,
        )

    return _ResidualSystem(
        variable_names=variable_names,
        algebraic_indices=algebraic_indices,
        equations=equations,
        params=params,
        y0=y0,
        ydot0=ydot0,
        initial_residual_norms=initial_residual_norms,
        initial_condition_repaired=initial_condition_repaired,
    )


def _make_sksundae_residual(
    system: _ResidualSystem,
) -> Callable[[float, np.ndarray, np.ndarray, np.ndarray | None], np.ndarray | None]:
    def residual(t: float, y: np.ndarray, ydot: np.ndarray, out: np.ndarray | None = None):
        values = np.asarray(_residual_values(system, t, y, ydot), dtype=float)
        if out is not None:
            out[:] = values
            return None
        return values

    return residual


def _residual_values(
    system: _ResidualSystem, t: float, y: np.ndarray, ydot: np.ndarray
) -> np.ndarray:
    values_by_name = _context_from_state(system.variable_names, system.params, t, y)
    residual = np.zeros(len(system.equations), dtype=float)
    for eq_idx, equation in enumerate(system.equations):
        if equation.kind == "differential":
            rhs = 0.0 if equation.expr is None else equation.expr.evaluate(values_by_name)
            residual[eq_idx] = float(ydot[equation.variable_index]) - rhs  # type: ignore[index]
        elif equation.kind == "explicit_algebraic":
            expected = equation.expr.evaluate(values_by_name)  # type: ignore[union-attr]
            residual[eq_idx] = float(y[equation.variable_index]) - expected  # type: ignore[index]
        else:
            residual[eq_idx] = equation.expr.evaluate(values_by_name)  # type: ignore[union-attr]
    return residual


def _extract_solution_arrays(solution: Any) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray | None, Any, str]:
    if isinstance(solution, dict):
        success = bool(solution.get("success", True))
        t = np.asarray(solution.get("t", []), dtype=float)
        y = np.asarray(solution.get("y", []), dtype=float)
        ydot = solution.get("yp", solution.get("ydot"))
        status = solution.get("status", solution.get("flag"))
        message = str(solution.get("message", ""))
    elif hasattr(solution, "t") and hasattr(solution, "y"):
        success = bool(getattr(solution, "success", True))
        t = np.asarray(solution.t, dtype=float)
        y = np.asarray(solution.y, dtype=float)
        ydot = getattr(solution, "yp", getattr(solution, "ydot", None))
        status = getattr(solution, "status", getattr(solution, "flag", None))
        message = str(getattr(solution, "message", ""))
    elif hasattr(solution, "values"):
        values = solution.values
        flag = getattr(solution, "flag", None)
        success = flag is None or int(flag) >= 0
        t = np.asarray(values.t, dtype=float)
        y = np.asarray(values.y, dtype=float)
        ydot = getattr(values, "ydot", getattr(values, "yp", None))
        status = flag
        message = str(getattr(solution, "message", ""))
    else:
        raise RuntimeError(f"Unsupported IDA solution object: {type(solution)!r}")

    if y.ndim != 2:
        raise RuntimeError(f"IDA returned y with shape {y.shape}, expected a 2-D array")
    if t.ndim != 1:
        raise RuntimeError(f"IDA returned t with shape {t.shape}, expected a 1-D array")
    if y.shape[0] != len(t) and y.shape[1] == len(t):
        y = y.T
    if y.shape[0] != len(t):
        raise RuntimeError(
            f"IDA returned incompatible time/state shapes: t={t.shape}, y={y.shape}"
        )

    ydot_array = None if ydot is None else np.asarray(ydot, dtype=float)
    if ydot_array is not None:
        if ydot_array.ndim != 2:
            raise RuntimeError(
                f"IDA returned ydot with shape {ydot_array.shape}, expected a 2-D array"
            )
        if ydot_array.shape[0] != len(t) and ydot_array.shape[1] == len(t):
            ydot_array = ydot_array.T
    return success, t, y, ydot_array, status, message


def _trajectory_algebraic_residuals(
    system: _ResidualSystem,
    t: np.ndarray,
    y: np.ndarray,
    ydot: np.ndarray | None,
) -> dict[str, float]:
    residuals = {
        equation.name: []
        for equation in system.equations
        if equation.kind != "differential"
    }
    if not residuals:
        return {}

    if ydot is None:
        ydot = np.zeros_like(y)
        for row_idx, row in enumerate(y):
            values = _context_from_state(system.variable_names, system.params, t[row_idx], row)
            for equation in system.equations:
                if equation.kind == "differential" and equation.variable_index is not None:
                    ydot[row_idx, equation.variable_index] = (
                        0.0 if equation.expr is None else equation.expr.evaluate(values)
                    )

    for row_idx, row in enumerate(y):
        values = _residual_values(system, float(t[row_idx]), row, ydot[row_idx])
        for eq_idx, equation in enumerate(system.equations):
            if equation.kind != "differential":
                residuals[equation.name].append(float(abs(values[eq_idx])))

    return {
        name: float(np.max(values)) if values else 0.0
        for name, values in residuals.items()
    }


def simulate_with_ida_sundials(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a DAE system with IDA from SUNDIALS via scikit-SUNDAE."""
    options = options or {}
    rtol = float(options.get("relative_tolerance", 1.0e-6))
    atol = float(options.get("absolute_tolerance", 1.0e-9))

    try:
        binding = _load_ida_binding()
    except IDASundialsUnavailable as exc:
        return _failure(str(exc), unsupported=True)

    try:
        system = _build_residual_system(model_ir, y0_override, options)
    except UnsupportedDAESystem as exc:
        return _failure(str(exc), unsupported=True)
    except InconsistentInitialCondition as exc:
        return _failure(
            f"IDA/SUNDIALS initial-condition setup failed: {exc}",
            unsupported=False,
            initial_residual_norms=exc.residuals,
        )
    except Exception as exc:
        return _failure(
            f"IDA/SUNDIALS initial-condition setup failed: {exc}",
            unsupported=False,
        )

    t_eval = np.linspace(float(t0), float(t_end), int(n_points))

    residual = _make_sksundae_residual(system)
    solver_options = {
        "rtol": rtol,
        "atol": atol,
    }
    if system.algebraic_indices:
        solver_options["algebraic_idx"] = np.asarray(system.algebraic_indices, dtype=int)

    integrator_stats: dict[str, Any] = {
        "package": binding.package,
        "package_version": binding.version,
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "algebraic_indices": list(system.algebraic_indices),
        "variable_names": list(system.variable_names),
        "initial_residual_norms": dict(system.initial_residual_norms),
        "initial_condition_repaired": system.initial_condition_repaired,
    }

    try:
        solver = binding.solver_class(residual, **solver_options)
        solution = solver.solve(t_eval, system.y0, system.ydot0)
        success, t, y, ydot, status, message = _extract_solution_arrays(solution)
        integrator_stats["return_status"] = status
        integrator_stats["solver_message"] = message

        if not success:
            return _failure(
                f"IDA/SUNDIALS solver failed: {message or status}",
                unsupported=False,
                integrator_stats=integrator_stats,
                initial_residual_norms=system.initial_residual_norms,
            )

        algebraic_residuals = _trajectory_algebraic_residuals(system, t, y, ydot)
        integrator_stats["algebraic_residual_norms"] = algebraic_residuals

        return {
            "t": t,
            "y": y,
            "state_names": list(system.variable_names),
            "success": True,
            "message": "",
            "backend": "ida_sundials",
            "solver_requirement": SolverRequirement.DAE_REQUIRED.value,
            "unsupported_solver_requirement": False,
            "integrator_stats": integrator_stats,
            "algebraic_residuals": algebraic_residuals,
            "initial_residual_norms": dict(system.initial_residual_norms),
        }
    except Exception as exc:
        integrator_stats["solver_diagnostics"] = str(exc)
        return _failure(
            f"IDA/SUNDIALS solver crashed: {exc}",
            unsupported=False,
            integrator_stats=integrator_stats,
            initial_residual_norms=system.initial_residual_norms,
        )
