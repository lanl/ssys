"""System and solver classification shared by core code and notebook helpers."""

import re
from collections.abc import Callable

import sympy as sp

from ssys.math_utils import (
    _get_coefficient_sign,
    _is_term_monomial,
    expand_to_terms,
    exponents_match,
)
from ssys.metadata import normalize_solver_requirement
from ssys.types import RecastStatus, SolverRequirement, SystemClass


def _is_clock_definition(defn: sp.Expr) -> bool:
    return defn == sp.Symbol("time") or str(defn).lower() in {"time", "t"}


def _constraint_mentions_state(constraint: str | sp.Expr, state_names: set[str]) -> bool:
    """Return true if an algebraic constraint is coupled to differential states."""
    if not state_names:
        return False
    try:
        expr = sp.sympify(str(constraint).replace("^", "**"))
    except Exception:
        return True
    return any(sym.name in state_names for sym in expr.free_symbols)


def _has_non_identity_mapping(result) -> bool:
    for orig, aux_list in result.factor_map.items():
        if len(aux_list) != 1 or aux_list[0] != orig:
            return True
    return False


def classify_solver_requirement(result, *, lifted_mode: str = "ode") -> SolverRequirement:
    """
    Classify the backend requirement for a generated recast output.

    `ode_with_assignment_rules` means an ODE integrator that honors explicit
    assignment rules is sufficient. `dae_required` is reserved for algebraic
    constraints coupled to differential states, such as ODE-mode lifted
    auxiliaries that must remain on a defining manifold.
    """
    state_names = {var.name for var in result.variables}
    for eq in result.equations:
        state_names.add(eq.var.name)
    for eq in result.gma_equations:
        state_names.add(eq.var.name)

    if result.algebraic_constraints and any(
        _constraint_mentions_state(constraint, state_names)
        for constraint in result.algebraic_constraints
    ):
        return SolverRequirement.DAE_REQUIRED

    if result.assignment_rules and any(name in state_names for name in result.assignment_rules):
        return SolverRequirement.DAE_REQUIRED

    lifted_aux_defs = {
        aux.name: defn
        for aux, defn in result.auxiliary_defs.items()
        if aux.name != "dummy_const" and not _is_clock_definition(defn)
    }
    if lifted_aux_defs:
        if lifted_mode == "assignment":
            return SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
        for aux_name, defn in lifted_aux_defs.items():
            if aux_name in state_names and any(sym.name in state_names for sym in defn.free_symbols):
                return SolverRequirement.DAE_REQUIRED

    if result.assignment_rules or _has_non_identity_mapping(result):
        return SolverRequirement.ODE_WITH_ASSIGNMENT_RULES

    return SolverRequirement.ODE_ONLY


def classify_sym_system_solver_requirement(sym) -> SolverRequirement:
    """Classify the solver requirement for an already parsed symbolic model."""
    configured = normalize_solver_requirement(getattr(sym, "solver_requirement", None))
    if configured == SolverRequirement.DAE_REQUIRED:
        return configured

    state_names = {var.name for var in sym.vars}
    constraints = getattr(sym, "algebraic_constraints", [])
    if constraints and any(_constraint_mentions_state(constraint, state_names) for constraint in constraints):
        return SolverRequirement.DAE_REQUIRED

    if sym.assignment_rules:
        if any(name in state_names for name in sym.assignment_rules):
            return SolverRequirement.DAE_REQUIRED
        return SolverRequirement.ODE_WITH_ASSIGNMENT_RULES

    return configured or SolverRequirement.ODE_ONLY


def _assignment_rule_parse_locals(sym) -> dict[str, sp.Symbol]:
    all_syms = {s.name: s for s in sym.vars}
    for param_name in sym.params:
        all_syms[param_name] = sp.Symbol(param_name, positive=True)
    all_syms["time"] = sp.Symbol("time", positive=True)
    for rule_name in sym.assignment_rules:
        if rule_name not in all_syms:
            all_syms[rule_name] = sp.Symbol(rule_name, positive=True)
    return all_syms


def _parse_assignment_rule_by_name(
    rule_name: str,
    assignment_rules: dict[str, str],
    all_syms: dict[str, sp.Symbol],
    parse_cache: dict[str, sp.Expr | None],
) -> sp.Expr | None:
    if rule_name in parse_cache:
        return parse_cache[rule_name]
    try:
        parse_cache[rule_name] = sp.sympify(
            assignment_rules[rule_name],
            locals=all_syms,
        )
    except Exception:
        parse_cache[rule_name] = None
    return parse_cache[rule_name]


def _expand_assignment_rule_by_name(
    rule_name: str,
    assignment_rules: dict[str, str],
    all_syms: dict[str, sp.Symbol],
    parse_cache: dict[str, sp.Expr | None],
    expanded_cache: dict[str, sp.Expr | None],
    visiting: set[str],
) -> sp.Expr | None:
    if rule_name in expanded_cache:
        return expanded_cache[rule_name]

    rule_expr = _parse_assignment_rule_by_name(
        rule_name,
        assignment_rules,
        all_syms,
        parse_cache,
    )
    if rule_expr is None:
        expanded_cache[rule_name] = None
        return None

    visiting.add(rule_name)
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for free_symbol in rule_expr.free_symbols:
        dependency_name = free_symbol.name
        if dependency_name not in assignment_rules or dependency_name in visiting:
            continue
        dependency_expr = _expand_assignment_rule_by_name(
            dependency_name,
            assignment_rules,
            all_syms,
            parse_cache,
            expanded_cache,
            visiting,
        )
        if dependency_expr is not None:
            substitutions[free_symbol] = dependency_expr
    visiting.remove(rule_name)

    expanded_expr = rule_expr.xreplace(substitutions) if substitutions else rule_expr
    expanded_cache[rule_name] = expanded_expr
    return expanded_expr


def _expand_assignment_rules_in_expr(
    expr: sp.Expr,
    assignment_rules: dict[str, str],
    all_syms: dict[str, sp.Symbol],
    parse_cache: dict[str, sp.Expr | None],
    expanded_cache: dict[str, sp.Expr | None],
) -> sp.Expr:
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for free_symbol in expr.free_symbols:
        rule_name = free_symbol.name
        if rule_name not in assignment_rules:
            continue
        rule_expr = _expand_assignment_rule_by_name(
            rule_name,
            assignment_rules,
            all_syms,
            parse_cache,
            expanded_cache,
            set(),
        )
        if rule_expr is not None:
            substitutions[free_symbol] = rule_expr
    return expr.xreplace(substitutions) if substitutions else expr


def _assignment_rule_substitutions(sym) -> dict[sp.Symbol, sp.Expr]:
    rule_subs: dict[sp.Symbol, sp.Expr] = {}
    if not sym.assignment_rules:
        return rule_subs

    all_syms = _assignment_rule_parse_locals(sym)
    parse_cache: dict[str, sp.Expr | None] = {}
    expanded_cache: dict[str, sp.Expr | None] = {}
    for rule_name in sym.assignment_rules:
        rule_expr = _expand_assignment_rule_by_name(
            rule_name,
            sym.assignment_rules,
            all_syms,
            parse_cache,
            expanded_cache,
            set(),
        )
        if rule_expr is not None:
            rule_subs[all_syms[rule_name]] = rule_expr

    return rule_subs


def _notify_progress(
    progress_callback: Callable[[str], None] | None,
    phase: str,
) -> None:
    """Best-effort progress hook for classification subphases."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase)
    except Exception:
        return


def _has_non_monomial_denominator(expr: sp.Expr) -> bool:
    """Return True when a rational expression cannot be monomial after expansion."""
    _numerator, denominator = expr.as_numer_denom()
    return denominator != 1 and not _is_term_monomial(denominator)


def classify_system(
    sym,
    *,
    progress_callback: Callable[[str], None] | None = None,
    progress_prefix: str = "classification",
) -> SystemClass:
    """
    Classify a symbolic system using expanded assignment-rule semantics.

    Assignment rules are expanded before classification so hidden rational
    functions, time dependence, and symbolic exponents are interpreted the same
    way across the core package and notebooks.
    """
    _notify_progress(progress_callback, f"{progress_prefix}:assignment_substitutions")
    assignment_rules = sym.assignment_rules
    assignment_rule_locals = (
        _assignment_rule_parse_locals(sym) if assignment_rules else {}
    )
    assignment_rule_parse_cache: dict[str, sp.Expr | None] = {}
    assignment_rule_expanded_cache: dict[str, sp.Expr | None] = {}

    is_canonical = True
    is_ssystem = True
    is_gma = True

    for _var, ode in sym.odes.items():
        var_name = str(_var)
        if assignment_rules:
            _notify_progress(progress_callback, f"{progress_prefix}:substitute:{var_name}")
            ode = _expand_assignment_rules_in_expr(
                ode,
                assignment_rules,
                assignment_rule_locals,
                assignment_rule_parse_cache,
                assignment_rule_expanded_cache,
            )

        _notify_progress(progress_callback, f"{progress_prefix}:denominator:{var_name}")
        if _has_non_monomial_denominator(ode):
            return SystemClass.GENERAL

        _notify_progress(progress_callback, f"{progress_prefix}:expand:{var_name}")
        terms = expand_to_terms(sp.expand(ode))

        pos_monomials = []
        neg_monomials = []

        for term in terms:
            if term == 0:
                continue
            if not _is_term_monomial(term):
                return SystemClass.GENERAL
            sign = _get_coefficient_sign(term)
            if sign > 0:
                pos_monomials.append(term)
            else:
                neg_monomials.append(term)

        if len(pos_monomials) != 1 or len(neg_monomials) != 1:
            is_canonical = False

        if len(pos_monomials) > 1 or len(neg_monomials) > 1:
            is_ssystem = False

    if is_canonical:
        return SystemClass.CANONICAL_SSYSTEM
    if is_ssystem:
        return SystemClass.SSYSTEM
    if is_gma:
        return SystemClass.GMA
    return SystemClass.GENERAL


def classify_result(result, mode: str = "simplified") -> SystemClass:
    """Classify a RecastResult based on its output structure."""
    has_time_varying_coeffs = False
    if result.assignment_rules:
        for _rule_name, rule_expr in result.assignment_rules.items():
            if re.search(r"\bT\b", str(rule_expr)):
                has_time_varying_coeffs = True
                break

    if has_time_varying_coeffs:
        return SystemClass.GMA_TIME_VARYING

    if result.status == RecastStatus.GMA:
        has_multi_term = False
        for eq in result.gma_equations:
            if len(eq.production) > 1:
                first_exps = eq.production[0][1]
                for _, exps in eq.production[1:]:
                    if not exponents_match(first_exps, exps):
                        has_multi_term = True
                        break

            if len(eq.degradation) > 1:
                first_exps = eq.degradation[0][1]
                for _, exps in eq.degradation[1:]:
                    if not exponents_match(first_exps, exps):
                        has_multi_term = True
                        break

            if has_multi_term:
                break

        if has_multi_term:
            return SystemClass.GMA

        for gma_eq in result.gma_equations:
            if len(gma_eq.production) != 1 or len(gma_eq.degradation) != 1:
                return SystemClass.GMA
        return SystemClass.CANONICAL_SSYSTEM

    if result.status == RecastStatus.CANONICAL_SSYSTEM:
        is_canonical = True
        is_ssystem = True

        for ssys_eq in result.equations:
            g_coeff = ssys_eq.growth[0]
            d_coeff = ssys_eq.decay[0]

            if isinstance(g_coeff, (int, float)):
                g_nonzero = g_coeff != 0
            elif isinstance(g_coeff, sp.Expr):
                g_nonzero = g_coeff != sp.Integer(0)
            else:
                g_nonzero = False

            if isinstance(d_coeff, (int, float)):
                d_nonzero = d_coeff != 0
            elif isinstance(d_coeff, sp.Expr):
                d_nonzero = d_coeff != sp.Integer(0)
            else:
                d_nonzero = False

            if mode == "canonical":
                has_zero = (not g_nonzero) or (not d_nonzero)
                has_nonzero = g_nonzero or d_nonzero
                if has_zero and has_nonzero:
                    nonzero_count = 2
                else:
                    nonzero_count = (1 if g_nonzero else 0) + (1 if d_nonzero else 0)
            else:
                nonzero_count = (1 if g_nonzero else 0) + (1 if d_nonzero else 0)

            if nonzero_count != 2:
                is_canonical = False
            if nonzero_count < 1 or nonzero_count > 2:
                is_ssystem = False

        if is_canonical:
            return SystemClass.CANONICAL_SSYSTEM
        if is_ssystem:
            return SystemClass.SSYSTEM
        return SystemClass.GMA

    return SystemClass.GENERAL


__all__ = [
    "_constraint_mentions_state",
    "_is_clock_definition",
    "classify_result",
    "classify_solver_requirement",
    "classify_sym_system_solver_requirement",
    "classify_system",
]
