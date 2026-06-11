"""System and solver classification shared by core code and notebook helpers."""

import re

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


def _assignment_rule_substitutions(sym) -> dict[sp.Symbol, sp.Expr]:
    rule_subs = {}
    if not sym.assignment_rules:
        return rule_subs

    all_syms = {s.name: s for s in sym.vars}
    for param_name in sym.params:
        all_syms[param_name] = sp.Symbol(param_name, positive=True)
    all_syms["time"] = sp.Symbol("time", positive=True)
    for rule_name in sym.assignment_rules:
        if rule_name not in all_syms:
            all_syms[rule_name] = sp.Symbol(rule_name, positive=True)

    for rule_name, rule_str in sym.assignment_rules.items():
        try:
            rule_expr = sp.sympify(rule_str, locals=all_syms)
            rule_subs[all_syms[rule_name]] = rule_expr
        except Exception:
            pass

    for _ in range(10):
        changed = False
        for rule_sym, rule_expr in list(rule_subs.items()):
            new_expr = rule_expr.subs(rule_subs)
            if new_expr != rule_expr:
                rule_subs[rule_sym] = new_expr
                changed = True
        if not changed:
            break

    return rule_subs


def classify_system(sym) -> SystemClass:
    """
    Classify a symbolic system using expanded assignment-rule semantics.

    Assignment rules are expanded before classification so hidden rational
    functions, time dependence, and symbolic exponents are interpreted the same
    way across the core package and notebooks.
    """
    rule_subs = _assignment_rule_substitutions(sym)
    name_to_expr = {rule_sym.name: expr for rule_sym, expr in rule_subs.items()}

    is_canonical = True
    is_ssystem = True
    is_gma = True

    for _var, ode in sym.odes.items():
        if rule_subs:
            ode = ode.subs(rule_subs)
            for sym_in_ode in ode.free_symbols:
                if sym_in_ode.name in name_to_expr:
                    ode = ode.subs(sym_in_ode, name_to_expr[sym_in_ode.name])
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
