"""Typed state shared by validation mixins."""

from typing import Any

import sympy as sp

from ssys.types import SolverRequirement, SymSystem


class ValidatorState:
    """Attribute contract populated by RecastValidator before mixin checks run."""

    orig_ir: Any
    recast_ir: Any
    orig_system: SymSystem
    recast_system: SymSystem
    orig_odes: dict[sp.Symbol, sp.Expr]
    orig_odes_expanded: dict[sp.Symbol, sp.Expr]
    recast_odes: dict[sp.Symbol, sp.Expr]
    recast_odes_expanded: dict[sp.Symbol, sp.Expr]
    recast_state_vars: list[sp.Symbol]
    mapping: dict[sp.Symbol, sp.Expr]
    factor_map: dict[sp.Symbol, Any]
    auxiliary_defs: dict[sp.Symbol, sp.Expr]
    canonical_symbols: dict[str, sp.Symbol]
    orig_solver_requirement: SolverRequirement
    recast_solver_requirement: SolverRequirement

    def _canonical_expr(self, expr: sp.Expr) -> sp.Expr:
        raise NotImplementedError

    def _is_clock_definition(self, expr: sp.Expr) -> bool:
        raise NotImplementedError

    def _parse_expr_with_canonical_symbols(self, expr: str | sp.Expr) -> sp.Expr:
        raise NotImplementedError
