# mypy: ignore-errors
# ruff: noqa: F401,I001
"""Shared imports and expression helpers for validation internals."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp
from sympy import Matrix, lambdify

from ssys.classification import (
    classify_sym_system_solver_requirement,
    classify_system,
)
from ssys.parsing import (
    build_sym_system,
    parse_antimony,
    parse_antimony_via_sbml,
)
from ssys.types import SolverRequirement, SystemClass

def _canonicalize_expr_by_name(expr: sp.Expr, symbols_by_name: dict[str, sp.Symbol]) -> sp.Expr:
    """Return an expression whose symbols are the canonical objects for each name."""
    if not hasattr(expr, "free_symbols"):
        expr = sp.sympify(expr)
    if not expr.free_symbols:
        expr_str = str(expr)
        if expr_str in symbols_by_name:
            return symbols_by_name[expr_str]
        return expr
    subs = {sym: symbols_by_name[sym.name] for sym in expr.free_symbols if sym.name in symbols_by_name}
    return expr.subs(subs) if subs else expr


def _substitute_symbols_by_name(
    expr: sp.Expr, replacements_by_name: dict[str, sp.Symbol | sp.Expr]
) -> sp.Expr:
    """Substitute expression symbols by matching symbol names, not object identity."""
    subs = {sym: replacements_by_name[sym.name] for sym in expr.free_symbols if sym.name in replacements_by_name}
    return expr.subs(subs) if subs else expr


def _simplify_identity_difference(expr: sp.Expr) -> sp.Expr:
    """Apply the validator's common simplification strategy to an identity difference."""
    try:
        expr = sp.nsimplify(expr, rational=True, tolerance=1e-10)
    except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
        pass

    candidates = [
        sp.simplify,
        lambda e: sp.cancel(sp.expand(e)),
        sp.ratsimp,
        lambda e: sp.simplify(sp.factor(e)),
        lambda e: sp.simplify(sp.expand(e)),
    ]

    current = expr
    for simplify_func in candidates:
        if current == 0:
            return sp.Integer(0)
        try:
            current = simplify_func(current)
        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
            continue

    if current != 0:
        try:
            numer, _denom = current.as_numer_denom()
            if sp.simplify(sp.expand(numer)) == 0:
                return sp.Integer(0)
        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
            pass

    return current

def _is_dev_mode() -> bool:
    """
    Detect if we're running in development mode.

    Development mode is indicated by having pytest installed (from [dev] extras).
    In dev mode, we run both JAX and non-JAX numerical tests for debugging.
    In production mode, we run JAX if available, else non-JAX.
    """
    try:
        import pytest

        return True
    except ImportError:
        return False

__all__ = [name for name in globals() if not name.startswith("__")]
