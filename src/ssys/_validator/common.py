"""Shared imports and expression helpers for validation internals."""

from importlib.util import find_spec

import sympy as sp


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


def _is_near_zero_float_residual(expr: sp.Expr, *, atol: float = 1e-9) -> bool:
    """Return True when a residual only contains tiny floating coefficients."""
    terms = sp.Add.make_args(expr)
    saw_float = False
    for term in terms:
        coeff, _rest = term.as_coeff_Mul()
        if not coeff.is_number:
            return False
        if coeff.atoms(sp.Float):
            saw_float = True
        try:
            magnitude = abs(float(coeff.evalf()))
        except (TypeError, ValueError, OverflowError):
            return False
        if magnitude > atol:
            return False
    return saw_float


def _cheap_zero_simplification(expr: sp.Expr) -> sp.Expr | None:
    """Try bounded identity checks before general SymPy simplification."""
    if sp.count_ops(expr) > 200:
        return sp.Integer(0) if expr == 0 else None

    candidates = [expr]
    for transform in (
        sp.expand_mul,
        lambda e: sp.cancel(sp.expand_mul(e)),
        sp.together,
        sp.factor_terms,
    ):
        try:
            candidates.append(transform(expr))
        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
            continue

    for candidate in candidates:
        if candidate == 0 or _is_near_zero_float_residual(candidate):
            return sp.Integer(0)
        try:
            numer, _denom = sp.together(candidate).as_numer_denom()
            if _is_near_zero_float_residual(sp.expand_mul(numer)):
                return sp.Integer(0)
        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
            continue
    return None


def _simplify_identity_difference(expr: sp.Expr) -> sp.Expr:
    """Apply the validator's common simplification strategy to an identity difference."""
    cheap = _cheap_zero_simplification(expr)
    if cheap == 0:
        return cheap

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
        cheap = _cheap_zero_simplification(current)
        if cheap == 0:
            return cheap

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
    return find_spec("pytest") is not None

__all__ = [
    "_canonicalize_expr_by_name",
    "_is_dev_mode",
    "_simplify_identity_difference",
    "_substitute_symbols_by_name",
]
