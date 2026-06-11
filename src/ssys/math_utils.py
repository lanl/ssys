"""Shared symbolic math utilities used by core code and notebook helpers."""

import sympy as sp


def expand_to_terms(expr: sp.Expr) -> list[sp.Expr]:
    """Expand an expression and return its additive terms."""
    expr = sp.expand(expr)
    if expr.is_Add:
        return list(expr.args)
    return [expr]


def expand_exps_through_factors(exps, factor_map):
    """Expand exponent maps through a variable-to-factor map."""
    new: dict[sp.Symbol, sp.Expr] = {}
    for symbol, exponent in exps.items():
        if symbol in factor_map:
            for factor in factor_map[symbol]:
                new[factor] = new.get(factor, sp.sympify(0)) + exponent
        else:
            new[symbol] = new.get(symbol, sp.sympify(0)) + exponent
    return new


def _expand_exps_through_factors(exps, factor_map):
    """Backward-compatible alias for the shared exponent expansion utility."""
    return expand_exps_through_factors(exps, factor_map)


def _coerce_exponent(exponent):
    if isinstance(exponent, int) or (
        isinstance(exponent, float) and exponent == int(exponent)
    ):
        return sp.Integer(int(exponent))
    if isinstance(exponent, float):
        return sp.Float(exponent)
    return sp.simplify(exponent)


def product_expr(coeff, exps) -> sp.Expr:
    """Build a symbolic product expression from a coefficient and exponent map."""
    if isinstance(coeff, sp.Expr):
        expr = coeff
    elif isinstance(coeff, int) or (isinstance(coeff, float) and coeff == int(coeff)):
        expr = sp.Integer(int(coeff))
    else:
        expr = sp.Float(coeff)

    for symbol, exponent in sorted(exps.items(), key=lambda kv: str(kv[0])):
        exp_sym = _coerce_exponent(exponent)
        if exp_sym == 0:
            continue
        if exp_sym.is_number:
            try:
                if abs(float(exp_sym)) < 1e-14:
                    continue
            except (TypeError, ValueError):
                pass
        expr *= symbol**exp_sym

    return sp.simplify(expr)


def _is_term_monomial(term: sp.Expr) -> bool:
    """
    Return true if a term is a monomial product of numbers, symbols, and symbol powers.

    Symbolic exponents are allowed because GMA/S-system exponents may be symbolic
    parameters. Non-symbol bases such as sums or functions are not monomial terms.
    """
    if term.is_Number:
        return True
    if isinstance(term, sp.Symbol):
        return True
    if isinstance(term, sp.Pow):
        base, _exp = term.args
        return isinstance(base, sp.Symbol)
    if term.is_Mul:
        return all(_is_term_monomial(factor) for factor in term.args)
    return False


def _get_coefficient_sign(term: sp.Expr) -> int:
    """Get the sign of a term's numeric coefficient."""
    if term.is_Number:
        return 1 if float(term) >= 0 else -1
    if term.is_Mul:
        coeff = 1.0
        for factor in term.args:
            if factor.is_Number:
                coeff *= float(factor)
        return 1 if coeff >= 0 else -1
    return 1


def exponents_match(exps1: dict[sp.Symbol, float], exps2: dict[sp.Symbol, float]) -> bool:
    """Check if two exponent patterns match within tolerance."""
    all_vars = set(exps1.keys()) | set(exps2.keys())
    for var in all_vars:
        e1 = exps1.get(var, 0.0)
        e2 = exps2.get(var, 0.0)

        if isinstance(e1, sp.Expr) or isinstance(e2, sp.Expr):
            diff = sp.simplify(sp.sympify(e1) - sp.sympify(e2))
            if diff == 0:
                continue
            if diff.is_number:
                try:
                    if abs(float(diff)) > 1e-10:
                        return False
                except (TypeError, ValueError):
                    return False
            else:
                return False
        else:
            try:
                if abs(float(e1) - float(e2)) > 1e-10:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _exponents_match(exps1: dict[sp.Symbol, float], exps2: dict[sp.Symbol, float]) -> bool:
    """Backward-compatible alias for the shared exponent-pattern comparison."""
    return exponents_match(exps1, exps2)


__all__ = [
    "_expand_exps_through_factors",
    "_exponents_match",
    "_get_coefficient_sign",
    "_is_term_monomial",
    "expand_exps_through_factors",
    "expand_to_terms",
    "exponents_match",
    "product_expr",
]
