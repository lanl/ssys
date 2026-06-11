"""LaTeX formatting helpers for ODE and S-system output."""

import sympy as sp

from ssys.math_utils import _expand_exps_through_factors
from ssys.types import RecastResult, SymSystem


def latex_odes(sym: "SymSystem") -> str:
    lines = []
    for v in sorted(sym.odes.keys(), key=lambda s: s.name):
        rhs = sp.simplify(sym.odes[v])
        lines.append(rf"\dot{{{sp.latex(v)}}} = {sp.latex(rhs)}")
    return r"\\begin{aligned}" + r"\\\\\n".join(lines) + r"\\end{aligned}"


def _latex_power_law(coeff, exps: dict[sp.Symbol, float]) -> str:
    """
    Format a power-law term as clean LaTeX.
    - Skip coefficients of 1
    - Skip exponents of 1
    - Display integers as integers (not floats)
    """
    parts = []

    # Handle coefficient
    if isinstance(coeff, sp.Expr):
        coeff_simplified = sp.simplify(coeff)
        if coeff_simplified == 0:
            return "0"
        elif coeff_simplified != 1:
            # Use sympy's latex for symbolic coefficients
            parts.append(sp.latex(coeff_simplified))
    else:
        # Numeric coefficient
        if coeff == 0:
            return "0"
        elif coeff != 1:
            # Display integers as integers
            if isinstance(coeff, int) or (isinstance(coeff, float) and coeff == int(coeff)):
                parts.append(str(int(coeff)))
            else:
                parts.append(f"{coeff:g}")

    # Handle power-law terms
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        if abs(e) < 1e-14:
            continue

        var_latex = sp.latex(s)

        # Skip exponent if it's 1
        if abs(e - 1.0) < 1e-14:
            parts.append(var_latex)
        else:
            # Display integer exponents as integers
            if isinstance(e, int) or (isinstance(e, float) and e == int(e)):
                parts.append(f"{var_latex}^{{{int(e)}}}")
            else:
                parts.append(f"{var_latex}^{{{e:g}}}")

    if not parts:
        return "0"

    # Join with space (LaTeX handles multiplication)
    return " ".join(parts)


def latex_ssys(result: "RecastResult") -> str:
    """
    Generate clean LaTeX representation of S-system equations.
    - Skip coefficients of 1
    - Skip exponents of 1
    - Display integers as integers
    - Apply slack variable transformation for canonical form (matches Antimony output)
    """
    lines = []
    for eq in result.equations:
        # Expand exponents through factor map
        g_exps = _expand_exps_through_factors(eq.growth[1], result.factor_map)
        h_exps = _expand_exps_through_factors(eq.decay[1], result.factor_map)
        g_coeff = eq.growth[0]
        h_coeff = eq.decay[0]

        # Check if growth or decay is zero (need slack variable transformation)
        g_is_zero = (isinstance(g_coeff, (int, float)) and g_coeff == 0) or (
            isinstance(g_coeff, sp.Expr) and g_coeff == sp.Integer(0)
        )
        h_is_zero = (isinstance(h_coeff, (int, float)) and h_coeff == 0) or (
            isinstance(h_coeff, sp.Expr) and h_coeff == sp.Integer(0)
        )

        if g_is_zero and not h_is_zero:
            # Pure decay: X' = 0 - h  =>  X' = epsilon*monomial - (epsilon + h)*monomial
            # Special case: pure constant decay (empty exponents)
            if not h_exps:
                if isinstance(h_coeff, sp.Expr):
                    combined = sp.Symbol("epsilon") + h_coeff
                else:
                    combined = sp.Symbol("epsilon") + sp.Float(h_coeff)
                g_latex = r"\epsilon"
                h_latex = sp.latex(combined)
            else:
                g_latex = _latex_power_law(sp.Symbol("epsilon"), h_exps)
                if isinstance(h_coeff, sp.Expr):
                    combined_coeff = sp.Symbol("epsilon") + h_coeff
                else:
                    combined_coeff = sp.Symbol("epsilon") + sp.Float(h_coeff)
                h_latex = _latex_power_law(combined_coeff, h_exps)
        elif h_is_zero and not g_is_zero:
            # Pure growth: X' = g - 0  =>  X' = (g + epsilon)*monomial - epsilon*monomial
            # Special case: pure constant growth (empty exponents)
            if not g_exps:
                if isinstance(g_coeff, sp.Expr):
                    combined = g_coeff + sp.Symbol("epsilon")
                else:
                    combined = sp.Float(g_coeff) + sp.Symbol("epsilon")
                g_latex = sp.latex(combined)
                h_latex = r"\epsilon"
            else:
                if isinstance(g_coeff, sp.Expr):
                    combined_coeff = g_coeff + sp.Symbol("epsilon")
                else:
                    combined_coeff = sp.Float(g_coeff) + sp.Symbol("epsilon")
                g_latex = _latex_power_law(combined_coeff, g_exps)
                h_latex = _latex_power_law(sp.Symbol("epsilon"), g_exps)
        else:
            # Both terms present (or both zero) - use as-is
            g_latex = _latex_power_law(g_coeff, g_exps)
            h_latex = _latex_power_law(h_coeff, h_exps)

        # Build equation
        var_latex = sp.latex(eq.var)
        lines.append(rf"\dot{{{var_latex}}} &= {g_latex} - {h_latex}")

    return r"\begin{aligned}" + "\\\\\n".join(lines) + r"\end{aligned}"
