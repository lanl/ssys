"""Helper functions for Jupyter notebook verification."""

import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Code

import ssys
from ssys.recaster import RecastStatus, SystemClass, classify_system, classify_result

FILL_MISSING_PARAMS = False  # set True to auto-fill absent params with 1.0


def rk4(f, t_span, y0, n_steps):
    """Simple RK4 integrator."""
    t0, t1 = t_span
    t = np.linspace(t0, t1, n_steps+1)
    h = (t1 - t0) / n_steps
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = np.array(y0, dtype=float)
    for i in range(n_steps):
        ti = t[i]
        yi = y[i]
        k1 = f(ti, yi)
        k2 = f(ti + 0.5*h, yi + 0.5*h*k1)
        k3 = f(ti + 0.5*h, yi + 0.5*h*k2)
        k4 = f(ti + h, yi + h*k3)
        y[i+1] = yi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return t, y


def build_rhs_from_sympy(vars_syms, rhs_exprs, param_vals):
    """Build numerical RHS function from symbolic expressions."""
    var_set = set(vars_syms)
    rhs_free = set().union(*[expr.free_symbols for expr in rhs_exprs])

    param_subs = {}
    missing = []
    for s in sorted(rhs_free - var_set, key=lambda z: z.name):
        name = s.name
        if name in param_vals:
            param_subs[s] = float(param_vals[name])
        else:
            missing.append(name)

    if missing and not FILL_MISSING_PARAMS:
        raise ValueError("Missing numeric values for parameters: " + ", ".join(sorted(set(missing))))
    if missing and FILL_MISSING_PARAMS:
        for sname in set(missing):
            sym = next(ss for ss in rhs_free if ss.name == sname and ss not in var_set)
            param_subs[sym] = 1.0
        display(Markdown("> **Warning**: filled missing params with 1.0 → " + ", ".join(sorted(set(missing)))))

    def f(t, y):
        subs = {s: float(y[i]) for i, s in enumerate(vars_syms)}
        subs.update(param_subs)
        vals = [float(expr.evalf(subs=subs)) for expr in rhs_exprs]
        return np.array(vals, dtype=float)
    return f


def _expand_exps_through_factors(exps, factor_map):
    """Expand exponents through factor map. Handles both numeric and symbolic exponents."""
    new = {}
    for s, e in exps.items():
        if s in factor_map:
            for v in factor_map[s]:
                if v in new:
                    new[v] = new[v] + e
                else:
                    new[v] = e
        else:
            if s in new:
                new[s] = new[s] + e
            else:
                new[s] = e
    return new


def product_expr(coeff, exps):
    """Build symbolic product expression from coefficient and exponents."""
    if isinstance(coeff, sp.Expr):
        expr = coeff
    else:
        if isinstance(coeff, int) or (isinstance(coeff, float) and coeff == int(coeff)):
            expr = sp.Integer(int(coeff))
        else:
            expr = sp.Float(coeff)
    
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        if isinstance(e, (int, float)):
            if abs(e) < 1e-14:
                continue
            if isinstance(e, int) or (isinstance(e, float) and e == int(e)):
                exp_sym = sp.Integer(int(e))
            else:
                exp_sym = sp.Float(e)
        else:
            exp_sym = sp.simplify(e)
            if exp_sym == 0:
                continue
        expr *= s**exp_sym
    
    return sp.simplify(expr)


def latex_odes_from_sym(sym):
    """Generate LaTeX for original ODEs."""
    lines = []
    for v in sorted(sym.odes.keys(), key=lambda s: s.name):
        rhs = sp.simplify(sym.odes[v])
        lines.append(r"\dot{%s} &= %s" % (sp.latex(v), sp.latex(rhs)))
    return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"


def parse_antimony_odes(antimony_text):
    """
    Parse ODE equations from Antimony text.
    
    Returns: List of (var_name, rhs_expr_string) tuples
    """
    import re
    odes = []
    # Match lines like: X_1' = ... or X' = ...
    ode_pattern = re.compile(r"^\s*([A-Za-z_]\w*)'\s*=\s*(.+?)\s*;?\s*$")
    
    for line in antimony_text.splitlines():
        # Remove comments
        line = line.split("//")[0].strip()
        if not line:
            continue
        
        match = ode_pattern.match(line)
        if match:
            var_name = match.group(1)
            rhs = match.group(2).rstrip(';').strip()
            odes.append((var_name, rhs))
    
    return odes


def _beautify_latex(latex_str):
    """
    Apply beautification rules to LaTeX string.
    Single source of truth: the Antimony file.
    Only cosmetic improvements, no restructuring.
    
    Rules applied in order:
    1. Greek letter conversion (alpha, beta, gamma, mu, etc.)
    2. epsilon conversion (eps → ε)
    3. Subscripting (k1 → k_1, but not Greek letters)
    4. Variable reordering (constants before variables)
    """
    import re
    
    # 1. Convert Greek letter names to LaTeX symbols
    # Must do this BEFORE subscripting to avoid mangling
    greek_letters = {
        'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
        'delta': r'\delta', 'epsilon': r'\varepsilon', 'zeta': r'\zeta',
        'eta': r'\eta', 'theta': r'\theta', 'iota': r'\iota',
        'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
        'nu': r'\nu', 'xi': r'\xi', 'pi': r'\pi',
        'rho': r'\rho', 'sigma': r'\sigma', 'tau': r'\tau',
        'upsilon': r'\upsilon', 'phi': r'\phi', 'chi': r'\chi',
        'psi': r'\psi', 'omega': r'\omega'
    }
    
    for name, symbol in greek_letters.items():
        # Use word boundaries to avoid partial matches
        # But SKIP if already escaped (i.e., preceded by backslash)
        # Negative lookbehind: don't match if preceded by \
        latex_str = re.sub(rf'(?<!\\)\b{name}\b', lambda m: symbol, latex_str)
    
    # 2. Replace 'eps' with epsilon (if not already escaped)
    latex_str = latex_str.replace(r'\varepsilon', r'\epsilon')
    # Replace eps with epsilon only if not already escaped
    latex_str = re.sub(r'(?<!\\)\beps\b', lambda m: '\\epsilon', latex_str)
    
    # 3. Convert common decimal exponents to fractions
    # 0.666... → 2/3, 0.333... → 1/3, 0.5 → 1/2, etc.
    def decimal_to_frac(match):
        val = float(match.group(1))
        # Check common fractions
        fracs = {
            0.5: '1/2', -0.5: '-1/2',
            0.333333: '1/3', -0.333333: '-1/3',
            0.666666: '2/3', -0.666666: '-2/3',
            0.25: '1/4', -0.25: '-1/4',
            0.75: '3/4', -0.75: '-3/4',
            0.2: '1/5', -0.2: '-1/5',
            0.4: '2/5', -0.4: '-2/5',
            0.6: '3/5', -0.6: '-3/5',
            0.8: '4/5', -0.8: '-4/5',
        }
        for frac_val, frac_str in fracs.items():
            if abs(val - frac_val) < 0.0001:
                return '^{' + frac_str + '}'
        # Return as-is if not a common fraction
        return match.group(0)
    
    latex_str = re.sub(r'\^{(-?\d+\.\d+)}', decimal_to_frac, latex_str)
    
    # 3. Auto-subscript: letter + number (but skip if it's part of Greek)
    # Only subscript patterns like k1, K2, d1, etc.
    def subscript_params(match):
        param = match.group(0)
        base = param[0]
        suffix = param[1:]
        return f"{base}_{{{suffix}}}"
    
    # Pattern: single letter + digit(s)
    latex_str = re.sub(r'\b([a-zA-Z])(\d+)\b', subscript_params, latex_str)
    
    return latex_str


def latex_ssys_from_antimony(antimony_text):
    """
    Generate LaTeX for S-system equations from Antimony file.
    
    CRITICAL: The Antimony file is the single source of truth.
    This function only converts it to LaTeX format with minimal
    cosmetic improvements. NO restructuring or reorganization.
    """
    odes = parse_antimony_odes(antimony_text)
    
    if not odes:
        return "\\text{(No ODEs found)}"
    
    # Extract all variable names and create symbol dictionary
    # This is critical for sympify to parse the expressions correctly
    import re
    all_vars = set()
    
    # Get all variable names from LHS
    for var_name, rhs in odes:
        all_vars.add(var_name)
    
    # Extract potential variable names from RHS expressions
    # Match patterns like X_1, epsilon, k1, etc.
    for var_name, rhs in odes:
        # Find all identifier-like tokens
        tokens = re.findall(r'\b[A-Za-z_]\w*\b', rhs)
        all_vars.update(tokens)
    
    # Create symbol dictionary for sympify
    # Include common parameters and epsilon
    symbol_dict = {name: sp.symbols(name, positive=True) for name in all_vars}
    
    rows = []
    for var_name, rhs in odes:
        try:
            # Direct conversion: Antimony string → sympy → LaTeX
            # CRITICAL: Pass locals so sympify knows about all variables
            rhs_expr = sp.sympify(rhs, locals=symbol_dict)
            rhs_latex = sp.latex(rhs_expr)
            
            # Remove line breaks that sympy adds (we want single-line equations)
            rhs_latex = rhs_latex.replace(' \\\\', '')
            rhs_latex = rhs_latex.replace('\\\\', '')
            
            # Apply cosmetic beautification only
            rhs_latex = _beautify_latex(rhs_latex)
            
            # Variable name to LaTeX
            var_latex = _beautify_latex(sp.latex(sp.Symbol(var_name)))
            
            rows.append(f"\\dot{{{var_latex}}} &= {rhs_latex}")
        except Exception as e:
            # Fallback: format as-is without \text (we're already in math mode)
            # Just escape special characters and use the raw expression
            rhs_escaped = rhs.replace('_', r'\_')
            rows.append(rf"\dot{{{var_name}}} &= {rhs_escaped}")
    
    # Join with proper LaTeX line break (double backslash + newline)
    return "\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"


def _is_already_ssystem(sym):
    """Check if input is already in canonical S-system form."""
    for var, ode in sym.odes.items():
        terms = []
        expanded = sp.expand(ode)
        if expanded.is_Add:
            terms = list(expanded.args)
        else:
            terms = [expanded]
        
        # Count positive and negative monomial terms
        pos_count = 0
        neg_count = 0
        for term in terms:
            if term == 0:
                continue
            # Check if it's a monomial (product of powers)
            if not _is_monomial(term):
                return False
            # Check sign
            if _get_term_sign(term) > 0:
                pos_count += 1
            else:
                neg_count += 1
        
        # S-system: exactly 1 positive and 1 negative monomial
        if pos_count != 1 or neg_count != 1:
            return False
    
    return True


def _is_already_gma(sym):
    """Check if input is already in GMA form (all terms are monomials)."""
    for var, ode in sym.odes.items():
        terms = []
        expanded = sp.expand(ode)
        if expanded.is_Add:
            terms = list(expanded.args)
        else:
            terms = [expanded]
        
        # Check if all terms are monomials
        for term in terms:
            if term == 0:
                continue
            if not _is_monomial(term):
                return False
    
    return True


def _is_monomial(term):
    """Check if a term is a monomial (product of powers with numeric coefficient)."""
    if term.is_Number:
        return True
    if isinstance(term, sp.Symbol):
        return True
    if isinstance(term, sp.Pow):
        base, exp = term.args
        # Base must be a symbol, exponent must be numeric
        return isinstance(base, sp.Symbol) and exp.is_number
    if term.is_Mul:
        # All factors must be numbers, symbols, or powers with numeric exponents
        for factor in term.args:
            if factor.is_Number or isinstance(factor, sp.Symbol):
                continue
            if isinstance(factor, sp.Pow):
                base, exp = factor.args
                if not (isinstance(base, sp.Symbol) and exp.is_number):
                    return False
            else:
                return False
        return True
    return False


def _get_term_sign(term):
    """Get the sign of a term's coefficient."""
    if term.is_Number:
        return 1 if float(term) >= 0 else -1
    if term.is_Mul:
        # Extract numeric coefficient
        coeff = 1.0
        for factor in term.args:
            if factor.is_Number:
                coeff *= float(factor)
        return 1 if coeff >= 0 else -1
    return 1  # Assume positive if no numeric coefficient


def latex_factor_map(rec):
    """Generate LaTeX for factor mapping."""
    if not rec.factor_map:
        return "\\text{(No factorization - direct form)}"
    rows = []
    for orig, aux_list in sorted(rec.factor_map.items(), key=lambda kv: kv[0].name):
        left = sp.latex(orig)
        right = " \\cdot ".join(sp.latex(a) for a in aux_list) if aux_list else "1"
        rows.append(f"{left} &= {right}")
    return "\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"


def load_and_report(ant_path, recast_path, T=20.0, steps=400,
                     mode="simplified"):
    """Load and report on a single recast model.
    
    Args:
        ant_path: Path to input Antimony file
        recast_path: Path to recast output Antimony file
        T: Simulation time
        steps: Number of simulation steps
        mode: Output mode ('simplified' or 'canonical')
    """
    ant_text = open(ant_path).read()
    rec_text = open(recast_path).read()
    display(Markdown("**Files**"))
    display(Markdown(f"- Antimony input: `{os.path.basename(ant_path)}`"
                     f"<br>- Recast output: "
                     f"`{os.path.basename(recast_path)}`"))
    display(Markdown("**Original Antimony**"))
    display(Code(ant_text, language="text"))
    display(Markdown("**Recast Antimony**"))
    display(Code(rec_text, language="text"))

    ir = ssys.parse_antimony(ant_text)
    sym = ssys.build_sym_system(ir)
    rec = ssys.recast_to_ssystem(sym, mode=mode)

    params = dict(sym.params)
    
    # Classify input and output
    input_class = classify_system(sym)
    output_class = classify_result(rec, mode=mode)
    
    # Display classification: Input → Output
    display(Markdown(f"**Classification:** {input_class.value} → {output_class.value}"))

    display(Markdown("**Mapping (original -> product of auxiliaries)**"))
    display(Markdown("$$\n" + latex_factor_map(rec) + "\n$$"))

    display(Markdown("**Original ODEs (LaTeX)**"))
    display(Markdown("$$\n" + latex_odes_from_sym(sym) + "\n$$"))
    display(Markdown("**Recast ODEs (LaTeX)**"))
    display(Markdown("$$\n" + latex_ssys_from_antimony(rec_text) + "\n$$"))

    var_syms = sorted(sym.odes.keys(), key=lambda s: s.name)
    rhs_exprs = [sp.simplify(sym.odes[s]) for s in var_syms]
    y0 = [float(sym.initials[s]) for s in var_syms]
    f_orig = build_rhs_from_sympy(var_syms, rhs_exprs, params)
    t_orig, y_orig = rk4(f_orig, (0.0, T), y0, steps)

    aux_syms = list(sorted(rec.variables, key=lambda s: s.name))
    
    if rec.status == RecastStatus.GMA:
        rec_rhs = []
        for eq in rec.gma_equations:
            prod = sp.Integer(0)
            for c, e in eq.production:
                prod += product_expr(c, e)
            deg = sp.Integer(0)
            for c, e in eq.degradation:
                deg += product_expr(c, e)
            rec_rhs.append(sp.simplify(prod - deg))
    else:
        rec_rhs = [
            sp.simplify(
                product_expr(eq.growth[0], _expand_exps_through_factors(eq.growth[1], rec.factor_map)) -
                product_expr(eq.decay[0], _expand_exps_through_factors(eq.decay[1], rec.factor_map))
            )
            for eq in rec.equations
        ]
    
    aux_y0 = [float(rec.initials[s]) for s in aux_syms]
    f_rec = build_rhs_from_sympy(aux_syms, rec_rhs, params)
    t_rec, y_rec = rk4(f_rec, (0.0, T), aux_y0, steps)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), dpi=120, sharey=True)

    ax = axes[0]
    for i, s in enumerate(var_syms):
        ax.plot(t_orig, y_orig[:, i], label=str(s))
    ax.set_title("Original system", fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("state", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    if rec.factor_map:
        aux_idx = {s: i for i, s in enumerate(aux_syms)}
        for k, factors in sorted(rec.factor_map.items(), key=lambda kv: kv[0].name):
            prod = np.ones(y_rec.shape[0], dtype=float)
            for s in factors:
                prod *= y_rec[:, aux_idx[s]]
            ax.plot(t_rec, prod, label=str(k), linestyle='--')
    else:
        for i, s in enumerate(aux_syms):
            ax.plot(t_rec, y_rec[:, i], label=str(s), linestyle='--')
    
    ax.set_title("Reconstructed from recast", fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("state", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
