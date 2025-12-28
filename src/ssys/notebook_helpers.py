"""Helper functions for Jupyter notebook verification."""

import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import Code, Markdown, display

import ssys
from ssys.recaster import (
    classify_result,
    classify_system,
    parse_antimony_via_sbml,
)

FILL_MISSING_PARAMS = False  # set True to auto-fill absent params with 1.0


def build_rhs_from_sympy(vars_syms, rhs_exprs, param_vals, assignment_rules=None):
    """Build numerical RHS function from symbolic expressions.

    Special handling for 'time' symbol: recognized as the independent variable
    (integration time t), not a parameter requiring a numeric value.

    Args:
        vars_syms: List of sympy symbols for state variables
        rhs_exprs: List of sympy expressions for ODE RHS
        param_vals: Dict mapping parameter names to numeric values
        assignment_rules: Optional dict mapping rule name to expression string.
            These are substituted into the ODEs before evaluation.
    """
    var_set = set(vars_syms)

    # Build assignment rule substitutions if provided
    # Assignment rules may depend on each other, so we need to repeatedly substitute
    # until no more rules can be applied
    if assignment_rules:
        # Build a mapping from symbol name to existing sympy symbol (to preserve identity)
        all_existing_syms = set().union(*[expr.free_symbols for expr in rhs_exprs])
        name_to_sym = {s.name: s for s in all_existing_syms}
        name_to_sym.update({s.name: s for s in vars_syms})

        # IMPORTANT: Also add all parameter names to prevent SymPy from interpreting
        # common names like 'beta', 'gamma' as built-in functions
        for param_name in param_vals.keys():
            if param_name not in name_to_sym:
                name_to_sym[param_name] = sp.Symbol(param_name)

        # Pre-create symbols for ALL assignment rule names BEFORE parsing
        # This ensures that inter-rule dependencies use the same symbol instances
        for rule_name in assignment_rules.keys():
            if rule_name not in name_to_sym:
                name_to_sym[rule_name] = sp.Symbol(rule_name)

        # Now parse all assignment rules into sympy expressions
        # Use name_to_sym to ensure consistent symbol identity
        rule_exprs = {}
        for rule_name, rule_str in assignment_rules.items():
            try:
                # Parse expression using existing/pre-created symbols
                rule_expr = sp.sympify(rule_str, locals=name_to_sym)
                rule_sym = name_to_sym[rule_name]  # Use the pre-created symbol
                rule_exprs[rule_sym] = rule_expr
            except Exception:
                # Skip rules that can't be parsed
                pass

        # Repeatedly substitute rules into each other until fixed point
        # This handles dependencies like: A := B*C, B := X+Y
        max_iterations = 20  # Increased for longer dependency chains
        for _ in range(max_iterations):
            changed = False
            for rule_sym, rule_expr in list(rule_exprs.items()):
                new_expr = rule_expr.subs(rule_exprs)
                if new_expr != rule_expr:
                    rule_exprs[rule_sym] = new_expr
                    changed = True
            if not changed:
                break

        # Now substitute all assignment rules into the ODE RHS expressions
        rhs_exprs = [expr.subs(rule_exprs) for expr in rhs_exprs]

    rhs_free = set().union(*[expr.free_symbols for expr in rhs_exprs])

    # Identify 'time' symbol if present (this is the independent variable, not a parameter)
    time_sym = None
    for s in rhs_free:
        if s.name == "time":
            time_sym = s
            break

    param_subs = {}
    missing = []
    for s in sorted(rhs_free - var_set, key=lambda z: z.name):
        name = s.name
        # Skip 'time' - it's the independent variable, not a parameter
        if name == "time":
            continue
        if name in param_vals:
            param_subs[s] = float(param_vals[name])
        else:
            missing.append(name)

    if missing and not FILL_MISSING_PARAMS:
        raise ValueError(
            "Missing numeric values for parameters: " + ", ".join(sorted(set(missing)))
        )
    if missing and FILL_MISSING_PARAMS:
        for sname in set(missing):
            sym = next(ss for ss in rhs_free if ss.name == sname and ss not in var_set)
            param_subs[sym] = 1.0
        display(
            Markdown(
                "> **Warning**: filled missing params with 1.0 → " + ", ".join(sorted(set(missing)))
            )
        )

    def f(t, y):
        subs = {s: float(y[i]) for i, s in enumerate(vars_syms)}
        subs.update(param_subs)
        # If 'time' symbol exists, substitute the current integration time
        if time_sym is not None:
            subs[time_sym] = float(t)
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
        lines.append(rf"\dot{{{sp.latex(v)}}} &= {sp.latex(rhs)}")
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
            rhs = match.group(2).rstrip(";").strip()
            odes.append((var_name, rhs))

    return odes


def _beautify_latex(latex_str):
    r"""Apply beautification rules to LaTeX string.

    Single source of truth: the Antimony file.
    Only cosmetic improvements, no restructuring.

    Rules applied in order:
    1. Handle compound names with Greek prefixes (gamma_rate → \gamma_{rate})
    2. Handle subscripts (Name_suffix → Name_{suffix})
    3. Convert standalone Greek letters
    4. Convert decimal exponents to fractions
    5. Auto-subscript single letter + digits (k1 → k_1)
    """
    import re

    greek_letters = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "zeta": r"\zeta",
        "eta": r"\eta",
        "theta": r"\theta",
        "iota": r"\iota",
        "kappa": r"\kappa",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "nu": r"\nu",
        "xi": r"\xi",
        "pi": r"\pi",
        "rho": r"\rho",
        "sigma": r"\sigma",
        "tau": r"\tau",
        "upsilon": r"\upsilon",
        "phi": r"\phi",
        "chi": r"\chi",
        "psi": r"\psi",
        "omega": r"\omega",
    }

    # Build pattern for Greek letter names
    greek_pattern = "|".join(greek_letters.keys())

    # Step 1: Handle Greek_suffix patterns (e.g., gamma_rate → \gamma_{rate})
    # This must be done BEFORE general subscript handling
    def greek_with_suffix(match):
        greek_name = match.group(1)
        suffix = match.group(2)
        greek_symbol = greek_letters.get(greek_name, greek_name)
        return greek_symbol + "_{" + suffix + "}"

    latex_str = re.sub(
        r"\b(" + greek_pattern + r")_([a-zA-Z_][a-zA-Z0-9_]*)", greek_with_suffix, latex_str
    )

    # Also handle Greek + digit (no underscore): alpha1 → \alpha_{1}
    def greek_with_digit(match):
        greek_name = match.group(1)
        digits = match.group(2)
        greek_symbol = greek_letters.get(greek_name, greek_name)
        return greek_symbol + "_{" + digits + "}"

    latex_str = re.sub(r"\b(" + greek_pattern + r")(\d+)\b", greek_with_digit, latex_str)

    # Step 2: Handle Name_suffix patterns (multi-char subscripts)
    # Patterns like: S_in, Y_xs, mu_max, k_m_1, E_T
    # Convert to: S_{in}, Y_{xs}, mu_{max}, k_{m_1}, E_{T}
    def subscript_compound(match):
        base = match.group(1)
        suffix = match.group(2)
        return f"{base}_{{{suffix}}}"

    # Match: word_suffix where suffix is alphanumeric (including underscores)
    # But NOT if base is a single letter followed by digits only (handled later)
    latex_str = re.sub(
        r"\b([A-Za-z][A-Za-z0-9]*)_([A-Za-z][A-Za-z0-9_]*)\b", subscript_compound, latex_str
    )

    # Also handle single letter with any suffix: S_in, X_r, Z_1, etc.
    # But NOT inside braces (already processed content)
    latex_str = re.sub(
        r"(?<!\{)\b([A-Za-z])_([A-Za-z0-9][A-Za-z0-9_]*)\b", subscript_compound, latex_str
    )

    # Handle single letter with single digit subscript: Z_1 → Z_{1}
    # But NOT if the underscore is already inside braces
    # Use negative lookbehind for { and positive lookbehind for word boundary
    def safe_subscript(match):
        # Check if we're inside braces by looking at context
        base = match.group(1)
        suffix = match.group(2)
        return f"{base}_{{{suffix}}}"

    # Only match at start of string or after non-brace character
    latex_str = re.sub(r"(?<![{\w])([A-Za-z])_(\d+)\b", safe_subscript, latex_str)

    # Step 3: Convert standalone Greek letters (not already escaped)
    # Only match when NOT followed by underscore (those were handled above)
    for name, symbol in greek_letters.items():
        # Match Greek name at word boundary, not followed by underscore
        # Use lambda to avoid backslash interpretation issues
        latex_str = re.sub(r"(?<!\\)\b" + name + r"\b(?!_)", lambda m, s=symbol: s, latex_str)

    # Step 4: Replace 'eps' with epsilon (if not already escaped)
    latex_str = re.sub(r"(?<!\\)\beps\b(?!_)", r"\\epsilon", latex_str)

    # Step 5: Convert common decimal exponents to fractions
    def decimal_to_frac(match):
        val = float(match.group(1))
        fracs = [
            (1 / 2, "1/2"),
            (-1 / 2, "-1/2"),
            (1 / 3, "1/3"),
            (-1 / 3, "-1/3"),
            (2 / 3, "2/3"),
            (-2 / 3, "-2/3"),
            (1 / 4, "1/4"),
            (-1 / 4, "-1/4"),
            (3 / 4, "3/4"),
            (-3 / 4, "-3/4"),
            (1 / 5, "1/5"),
            (-1 / 5, "-1/5"),
            (2 / 5, "2/5"),
            (-2 / 5, "-2/5"),
            (3 / 5, "3/5"),
            (-3 / 5, "-3/5"),
            (4 / 5, "4/5"),
            (-4 / 5, "-4/5"),
        ]
        for frac_val, frac_str in fracs:
            if abs(val - frac_val) < 1e-4:
                return "^{" + frac_str + "}"
        return match.group(0)

    latex_str = re.sub(r"\^{(-?\d+\.\d+)}", decimal_to_frac, latex_str)

    # Step 6: Auto-subscript single letter + digit(s) ONLY
    # Pattern: k1, K2, d1 → k_{1}, K_{2}, d_{1}
    # But NOT if already has subscript braces
    def subscript_letter_digit(match):
        letter = match.group(1)
        digits = match.group(2)
        return f"{letter}_{{{digits}}}"

    # Only match if NOT already subscripted (not preceded by _{ )
    latex_str = re.sub(r"(?<!_)\b([a-zA-Z])(\d+)\b", subscript_letter_digit, latex_str)

    return latex_str


def _simplify_exponent_content(content):
    """Simplify content inside an exponent.

    - Convert X.0 to X (e.g., 1.0 → 1, -2.0 → -2)
    - Clean up spacing
    """
    import re

    # Convert decimal integers to integers: 1.0 → 1, -2.0 → -2
    content = re.sub(r"(\d+)\.0\b", r"\1", content)

    # Clean up spacing around operators
    content = content.strip()

    return content


def _antimony_to_latex_direct(expr_str):
    """
    Convert Antimony expression string directly to LaTeX without sympy evaluation.

    This preserves the EXACT structure of the expression, including:
    - Parentheses groupings
    - Term ordering
    - Operator placement

    Only applies cosmetic improvements (Greek letters, subscripts).
    """
    import re

    result = expr_str

    # Step 0: Pre-process - convert X.0 to X throughout (1.0 → 1, etc.)
    result = re.sub(r"(\d+)\.0\b", r"\1", result)

    # Step 1a: Handle symbolic exponents in parentheses: ^(a - 1) → ^{a - 1}
    # Must be done BEFORE numeric exponent handling
    def process_paren_exponent(match):
        content = match.group(1)
        # Simplify the content
        content = _simplify_exponent_content(content)
        return f"^{{{content}}}"

    result = re.sub(r"\^\(([^)]+)\)", process_paren_exponent, result)

    # Step 1b: Convert Antimony ^ to LaTeX ^ with grouping for numeric exponents
    # Handle ALL exponents: integers, negatives, AND decimals
    # Pattern: ^-1, ^2, ^0.5, ^-0.333333, etc.
    result = re.sub(r"\^(-?\d+\.?\d*)", lambda m: f"^{{{m.group(1)}}}", result)

    # Step 1c: Handle symbolic exponents without parentheses: ^n, ^a, ^-n
    # Pattern: ^symbol or ^-symbol (single word identifier)
    result = re.sub(r"\^(-?[a-zA-Z_]\w*)", lambda m: f"^{{{m.group(1)}}}", result)

    # Step 2: Convert * to space (implicit multiplication in LaTeX)
    result = result.replace("*", " ")

    # Step 3: Apply beautification (Greek letters, subscripts, fraction conversion)
    result = _beautify_latex(result)

    return result


def latex_ssys_from_antimony(antimony_text):
    """
    Generate LaTeX for S-system equations from Antimony file.

    CRITICAL: The Antimony file is the single source of truth.
    This function converts Antimony directly to LaTeX WITHOUT using sympy
    to parse/evaluate expressions. This preserves the exact structure
    including term ordering like (epsilon + 1) - epsilon.
    """
    odes = parse_antimony_odes(antimony_text)

    if not odes:
        return "\\text{(No ODEs found)}"

    rows = []
    for var_name, rhs in odes:
        # Direct string conversion - no sympy parsing
        rhs_latex = _antimony_to_latex_direct(rhs)

        # Variable name to LaTeX (just beautify it)
        var_latex = _beautify_latex(var_name)

        rows.append(f"\\dot{{{var_latex}}} &= {rhs_latex}")

    # Join with proper LaTeX line break (double backslash + newline)
    return "\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"


def _is_already_ssystem(sym):
    """Check if input is already in canonical S-system form."""
    for _var, ode in sym.odes.items():
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
    for _var, ode in sym.odes.items():
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


def is_nonautonomous(sym):
    """Check if a SymSystem has explicit time dependence.

    Returns True if any ODE contains the 'time' symbol (Antimony keyword).
    """
    for ode_expr in sym.odes.values():
        free = ode_expr.free_symbols
        if any(s.name == "time" for s in free):
            return True
    return False


def was_nonautonomous(sym):
    """Check if system was originally nonautonomous but lifted.

    Detects either:
    - Explicit 'time' symbol in ODEs
    - Clock variable (t' = 1) indicating lifted time

    Returns True if system has/had explicit time dependence.
    """
    # Check for 'time' keyword
    if is_nonautonomous(sym):
        return True

    # Check for clock variable (was lifted from nonautonomous)
    if find_clock_variable(sym) is not None:
        return True

    return False


def find_clock_variable(sym):
    """Find clock variable in a SymSystem (lifted time).

    A clock variable has ODE = 1 (constant derivative, tracks time).
    Returns the variable name if found, None otherwise.
    """
    for var, ode_expr in sym.odes.items():
        # Check if ODE == 1 (simplify to handle symbolic forms)
        simplified = sp.simplify(ode_expr - 1)
        if simplified == 0:
            return str(var)
    return None


def get_autonomy_label(sym, is_recast=False, orig_was_nonautonomous=False):
    """Get autonomy label for display.

    Args:
        sym: SymSystem to analyze
        is_recast: True if this is a recast system
        orig_was_nonautonomous: True if original was nonautonomous

    Returns:
        Tuple of (label, clock_var_name or None)
        Label is one of: "autonomous", "nonautonomous", "autonomous, lifted"
    """
    nonauto = is_nonautonomous(sym)

    if nonauto:
        return ("nonautonomous", None)

    # System is autonomous - check if it was lifted
    clock_var = find_clock_variable(sym)

    if clock_var and (is_recast or orig_was_nonautonomous):
        return ("autonomous, lifted", clock_var)

    return ("autonomous", None)


def load_and_report(
    ant_path,
    recast_path,
    T=None,
    T_start=None,
    steps=None,
    mode="simplified",
    validation_json=None,
):
    """Load and report on a single recast model.

    Args:
        ant_path: Path to input Antimony file
        recast_path: Path to recast output Antimony file
        T: Simulation end time (if None, uses @SIM T_END from file, default 20.0)
        T_start: Simulation start time (if None, uses @SIM T_START from file, default 0.0)
        steps: Number of simulation steps (if None, uses @SIM N_STEPS from file, default 400)
        mode: Output mode ('simplified' or 'canonical')
        validation_json: Optional path to validation JSON file
    """
    ant_text = open(ant_path).read()
    rec_text = open(recast_path).read()

    # Parse original model using SBML-based parser (same as CLI)
    # This ensures consistent ODE extraction across CLI and notebook
    sym = parse_antimony_via_sbml(ant_text)

    # Also parse with hand-rolled parser for simulation (ModelIR needed by simulate_ode)
    ir = ssys.parse_antimony(ant_text)

    # Extract simulation metadata from hand-rolled parser (ir has @SIM data)
    # SBML doesn't have @SIM concept, so we get it from the Antimony parser
    sim_t_start = ir.sim_t_start
    sim_t_end = ir.sim_t_end
    sim_n_steps = ir.sim_n_steps

    # Model @SIM values take PRECEDENCE over function parameters
    # This ensures per-model simulation settings are respected
    if sim_t_start is not None:
        T_start = sim_t_start
    elif T_start is None:
        T_start = 0.0  # default

    if sim_t_end is not None:
        T = sim_t_end
    elif T is None:
        T = 20.0  # default

    if sim_n_steps is not None:
        steps = sim_n_steps
    elif steps is None:
        steps = 400  # default
    display(Markdown("**Files**"))
    display(
        Markdown(
            f"- Antimony input: `{os.path.basename(ant_path)}`"
            f"<br>- Recast output: "
            f"`{os.path.basename(recast_path)}`"
        )
    )
    display(Markdown("**Original Antimony**"))
    display(Code(ant_text, language="text"))
    display(Markdown("**Recast Antimony**"))
    display(Code(rec_text, language="text"))

    # Recast the symbolic system (already parsed above using SBML-based parser)
    rec = ssys.recast_to_ssystem(sym, mode=mode)

    dict(sym.params)

    # Classify input and output
    input_class = classify_system(sym)
    output_class = classify_result(rec, mode=mode)

    # Check autonomy status - use was_nonautonomous to catch pre-lifted models
    orig_has_time = was_nonautonomous(sym)
    orig_clock = find_clock_variable(sym)

    # Parse recast system to check for clock variable
    rec_sym = parse_antimony_via_sbml(rec_text)
    rec_autonomy_label, clock_var = get_autonomy_label(
        rec_sym, is_recast=True, orig_was_nonautonomous=orig_has_time
    )

    # Build original autonomy label
    if is_nonautonomous(sym):
        orig_autonomy_label = "nonautonomous"
    elif orig_clock:
        orig_autonomy_label = f"autonomous, lifted, clock: {orig_clock}"
    else:
        orig_autonomy_label = "autonomous"

    # Build classification string with autonomy info
    input_str = f"{input_class.value} ({orig_autonomy_label})"
    if clock_var:
        output_str = f"{output_class.value} ({rec_autonomy_label}, clock: {clock_var})"
    else:
        output_str = f"{output_class.value} ({rec_autonomy_label})"

    # Display classification: Input → Output
    display(Markdown(f"**Classification:** {input_str} → {output_str}"))

    # Display validation results if available
    if validation_json and os.path.exists(validation_json):
        import json

        with open(validation_json) as f:
            validation_data = json.load(f)

        # Extract key info
        overall_pass = validation_data.get("overall_pass", False)
        summary = validation_data.get("summary", "")
        tests = validation_data.get("tests", {})

        # Display overall status with color
        status_emoji = "✓" if overall_pass else "✗"
        status_color = "green" if overall_pass else "red"
        display(
            Markdown(
                f"### <span style='color:{status_color}'>{status_emoji} Validation: {summary}</span>"
            )
        )

        # Display test results table
        table_rows = []
        table_rows.append("| Test | Result | Max Error | Mean Error |")
        table_rows.append("|------|--------|-----------|------------|")

        for test_name, test_data in tests.items():
            if test_data and test_name != "auxiliaries":
                result = test_data.get("result", "N/A")
                max_err = test_data.get("max_error")
                mean_err = test_data.get("mean_error")

                result_emoji = {"pass": "✓", "fail": "✗", "timeout": "⏱", "not_attempted": "⊘"}.get(
                    result, "?"
                )

                max_str = f"{max_err:.2e}" if max_err is not None else "N/A"
                mean_str = f"{mean_err:.2e}" if mean_err is not None else "N/A"

                table_rows.append(
                    f"| {test_name} | {result_emoji} {result} | {max_str} | {mean_str} |"
                )

        display(Markdown("\n".join(table_rows)))

        # Show details for failed tests
        for test_name, test_data in tests.items():
            if test_data and test_data.get("result") == "fail" and test_name != "auxiliaries":
                details = test_data.get("details", "")
                if details:
                    display(Markdown(f"**{test_name} details:** {details}"))

    display(Markdown("**Mapping (original -> product of auxiliaries)**"))
    display(Markdown("$$\n" + latex_factor_map(rec) + "\n$$"))

    display(Markdown("**Original ODEs (LaTeX)**"))
    display(Markdown("$$\n" + latex_odes_from_sym(sym) + "\n$$"))
    display(Markdown("**Recast ODEs (LaTeX)**"))
    display(Markdown("$$\n" + latex_ssys_from_antimony(rec_text) + "\n$$"))

    # Initialize state name lists for column mapping (will be populated by simulation)
    orig_state_names = []
    rec_state_names = []

    # Use ODE backend abstraction
    from ssys.ode_backends import simulate_ode

    # Simulate original model
    result_orig = simulate_ode(ir, T_start, T, steps + 1)
    if not result_orig["success"]:
        display(
            Markdown(
                f"**❌ Original simulation failed:** {result_orig['message']}"
            )
        )
        display(Markdown("*Cannot proceed without successful simulation.*"))
        return  # Exit early
    else:
        t_orig = result_orig["t"]
        y_orig = result_orig["y"]
        # Get state names from simulation result for correct column mapping
        orig_state_names = result_orig.get("state_names", [])

    # Build recast IR for simulation
    recast_ir = ssys.parse_antimony(rec_text)
    result_rec = simulate_ode(recast_ir, T_start, T, steps + 1)
    if not result_rec["success"]:
        display(
            Markdown(f"**❌ Recast simulation failed:** {result_rec['message']}")
        )
        display(Markdown("*Cannot proceed without successful simulation.*"))
        return  # Exit early
    else:
        t_rec = result_rec["t"]
        y_rec = result_rec["y"]
        # Get state names from simulation result for correct column mapping
        rec_state_names = result_rec.get("state_names", [])
        # Use actual symbols from rec.variables for plotting factor_map
        aux_syms = sorted(rec.variables, key=lambda s: s.name)

    var_syms = sorted(sym.odes.keys(), key=lambda s: s.name)

    # Build column index maps from state_names (if available from RoadRunner)
    # This fixes the ordering mismatch between alphabetical sort and Antimony definition order
    if orig_state_names:
        orig_name_to_idx = {name: i for i, name in enumerate(orig_state_names)}
    else:
        orig_name_to_idx = {str(s): i for i, s in enumerate(var_syms)}

    if rec_state_names:
        rec_name_to_idx = {name: i for i, name in enumerate(rec_state_names)}
    else:
        rec_name_to_idx = {str(s): i for i, s in enumerate(aux_syms)}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), dpi=120, sharey=True)

    ax = axes[0]
    for s in var_syms:
        idx = orig_name_to_idx.get(str(s))
        if idx is not None:
            ax.plot(t_orig, y_orig[:, idx], label=str(s))
    ax.set_title("Original system", fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("state", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    if rec.factor_map:
        # Use rec_name_to_idx for correct column mapping
        for k, factors in sorted(rec.factor_map.items(), key=lambda kv: kv[0].name):
            prod = np.ones(y_rec.shape[0], dtype=float)
            for s in factors:
                idx = rec_name_to_idx.get(str(s))
                if idx is not None:
                    prod *= y_rec[:, idx]
            ax.plot(t_rec, prod, label=str(k), linestyle="--")
    else:
        for s in aux_syms:
            idx = rec_name_to_idx.get(str(s))
            if idx is not None:
                ax.plot(t_rec, y_rec[:, idx], label=str(s), linestyle="--")

    ax.set_title("Reconstructed from recast", fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("state", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    plt.show()

    # ==============================
    # Trajectory Comparison Table
    # ==============================
    # Compute scaled relative error between original and reconstructed trajectories
    # Error = |X_orig - X_recast| / (1 + max(|X_orig|, |X_recast|))

    display(Markdown("### Trajectory Comparison"))

    # Build reconstructed values from recast simulation
    n_points = len(t_orig)
    n_vars = len(var_syms)
    X_orig_array = np.zeros((n_points, n_vars))
    X_recast_array = np.zeros((n_points, n_vars))

    # Fill original values
    for i, s in enumerate(var_syms):
        idx = orig_name_to_idx.get(str(s))
        if idx is not None:
            X_orig_array[:, i] = y_orig[:, idx]

    # Fill reconstructed values from recast
    if rec.factor_map:
        for i, (k, factors) in enumerate(sorted(rec.factor_map.items(), key=lambda kv: kv[0].name)):
            prod = np.ones(n_points, dtype=float)
            for s in factors:
                idx = rec_name_to_idx.get(str(s))
                if idx is not None:
                    prod *= y_rec[:, idx]
            # Find which original variable this corresponds to
            var_idx = next((j for j, v in enumerate(var_syms) if str(v) == str(k)), None)
            if var_idx is not None:
                X_recast_array[:, var_idx] = prod
    else:
        for i, s in enumerate(var_syms):
            idx = rec_name_to_idx.get(str(s))
            if idx is not None:
                X_recast_array[:, i] = y_rec[:, idx]

    # Compute error relative to characteristic scale (max value over trajectory)
    # This avoids false positives from small absolute differences at early time
    # when values are near zero.
    #
    # For each variable:
    #   scale = max(|orig|, |recast|) over all time points
    #   error = |orig - recast| / scale
    #
    # This gives intuitive results: if curves look the same, error is small.
    scale = np.maximum(np.max(np.abs(X_orig_array), axis=0), np.max(np.abs(X_recast_array), axis=0))
    scale = np.maximum(scale, 1e-10)  # Avoid division by zero for zero trajectories

    errors = np.abs(X_orig_array - X_recast_array) / scale[np.newaxis, :]

    # Build table
    # 1.5% threshold for GMA recasts with auxiliary variables
    # The coupled systems can have small numerical integration differences
    threshold = 1.5e-2  # 1.5% error threshold
    table_rows = []
    table_rows.append("| Variable | Max Error | Mean Error | Status |")
    table_rows.append("|----------|-----------|------------|--------|")

    overall_pass = True
    for i, s in enumerate(var_syms):
        var_errors = errors[:, i]
        max_err = float(np.max(var_errors))
        mean_err = float(np.mean(var_errors))

        status = "✓" if max_err < threshold else "✗"
        if max_err >= threshold:
            overall_pass = False

        table_rows.append(f"| {s} | {max_err:.2e} | {mean_err:.2e} | {status} |")

    # Overall row
    overall_max = float(np.max(errors))
    overall_mean = float(np.mean(errors))
    overall_status = "✓" if overall_pass else "✗"
    table_rows.append(
        f"| **Overall** | **{overall_max:.2e}** | **{overall_mean:.2e}** | **{overall_status}** |"
    )

    display(Markdown("\n".join(table_rows)))

    # Show interpretation
    if overall_pass:
        display(Markdown(f"*Trajectories match within {threshold * 100:.1f}% tolerance.*"))
    else:
        display(
            Markdown(
                f"⚠️ *Trajectories diverge. Max scaled error: {overall_max:.2e} exceeds {threshold * 100:.1f}% threshold.*"
            )
        )
