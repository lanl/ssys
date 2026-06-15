"""Symbolic lifting for rational, composite, and time-dependent terms."""

from dataclasses import dataclass
from typing import Any

import sympy as sp

from ssys._recaster.parsing import _sympy_to_antimony_syntax
from ssys.math_utils import expand_to_terms
from ssys.types import SymSystem

_RATIONAL_LIFT_SIMPLIFY_MAX_OPS = 20
_COMPOSITE_LIFT_SIMPLIFY_MAX_OPS = 20


class UnsupportedCompositeDerivativeError(RuntimeError):
    """Raised when composite lifting would emit unsupported symbolic derivatives."""

    def __init__(
        self,
        *,
        context: str,
        function_names: list[str],
        source_expr: sp.Expr,
        derivative_expr: sp.Expr,
    ) -> None:
        self.context = context
        self.function_names = function_names
        self.source_expr = source_expr
        self.derivative_expr = derivative_expr
        source_preview = _preview_expr(source_expr)
        derivative_preview = _preview_expr(derivative_expr)
        functions = ", ".join(function_names) if function_names else "unknown"
        super().__init__(
            "unsupported_generated_output: unsupported composite derivative "
            f"for function(s) {functions}; context={context}; "
            "SymPy left an unevaluated Derivative/Subs expression that cannot "
            "be emitted to Antimony safely; "
            f"source_expr={source_preview}; derivative_expr={derivative_preview}"
        )


def _preview_expr(expr: sp.Expr, *, limit: int = 240) -> str:
    preview = str(expr)
    if len(preview) > limit:
        preview = preview[: limit - 3] + "..."
    return preview


def _function_names_for_unsupported_derivative(
    *,
    source_expr: sp.Expr,
    derivative_expr: sp.Expr,
) -> list[str]:
    names: set[str] = set()
    for derivative in derivative_expr.atoms(sp.Derivative):
        for func in derivative.expr.atoms(sp.Function):
            names.add(func.func.__name__)
        if isinstance(derivative.expr, sp.Function):
            names.add(derivative.expr.func.__name__)
    for func in source_expr.atoms(sp.Function):
        names.add(func.func.__name__)
    if isinstance(source_expr, sp.Function):
        names.add(source_expr.func.__name__)
    return sorted(names)


def _raise_for_unsupported_derivative_printer(
    derivative_expr: sp.Expr,
    *,
    source_expr: sp.Expr,
    context: str,
) -> None:
    """Fail closed before raw SymPy Derivative/Subs text reaches Antimony."""
    if not derivative_expr.has(sp.Derivative) and not derivative_expr.has(sp.Subs):
        return
    raise UnsupportedCompositeDerivativeError(
        context=context,
        function_names=_function_names_for_unsupported_derivative(
            source_expr=source_expr,
            derivative_expr=derivative_expr,
        ),
        source_expr=source_expr,
        derivative_expr=derivative_expr,
    )


def _sbml_piecewise(*args) -> sp.Piecewise:
    """Convert SBML-style piecewise(value, condition, ..., otherwise) syntax."""
    if not args:
        return sp.Piecewise((sp.nan, True))

    pairs: list[tuple[Any, Any]] = []
    if len(args) % 2 == 1:
        pairs.extend((args[i], args[i + 1]) for i in range(0, len(args) - 1, 2))
        pairs.append((args[-1], True))
    else:
        pairs.extend((args[i], args[i + 1]) for i in range(0, len(args), 2))
    return sp.Piecewise(*pairs)


def _assignment_rule_math_locals() -> dict[str, object]:
    """Return math helpers needed when parsing assignment-rule strings."""
    return {
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "ceiling": sp.ceiling,
        "cos": sp.cos,
        "eq": sp.Eq,
        "exp": sp.exp,
        "floor": sp.floor,
        "geq": sp.Ge,
        "gt": sp.Gt,
        "leq": sp.Le,
        "ln": sp.log,
        "log": sp.log,
        "lt": sp.Lt,
        "neq": sp.Ne,
        "piecewise": _sbml_piecewise,
        "pow": lambda base, exponent: base**exponent,
        "sin": sp.sin,
        "sqrt": sp.sqrt,
        "tan": sp.tan,
        "tanh": sp.tanh,
    }


def _substitute_symbols_by_name(
    expr: sp.Expr,
    replacements_by_name: dict[str, sp.Expr],
) -> sp.Expr:
    """Substitute symbols by matching names instead of object identity."""
    if not replacements_by_name:
        return expr
    substitutions = {
        symbol: replacements_by_name[str(symbol)]
        for symbol in expr.free_symbols
        if str(symbol) in replacements_by_name
    }
    return expr.subs(substitutions) if substitutions else expr


def _assignment_rule_parse_locals(sym: SymSystem) -> dict[str, object]:
    """Build locals that preserve model identifiers when sympifying rules."""
    local_symbols: dict[str, object] = {}
    for var in sym.vars:
        local_symbols[var.name] = var
    for name in sym.params:
        local_symbols.setdefault(name, sp.Symbol(name, positive=True))
    for name in sym.compartments:
        local_symbols.setdefault(name, sp.Symbol(name, positive=True))
    for name in sym.assignment_rules:
        local_symbols.setdefault(name, sp.Symbol(name, positive=True))
    local_symbols.setdefault("time", sp.Symbol("time", positive=True))
    local_symbols.setdefault("t", sp.Symbol("t", positive=True))
    local_symbols.setdefault("T", sp.Symbol("T", positive=True))
    for name, helper in _assignment_rule_math_locals().items():
        local_symbols.setdefault(name, helper)
    return local_symbols


def _parse_assignment_rule_expressions(sym: SymSystem) -> dict[str, sp.Expr]:
    """Parse assignment rules and recursively expand nested rule references."""
    if not sym.assignment_rules:
        return {}

    local_symbols = _assignment_rule_parse_locals(sym)
    parsed: dict[str, sp.Expr] = {}
    for name, expr_text in sym.assignment_rules.items():
        try:
            parsed[name] = sp.sympify(expr_text, locals=local_symbols)
        except (TypeError, ValueError, sp.SympifyError):
            continue

    expanded = dict(parsed)
    for _ in range(len(expanded) + 1):
        changed = False
        for name, expr in list(expanded.items()):
            replacements = {
                other_name: other_expr
                for other_name, other_expr in expanded.items()
                if other_name != name
            }
            next_expr = _substitute_symbols_by_name(expr, replacements)
            if next_expr != expr:
                expanded[name] = next_expr
                changed = True
        if not changed:
            break
    return expanded


def _expand_assignment_rules_by_name(
    expr: sp.Expr,
    assignment_exprs: dict[str, sp.Expr],
) -> sp.Expr:
    """Inline assignment-rule expressions into ``expr`` by identifier name."""
    if not assignment_exprs:
        return expr
    expanded = expr
    for _ in range(len(assignment_exprs) + 1):
        next_expr = _substitute_symbols_by_name(expanded, assignment_exprs)
        if next_expr == expanded:
            break
        expanded = next_expr
    return expanded


def _time_symbols(expr: sp.Expr) -> list[sp.Symbol]:
    """Return free symbols named time, independent of SymPy assumptions."""
    return [symbol for symbol in expr.free_symbols if symbol.name == "time"]


def _has_compound_symbolic_power(expr: sp.Expr) -> bool:
    """Return true for symbolic powers that make global simplify expensive."""
    for pow_expr in expr.atoms(sp.Pow):
        base, exp = pow_expr.as_base_exp()
        if exp.is_number:
            continue
        if isinstance(base, sp.Symbol) or base.is_Number:
            continue
        return True
    return False


def _bounded_rational_lift_simplify(expr: sp.Expr) -> sp.Expr:
    """Avoid global simplify on large lifted rational expressions."""
    if _has_compound_symbolic_power(expr):
        return expr
    try:
        if sp.count_ops(expr, visual=False) > _RATIONAL_LIFT_SIMPLIFY_MAX_OPS:
            return expr
    except (TypeError, ValueError):
        return expr
    return sp.simplify(expr)


def _bounded_composite_lift_simplify(expr: sp.Expr) -> sp.Expr:
    """Avoid global simplify on large lifted composite-function expressions."""
    if _has_compound_symbolic_power(expr):
        return expr
    try:
        if sp.count_ops(expr, visual=False) > _COMPOSITE_LIFT_SIMPLIFY_MAX_OPS:
            return expr
    except (TypeError, ValueError):
        return expr
    return sp.simplify(expr)


def _bounded_composite_substitute(
    expr: sp.Expr,
    substitutions: dict[sp.Expr, sp.Expr],
) -> sp.Expr:
    """Substitute composite auxiliaries without expensive global matching."""
    if not substitutions:
        return expr

    try:
        if (
            sp.count_ops(expr, visual=False) <= _COMPOSITE_LIFT_SIMPLIFY_MAX_OPS
            and not _has_compound_symbolic_power(expr)
        ):
            return expr.subs(substitutions)
    except (TypeError, ValueError):
        pass

    replaced = expr.xreplace(substitutions)
    if _has_compound_symbolic_power(replaced):
        return replaced
    try:
        if sp.count_ops(replaced, visual=False) > _COMPOSITE_LIFT_SIMPLIFY_MAX_OPS:
            return replaced
    except (TypeError, ValueError):
        return replaced
    return replaced.subs(substitutions)


def find_rational_denominators(expr: sp.Expr) -> set[sp.Expr]:
    """
    Find all unique non-trivial denominators in an expression.
    Returns set of denominator expressions that need auxiliary variables.

    A denominator is "non-trivial" if it's not:
    - A constant
    - A single variable (already power-law)
    """
    denoms = set()

    def visit(e):
        if isinstance(e, sp.Pow):
            base, exp = e.args
            # Check for negative exponents (divisions)
            if exp.is_number and float(exp) < 0:
                # If base is not a simple symbol, it needs lifting
                if not isinstance(base, sp.Symbol):
                    denoms.add(base)
            # Recurse into base
            visit(base)
        elif isinstance(e, (sp.Add, sp.Mul)):
            for arg in e.args:
                visit(arg)

    visit(expr)
    return denoms


def find_composite_functions(expr: sp.Expr) -> set[sp.Expr]:
    """
    Find all composite functions (non-algebraic functions) in an expression.
    Returns set of function applications that need auxiliary variables.

    A composite function is any sympy function application (exp, sin, log, etc.)
    that is not a simple algebraic operation (Add, Mul, Pow with numeric exponent).
    """
    functions = set()

    def visit(e):
        # Check if this is a function application
        if isinstance(e, sp.Function):
            # This is a function like exp(X), sin(X), etc.
            functions.add(e)
            # Also recurse into arguments
            for arg in e.args:
                visit(arg)
        elif isinstance(e, sp.Pow):
            # Recurse into base and exponent
            visit(e.args[0])
            if not e.args[1].is_number:
                visit(e.args[1])
        elif isinstance(e, (sp.Add, sp.Mul)):
            for arg in e.args:
                visit(arg)
        # Note: we don't add Symbol, Number, or Pow with numeric exponent

    visit(expr)
    return functions


def find_sqrt_of_sums(expr: sp.Expr) -> set[sp.Expr]:
    """
    Find all sqrt(sum) patterns in an expression.

    These are Pow(base, exp) where:
    - exp is 0.5 or 1/2 (square root)
    - base is an Add (sum of terms)

    Such expressions are NOT monomials and need to be lifted to auxiliary variables.
    Returns set of sqrt(sum) expressions that need auxiliary variables.
    """
    sqrt_sums = set()

    def visit(e):
        if isinstance(e, sp.Pow):
            base, exp = e.args
            # Check if this is a square root (exp = 0.5 or 1/2)
            is_sqrt = False
            if exp.is_number:
                try:
                    exp_val = float(exp)
                    is_sqrt = abs(exp_val - 0.5) < 1e-10
                except (TypeError, ValueError):
                    pass
            elif exp == sp.Rational(1, 2):
                is_sqrt = True

            # Check if base is a sum (Add)
            if is_sqrt and base.is_Add:
                sqrt_sums.add(e)

            # Recurse into base
            visit(base)
        elif isinstance(e, sp.Function):
            # Recurse into function arguments
            for arg in e.args:
                visit(arg)
        elif isinstance(e, (sp.Add, sp.Mul)):
            for arg in e.args:
                visit(arg)

    visit(expr)
    return sqrt_sums


def create_auxiliary_for_denominator(
    denom: sp.Expr, var_odes: dict[sp.Symbol, sp.Expr], aux_counter: int, prefix: str = "W"
) -> tuple[sp.Symbol, sp.Expr]:
    """
    Create auxiliary W = 1/denom and compute W' via chain rule.

    For W = 1/D(X):
        W' = -W^2 * dD/dt
    where dD/dt = sum_i (∂D/∂X_i) * dX_i/dt

    Returns: (W_symbol, W_ode)
    """
    # Create auxiliary symbol
    W = sp.symbols(f"{prefix}_{aux_counter}", positive=True)

    # Compute dD/dt using chain rule
    denom_prime = sp.Integer(0)
    for var, var_ode in var_odes.items():
        if var in denom.free_symbols:
            partial = sp.diff(denom, var)
            denom_prime += partial * var_ode

    # W' = -W^2 * dD/dt
    W_ode = -(W**2) * denom_prime
    W_ode = sp.simplify(W_ode)

    return W, W_ode


def _is_composite_function_expr(expr: sp.Expr) -> bool:
    """
    Check if expression is a composite function (exp, log, sin, etc.) or contains one.
    Returns True if expr is or contains a function application.
    """
    if isinstance(expr, sp.Function):
        # Direct function application like exp(X), log(Y), sin(Z)
        return True
    if isinstance(expr, (sp.Add, sp.Mul)):
        # Check if any subexpression is a function
        for arg in expr.args:
            if _is_composite_function_expr(arg):
                return True
    if isinstance(expr, sp.Pow):
        # Check base (exponent can be numeric or symbolic parameter)
        if _is_composite_function_expr(expr.args[0]):
            return True
    return False


def lift_rational_functions(
    sym: SymSystem, composite_aux_defs: dict[sp.Symbol, sp.Expr] | None = None
) -> tuple[SymSystem, dict[sp.Symbol, sp.Expr]]:
    """
    Augment system with auxiliary variables for all rational terms.

    Returns:
        Tuple of (augmented SymSystem, auxiliary_defs dict mapping Y -> definition)

    Strategy:
    1. First substitute all constant denominators with their numeric values
    2. Then lift dynamic denominators that depend on state variables
    3. Recursively repeat until no more rational functions remain
    4. CRITICAL: Skip denominators that are:
       a. Simple symbols (Z, Z_1, etc.) - use negative exponents directly
       b. Composite functions (log(Z_1), exp(Z), etc.) - already power-law compatible

    For each unique non-trivial denominator D(X):
    - If D depends only on constants: substitute its numeric value directly
    - If D is a simple symbol or composite function: SKIP - use negative exponent directly
    - If D is a complex algebraic expression:
       a. Create auxiliary Y = D (denominator itself)
       b. Add ODE: Y' = dD/dt
       c. Replace D with Y in all ODEs, use Y^(-1) for 1/D
       d. Set Y(0) = D(X(0))

    This produces exact S-system form with negative exponents.

    Returns augmented SymSystem with rational terms eliminated.

    Args:
        sym: System to lift
        composite_aux_defs: Definitions of composite function auxiliaries (to avoid re-lifting)
    """
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    aux_counter = 1

    # Accumulate ALL auxiliary definitions across iterations
    all_aux_defs: dict[sp.Symbol, sp.Expr] = {}

    # Track which expressions are already lifted composite auxiliaries
    # Build a set of lifted auxiliary expressions for fast checking
    lifted_aux_exprs = set()
    if composite_aux_defs:
        for aux, defn in composite_aux_defs.items():
            lifted_aux_exprs.add(defn)

    while iteration < max_iterations:
        iteration += 1

        # Find all unique denominators across all ODEs
        all_denoms = set()
        for var, ode in sym.odes.items():
            denoms = find_rational_denominators(ode)
            all_denoms.update(denoms)

        if not all_denoms:
            # No more rational functions to lift
            break

        # Separate denominators into constant vs. dynamic vs. simple state variables
        state_vars = set(sym.vars)
        assignment_rule_names = set(sym.assignment_rules)
        const_denoms = set()  # denominators that depend only on constants
        dynamic_denoms = set()  # denominators that depend on state variables AND need lifting

        for denom in all_denoms:
            denom_vars = denom.free_symbols & state_vars
            denom_assignment_names = {str(sym_) for sym_ in denom.free_symbols} & assignment_rule_names
            if denom_assignment_names:
                # Assignment-rule targets are algebraic functions, not constants.
                # SBML parsing may also record placeholder numeric values for
                # them in params; using those placeholders would erase real
                # denominators such as 1 + f_Nstar__k_cnd_D.
                continue
            if not denom_vars:
                # Denominator has no state variables - it's constant
                const_denoms.add(denom)
            elif isinstance(denom, sp.Symbol):
                # Denominator is a simple symbol (state variable or parameter)
                # S-systems naturally support negative exponents like Z^(-1) or Z_1^(-1)
                # Skip this denominator - it will remain as a negative exponent
                # This applies to both original variables AND lifted auxiliaries
                continue
            elif _is_composite_function_expr(denom):
                # Denominator is or contains a composite function (log(Z), exp(X), etc.)
                # These are power-law compatible through negative exponents
                # Skip - use negative exponent directly (e.g., log(Z_1)^-1)
                continue
            else:
                # Denominator is a complex algebraic expression - needs lifting
                dynamic_denoms.add(denom)

        # First, substitute constant denominators with their numeric values
        new_odes: dict[sp.Symbol, sp.Expr] = {}
        for var, ode in sym.odes.items():
            new_ode = ode
            # Substitute constant denominators directly with their reciprocal values
            for denom in const_denoms:
                # Evaluate denominator numerically
                denom_val = denom
                for param_sym in list(denom.free_symbols):
                    if param_sym.name in sym.params:
                        denom_val = denom_val.subs(param_sym, sym.params[param_sym.name])
                try:
                    recip_val = float(1.0 / denom_val)
                    replacements = {
                        denom ** (-n): sp.Float(recip_val**n)
                        for n in range(1, 6)
                    }
                    new_ode = new_ode.xreplace(replacements)
                except (TypeError, ValueError, ZeroDivisionError):
                    # If numeric evaluation fails, leave the denominator symbolic.
                    continue
            new_odes[var] = new_ode

        # Now create auxiliary symbols only for dynamic denominators
        # Y = D (denominator itself, not reciprocal)
        # CRITICAL: Skip denominators that are already lifted composite auxiliaries
        # ALSO: Skip denominators that already have an auxiliary from previous iterations
        denom_to_aux: dict[sp.Expr, sp.Symbol] = {}

        # Build reverse lookup: normalized_denom -> existing auxiliary
        existing_denom_to_aux: dict[sp.Expr, sp.Symbol] = {}
        for aux, defn in all_aux_defs.items():
            existing_denom_to_aux[defn] = aux

        for denom in sorted(dynamic_denoms, key=str):
            # Check if this denominator is already a lifted composite auxiliary

            if denom in lifted_aux_exprs:
                # This denominator is already a lifted auxiliary - SKIP
                # Use negative exponent directly (e.g., log(Z_1)^-1)
                continue

            # Check if we already have an auxiliary for this denominator (from previous iteration)
            if denom in existing_denom_to_aux:
                # Reuse existing auxiliary
                denom_to_aux[denom] = existing_denom_to_aux[denom]
                continue

            # Not a lifted auxiliary and no existing auxiliary - create new Y
            Y = sp.symbols(f"Y_{aux_counter}", positive=True)
            denom_to_aux[denom] = Y
            all_aux_defs[Y] = denom  # Accumulate definitions
            existing_denom_to_aux[denom] = Y  # Track for future denoms in this iteration
            aux_counter += 1

        # Substitute dynamic denominators with auxiliaries
        # ONLY replace when appearing as negative powers (denominators)
        # CRITICAL FIX: Handle ALL powers of denom (including fractional like -0.5)
        for var in sym.vars:
            new_ode = new_odes[var]
            for denom, Y in denom_to_aux.items():
                # Find all Pow atoms and check if their base matches denom
                for atom in list(new_ode.atoms(sp.Pow)):
                    base, exp = atom.as_base_exp()
                    # Denominators are collected from Pow bases, so structural
                    # identity avoids expensive algebraic matching here.
                    if base == denom:
                        # Replace denom^exp with Y^exp
                        new_ode = new_ode.subs(atom, Y**exp)
            new_odes[var] = _bounded_rational_lift_simplify(new_ode)

        # Compute Y' for dynamic auxiliaries using the LIFTED ODEs
        # Y' = dD/dt (direct derivative, no chain rule needed)
        new_aux_odes: dict[sp.Symbol, sp.Expr] = {}
        for denom, Y in denom_to_aux.items():
            # Compute dD/dt using the lifted ODEs
            denom_prime = sp.Integer(0)
            for var in sym.vars:
                if var in denom.free_symbols:
                    partial = sp.diff(denom, var)
                    # Use the NEW (lifted) ODE for var
                    denom_prime += partial * new_odes[var]

            # denom_prime is already computed from lifted ODEs (which have Y in them)
            # No additional substitution needed - it would cause spurious replacements
            Y_ode = denom_prime
            new_aux_odes[Y] = _bounded_rational_lift_simplify(Y_ode)

        # Combine original and auxiliary ODEs
        combined_odes = {**new_odes, **new_aux_odes}

        # Compute initial conditions for auxiliaries
        new_initials = dict(sym.initials)

        # Build name-based lookup for initial conditions to handle symbol object mismatch
        # (Different Symbol objects with same name won't match in dict lookup)
        # CRITICAL: SBML parser may put species ICs in params instead of initials
        # So we check BOTH sources, with initials taking precedence
        initials_by_name = {}
        # First add params (lower priority)
        for k, v in sym.params.items():
            initials_by_name[k] = v
        # Then add initials (higher priority - overwrites params)
        for k, v in sym.initials.items():
            if isinstance(k, sp.Symbol):
                initials_by_name[str(k)] = v

        for denom, Y in denom_to_aux.items():
            # Evaluate denominator at t=0
            denom_at_0 = denom
            # First substitute state variables - use name-based lookup
            for var in sym.vars:
                if var in denom.free_symbols:
                    var_name = str(var)
                    ic_value = initials_by_name.get(var_name, 1.0)
                    denom_at_0 = denom_at_0.subs(var, ic_value)
            # Then substitute parameters - use actual symbols from expression
            for param_sym in denom_at_0.free_symbols:
                param_name = param_sym.name
                if param_name in sym.params:
                    denom_at_0 = denom_at_0.subs(param_sym, sym.params[param_name])
            # Y(0) = D(X(0))
            try:
                Y_init = float(denom_at_0)
            except (TypeError, ValueError, sp.SympifyError):
                Y_init = 1.0  # Fallback if evaluation fails
            new_initials[Y] = Y_init

        # Create new variable list: keep original vars, add Y auxiliaries
        new_vars = list(sym.vars) + list(denom_to_aux.values())

        # Update sym for next iteration (preserve all metadata)
        sym = SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=combined_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,  # Propagate symbolic IC expressions
            assignment_rules=sym.assignment_rules,  # Preserve original assignment rules
            compartments=sym.compartments,  # Propagate compartments
            sim_t_start=sym.sim_t_start,  # Propagate sim metadata
            sim_t_end=sym.sim_t_end,
            sim_n_steps=sym.sim_n_steps,
            eps_init=sym.eps_init,
            eps_slack=sym.eps_slack,
        )

    # Return final system and ALL accumulated auxiliary definitions
    return sym, all_aux_defs


def add_dummy_for_constants(sym: SymSystem) -> tuple[SymSystem, dict[sp.Symbol, sp.Expr]]:
    """
    Add dummy auxiliary variable for equations with constant terms.

    S-systems cannot represent constant terms directly. This function transforms:
        X' = C + other_terms
    Into:
        X' = C * dummy^0 + other_terms
        dummy' = 0
        dummy(0) = 1

    Since dummy^0 = 1 for all time, this preserves the mathematical equivalence
    while expressing the constant in power-law form.

    This approach follows Voit's literature on S-system recasting.

    Returns:
        Tuple of (augmented SymSystem, auxiliary_defs dict mapping dummy -> 1)
    """
    # Identify variables with constant terms
    constant_terms = {}  # {variable: constant_value}
    for var in sym.vars:
        ode = sym.odes[var]
        terms = expand_to_terms(sp.expand(ode))
        for term in terms:
            if term.is_Number and term != 0:
                # Found a non-zero constant term
                constant_terms[var] = term
                break  # Only expect one constant per equation

    if not constant_terms:
        # No constant terms - return unchanged
        return sym, {}

    # Create dummy auxiliary variable
    dummy = sp.symbols("dummy_const", positive=True)

    # Transform ODEs: replace constant C with C * dummy^0
    new_odes = {}
    for var in sym.vars:
        old_ode = sym.odes[var]

        if var in constant_terms:
            # This variable has a constant term to replace
            const_value = constant_terms[var]

            # Expand and process each term
            terms = expand_to_terms(sp.expand(old_ode))
            new_terms = []
            const_replaced = False

            for term in terms:
                if term.is_Number and term != 0 and term == const_value and not const_replaced:
                    # Replace first occurrence of constant with C * dummy^0
                    # Use Pow with evaluate=False to prevent sympy from simplifying dummy^0 to 1
                    new_terms.append(
                        sp.Mul(const_value, sp.Pow(dummy, 0, evaluate=False), evaluate=False)
                    )
                    const_replaced = True
                else:
                    new_terms.append(term)

            # Use sp.Add with evaluate=False to prevent evaluation of dummy^0
            if len(new_terms) == 0:
                new_odes[var] = sp.Integer(0)
            elif len(new_terms) == 1:
                new_odes[var] = new_terms[0]
            else:
                new_odes[var] = sp.Add(*new_terms, evaluate=False)
        else:
            # No constant term - keep as is
            new_odes[var] = old_ode

    # IMPORTANT: Do NOT add dummy' = 0 as an ODE - it causes GMA classification
    # Instead, treat dummy_const as a PARAMETER (constant value = 1)
    # This way X' = C * dummy_const^0 simplifies correctly since dummy_const = 1

    # Keep original initials (don't add dummy as a state variable)
    new_initials = dict(sym.initials)

    # Keep original variable list (don't add dummy)
    new_vars = list(sym.vars)

    # Add dummy_const = 1 as a parameter
    new_params = dict(sym.params)
    new_params["dummy_const"] = 1.0

    # Auxiliary definition: dummy is constant = 1
    aux_defs = {dummy: sp.Integer(1)}

    return (
        SymSystem(
            vars=new_vars,
            params=new_params,
            odes=new_odes,
            initials=new_initials,
            compartments=sym.compartments,  # Propagate compartments
            sim_t_start=sym.sim_t_start,  # Propagate sim metadata
            sim_t_end=sym.sim_t_end,
            sim_n_steps=sym.sim_n_steps,
            eps_init=sym.eps_init,
            eps_slack=sym.eps_slack,
        ),
        aux_defs,
    )


def _build_composite_inverse_mappings(
    func_to_aux: dict[sp.Expr, sp.Symbol],
    func_to_offset: dict[sp.Expr, float],
    original_vars: list[sp.Symbol],
) -> dict[sp.Expr, sp.Expr]:
    """
    Build comprehensive inverse mappings for nested composite functions.

    This handles cases like:
    - If Z_1 = exp(Z_2^2) and Z_2 = log(Z), then log(Z_1) = Z_2^2
    - If Z_2 = log(Z), then Z = exp(Z_2), 1/Z = exp(-Z_2), Z^(-n) = exp(-n*Z_2)

    Args:
        func_to_aux: Mapping from composite functions to their auxiliary symbols
        func_to_offset: Mapping from functions to their offsets (for sin/cos)
        original_vars: List of original variable symbols

    Returns:
        Dictionary mapping composite expressions to their simplified forms
    """
    inverse_map: dict[sp.Expr, sp.Expr] = {}

    # Build mappings for each auxiliary variable
    for func, aux_sym in func_to_aux.items():
        offset = func_to_offset.get(func, 0.0)

        # Handle exp functions: if aux = exp(arg), then log(aux) = arg AND exp(arg) = aux
        if func.func == sp.exp and offset == 0:
            arg = func.args[0]
            # CRITICAL: Add forward mapping: exp(arg) -> aux
            # This allows us to recognize exp(Z_2^2) as Z_1 directly
            inverse_map[func] = aux_sym

            # log(aux) = arg
            inverse_map[sp.log(aux_sym)] = arg

            # If arg is another auxiliary or expression, try to expand further
            # For example: if Z_1 = exp(Z_2^2), then log(Z_1) = Z_2^2
            # This happens automatically since arg = Z_2^2

        # Handle log functions: if aux = log(var), then exp(aux) = var and 1/var = exp(-aux)
        elif func.func == sp.log and offset == 0:
            arg = func.args[0]

            # Check if arg is an original variable (single symbol)
            if isinstance(arg, sp.Symbol) and arg in original_vars:
                # aux = log(var) => var = exp(aux)
                inverse_map[arg] = sp.exp(aux_sym)

                # CRITICAL: Add all power forms of the original variable
                # var^(-1) = exp(-aux)
                inverse_map[arg ** (-1)] = sp.exp(-aux_sym)
                # Also handle 1/var explicitly (sympy might not always convert to Pow)
                inverse_map[1 / arg] = sp.exp(-aux_sym)

                # Add common negative powers: var^(-2), var^(-3), etc.
                for n in range(2, 6):
                    inverse_map[arg ** (-n)] = sp.exp(-n * aux_sym)

    # Handle nested cases: if we have both Z_1 = exp(f(Z_2)) and Z_2 = log(Z)
    # Then we need to recognize that log(Z_1) should be expressed in terms of Z_2
    for func1, _aux1 in func_to_aux.items():
        if func1.func == sp.exp and func_to_offset.get(func1, 0.0) == 0:
            arg1 = func1.args[0]
            # Check if arg1 contains other auxiliaries
            for _func2, aux2 in func_to_aux.items():
                if aux2 in arg1.free_symbols:
                    # arg1 contains aux2
                    # So aux1 = exp(expr(aux2))
                    # Therefore log(aux1) = expr(aux2)
                    # We already have inverse_map[log(aux1)] = arg1
                    # which is correct since arg1 = expr(aux2)
                    pass

    return inverse_map


def _requires_positivity_transform(func: sp.Expr) -> tuple[bool, float]:
    """
    Check if function requires positivity transformation (X = Z + c).

    Sign-changing functions like sin and cos need offset to ensure positivity
    for power-law representation.

    Returns: (needs_transform, offset_amount)
    """
    if func.func == sp.sin:
        # sin(x) ∈ [-1, 1] → add 2 → [1, 3]
        return True, 2.0
    if func.func == sp.cos:
        # cos(x) ∈ [-1, 1] → add 2 → [1, 3]
        return True, 2.0
    # Other functions (exp, log) are positive for positive args - no offset needed
    return False, 0.0


# =============================================================================
# AUTONOMOUS LIFTING: Convert time-dependent functions to state variables
# =============================================================================


@dataclass
class AutonomousLiftResult:
    """Result of autonomous lifting for a time-dependent expression."""

    new_vars: list[sp.Symbol]  # New state variables to add
    new_odes: dict[sp.Symbol, sp.Expr]  # ODEs for new variables
    new_initials: dict[sp.Symbol, sp.Expr]  # Initial conditions (symbolic)
    substitution: sp.Expr  # Expression to substitute for original function
    aux_defs: dict[sp.Symbol, sp.Expr]  # Auxiliary definitions for documentation


def _detect_exp_decay_pattern(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr] | None:
    """
    Detect exponential decay pattern: exp(-k*time) or exp(k*time) where k is constant.

    Returns: (coefficient, time_coeff) where expr = exp(time_coeff * time)
             or None if not matching pattern.

    Examples:
        exp(-k_0 * time) → (1, -k_0)
        exp(-0.5 * time) → (1, -0.5)
        2*exp(-k*time) → (2, -k)
    """
    time_sym = sp.Symbol("time")

    # Handle case where expr is just exp(...)
    if expr.func == sp.exp:
        arg = expr.args[0]
        # Check if arg is linear in time: coeff * time
        if arg.is_Mul and time_sym in arg.free_symbols:
            # Extract coefficient of time
            time_coeff = arg / time_sym
            # Check that time_coeff doesn't contain time
            if time_sym not in time_coeff.free_symbols:
                return (sp.Integer(1), sp.simplify(time_coeff))
        elif arg == time_sym:
            return (sp.Integer(1), sp.Integer(1))
        elif arg == -time_sym:
            return (sp.Integer(1), sp.Integer(-1))

    return None


def _detect_harmonic_pattern(expr: sp.Expr) -> tuple[str, sp.Expr, sp.Expr] | None:
    """
    Detect harmonic pattern: cos(ω*time + φ) or sin(ω*time + φ)

    Returns: (func_type, omega, phase) where func_type is 'cos' or 'sin'
             or None if not matching pattern.

    Examples:
        cos(2*pi*time/30) → ('cos', pi/15, 0)
        sin(omega*time) → ('sin', omega, 0)
        cos(pi*time/15) → ('cos', pi/15, 0)
    """
    time_sym = sp.Symbol("time")

    # Check if this is cos or sin
    if expr.func == sp.cos:
        func_type = "cos"
    elif expr.func == sp.sin:
        func_type = "sin"
    else:
        return None

    arg = expr.args[0]

    # Check if arg contains time
    if time_sym not in arg.free_symbols:
        return None

    # Try to decompose arg = omega * time + phase
    # Collect coefficients of time
    arg_expanded = sp.expand(arg)

    # Get coefficient of time (omega) and constant term (phase)
    omega = arg_expanded.diff(time_sym)
    if time_sym in omega.free_symbols:
        # omega shouldn't depend on time for linear case
        return None

    # Compute phase: arg - omega*time at time=0
    phase = arg_expanded.subs(time_sym, 0)

    return (func_type, sp.simplify(omega), sp.simplify(phase))


def _detect_tanh_sigmoid_pattern(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr] | None:
    """
    Detect tanh sigmoid pattern: tanh(k*(time - a)) or tanh(k*(a - time))

    Returns: (k, a) where expr = tanh(k*(time - a)) or tanh(k*(a - time))
             or None if not matching pattern.

    Note: tanh(x) = 2*sigmoid(2x) - 1, where sigmoid(x) = 1/(1+exp(-x))
    We lift to logistic form: h' = 2k*h - 2k*h² for h = sigmoid(2k*(t-a))

    Examples:
        tanh(k_steep*(time - 5)) → (k_steep, 5) [increasing sigmoid]
        tanh(k_steep*(70 - time)) → (-k_steep, 70) [decreasing sigmoid]
    """
    time_sym = sp.Symbol("time")

    if not hasattr(expr, "func") or expr.func != sp.tanh:
        return None

    arg = expr.args[0]

    # Check if arg contains time
    if time_sym not in arg.free_symbols:
        return None

    # Try to decompose arg = k * (time - a) or k * (a - time)
    # arg should be linear in time
    arg_expanded = sp.expand(arg)

    # Get coefficient of time (this is k or -k)
    k_coeff = arg_expanded.diff(time_sym)
    if time_sym in k_coeff.free_symbols:
        return None  # Not linear in time

    # Compute constant term: arg at time=0
    const_term = arg_expanded.subs(time_sym, 0)

    # arg = k_coeff * time + const_term
    # For tanh(k*(time - a)): k_coeff = k, const_term = -k*a → a = -const_term/k
    # For tanh(k*(a - time)): k_coeff = -k, const_term = k*a → a = const_term/(-k_coeff) = -const_term/k_coeff

    if k_coeff == 0:
        return None

    # a = -const_term / k_coeff
    a = sp.simplify(-const_term / k_coeff)

    return (k_coeff, a)


def lift_exp_decay(
    expr: sp.Expr, aux_counter: int, params: dict[str, float]
) -> AutonomousLiftResult | None:
    """
    Lift exponential decay exp(-k*time) to autonomous ODE.

    For E = exp(c*time) where c is the time coefficient:
        E' = c * E
        E(0) = 1

    This is already in GMA form (single term).
    """
    pattern = _detect_exp_decay_pattern(expr)
    if pattern is None:
        return None

    outer_coeff, time_coeff = pattern

    # Create new state variable
    E = sp.Symbol(f"E_{aux_counter}", positive=True)

    # ODE: E' = time_coeff * E
    E_ode = time_coeff * E

    # Initial condition: E(0) = exp(time_coeff * 0) = 1
    E_init = sp.Integer(1)

    return AutonomousLiftResult(
        new_vars=[E],
        new_odes={E: E_ode},
        new_initials={E: E_init},
        substitution=outer_coeff * E,  # exp(-k*t) → E
        aux_defs={E: expr},
    )


def lift_harmonic(
    expr: sp.Expr,
    aux_counter: int,
    params: dict[str, float],
    existing_harmonics: dict[sp.Expr, tuple[sp.Symbol, sp.Symbol]] | None = None,
) -> AutonomousLiftResult | None:
    """
    Lift harmonic function cos(ω*time + φ) or sin(ω*time + φ) to autonomous ODEs.

    For coupled oscillator:
        c' = -ω * s
        s' = ω * c
        c(0) = cos(φ)
        s(0) = sin(φ)

    where c = cos(ω*time + φ), s = sin(ω*time + φ)

    This is GMA form (single term per ODE).

    Args:
        existing_harmonics: Dict mapping omega -> (c_symbol, s_symbol) for reuse
    """
    pattern = _detect_harmonic_pattern(expr)
    if pattern is None:
        return None

    func_type, omega, phase = pattern

    # Check if we already have this omega (can reuse oscillator)
    if existing_harmonics and omega in existing_harmonics:
        c_sym, s_sym = existing_harmonics[omega]
        if func_type == "cos":
            # cos(ω*t + φ) = cos(ω*t)cos(φ) - sin(ω*t)sin(φ)
            if phase == 0:
                return AutonomousLiftResult(
                    new_vars=[], new_odes={}, new_initials={}, substitution=c_sym, aux_defs={}
                )
            else:
                return AutonomousLiftResult(
                    new_vars=[],
                    new_odes={},
                    new_initials={},
                    substitution=c_sym * sp.cos(phase) - s_sym * sp.sin(phase),
                    aux_defs={},
                )
        else:  # sin
            # sin(ω*t + φ) = sin(ω*t)cos(φ) + cos(ω*t)sin(φ)
            if phase == 0:
                return AutonomousLiftResult(
                    new_vars=[], new_odes={}, new_initials={}, substitution=s_sym, aux_defs={}
                )
            else:
                return AutonomousLiftResult(
                    new_vars=[],
                    new_odes={},
                    new_initials={},
                    substitution=s_sym * sp.cos(phase) + c_sym * sp.sin(phase),
                    aux_defs={},
                )

    # Create new coupled oscillator
    c_sym = sp.Symbol(f"c_{aux_counter}", positive=True)
    s_sym = sp.Symbol(f"s_{aux_counter}", positive=True)

    # ODEs: c' = -ω*s, s' = ω*c (GMA form)
    c_ode = -omega * s_sym
    s_ode = omega * c_sym

    # Initial conditions for oscillator with phase
    # c(0) = cos(φ), s(0) = sin(φ)
    c_init = sp.cos(phase)
    s_init = sp.sin(phase)

    # Determine substitution based on function type
    if func_type == "cos":
        substitution = c_sym
    else:  # sin
        substitution = s_sym

    return AutonomousLiftResult(
        new_vars=[c_sym, s_sym],
        new_odes={c_sym: c_ode, s_sym: s_ode},
        new_initials={c_sym: c_init, s_sym: s_init},
        substitution=substitution,
        aux_defs={
            c_sym: sp.cos(omega * sp.Symbol("time") + phase),
            s_sym: sp.sin(omega * sp.Symbol("time") + phase),
        },
    )


# Perturbation constant for logistic ICs near fixed points
EPS_LOGISTIC = 1e-2


def lift_tanh_sigmoid(
    expr: sp.Expr, aux_counter: int, params: dict[str, float]
) -> AutonomousLiftResult | None:
    """
    Lift tanh sigmoid to autonomous logistic ODE.

    For tanh(k*(time - a)):
        Let h = sigmoid(2k*(t-a)) = 1/(1 + exp(-2k*(t-a)))
        Then tanh(k*(t-a)) = 2*h - 1

    The logistic equation is:
        h' = 2k * h * (1 - h) = 2k*h - 2k*h²
        h(0) = 1/(1 + exp(2k*a))

    This is GMA form (two terms: growth and decay).

    CRITICAL: h=0 and h=1 are fixed points of the logistic equation.
    If h(0) is at or very near a fixed point, the dynamics don't evolve.
    We perturb ICs away from fixed points to ensure proper gate dynamics.

    For tanh(k*(a - time)) (decreasing sigmoid):
        This is -tanh(k*(time - a)) = 1 - 2*h
        where h follows the same logistic equation
    """
    pattern = _detect_tanh_sigmoid_pattern(expr)
    if pattern is None:
        return None

    k, a = pattern

    # Create new state variable for logistic
    h = sp.Symbol(f"h_{aux_counter}", positive=True)

    # Determine if this is increasing (k > 0) or decreasing (k < 0)
    # ODE: h' = 2|k|*h - 2|k|*h² (always positive rate constant for GMA)
    # The sign of k determines direction of sigmoid

    # ODE coefficient: we use the absolute value of k for the rate
    # h' = 2*|k|*h*(1-h) but k already encodes direction in the substitution
    # Actually, for correct dynamics:
    # If k > 0: h increases from 0 to 1 as t increases (standard logistic)
    # If k < 0: h decays from h(0) toward 0

    # For standard logistic with rate coefficient r:
    # h' = r*h*(1-h) = r*h - r*h²
    # Here r = 2*k (where k is the coefficient of time in tanh argument)

    # The ODE is: h' = 2*k*h - 2*k*h²
    # When k > 0: h grows from h(0) toward 1
    # When k < 0: h decays from h(0) toward 0
    rate = 2 * k
    h_ode = rate * h - rate * h**2

    # Initial condition: h(0) = 1/(1 + exp(2*k*a))
    # Note: exp(2*k*a) where a = time offset
    #
    # CRITICAL: h=0 and h=1 are fixed points of h' = r*h*(1-h).
    # If h_init is at or very near a fixed point, the gate never moves.
    # The clamping away from fixed points is applied AFTER numeric evaluation
    # in lift_time_functions_to_autonomous() using EPS_LOGISTIC.
    # Here we just store the exact symbolic expression.
    h_init = 1 / (1 + sp.exp(2 * k * a))

    # Substitution: tanh(k*(t-a)) = 2*h - 1
    substitution = 2 * h - 1

    return AutonomousLiftResult(
        new_vars=[h],
        new_odes={h: h_ode},
        new_initials={h: h_init},
        substitution=substitution,
        aux_defs={h: (1 + expr) / 2},  # h = (1 + tanh(...))/2 = sigmoid(2*arg)
    )


def _detect_sqrt_of_squared_pattern(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr] | None:
    """
    Detect sqrt(X² + c) pattern for squared variable lifting.

    Returns: (X, c) where expr = sqrt(X² + c)
             or None if not matching pattern.

    This handles smooth ReLU approximations like:
        sqrt(raw² + ε²)
        sqrt(X² + 0.01)

    Examples:
        sqrt(raw^2 + eps_k^2) → (raw, eps_k^2)
        sqrt(X^2 + 1) → (X, 1)
    """
    # Check if this is a square root: Pow(base, 0.5) or Pow(base, 1/2)
    if not isinstance(expr, sp.Pow):
        return None

    base, exp = expr.args

    # Check if exponent is 0.5
    is_sqrt = False
    if exp.is_number:
        try:
            exp_val = float(exp)
            is_sqrt = abs(exp_val - 0.5) < 1e-10
        except (TypeError, ValueError):
            pass
    elif exp == sp.Rational(1, 2):
        is_sqrt = True

    if not is_sqrt:
        return None

    # Check if base is X² + c (a sum with a squared term and a constant)
    if not base.is_Add:
        return None

    # Expand and look for pattern
    base_expanded = sp.expand(base)

    # Collect terms: look for X² and constants
    squared_term = None
    constant = sp.Integer(0)
    other_terms = []

    for term in base_expanded.as_ordered_terms():
        # Check if term is X² (a symbol squared)
        if isinstance(term, sp.Pow):
            term_base, term_exp = term.args
            if term_exp == 2:
                # Found X²
                if squared_term is None:
                    squared_term = term_base
                else:
                    # Multiple squared terms - not our pattern
                    other_terms.append(term)
        elif term.is_number:
            constant += term
        elif term.is_Mul:
            # Check if it's coeff * X²
            has_square = False
            for factor in term.args:
                if isinstance(factor, sp.Pow) and factor.args[1] == 2:
                    has_square = True
                    break
            if has_square:
                other_terms.append(term)
            else:
                # Could be a constant expression with parameters
                # Check if it contains any state variables
                # For now, treat as constant if no free symbols or only parameters
                other_terms.append(term)
        else:
            other_terms.append(term)

    if squared_term is None:
        return None

    if other_terms:
        # Has terms that don't fit pattern - not simple sqrt(X² + c)
        return None

    return (squared_term, constant)


def lift_squared_for_sqrt(
    expr: sp.Expr, aux_counter: int, sym: SymSystem
) -> AutonomousLiftResult | None:
    """
    Lift sqrt(X² + c) to squared variable u = X² + c with ODE.

    For u = X² + c:
        u' = 2*X*X'
        u(0) = X(0)² + c

    Then sqrt(X² + c) = u^(0.5) which is a GMA monomial.
    """
    pattern = _detect_sqrt_of_squared_pattern(expr)
    if pattern is None:
        return None

    X, c = pattern

    # Create new state variable for squared expression
    u = sp.Symbol(f"u_{aux_counter}", positive=True)

    # ODE: u' = 2*X*X'
    # Need to compute X' from the SymSystem
    # X might be a state variable or an expression involving state variables

    # If X is a state variable, use its ODE directly
    if isinstance(X, sp.Symbol) and X in sym.odes:
        X_prime = sym.odes[X]
    else:
        # X is an expression - compute X' via chain rule
        X_prime = sp.Integer(0)
        for var, ode in sym.odes.items():
            if var in X.free_symbols:
                partial = sp.diff(X, var)
                X_prime += partial * ode

    u_ode = 2 * X * X_prime

    # Initial condition: u(0) = X(0)² + c
    # Build name-based lookup for initial conditions (handles symbol object mismatch)
    initials_by_name = {}
    for s, v in sym.initials.items():
        if hasattr(s, 'name'):
            initials_by_name[s.name] = v

    X_at_0 = X
    # Substitute state variables by NAME (symbol identity may differ)
    for sym_in_X in list(X.free_symbols):
        if sym_in_X.name in initials_by_name:
            X_at_0 = X_at_0.subs(sym_in_X, initials_by_name[sym_in_X.name])
    # Substitute parameters
    for param_name, param_val in sym.params.items():
        X_at_0 = X_at_0.subs(sp.Symbol(param_name), param_val)

    # Evaluate c at t=0
    c_at_0 = c
    for param_name, param_val in sym.params.items():
        c_at_0 = c_at_0.subs(sp.Symbol(param_name), param_val)

    try:
        u_init = float(X_at_0) ** 2 + float(c_at_0)
    except (TypeError, ValueError):
        u_init = 1.0  # Fallback

    # Substitution: sqrt(X² + c) → u^(0.5)
    substitution = u ** sp.Rational(1, 2)

    return AutonomousLiftResult(
        new_vars=[u],
        new_odes={u: sp.simplify(u_ode)},
        new_initials={u: sp.Float(u_init)},
        substitution=substitution,
        aux_defs={u: X**2 + c},
    )


def lift_time_functions_to_autonomous(
    sym: SymSystem, aux_counter_start: int = 1
) -> tuple[SymSystem, dict[sp.Symbol, sp.Expr], int]:
    """
    Transform time-dependent systems to autonomous form using clock state.

    CLOCK APPROACH (correct):
    - Add clock state: T' = 1, T(0) = 0
    - Substitute `time` → `T` everywhere (ODEs and assignment rules)
    - Keep time-dependent expressions as assignment rules (not ODEs)

    This is mathematically exact and numerically robust.

    Also handles:
    - sqrt(X² + c) → squared variable ODE (Phase 2, state-dependent)

    Args:
        sym: SymSystem to transform
        aux_counter_start: Starting index for auxiliary variable names

    Returns:
        Tuple of (transformed SymSystem, auxiliary definitions, next aux counter)
    """
    # CRITICAL: Create time symbol with positive=True to match SBML parser
    time_sym = sp.Symbol("time", positive=True)

    # Build locals dict for sympify to avoid conflicts with SymPy reserved names
    # (e.g., Q, S, I, E, O, N are commonly used in biology but reserved in SymPy)
    sympify_locals: dict[str, sp.Symbol] = {"time": time_sym}
    for var in sym.vars:
        sympify_locals[var.name] = var
    for param_name in sym.params:
        sympify_locals[param_name] = sp.Symbol(param_name, positive=True)
    # Also include assignment rule names as symbols
    for rule_name in sym.assignment_rules:
        if rule_name not in sympify_locals:
            sympify_locals[rule_name] = sp.Symbol(rule_name, positive=True)

    # Helper function to check if ODE contains any symbol named "time"
    # (handles both Symbol("time") and Symbol("time", positive=True))
    def _contains_time(expr: sp.Expr) -> bool:
        return any(s.name == "time" for s in expr.free_symbols)

    # Check if system contains explicit time dependence
    has_time_dependence = False
    for var, ode in sym.odes.items():
        if _contains_time(ode):
            has_time_dependence = True
            break

    # Also check assignment rules for time dependence
    for rule_name, rule_expr_str in sym.assignment_rules.items():
        rule_expr = sp.sympify(rule_expr_str, locals=sympify_locals)
        if _contains_time(rule_expr):
            has_time_dependence = True
            break

    # Track auxiliary definitions
    all_aux_defs: dict[sp.Symbol, sp.Expr] = {}
    aux_counter = aux_counter_start

    # If time-dependent, add clock state T' = 1
    if has_time_dependence:
        T = sp.Symbol("T", positive=True)

        # Substitute time → T in all ODEs
        # Handle both Symbol("time") and Symbol("time", positive=True)
        new_odes: dict[sp.Symbol, sp.Expr] = {}
        for var, ode in sym.odes.items():
            new_ode = ode
            for s in list(ode.free_symbols):
                if s.name == "time":
                    new_ode = new_ode.subs(s, T)
            new_odes[var] = new_ode

        # Substitute time → T in assignment rules
        new_assignment_rules: dict[str, str] = {}
        for rule_name, rule_expr_str in sym.assignment_rules.items():
            # Simple string replacement for time → T
            new_rule = rule_expr_str.replace("time", "T")
            new_assignment_rules[rule_name] = new_rule

        # Add clock ODE: T' = 1
        new_odes[T] = sp.Integer(1)

        # Add clock IC: T(0) = 0
        new_initials = dict(sym.initials)
        new_initials[T] = 0.0

        # Add clock to variable list
        new_vars = list(sym.vars) + [T]

        # Document clock auxiliary
        all_aux_defs[T] = time_sym  # T represents time

        # Create updated system (preserve all metadata)
        sym = SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=new_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,
            assignment_rules=new_assignment_rules,
            compartments=sym.compartments,  # Propagate compartments
            sim_t_start=sym.sim_t_start,  # Propagate sim metadata
            sim_t_end=sym.sim_t_end,
            sim_n_steps=sym.sim_n_steps,
            eps_init=sym.eps_init,
            eps_slack=sym.eps_slack,
        )

    # Phase 2: Handle sqrt(X² + c) patterns for STATE-dependent expressions only
    # (time-dependent sqrt is handled via assignment rules with T substitution)
    sqrt_exprs: set[sp.Expr] = set()
    state_vars = set(sym.vars)

    for var, ode in sym.odes.items():
        for atom in ode.atoms(sp.Pow):
            if len(atom.args) == 2:
                base, exp = atom.args
                is_sqrt = False
                if exp.is_number:
                    try:
                        exp_val = float(exp)
                        is_sqrt = abs(exp_val - 0.5) < 1e-10
                    except (TypeError, ValueError):
                        pass
                elif exp == sp.Rational(1, 2):
                    is_sqrt = True

                if is_sqrt and base.is_Add:
                    # Check if this depends on state variables (not just T/time)
                    base_symbols = base.free_symbols
                    has_state_var = any(s in state_vars and s.name != "T" for s in base_symbols)
                    if has_state_var:
                        sqrt_exprs.add(atom)

    # Track which expressions have been lifted
    expr_to_sub: dict[sp.Expr, sp.Expr] = {}
    all_new_vars: list[sp.Symbol] = []
    all_new_odes: dict[sp.Symbol, sp.Expr] = {}
    all_new_initials: dict[sp.Symbol, sp.Expr] = {}

    # Process sqrt(X² + c) patterns
    for sqrt_expr in sorted(sqrt_exprs, key=str):
        if sqrt_expr in expr_to_sub:
            continue

        result = lift_squared_for_sqrt(sqrt_expr, aux_counter, sym)
        if result is not None:
            expr_to_sub[sqrt_expr] = result.substitution
            all_new_vars.extend(result.new_vars)
            all_new_odes.update(result.new_odes)
            all_new_initials.update(result.new_initials)
            all_aux_defs.update(result.aux_defs)
            aux_counter += len(result.new_vars)

    if not expr_to_sub and not has_time_dependence:
        # No patterns matched and no time dependence
        return sym, {}, aux_counter_start

    # Apply sqrt substitutions to ODEs
    if expr_to_sub:
        new_odes = {}
        for var, ode in sym.odes.items():
            new_ode = ode
            for expr, sub in expr_to_sub.items():
                new_ode = new_ode.subs(expr, sub)
            new_odes[var] = sp.simplify(new_ode)

        # Combine with new auxiliary ODEs
        combined_odes = {**new_odes, **all_new_odes}

        # Compute numeric initial conditions
        new_initials = dict(sym.initials)
        for var, init_expr in all_new_initials.items():
            init_val = init_expr
            for param_name, param_val in sym.params.items():
                for sym_in_expr in init_val.free_symbols:
                    if sym_in_expr.name == param_name:
                        init_val = init_val.subs(sym_in_expr, param_val)
            try:
                new_initials[var] = float(init_val)
            except (TypeError, ValueError):
                new_initials[var] = 1.0

        # Create new variable list
        new_vars = list(sym.vars) + all_new_vars

        sym = SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=combined_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,
            assignment_rules=sym.assignment_rules,
            compartments=sym.compartments,  # Propagate compartments
            sim_t_start=sym.sim_t_start,  # Propagate sim metadata
            sim_t_end=sym.sim_t_end,
            sim_n_steps=sym.sim_n_steps,
            eps_init=sym.eps_init,
            eps_slack=sym.eps_slack,
        )

    return sym, all_aux_defs, aux_counter


def _is_time_only_function(func: sp.Expr, state_vars: set[sp.Symbol]) -> bool:
    """
    Check if a composite function depends only on time (and parameters), not state variables.

    Time-only functions should be assignment rules, not state variables with ODEs,
    because they can be computed directly from time without differential equations.

    Args:
        func: The function expression to check
        state_vars: Set of state variable symbols

    Returns:
        True if function depends only on time/parameters, False if it depends on state variables
    """
    func_symbols = func.free_symbols
    # Check if any state variable appears in the function
    for sym in func_symbols:
        if sym in state_vars:
            return False
    return True


def lift_composite_functions(sym: SymSystem) -> tuple[SymSystem, dict[sp.Symbol, sp.Expr]]:
    """
    Augment system with auxiliary variables for all composite functions.

    CRITICAL DISTINCTION:
    - Functions of STATE VARIABLES (exp(X), log(Y)) → lift with ODEs via chain rule
    - Functions of TIME ONLY (sin(t), cos(t), tanh(k*t)) → assignment rules, NOT state ODEs

    For state-dependent functions f(X):
    1. Check if f requires positivity transformation (sin/cos need offset)
    2. Create auxiliary Z = f(X) + offset
    3. For sin/cos: create BOTH auxiliaries as a coupled pair
    4. Add ODEs with proper coupling for sin/cos derivatives
    5. Replace f(X) with (Z - offset) in all ODEs
    6. Set Z(0) = f(X(0)) + offset

    For time-only functions f(t):
    1. Create auxiliary Z = f(t) + offset
    2. Output as ASSIGNMENT RULE (Z := f(t) + offset), NOT ODE
    3. Replace f(t) with (Z - offset) in all ODEs

    Also lifts sqrt(sum) patterns: sqrt(a + b + ...) into auxiliary Z.

    Returns:
        Tuple of (augmented SymSystem, auxiliary_defs dict mapping Z -> f(X)+offset)
    """
    # Find all unique composite functions across all ODEs
    all_functions = set()
    all_sqrt_sums = set()
    for var, ode in sym.odes.items():
        funcs = find_composite_functions(ode)
        all_functions.update(funcs)
        sqrt_sums = find_sqrt_of_sums(ode)
        all_sqrt_sums.update(sqrt_sums)

    if not all_functions and not all_sqrt_sums:
        # No composite functions or sqrt(sum) to lift
        return sym, {}

    # CRITICAL: Separate time-only functions from state-dependent functions
    # Time-only → assignment rules (no ODE needed)
    # State-dependent → ODEs via chain rule
    state_vars = set(sym.vars)
    assignment_exprs = _parse_assignment_rule_expressions(sym)

    time_only_functions = set()
    state_dependent_functions = set()

    for func in all_functions:
        expanded_func = _expand_assignment_rules_by_name(func, assignment_exprs)
        if _is_time_only_function(expanded_func, state_vars):
            time_only_functions.add(func)
        else:
            state_dependent_functions.add(func)

    # Group functions by type and argument for coupled handling (sin/cos pairs)
    # CLASSICAL S-SYSTEM: ALL functions get ODEs (including time-only)
    # This follows Savageau 1987: sin(time), cos(time) become coupled oscillator state variables
    sin_cos_pairs: dict[
        sp.Expr, dict[str, sp.Expr]
    ] = {}  # arg -> {"sin": sin(arg), "cos": cos(arg)}
    other_functions = set()

    # Process ALL functions (time-only AND state-dependent) - all get ODEs
    for func in all_functions:
        arg = func.args[0] if func.args else None
        if arg is None:
            other_functions.add(func)
            continue

        # Check if this is sin or cos - use direct class comparison
        if func.func == sp.sin:
            # This is sin(arg)
            if arg not in sin_cos_pairs:
                sin_cos_pairs[arg] = {}
            sin_cos_pairs[arg]["sin"] = func
        elif func.func == sp.cos:
            # This is cos(arg)
            if arg not in sin_cos_pairs:
                sin_cos_pairs[arg] = {}
            sin_cos_pairs[arg]["cos"] = func
        else:
            # Other function (exp, log, etc.)
            other_functions.add(func)

    # Create auxiliary symbols for each function with offsets
    func_to_aux: dict[sp.Expr, sp.Symbol] = {}
    func_to_offset: dict[sp.Expr, float] = {}
    time_only_aux: dict[sp.Expr, sp.Symbol] = {}  # Time-only functions → assignment rules
    aux_counter = 1

    # Time-only functions (sin(time), cos(time), etc.) are handled the SAME as state-dependent
    # functions - they become state variables with coupled oscillator ODEs (classical S-system approach).
    # This follows Savageau 1987: all functions are lifted to autonomous state variables.
    assignment_rules: dict[str, str] = dict(sym.assignment_rules)  # Copy existing rules

    # NOTE: time_only_functions and time_only_aux are now UNUSED - all functions get ODEs

    def _function_subs_map() -> dict[sp.Expr, sp.Expr]:
        substitutions: dict[sp.Expr, sp.Expr] = {}
        for known_func, known_aux in func_to_aux.items():
            offset = func_to_offset[known_func]
            replacement = known_aux - offset if offset > 0 else known_aux
            substitutions[known_func] = replacement
            expanded_known_func = _expand_assignment_rules_by_name(known_func, assignment_exprs)
            substitutions[expanded_known_func] = replacement
        return substitutions

    # Handle sin/cos pairs (state-dependent only) - create BOTH auxiliaries even if only one appears
    for arg, funcs_dict in sin_cos_pairs.items():
        sin_func = funcs_dict.get("sin", sp.sin(arg))
        cos_func = funcs_dict.get("cos", sp.cos(arg))

        # Create auxiliary for sin
        Z_sin = sp.symbols(f"Z_{aux_counter}", positive=True)
        func_to_aux[sin_func] = Z_sin
        func_to_offset[sin_func] = 2.0  # sin ∈ [-1,1] → [1,3]
        aux_counter += 1

        # Create auxiliary for cos
        Z_cos = sp.symbols(f"Z_{aux_counter}", positive=True)
        func_to_aux[cos_func] = Z_cos
        func_to_offset[cos_func] = 2.0  # cos ∈ [-1,1] → [1,3]
        aux_counter += 1

    # Handle other functions (exp, log, etc.) - no offset needed
    for func in sorted(other_functions, key=str):
        Z = sp.symbols(f"Z_{aux_counter}", positive=True)
        func_to_aux[func] = Z
        func_to_offset[func] = 0.0  # No offset for exp, log, etc.
        aux_counter += 1

    # Handle sqrt(sum) expressions - these are NOT monomials
    # Create auxiliary Z = sqrt(base) with Z' = (d base/dt) / (2*Z)
    sqrt_to_aux: dict[sp.Expr, sp.Symbol] = {}
    for sqrt_expr in sorted(all_sqrt_sums, key=str):
        # Check if sqrt is time-only
        expanded_sqrt_expr = _expand_assignment_rules_by_name(sqrt_expr, assignment_exprs)
        if _is_time_only_function(expanded_sqrt_expr, state_vars):
            # Time-only sqrt → assignment rule
            Z = sp.symbols(f"Z_{aux_counter}", positive=True)
            time_only_aux[sqrt_expr] = Z
            assignment_rules[Z.name] = str(sqrt_expr)
            func_to_aux[sqrt_expr] = Z  # For substitution
            func_to_offset[sqrt_expr] = 0.0  # sqrt is always positive, no offset
            sqrt_to_aux[sqrt_expr] = Z  # Keep in sqrt_to_aux for substitution tracking
        else:
            Z = sp.symbols(f"Z_{aux_counter}", positive=True)
            sqrt_to_aux[sqrt_expr] = Z
            func_to_offset[sqrt_expr] = 0.0  # sqrt is always positive, no offset
        aux_counter += 1

    # CRITICAL: DO NOT substitute auxiliaries in original ODEs yet
    # We need the original functions present for the chain rule to work correctly
    # Keep original ODEs unchanged for now
    new_odes: dict[sp.Symbol, sp.Expr] = dict(sym.odes)
    function_subs = _function_subs_map()

    # Compute Z' using coupled derivatives for sin/cos
    new_aux_odes: dict[sp.Symbol, sp.Expr] = {}

    # Handle sin/cos pairs with coupled derivatives
    for arg, funcs_dict2 in sin_cos_pairs.items():
        sin_func = funcs_dict2.get("sin", sp.sin(arg))
        cos_func = funcs_dict2.get("cos", sp.cos(arg))
        Z_sin = func_to_aux[sin_func]
        Z_cos = func_to_aux[cos_func]

        # d/dt[sin(arg) + 2] = cos(arg) * d(arg)/dt = (Z_cos - 2) * d(arg)/dt
        # d/dt[cos(arg) + 2] = -sin(arg) * d(arg)/dt = -(Z_sin - 2) * d(arg)/dt = (2 - Z_sin) * d(arg)/dt

        # Compute d(arg)/dt using chain rule. Assignment-rule targets are
        # algebraic expressions, so expand them before differentiating.
        diff_arg = _expand_assignment_rules_by_name(arg, assignment_exprs)
        arg_prime = sp.Integer(0)
        for var in sym.vars:
            if var in diff_arg.free_symbols:
                partial = sp.diff(diff_arg, var)
                arg_prime += partial * new_odes[var]

        # Handle explicit time dependence: d(time)/dt = 1
        # CRITICAL: For sin(time), cos(time), the argument IS time, so arg' = 1
        for time_symbol in _time_symbols(diff_arg):
            partial_t = sp.diff(diff_arg, time_symbol)
            arg_prime += partial_t  # d(time)/dt = 1

        # Z_sin' = (Z_cos - 2) * arg'
        Z_sin_ode = (Z_cos - 2) * arg_prime
        new_aux_odes[Z_sin] = _bounded_composite_lift_simplify(Z_sin_ode)

        # Z_cos' = (2 - Z_sin) * arg'
        Z_cos_ode = (2 - Z_sin) * arg_prime
        new_aux_odes[Z_cos] = _bounded_composite_lift_simplify(Z_cos_ode)

    # Collect all variables that have ODEs at this point:
    # - Original variables (from sym.vars)
    # - Newly created sin/cos auxiliaries (keys in new_aux_odes)
    all_vars_with_odes = list(sym.vars) + list(new_aux_odes.keys())

    # Handle sqrt(sum) expressions: Z' = (d base/dt) / (2*Z)
    # This uses the chain rule: d/dt sqrt(f) = f' / (2*sqrt(f)) = f' / (2*Z)
    sqrt_aux_odes: dict[sp.Symbol, sp.Expr] = {}
    for sqrt_expr, Z in sqrt_to_aux.items():
        # Skip time-only sqrts - they're assignment rules, not state variables
        if sqrt_expr in time_only_aux:
            continue
        base = sqrt_expr.args[0]  # The base of sqrt(base)
        diff_base = _expand_assignment_rules_by_name(base, assignment_exprs)

        # Compute d(base)/dt using chain rule
        base_prime = sp.Integer(0)
        for var in sym.vars:
            if var in diff_base.free_symbols:
                partial = sp.diff(diff_base, var)
                base_prime += partial * new_odes[var]

        # Handle time dependence (time_sym defined above at start of sin/cos loop)
        for time_symbol in _time_symbols(diff_base):
            partial_t = sp.diff(diff_base, time_symbol)
            base_prime += partial_t  # d(time)/dt = 1

        # Z' = base' / (2*Z)
        # CRITICAL FIX: Substitute sqrt(base) → Z in base_prime BEFORE dividing
        # This enforces the identity Z = sqrt(base), producing clean compact form:
        #   Z' = (Ca - Ca_c) * Ca' / Z  (instead of unsimplified form with sqrt(...))
        # This is a generally applicable fix - any time we create Z = f(expr),
        # occurrences of f(expr) in the ODE should be replaced with Z.
        base_prime = base_prime.subs(sqrt_expr, Z)

        Z_ode = base_prime / (2 * Z)
        sqrt_aux_odes[Z] = Z_ode  # Skip expensive simplify - algebraically correct

    # Add sqrt auxiliary ODEs to new_aux_odes
    new_aux_odes.update(sqrt_aux_odes)

    # Update all_vars_with_odes to include sqrt auxiliaries
    all_vars_with_odes = list(sym.vars) + list(new_aux_odes.keys())

    # Handle other functions with standard chain rule
    for func in other_functions:
        Z = func_to_aux[func]
        diff_func = _expand_assignment_rules_by_name(func, assignment_exprs)

        # Compute df/dt using chain rule: df/dt = sum_i (∂f/∂X_i) * dX_i/dt
        func_prime = sp.Integer(0)
        # CRITICAL FIX: Use all_vars_with_odes which includes ALL variables with ODEs
        # (original variables + sin/cos auxiliaries created earlier)
        for var in all_vars_with_odes:
            if var in diff_func.free_symbols:
                partial = sp.diff(diff_func, var)
                _raise_for_unsupported_derivative_printer(
                    partial,
                    source_expr=diff_func,
                    context=f"d/d{var}",
                )

                # Use the ODE for var (either from new_odes or new_aux_odes)
                var_ode = new_odes.get(var) or new_aux_odes.get(var)
                if var_ode is not None:
                    # Compute the chain rule term
                    term = partial * var_ode

                    # Replace composite functions with auxiliaries AFTER multiplication
                    # Use .subs() instead of .replace() to handle algebraic simplifications
                    # (e.g., exp(2*x) = exp(x)^2)
                    term = _bounded_composite_substitute(term, function_subs)

                    func_prime += term

        # Handle explicit time dependence: d(time)/dt = 1
        # (time_sym defined above at start of sin/cos loop)
        for time_symbol in _time_symbols(diff_func):
            partial_t = sp.diff(diff_func, time_symbol)
            _raise_for_unsupported_derivative_printer(
                partial_t,
                source_expr=diff_func,
                context=f"d/d{time_symbol}",
            )
            # Substitute auxiliaries in the time derivative term
            partial_t = _bounded_composite_substitute(partial_t, function_subs)
            func_prime += partial_t

        # Store the computed ODE
        Z_ode = func_prime

        # CRITICAL: Final expansion and simplification pass
        # Expand products and collect like terms
        Z_ode = sp.expand(Z_ode)

        # Replace any remaining instances of composite functions with auxiliaries
        # Use .subs() instead of .replace() to handle algebraic simplifications
        Z_ode = _bounded_composite_substitute(Z_ode, function_subs)

        Z_ode = _bounded_composite_lift_simplify(Z_ode)
        _raise_for_unsupported_derivative_printer(
            Z_ode,
            source_expr=diff_func,
            context=f"auxiliary_ode:{Z}",
        )

        # CRITICAL: DO NOT apply inverse mappings to eliminate original variables
        # This violates the chain rule and creates incorrect dynamics.
        # The chain rule derivation MUST keep original variables in the auxiliary ODEs.
        #
        # Example: For Z' = k*exp((log(Z))^2) with auxiliaries:
        #   Z_1 = exp((log(Z))^2)
        #   Z_2 = log(Z)
        # The correct ODEs are:
        #   Z_1' = Z_1 * 2*Z_2 * Z_2'  (chain rule with Z, not with exp(Z_2))
        #        = Z_1 * 2*Z_2 * (1/Z * Z')
        #        = Z_1 * 2*Z_2 * (1/Z * k*Z_1)
        #        = 2*k * Z^(-1) * Z_1^2 * Z_2  ✓ Correct
        #
        # If we substitute Z → exp(Z_2), we get:
        #   Z_1' = Z_1 * 2*Z_2 * (1/exp(Z_2) * k*Z_1)
        #        = 2*k * exp(-Z_2) * Z_1^2 * Z_2
        #        = ... (becomes -k*Z_1^3 after simplification) ✗ Wrong!
        #
        # The inverse mappings break the chain rule relationships.

        new_aux_odes[Z] = Z_ode

    # NOW substitute composite functions with auxiliaries ONLY in original ODEs
    # This must happen AFTER computing all auxiliary ODEs via chain rule
    # CRITICAL: Do NOT modify auxiliary ODEs - they are already correct from chain rule
    for var in new_odes.keys():
        new_ode = new_odes[var]

        # Use .subs() instead of .replace() to handle algebraic simplifications
        subs_map = {}
        for func, Z in func_to_aux.items():
            offset = func_to_offset[func]
            if offset > 0:
                subs_map[func] = Z - offset
            else:
                subs_map[func] = Z
        new_ode = _bounded_composite_substitute(new_ode, subs_map)

        # Also substitute sqrt(sum) expressions
        for sqrt_expr, Z in sqrt_to_aux.items():
            new_ode = _bounded_composite_substitute(new_ode, {sqrt_expr: Z})

        new_odes[var] = _bounded_composite_lift_simplify(new_ode)

    # Combine original and auxiliary ODEs
    combined_odes = {**new_odes, **new_aux_odes}

    # Compute initial conditions for auxiliaries with offsets
    new_initials = dict(sym.initials)
    # Combine original and auxiliary ODEs
    combined_odes = {**new_odes, **new_aux_odes}

    # Compute initial conditions for auxiliaries with offsets (before recursive lifting)
    new_initials = dict(sym.initials)
    for func, Z in func_to_aux.items():
        # Skip sqrt expressions - they're handled separately below
        if func in sqrt_to_aux:
            continue
        # Evaluate function at t=0
        func_at_0 = _expand_assignment_rules_by_name(func, assignment_exprs)
        # CRITICAL: Substitute time=0 FIRST for time-only functions
        for time_symbol in _time_symbols(func_at_0):
            func_at_0 = func_at_0.subs(time_symbol, 0)
        # Then substitute state variables
        # CRITICAL: Check BOTH initials AND params - SBML parser puts species ICs in params
        for var in sym.vars:
            if var in func_at_0.free_symbols:
                # Try initials first, then params (using var.name for params dict)
                init_val = sym.initials.get(var, sym.params.get(var.name, 1.0))
                func_at_0 = func_at_0.subs(var, init_val)
        # Then substitute parameters - use actual symbols from expression
        for param_sym in func_at_0.free_symbols:
            param_name = param_sym.name
            if param_name in sym.params:
                func_at_0 = func_at_0.subs(param_sym, sym.params[param_name])
        # Z(0) = f(X(0)) + offset
        offset = func_to_offset.get(func, 0.0)  # Use .get() to handle missing keys
        try:
            Z_init = float(func_at_0) + offset
        except (TypeError, ValueError, sp.SympifyError):
            Z_init = 1.0 + offset  # Fallback if evaluation fails
        new_initials[Z] = Z_init

    # Compute initial conditions for sqrt auxiliaries
    # Build name-based lookup for initials (handles symbol object mismatch)
    initials_by_name = {}
    for s, v in sym.initials.items():
        if hasattr(s, 'name'):
            initials_by_name[s.name] = v

    for sqrt_expr, Z in sqrt_to_aux.items():
        # Evaluate sqrt at t=0
        sqrt_at_0 = _expand_assignment_rules_by_name(sqrt_expr, assignment_exprs)
        # First substitute time=0
        for time_symbol in _time_symbols(sqrt_at_0):
            sqrt_at_0 = sqrt_at_0.subs(time_symbol, 0)
        # Then substitute state variables by NAME (symbol identity may differ)
        for sym_in_sqrt in list(sqrt_at_0.free_symbols):
            if sym_in_sqrt.name in initials_by_name:
                sqrt_at_0 = sqrt_at_0.subs(sym_in_sqrt, initials_by_name[sym_in_sqrt.name])
        # Then substitute parameters
        for param_sym in list(sqrt_at_0.free_symbols):
            param_name = param_sym.name
            if param_name in sym.params:
                sqrt_at_0 = sqrt_at_0.subs(param_sym, sym.params[param_name])
        try:
            Z_init = float(sqrt_at_0)
        except (TypeError, ValueError, sp.SympifyError):
            Z_init = 1.0  # Fallback if evaluation fails
        new_initials[Z] = Z_init

    # Create new variable list: keep original vars, add Z auxiliaries (excluding time-only)
    # Time-only auxiliaries are assignment rules, NOT state variables with ODEs
    state_aux_vars = [Z for func, Z in func_to_aux.items() if func not in time_only_aux]
    sqrt_state_vars = [Z for sqrt_expr, Z in sqrt_to_aux.items() if sqrt_expr not in time_only_aux]
    new_vars = list(sym.vars) + state_aux_vars + sqrt_state_vars

    # Create auxiliary definitions with offsets: Z -> f(X) + offset
    aux_to_func_with_offset = {}
    for func, Z in func_to_aux.items():
        offset = func_to_offset[func]
        if offset > 0:
            # Z = f(X) + offset
            aux_to_func_with_offset[Z] = func + offset
        else:
            # Z = f(X) (no offset)
            aux_to_func_with_offset[Z] = func

    # Add sqrt(sum) auxiliary definitions
    for sqrt_expr, Z in sqrt_to_aux.items():
        aux_to_func_with_offset[Z] = sqrt_expr

    # FOURTH PASS: Recursively lift any NEW composite functions introduced by inverse mappings
    # This handles cases where inverse mappings create expressions like exp(-Z_2)
    # which are mathematically correct but still contain composite functions
    #
    # IMPORTANT: For time-dependent models with complex nested functions (like Weber2018),
    # the recursive lifting can create infinite loops. DISABLE recursive lifting entirely
    # when sqrt(sum) expressions are present since they indicate complex time-dependent
    # models that don't benefit from recursive lifting.
    max_recursive_lifts = 0 if sqrt_to_aux else 1  # Disable for sqrt(sum) models
    for _recursive_iteration in range(max_recursive_lifts):
        # Scan all ODEs for remaining composite functions
        has_composite = False
        all_new_functions = set()
        for var, ode in combined_odes.items():
            funcs = find_composite_functions(ode)
            if funcs:
                has_composite = True
                all_new_functions.update(funcs)

        if not has_composite:
            break  # All ODEs are now in power-law form

        # Check if all remaining functions are already lifted auxiliaries
        # If so, we're done (avoid infinite recursion)
        already_lifted = set(aux_to_func_with_offset.values())
        new_funcs_not_lifted = set()
        for func in all_new_functions:
            if func not in already_lifted:
                new_funcs_not_lifted.add(func)

        if not new_funcs_not_lifted:
            # All remaining functions are already lifted - we're done
            break

        # Found composite functions - recursively lift them
        # CRITICAL: Find max Z_n index to avoid duplicate names in recursive call
        max_z_index = 0
        for var in combined_odes.keys():
            var_name = var.name if hasattr(var, "name") else str(var)
            # Check for Z_n pattern
            if var_name.startswith("Z_"):
                try:
                    index = int(var_name.split("_")[1])
                    max_z_index = max(max_z_index, index)
                except (ValueError, IndexError):
                    pass

        # Create temporary system and manually rename composite functions to avoid conflicts
        current_vars = list(combined_odes.keys())
        temp_sym = SymSystem(
            vars=current_vars,
            params=sym.params,
            odes=combined_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,
            assignment_rules=sym.assignment_rules,
            compartments=sym.compartments,  # Propagate compartments
            sim_t_start=sym.sim_t_start,  # Propagate sim metadata
            sim_t_end=sym.sim_t_end,
            sim_n_steps=sym.sim_n_steps,
            eps_init=sym.eps_init,
            eps_slack=sym.eps_slack,
        )

        # Recursively lift and manually adjust auxiliary names to continue from max_z_index
        temp_sym, new_comp_aux_defs = lift_composite_functions(temp_sym)

        # Rename recursively created auxiliaries to avoid conflicts
        # Map Z_1, Z_2, ... from recursive call to Z_{max+1}, Z_{max+2}, ...
        rename_map: dict[sp.Symbol, sp.Symbol] = {}
        counter = 1
        for var in temp_sym.vars:
            if var not in current_vars:  # This is a newly created auxiliary
                var_name = var.name if hasattr(var, "name") else str(var)
                if var_name.startswith("Z_"):
                    try:
                        int(var_name.split("_")[1])
                        new_index = max_z_index + counter
                        new_var = sp.Symbol(f"Z_{new_index}", positive=True)
                        rename_map[var] = new_var
                        counter += 1
                    except (ValueError, IndexError):
                        pass

        # Apply renaming to ODEs, initials, and auxiliary definitions
        if rename_map:
            # Rename in ODEs
            renamed_odes = {}
            for var, ode in temp_sym.odes.items():
                new_var = rename_map.get(var, var)
                new_ode = ode
                for old, new in rename_map.items():
                    new_ode = new_ode.subs(old, new)
                renamed_odes[new_var] = new_ode

            # Rename in initials
            renamed_initials = {}
            for var, val in temp_sym.initials.items():
                new_var = rename_map.get(var, var)
                renamed_initials[new_var] = val

            # Rename in auxiliary definitions
            renamed_aux_defs = {}
            for aux, defn in new_comp_aux_defs.items():
                new_aux = rename_map.get(aux, aux)
                new_defn = defn
                for old, new in rename_map.items():
                    new_defn = new_defn.subs(old, new)
                renamed_aux_defs[new_aux] = new_defn

            # Update results - CRITICAL: deduplicate variables to avoid duplicate entries
            # Use a dict to preserve order while removing duplicates
            seen_vars = {}
            for var in renamed_odes.keys():
                if var not in seen_vars:
                    seen_vars[var] = True
            new_vars = list(seen_vars.keys())

            combined_odes = renamed_odes
            new_initials = renamed_initials
            # CRITICAL FIX: Only add auxiliary definitions for NEW auxiliaries
            # Don't overwrite existing definitions with recursive call results
            for aux, defn in renamed_aux_defs.items():
                if aux not in aux_to_func_with_offset:
                    aux_to_func_with_offset[aux] = defn
        else:
            # No renaming needed
            new_vars = temp_sym.vars
            combined_odes = temp_sym.odes
            new_initials = temp_sym.initials
            # CRITICAL FIX: Only add auxiliary definitions for NEW auxiliaries
            # Don't overwrite existing definitions with recursive call results
            for aux, defn in new_comp_aux_defs.items():
                if aux not in aux_to_func_with_offset:
                    aux_to_func_with_offset[aux] = defn

    # Recompute numeric auxiliary ICs from the final definitions. Bounded
    # substitution during ODE construction can leave intermediate ICs stale for
    # nested functions such as exp(log(Z)**2), while the final definition map is
    # still authoritative.
    initial_values_by_name = {str(name): value for name, value in sym.params.items()}
    for var, value in sym.initials.items():
        if hasattr(var, "name"):
            initial_values_by_name[var.name] = value
    for var, value in new_initials.items():
        if hasattr(var, "name"):
            initial_values_by_name[var.name] = value

    for aux_sym, aux_def in aux_to_func_with_offset.items():
        aux_at_0 = _expand_assignment_rules_by_name(aux_def, assignment_exprs)
        for time_symbol in _time_symbols(aux_at_0):
            aux_at_0 = aux_at_0.subs(time_symbol, 0)
        for free_symbol in list(aux_at_0.free_symbols):
            initial_value = initial_values_by_name.get(free_symbol.name)
            if initial_value is not None:
                aux_at_0 = aux_at_0.subs(free_symbol, initial_value)
        try:
            new_initials[aux_sym] = float(aux_at_0)
            initial_values_by_name[aux_sym.name] = new_initials[aux_sym]
        except (TypeError, ValueError, sp.SympifyError):
            pass

    # Build symbolic IC expressions for auxiliary variables
    # CRITICAL: Only use symbolic ICs when they DON'T depend on state variables
    # If an IC expression like Z_1 = exp(log(Z)^2) depends on state variable Z,
    # we must output the numeric value because Z isn't defined until later in
    # the Antimony output, causing initialization order dependency errors.
    new_initial_exprs = dict(sym.initial_exprs)  # Copy existing
    state_var_names = {v.name for v in sym.vars}
    for aux_sym, aux_def in aux_to_func_with_offset.items():
        # Check if the definition depends on any state variables
        def_free_names = {s.name for s in aux_def.free_symbols}
        depends_on_state = bool(def_free_names & state_var_names)

        if not depends_on_state:
            # Safe to use symbolic expression (only depends on params/constants)
            new_initial_exprs[aux_sym] = _sympy_to_antimony_syntax(str(aux_def))
        # else: use numeric value from new_initials (already computed above)

    # Return augmented system and auxiliary definitions
    return (
        SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=combined_odes,
            initials=new_initials,
            initial_exprs=new_initial_exprs,  # Include symbolic IC expressions for auxiliaries
            assignment_rules=assignment_rules,  # Time-only auxiliaries as assignment rules
            compartments=sym.compartments,  # Propagate compartments
            sim_t_start=sym.sim_t_start,  # Propagate sim metadata
            sim_t_end=sym.sim_t_end,
            sim_n_steps=sym.sim_n_steps,
            eps_init=sym.eps_init,
            eps_slack=sym.eps_slack,
        ),
        aux_to_func_with_offset,  # Dictionary mapping Z_i -> f(X) + offset
    )
