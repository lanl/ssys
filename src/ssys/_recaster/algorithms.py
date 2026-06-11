"""S-system and GMA recasting algorithms."""

import sympy as sp

from ssys._recaster.common import EPS_INIT
from ssys._recaster.lifting import (
    lift_composite_functions,
    lift_rational_functions,
    lift_time_functions_to_autonomous,
)
from ssys.classification import classify_solver_requirement
from ssys.math_utils import _exponents_match, expand_to_terms
from ssys.types import (
    GMAEquation,
    RecastResult,
    RecastStatus,
    SSysEquation,
    SymSystem,
)


def canonicalize_aux_names(res: "RecastResult", prefix: str = "Z") -> "RecastResult":
    """
    Rename every auxiliary variable to Z_1, Z_2, ... in first-appearance order.
    Updates equations, initials, variables, and factor_map consistently.
    Uses 'Z' prefix by default to avoid collision with original variable names.
    """
    # 1) Determine aux order by first appearance in equations
    aux_order, seen = [], set()
    for eq in res.equations:
        if eq.var not in seen:
            aux_order.append(eq.var)
            seen.add(eq.var)

    # 2) Map old aux -> new canonical aux
    name_map = {old: sp.Symbol(f"{prefix}_{i}") for i, old in enumerate(aux_order, start=1)}

    def remap_exps(exps: dict[sp.Symbol, sp.Expr]) -> dict[sp.Symbol, sp.Expr]:
        out: dict[sp.Symbol, sp.Expr] = {}
        for s, e in exps.items():
            out[name_map.get(s, s)] = e
        return out

    def remap_coeff(coeff: sp.Expr) -> sp.Expr:
        """Apply name_map substitutions to coefficient expression.

        This handles symbolic coefficients that contain original variable names
        (e.g., sqrt(Z2^2 + 1) should become sqrt(Z_2^2 + 1)).

        Note: We must match symbols by NAME, not by object identity, because
        the coefficient may contain symbols created during parsing that are
        different objects from those in name_map but have the same name.
        """
        if isinstance(coeff, sp.Expr) and not coeff.is_Number:
            result = coeff
            # Build name-to-new-symbol map
            name_to_new = {old_sym.name: new_sym for old_sym, new_sym in name_map.items()}

            # Find all symbols in the coefficient and substitute by name
            for sym in result.free_symbols:
                if sym.name in name_to_new:
                    result = result.subs(sym, name_to_new[sym.name])
            return result
        return coeff

    # 3) Remap factor_map FIRST (needed for coefficient substitution)
    new_factor_map = {
        orig: [name_map.get(a, a) for a in aux_list] for orig, aux_list in res.factor_map.items()
    }

    def remap_coeff_with_factors(coeff: sp.Expr) -> sp.Expr:
        """Apply name_map AND factor_map substitutions to coefficient expression.

        Two substitutions are needed:
        1. Pool auxiliaries: Z1_t1 -> Z_1 (via name_map)
        2. Original variables: Z2 -> Z_2 (via new_factor_map)

        This handles symbolic coefficients like sqrt(Z2^2 + 1) where Z2 is
        an original variable that should become Z_2.
        """
        if not isinstance(coeff, sp.Expr) or coeff.is_Number:
            return coeff

        result = coeff

        # Step 1: Remap pool auxiliary names (Z1_t1 -> Z_1)
        name_to_new = {old_sym.name: new_sym for old_sym, new_sym in name_map.items()}
        for sym in list(result.free_symbols):
            if sym.name in name_to_new:
                result = result.subs(sym, name_to_new[sym.name])

        # Step 2: Remap original variables to their factor products
        # e.g., Z2 -> Z_2 (the product of its renamed factors)
        for orig_var, aux_list in new_factor_map.items():
            if not aux_list:
                continue
            # Build product of factors
            factor_product = aux_list[0] if len(aux_list) == 1 else sp.Mul(*aux_list)
            # Substitute original variable by name matching
            for sym in list(result.free_symbols):
                if sym.name == orig_var.name and sym != factor_product:
                    result = result.subs(sym, factor_product)

        return result

    # 4) Remap equations (var, exponent maps, AND coefficients)
    new_eqs: list[SSysEquation] = []
    for eq in res.equations:
        # Remap coefficients to use canonical variable names
        new_eqs.append(
            SSysEquation(
                var=name_map.get(eq.var, eq.var),
                growth=(remap_coeff_with_factors(eq.growth[0]), remap_exps(eq.growth[1])),
                decay=(remap_coeff_with_factors(eq.decay[0]), remap_exps(eq.decay[1])),
            )
        )

    # 5) Remap initials (keys)
    new_initials = {name_map.get(s, s): float(v) for s, v in res.initials.items()}

    # 6) Canonical variables list, in canonical order
    new_variables = [name_map[old] for old in aux_order]

    return RecastResult(
        status=RecastStatus.CANONICAL_SSYSTEM,
        equations=new_eqs,
        initials=new_initials,
        variables=new_variables,
        factor_map=new_factor_map,
        params=res.params,
        compartments=res.compartments,  # Propagate compartments
    )


# --- end canonicalize_aux_names ---


def term_to_coeff_exps(
    term: sp.Expr, state_vars: set[sp.Symbol] | None = None
) -> tuple[sp.Expr, dict[sp.Symbol, float]]:
    """
    Extract coefficient and exponents from a power-law monomial term.
    Now returns symbolic coefficient (sp.Expr) instead of float.

    Args:
        term: The term to decompose
        state_vars: Set of state variable symbols. If provided, only these symbols
                   are treated as variables with exponents; all others go into coefficient.

    Returns: (coeff_expr, {symbol: exponent})
    """
    term = sp.simplify(term)
    coeff = sp.Integer(1)
    exps: dict[sp.Symbol, float] = {}

    if term.is_Number:
        # Check if dummy_const is in state_vars - if so, add it with exponent 0
        # This handles constant terms that were transformed by add_dummy_for_constants
        if state_vars:
            dummy_const = None
            for s in state_vars:
                if s.name == "dummy_const":
                    dummy_const = s
                    break
            if dummy_const is not None:
                exps[dummy_const] = 0.0
        return term, exps

    if isinstance(term, sp.Symbol):
        # Check if this is a state variable or a parameter
        if state_vars is None or term in state_vars:
            exps[term] = 1.0
        else:
            coeff = term
        return coeff, exps

    if term.is_Mul:
        for f in term.args:
            if f.is_Number:
                coeff *= f
            elif isinstance(f, sp.Symbol):
                # Only treat as variable if it's in state_vars
                if state_vars is None or f in state_vars:
                    exps[f] = exps.get(f, 0.0) + 1.0
                else:
                    # It's a parameter - add to coefficient
                    coeff *= f
            elif isinstance(f, sp.Pow):
                base, exp_val = f.args
                if isinstance(base, sp.Symbol):
                    # Check if base is a state variable
                    if state_vars is None or base in state_vars:
                        # Handle both numeric and symbolic exponents
                        if exp_val.is_number:
                            exps[base] = exps.get(base, 0.0) + float(exp_val)
                        else:
                            # Symbolic exponent - keep base as variable with symbolic exp
                            exps[base] = exps.get(base, 0) + exp_val
                    else:
                        # It's a parameter raised to a power - keep in coefficient
                        coeff *= f
                else:
                    # Complex base - keep in coefficient
                    coeff *= f
            else:
                # Non-power-law factor - keep in coefficient
                coeff *= f
        return coeff, exps

    if isinstance(term, sp.Pow):
        base, exp_val = term.args
        if isinstance(base, sp.Symbol):
            # Check if this is a state variable
            if state_vars is None or base in state_vars:
                if exp_val.is_number:
                    exps[base] = float(exp_val)
                    return coeff, exps
        # Not a state variable or complex - return as coefficient
        return term, exps

    # If we can't decompose it, return as pure coefficient
    return term, exps

def _analyze_ode_terms(
    terms: list[sp.Expr], state_vars: set[sp.Symbol] | None = None
) -> tuple[list[tuple[sp.Expr, dict]], list[tuple[sp.Expr, dict]]]:
    """
    Analyze ODE terms and separate into growth and decay.

    Args:
        terms: List of terms from the ODE
        state_vars: Set of state variable symbols

    Returns: (growth_terms, decay_terms) where each term is (coeff, exps)
    """
    growth_terms = []
    decay_terms = []

    for t in terms:
        if t == 0:
            continue
        try:
            coeff, exps = term_to_coeff_exps(t, state_vars)
            # Determine sign of coefficient
            # Handle symbolic coefficients by extracting the numeric part
            is_positive = _is_coefficient_positive(coeff)

            if is_positive:
                growth_terms.append((coeff, exps))
            else:
                # Use -coeff instead of sp.Abs(coeff) to handle symbolic coefficients
                # sp.Abs() doesn't evaluate for symbolic expressions like -J_2 → Abs(J_2)
                # but -(-J_2) → J_2 works correctly
                decay_terms.append((-coeff, exps))
        except (TypeError, ValueError, sp.SympifyError):
            continue

    return growth_terms, decay_terms


def _is_coefficient_positive(coeff: sp.Expr) -> bool:
    """
    Determine if a coefficient is positive or negative.

    For purely numeric coefficients, just check the sign.
    For symbolic coefficients like -V_1, extract the leading numeric factor.

    Returns True if positive, False if negative.
    """
    # Pure number case
    if coeff.is_Number:
        try:
            return float(coeff) >= 0
        except (TypeError, ValueError):
            return True

    # Try sp.sign() first (works for simple cases)
    try:
        sign_result = sp.sign(coeff)
        if sign_result.is_Number:
            return float(sign_result) >= 0
    except (TypeError, ValueError):
        pass

    # For Mul expressions like -1*V_1 or -V_1, check for leading negative
    if coeff.is_Mul:
        # as_coeff_Mul() returns (numeric_coeff, rest)
        # e.g., -V_1 → (-1, V_1), 2*V_1 → (2, V_1)
        numeric, _ = coeff.as_coeff_Mul()
        if numeric.is_Number:
            try:
                return float(numeric) >= 0
            except (TypeError, ValueError):
                pass

    # For expressions like -V_1 which sympy represents as Mul(-1, V_1)
    # Check if first arg is -1
    if hasattr(coeff, "args") and coeff.args:
        first_arg = coeff.args[0]
        if first_arg.is_Number:
            try:
                return float(first_arg) >= 0
            except (TypeError, ValueError):
                pass

    # Default: assume positive (reasonable for physical systems)
    return True


def _requires_gma(sym: SymSystem) -> bool:
    """
    Check if system requires GMA format (cannot be exact canonical S-system).
    Returns True if any ODE has multiple terms with different exponent patterns.
    """
    for _var, ode in sym.odes.items():
        terms = expand_to_terms(sp.simplify(ode))
        growth_terms, decay_terms = _analyze_ode_terms(terms)

        # Check if multiple growth terms have different exponent patterns
        if len(growth_terms) > 1:
            first_exps = growth_terms[0][1]
            for _coeff, exps in growth_terms[1:]:
                if not _exponents_match(first_exps, exps):
                    return True

        # Check if multiple decay terms have different exponent patterns
        if len(decay_terms) > 1:
            first_exps = decay_terms[0][1]
            for _coeff, exps in decay_terms[1:]:
                if not _exponents_match(first_exps, exps):
                    return True

    return False


# Safety constraints for pool construction
MAX_TERMS_PER_EQUATION = 6
MAX_DIM_FACTOR = 4
MAX_PRODUCT_LENGTH = 4
MAX_NEGATIVE_EXPONENT = -2


def _should_attempt_pool_construction(sym: SymSystem) -> tuple[bool, str | None]:
    """
    Pre-flight check: Is pool construction worth attempting?

    Returns: (should_attempt, refusal_reason)
    """
    n_vars = len(sym.vars)
    total_terms = 0
    max_terms_in_equation = 0

    for _var, ode in sym.odes.items():
        terms = expand_to_terms(sp.simplify(ode))
        n_terms = len([t for t in terms if t != 0])

        # Track max terms per equation
        if n_terms > max_terms_in_equation:
            max_terms_in_equation = n_terms

        # Per-equation check
        if n_terms > MAX_TERMS_PER_EQUATION:
            return False, f"equation has {n_terms} terms (max {MAX_TERMS_PER_EQUATION} allowed)"

        total_terms += n_terms

    # Dimension explosion check
    max_allowed_terms = MAX_DIM_FACTOR * n_vars
    if total_terms > max_allowed_terms:
        return (
            False,
            f"would create {total_terms} auxiliaries for {n_vars} variables (>{MAX_DIM_FACTOR}x expansion)",
        )

    return True, None


def _validate_pool_result(result: RecastResult) -> tuple[bool, str | None]:
    """
    Post-construction check: Is the pool result numerically sane?

    Returns: (is_valid, rejection_reason)
    """
    # Check product lengths
    max_product_length = 0
    for orig, factors in result.factor_map.items():
        if len(factors) > max_product_length:
            max_product_length = len(factors)
        if len(factors) > MAX_PRODUCT_LENGTH:
            return (
                False,
                f"variable {orig.name} mapped to product of {len(factors)} factors (max {MAX_PRODUCT_LENGTH} allowed)",
            )

    # Check for excessive negative exponents
    min_exponent = 0.0
    for eq in result.equations:
        for exps_dict in [eq.growth[1], eq.decay[1]]:
            for _var, exp in exps_dict.items():
                exp_val = float(exp) if not isinstance(exp, sp.Expr) else 0.0
                if exp_val < min_exponent:
                    min_exponent = exp_val
                if exp_val < MAX_NEGATIVE_EXPONENT:
                    return (
                        False,
                        f"equation for {eq.var.name} has exponent {exp_val:.1f} (< {MAX_NEGATIVE_EXPONENT})",
                    )

    return True, None


def recast_to_ssystem(sym: "SymSystem", mode: str = "simplified") -> "RecastResult":
    """
    Recast system to canonical S-system or GMA format.

    Strategy:
    1. Lift time-dependent functions to autonomous ODEs (exp(-kt), cos(ωt), tanh(k(t-a)))
    2. Lift remaining composite functions (exp, sin, log, etc.)
    3. Lift rational functions (1/(X+1), etc.)
    4. Check for constant terms (S-systems cannot represent these)
    5. Attempt canonical S-system recast:
       - If lifting occurred: use direct form
       - Otherwise: use pool construction
    6. Check if output has GMA characteristics (multi-term incompatible)
    7. If canonical failed, fall back to GMA format

    Args:
        sym: SymSystem to recast
        mode: Output mode ('simplified' or 'canonical')

    Returns:
        RecastResult with status indicating output form and auxiliary definitions
    """
    # Track original variables before lifting
    original_vars = set(sym.vars)

    # Collect auxiliary definitions from lifting operations
    all_auxiliary_defs: dict[sp.Symbol, sp.Expr] = {}

    # FIRST: Lift time-dependent functions to autonomous ODEs
    # This converts exp(-k*time), cos(ω*time), tanh(k*(time-a)) to state variables
    # with their own differential equations (strict GMA form)
    sym, time_aux_defs, _ = lift_time_functions_to_autonomous(sym)
    all_auxiliary_defs.update(time_aux_defs)

    # Lift remaining composite functions (exp, sin, log of state variables)
    sym, composite_aux_defs = lift_composite_functions(sym)
    all_auxiliary_defs.update(composite_aux_defs)

    # Then lift rational functions (1/(X+1), etc.)
    # Pass composite_aux_defs to prevent re-lifting composite functions
    sym, rational_aux_defs = lift_rational_functions(sym, composite_aux_defs)
    all_auxiliary_defs.update(rational_aux_defs)

    # Handle constant terms: in simplified mode, constants are acceptable as-is.
    # In canonical mode, the epsilon slack mechanism handles pure constant terms
    # by converting X' = C to X' = (C+ε) - ε, which provides the required
    # "two terms per ODE" form without needing a dummy variable.
    #
    # NOTE: add_dummy_for_constants() was REMOVED because:
    # 1. It's redundant with epsilon slack
    # 2. It created a new SymSystem that lost assignment_rules, causing bugs
    # 3. It added unnecessary complexity for no practical benefit

    # Identify lifted auxiliaries (those added during lifting)
    lifted_vars = set(sym.vars) - original_vars

    # CRITICAL: For composite function systems, DO NOT apply inverse mappings
    # Inverse mappings violate the chain rule by rewriting original variables in terms
    # of auxiliaries (e.g., Z → exp(Z_2)), which changes the functional relationships
    # and breaks mathematical equivalence.
    #
    # For composite systems, auxiliary ODEs are computed via chain rule during lifting,
    # and they MUST remain in terms of original variables to preserve the dynamics.
    #
    # Only apply inverse mappings for rational/algebraic auxiliaries (Y = f(X) identity)
    has_composite_aux = any(
        isinstance(defn, (sp.log, sp.exp, sp.sin, sp.cos))
        or (
            defn.is_Add
            and any(isinstance(arg, (sp.log, sp.exp, sp.sin, sp.cos)) for arg in defn.args)
        )
        for defn in all_auxiliary_defs.values()
    )

    if lifted_vars and all_auxiliary_defs and not has_composite_aux:
        # Only apply inverse mappings for non-composite systems (rational/identity mappings)
        # Build inverse map: original_var -> expression in terms of auxiliaries
        orig_to_aux_expr: dict[sp.Symbol, sp.Expr] = {}

        # For identity mappings: if Y_1 = Z (simple symbol equality)
        for aux, defn in all_auxiliary_defs.items():
            if aux in lifted_vars and isinstance(defn, sp.Symbol) and defn in original_vars:
                # Y_1 = Z => can substitute Y_1 for Z in other ODEs
                # But this is an identity, so no substitution needed
                pass

        # Apply inverse mappings (currently empty for composite systems)
        if orig_to_aux_expr:
            new_odes = {}
            for var, ode in sym.odes.items():
                new_ode = ode.subs(orig_to_aux_expr)
                new_odes[var] = sp.simplify(new_ode)

            new_vars = list(sym.vars)
            new_initials = dict(sym.initials)

            sym = SymSystem(
                vars=new_vars,
                params=sym.params,
                odes=new_odes,
                initials=new_initials,
                initial_exprs=sym.initial_exprs,
            )

    # Always attempt canonical S-system recast
    if lifted_vars:
        # Lifted systems use direct form
        result = _direct_ssystem_recast(sym, original_vars, mode=mode)
    else:
        # Pure polynomial systems - attempt pool construction with safety checks

        # Pre-flight check: would pool construction be reasonable?
        should_attempt, preflight_reason = _should_attempt_pool_construction(sym)

        if not should_attempt:
            # Pre-flight failed - use GMA
            result = _gma_recast(sym, original_vars)
            result.canonical_refusal_reason = preflight_reason
        else:
            # Attempt pool construction
            result = _pool_ssystem_recast(sym, mode=mode)

            # Post-flight validation: is result numerically sane?
            is_valid, validation_reason = _validate_pool_result(result)

            if not is_valid:
                # Pool result invalid - fallback to GMA
                result = _gma_recast(sym, original_vars)
                result.canonical_refusal_reason = validation_reason

    # Add auxiliary definitions to result
    result.auxiliary_defs = all_auxiliary_defs

    # Pass assignment rules (time-only auxiliaries) to result
    result.assignment_rules = sym.assignment_rules
    result.algebraic_constraints = list(sym.algebraic_constraints)
    result.solver_requirement = classify_solver_requirement(result)

    # Propagate simulation metadata from input SymSystem
    result.sim_t_start = sym.sim_t_start
    result.sim_t_end = sym.sim_t_end
    result.sim_n_steps = sym.sim_n_steps
    result.eps_slack = sym.eps_slack  # Propagate user-specified EPS_SLACK

    # CRITICAL: If any IC was perturbed to EPS_INIT (for zero approximation),
    # we must record the EPS_INIT value used in the output for reproducibility.
    # Check if any IC is approximately equal to EPS_INIT.
    eps_init_used = sym.eps_init if sym.eps_init is not None else EPS_INIT
    ic_was_perturbed = any(
        abs(v - eps_init_used) < 1e-12
        for v in result.initials.values()
        if isinstance(v, (int, float))
    )

    if ic_was_perturbed:
        # Record the actual EPS_INIT value used
        result.eps_init = eps_init_used
    else:
        # No perturbation - only propagate user-specified value
        result.eps_init = sym.eps_init

    return result


def _gma_recast(sym: SymSystem, original_vars: set[sp.Symbol]) -> RecastResult:
    """
    GMA (Generalized Mass Action) recast for systems with multiple flux channels.

    Preserves all production and degradation terms exactly without forcing them
    into canonical S-system form. Each ODE can have multiple terms on each side.
    """
    gma_equations: list[GMAEquation] = []
    new_initials: dict[sp.Symbol, float] = dict(sym.initials)
    new_variables: list[sp.Symbol] = list(sym.vars)
    factor_map: dict[sp.Symbol, list[sp.Symbol]] = {}

    for var in sorted(sym.vars, key=lambda s: s.name):
        # Get ODE - keep parameters symbolic
        rhs = sp.simplify(sym.odes[var])

        # Expand to terms
        terms = expand_to_terms(rhs)
        growth_terms, decay_terms = _analyze_ode_terms(terms)

        # Create GMA equation preserving all terms
        gma_equations.append(GMAEquation(var=var, production=growth_terms, degradation=decay_terms))

        # Original variables map to themselves
        if var in original_vars:
            factor_map[var] = [var]

    return RecastResult(
        status=RecastStatus.GMA,
        equations=[],  # GMA doesn't use SSysEquation format
        initials=new_initials,
        variables=new_variables,
        factor_map=factor_map,
        gma_equations=gma_equations,
        params=sym.params,
        compartments=sym.compartments,  # Propagate compartments from original
        initial_exprs=sym.initial_exprs,  # Propagate symbolic IC expressions
        assignment_rules=sym.assignment_rules,  # Preserve original assignment rules
    )


def _direct_ssystem_recast(
    sym: "SymSystem", original_vars: set[sp.Symbol], mode: str = "simplified"
) -> "RecastResult":
    """
    Direct S-system recast for systems with lifted rational/composite functions.

    Simply converts each ODE to growth-decay form without pool construction.
    This preserves the mathematical relationships of lifted auxiliaries.

    IMPORTANT: Checks if any equation has >2 monomial terms with different
    exponent patterns. If so, returns GMA format instead of claiming canonical.

    Args:
        sym: SymSystem to recast
        original_vars: Set of original variables before lifting
        mode: Output mode ('simplified' or 'canonical')
    """
    new_equations: list[SSysEquation] = []
    new_variables: list[sp.Symbol] = []
    new_initials: dict[sp.Symbol, float] = dict(sym.initials)
    factor_map: dict[sp.Symbol, list[sp.Symbol]] = {}
    state_vars = set(sym.vars)

    # CRITICAL: Deduplicate variables to avoid duplicate entries in output
    # Use dict to preserve order while removing duplicates
    seen_vars = {}
    for var in sym.vars:
        if var not in seen_vars:
            seen_vars[var] = True
    deduplicated_vars = list(seen_vars.keys())

    # Check if any ODE has multiple terms with different exponent patterns
    # If so, we need GMA format, not canonical S-system
    needs_gma = False

    for var in sorted(deduplicated_vars, key=lambda s: s.name):
        new_variables.append(var)

        # Get ODE - keep parameters symbolic
        rhs = sp.simplify(sym.odes[var])

        # Expand to terms
        terms = expand_to_terms(rhs)

        # Use robust sign analysis that handles symbolic coefficients
        growth_terms, decay_terms = _analyze_ode_terms(terms, state_vars)

        # Check if growth terms have different exponent patterns
        if len(growth_terms) > 1:
            first_exps = growth_terms[0][1]
            for _, exps in growth_terms[1:]:
                if not _exponents_match(first_exps, exps):
                    needs_gma = True
                    break

        # Check if decay terms have different exponent patterns
        if len(decay_terms) > 1:
            first_exps = decay_terms[0][1]
            for _, exps in decay_terms[1:]:
                if not _exponents_match(first_exps, exps):
                    needs_gma = True
                    break

        # Combine growth terms (sum coefficients, keep as symbolic)
        if growth_terms:
            g_coeff = sum((c for c, _ in growth_terms), sp.Integer(0))
            # For direct mode: don't average exponents, just use first term's exponents
            # (all terms should have same structure after lifting)
            g_exps = growth_terms[0][1] if growth_terms else {}
        else:
            g_coeff, g_exps = sp.Integer(0), {}

        # Combine decay terms (sum coefficients, keep as symbolic)
        if decay_terms:
            d_coeff = sum((c for c, _ in decay_terms), sp.Integer(0))
            # For direct mode: use first term's exponents
            d_exps = decay_terms[0][1] if decay_terms else {}
        else:
            d_coeff, d_exps = sp.Integer(0), {}

        # Add equation
        new_equations.append(SSysEquation(var, (g_coeff, g_exps), (d_coeff, d_exps)))

        # Original variables map to themselves (no factorization)
        if var in original_vars:
            factor_map[var] = [var]

    # If any equation needs GMA, return GMA format instead
    if needs_gma:
        return _gma_recast(sym, original_vars)

    # Build result (no name canonicalization needed for direct form)
    return RecastResult(
        status=RecastStatus.CANONICAL_SSYSTEM,
        equations=new_equations,
        initials=new_initials,
        variables=new_variables,
        factor_map=factor_map,
        params=sym.params,
        compartments=sym.compartments,  # Propagate compartments from original
        initial_exprs=sym.initial_exprs,  # Propagate symbolic IC expressions
        assignment_rules=sym.assignment_rules,  # Preserve original assignment rules
    )


def _pool_ssystem_recast(sym: "SymSystem", mode: str = "simplified") -> "RecastResult":
    """
    Pool construction S-system recast for pure polynomial systems.

    This is the original pool method that works well for systems without
    rational or composite functions.

    Args:
        sym: SymSystem to recast
        mode: Output mode ('simplified' or 'canonical')
    """
    new_equations: list[SSysEquation] = []
    new_variables: list[sp.Symbol] = []
    new_initials: dict[sp.Symbol, float] = dict(sym.initials)  # keep params and originals
    factor_map: dict[sp.Symbol, list[sp.Symbol]] = {}

    for Xi in sorted(sym.vars, key=lambda s: s.name):
        # Original variables: apply pool construction
        # 1) decompose RHS into signed monomial terms over ORIGINAL symbols
        rhs = sp.simplify(sym.odes[Xi])
        # Keep parameters symbolic - DO NOT substitute
        terms = expand_to_terms(rhs)
        state_vars = set(sym.vars)
        mono_terms: list[tuple[float, dict[sp.Symbol, float]]] = []
        for t in terms:
            if t == 0:
                continue
            coeff, exps = term_to_coeff_exps(t, state_vars)  # coeff may be ±
            mono_terms.append((coeff, exps))

        # Handle degenerate X' == 0
        # Use original variable's initial condition, not hardcoded 1.0
        if not mono_terms:
            V = sp.symbols(f"{Xi.name}_t1", positive=True)
            new_variables.append(V)
            xi0 = float(new_initials.get(Xi, 1.0))
            new_initials[V] = xi0 if xi0 != 0.0 else 1.0  # Preserve non-zero IC
            new_equations.append(SSysEquation(V, (0.0, {}), (0.0, {})))
            factor_map[Xi] = [V]
            continue

        # 2) create one auxiliary per term
        V_list: list[sp.Symbol] = []
        for j in range(len(mono_terms)):
            Vj = sp.symbols(f"{Xi.name}_t{j + 1}", positive=True)
            V_list.append(Vj)
            new_variables.append(Vj)
            new_initials.setdefault(Vj, 1.0)

        # 3) define each V_j' per the pool formula; EXCLUDE V_j from the denominator
        for j, (coeff, exps_orig) in enumerate(mono_terms):
            Vj = V_list[j]
            exps = dict(exps_orig)  # start with original-variable exponents

            # Multiply by (∏_{ℓ≠j} V_ℓ)^(-1)  → add -1 exponent for every V_k with k != j
            for k, Vk in enumerate(V_list):
                if k == j:  # exclude V_j itself!
                    continue
                exps[Vk] = exps.get(Vk, 0.0) - 1.0

            # Assign growth/decay by sign of coeff (works for symbolic and numeric)
            # Handle symbolic coefficients containing 'time' or other symbols
            is_positive = _is_coefficient_positive(coeff)

            if is_positive:
                new_equations.append(
                    SSysEquation(
                        var=Vj,
                        growth=(coeff, exps),  # Use coeff directly (already positive)
                        decay=(sp.Integer(0), {}),
                    )
                )
            else:
                # Use -coeff instead of sp.Abs(coeff) to handle symbolic coefficients
                new_equations.append(
                    SSysEquation(var=Vj, growth=(sp.Integer(0), {}), decay=(-coeff, exps))
                )

        # 4) mapping X = ∏_j V_j and initial consistency at t=0
        factor_map[Xi] = list(V_list)
        # Match by name to avoid symbol identity mismatch (SBML parser vs pool vars)
        # Also check params as fallback (SBML parser may put species ICs in params)
        xi0 = 1.0
        for s, v in new_initials.items():
            if hasattr(s, "name") and s.name == Xi.name:
                xi0 = float(v)
                break
        else:
            # Fallback: SBML parser may put species IC in params dict
            if Xi.name in sym.params:
                xi0 = float(sym.params[Xi.name])

        # Set initial conditions for pool auxiliaries
        if V_list:
            if xi0 > 0.0 and xi0 >= EPS_INIT:
                # Positive initial condition: first aux = xi0, others = 1.0
                # This ensures Xi(0) = xi0 * 1 * 1 * ... = xi0
                new_initials[V_list[0]] = xi0
                for Vj in V_list[1:]:
                    new_initials.setdefault(Vj, 1.0)
            else:
                # Zero or near-zero initial condition
                # Only use EPS_INIT if variable appears with negative exponents
                # (will be determined after all equations are built)
                new_initials[V_list[0]] = 0.0  # Placeholder, will adjust later
                for Vj in V_list[1:]:
                    new_initials.setdefault(Vj, 1.0)

    # 5) Detect which variables have negative exponents AFTER factor_map expansion
    # The exponent dict may contain original vars (x, y, z) that get expanded via factor_map.
    # We must expand the factor_map to get the TRUE final exponents before deciding which
    # variables need EPS_INIT for division-by-zero protection.
    #
    # Example: if z = Z_5*Z_6*Z_7 and an exponent dict has {z: 1, Z_5: -1, Z_7: -1}
    #   After expansion: Z_5^1 * Z_6^1 * Z_7^1 * Z_5^-1 * Z_7^-1 = Z_6^1
    #   So Z_5 and Z_7 don't actually appear with negative exponents after expansion!

    def expand_exponents_via_factor_map(exps: dict) -> dict:
        """Expand original variables to pool variables and sum exponents."""
        expanded: dict[sp.Symbol, float] = {}
        for var, exp in exps.items():
            # Handle different exponent types
            if isinstance(exp, (int, float)):
                exp_val = float(exp)
            elif isinstance(exp, sp.Expr):
                # Only convert to float if it's actually a number
                if exp.is_number:
                    try:
                        exp_val = float(exp)
                    except (TypeError, ValueError):
                        exp_val = 1.0  # Fallback for un-evaluable
                else:
                    # Symbolic exponent (parameter) - treat as positive
                    # to be conservative (won't trigger EPS_INIT)
                    exp_val = 1.0
            else:
                exp_val = 1.0

            if var in factor_map:
                # Original var: expand via factor_map (e.g., x -> Z_1*Z_2)
                for pool_var in factor_map[var]:
                    expanded[pool_var] = expanded.get(pool_var, 0.0) + exp_val
            else:
                # Already a pool var or parameter
                expanded[var] = expanded.get(var, 0.0) + exp_val
        return expanded

    vars_with_neg_exp = set()
    for eq in new_equations:
        # Check growth exponents (expanded)
        expanded_growth = expand_exponents_via_factor_map(eq.growth[1])
        for var, exp in expanded_growth.items():
            if isinstance(exp, (int, float)) and exp < 0:
                vars_with_neg_exp.add(var)
            elif isinstance(exp, sp.Expr) and exp.is_number and float(exp) < 0:
                vars_with_neg_exp.add(var)
        # Check decay exponents (expanded)
        expanded_decay = expand_exponents_via_factor_map(eq.decay[1])
        for var, exp in expanded_decay.items():
            if isinstance(exp, (int, float)) and exp < 0:
                vars_with_neg_exp.add(var)
            elif isinstance(exp, sp.Expr) and exp.is_number and float(exp) < 0:
                vars_with_neg_exp.add(var)

    # 6) Adjust zero initial conditions: use EPS_INIT only for vars with negative exponents
    # Use user-specified eps_init if available, otherwise use module default
    eps_init_value = sym.eps_init if sym.eps_init is not None else EPS_INIT
    for var in new_variables:
        if var in new_initials and abs(new_initials[var]) < 1e-14:
            # This variable has zero IC
            if var in vars_with_neg_exp:
                # Has negative exponents - use eps_init to prevent division by zero
                new_initials[var] = eps_init_value
            else:
                # No negative exponents - keep exact zero
                new_initials[var] = 0.0

    # 7) build result and canonicalize names to Z_1, Z_2, ...
    # Filter out compartments from params to avoid duplicate output
    filtered_params = {k: v for k, v in sym.params.items() if k not in sym.compartments}

    res = RecastResult(
        status=RecastStatus.CANONICAL_SSYSTEM,
        equations=new_equations,
        initials=new_initials,
        variables=new_variables,
        factor_map=factor_map,
        params=filtered_params,
        compartments=sym.compartments,  # Propagate compartments
    )
    return canonicalize_aux_names(res, prefix="Z")
