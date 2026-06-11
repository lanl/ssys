# mypy: ignore-errors
# ruff: noqa: F401, F403, F405, I001
"""Antimony formatting for GMA and S-system recast results."""

from ssys._recaster.common import *
from ssys._recaster.names import (
    _build_name_sanitization_map,
    _collect_antimony_names,
    _format_antimony_token,
    _sanitize_antimony_name,
)
from ssys._recaster.parsing import _sympy_to_antimony_syntax

def product_to_antimony(
    coeff, exps: dict[sp.Symbol, float], name_map: dict[str, str] | None = None
) -> str:
    """
    Format coefficient and exponents as Antimony expression string.
    coeff can be either float or sp.Expr (symbolic).
    Exponents can also be symbolic expressions.

    Args:
        coeff: Coefficient (numeric or symbolic)
        exps: Dictionary mapping symbols to exponents
        name_map: Optional mapping of original names to sanitized names
                  (for Antimony reserved keyword handling)
    """
    parts: list[str] = []

    # Check if we have dummy_const with exponent 0 (special case for constants)
    has_dummy_const_zero = any(
        s.name == "dummy_const"
        and (
            (isinstance(e, sp.Expr) and sp.simplify(e) == 0)
            or (not isinstance(e, sp.Expr) and abs(e) < 1e-14)
        )
        for s, e in exps.items()
    )

    # Handle coefficient (numeric or symbolic)
    if isinstance(coeff, sp.Expr):
        # Symbolic coefficient - format it cleanly
        coeff_simplified = sp.simplify(coeff)
        if coeff_simplified == 0:
            return "0"
        elif coeff_simplified != 1 or has_dummy_const_zero:
            # Always show coefficient if we have dummy_const^0 (even if coeff=1)
            # Check if coefficient is a sum (needs parentheses)
            if coeff_simplified.is_Add:
                # Format as a single parenthesized expression
                coeff_str = _format_antimony_token(coeff_simplified, name_map, expression=True)
                parts.append(f"({coeff_str})")
            else:
                # Break symbolic coefficient into factors for clean formatting
                coeff_factors = _format_symbolic_coeff(coeff_simplified, name_map)
                if coeff_factors:
                    parts.extend(coeff_factors)
    else:
        # Numeric coefficient
        if coeff == 0.0:
            return "0"
        elif coeff != 1.0 or has_dummy_const_zero:
            # Always show coefficient if we have dummy_const^0 (even if coeff=1)
            parts.append(_format_antimony_number(coeff))

    # Add power-law terms
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        # Special case: dummy_const with exponent 0 should always be shown
        # This represents constant terms as C * dummy_const^0
        is_dummy_const = s.name == "dummy_const"

        # Get sanitized name for this symbol
        s_name = _format_antimony_token(s, name_map)

        # Handle both numeric and symbolic exponents
        if isinstance(e, sp.Expr):
            e_simplified = sp.simplify(e)
            if e_simplified == 0 and not is_dummy_const:
                continue
            # Check if it's effectively 1.0 (including sympy Float)
            elif e_simplified == 1 or (
                e_simplified.is_Number and abs(float(e_simplified) - 1.0) < 1e-10
            ):
                parts.append(s_name)
            else:
                # Format as integer if it's an integer value
                if e_simplified.is_Number:
                    e_val = float(e_simplified)
                    if abs(e_val - round(e_val)) < 1e-10:
                        parts.append(f"{s_name}^{int(round(e_val))}")
                    else:
                        parts.append(f"{s_name}^{e_simplified}")
                else:
                    # Symbolic exponent - add parentheses if it's a sum/difference
                    if e_simplified.is_Add:
                        exp_str = _format_antimony_token(
                            e_simplified, name_map, expression=True
                        )
                        parts.append(f"{s_name}^({exp_str})")
                    else:
                        exp_str = _format_antimony_token(
                            e_simplified, name_map, expression=True
                        )
                        parts.append(f"{s_name}^{exp_str}")
        else:
            # Numeric exponent
            # Always show dummy_const even with exponent 0
            if abs(e) < 1e-14 and not is_dummy_const:
                continue
            if abs(e - 1.0) < 1e-14:
                parts.append(f"{s_name}")
            else:
                parts.append(f"{s_name}^{_format_antimony_number(e)}")

    # Special case: if parts is empty, return the coefficient string
    # This handles pure constants like "1" (coeff=1, exps={})
    if not parts:
        # If we have a non-zero coefficient but no variables, return coefficient as string
        if isinstance(coeff, sp.Expr):
            coeff_simplified = sp.simplify(coeff)
            if coeff_simplified == 0:
                return "0"
            else:
                return _format_antimony_token(coeff_simplified, name_map, expression=True)
        else:
            # Numeric coefficient
            if coeff == 0.0:
                return "0"
            else:
                return _format_antimony_number(coeff)
    return "*".join(parts)


def _format_symbolic_coeff(
    coeff: sp.Expr, name_map: dict[str, str] | None = None
) -> list[str]:
    """
    Format a symbolic coefficient cleanly by extracting factors.
    Returns list of string parts to be joined with '*'.
    """
    parts: list[str] = []

    # If it's a multiplication, extract factors
    if coeff.is_Mul:
        for factor in coeff.args:
            part = _format_factor(factor, name_map)
            if part:
                parts.append(part)
    else:
        # Single factor
        part = _format_factor(coeff, name_map)
        if part:
            parts.append(part)

    return parts


def _format_factor(factor: sp.Expr, name_map: dict[str, str] | None = None) -> str:
    """Format a single factor from a coefficient."""
    # Pure number
    if factor.is_Number:
        # Check if it's a rational number with non-trivial denominator - keep as fraction
        # This preserves exact values like 1/6 instead of 0.166667
        if isinstance(factor, sp.Rational) and factor.q != 1:
            # Format as (p/q) for Antimony - parentheses ensure correct precedence
            return f"({factor.p}/{factor.q})"
        val = float(factor)
        if val == int(val):
            return str(int(val))
        else:
            return _format_antimony_number(val)

    # Symbol (parameter)
    if isinstance(factor, sp.Symbol):
        return _format_antimony_token(factor, name_map)

    # Power: base^exp
    if isinstance(factor, sp.Pow):
        base, exp = factor.args

        # Check if exponent is 1.0 (skip the exponent entirely)
        if exp.is_Number:
            exp_val = float(exp)
            if abs(exp_val - 1.0) < 1e-10:
                return _format_factor(base)

        # Format base
        if isinstance(base, sp.Symbol):
            base_str = _format_antimony_token(base, name_map)
        elif base.is_Number:
            base_str = _format_antimony_number(base)
        else:
            base_str = f"({_format_factor(base, name_map)})"

        # Format exponent
        if exp.is_Number:
            exp_val = float(exp)
            if exp_val == int(exp_val):
                exp_str = str(int(exp_val))
            else:
                exp_str = _format_antimony_number(exp_val)
        else:
            # Symbolic exponent - wrap in parentheses if it's a sum/difference
            # CRITICAL: Without parentheses, (T+a)^-C-1 parses as (T+a)^(-C) - 1
            # instead of (T+a)^(-C-1), completely corrupting the equation
            if exp.is_Add:
                exp_str = f"({_format_antimony_token(exp, name_map, expression=True)})"
            else:
                exp_str = _format_antimony_token(exp, name_map, expression=True)

        return f"{base_str}^{exp_str}"

    # Sum expression - MUST be wrapped in parentheses to preserve operator precedence
    # This is critical for expressions like H*(A + B + C)*X^-1 where the sum is a factor
    if factor.is_Add:
        expr_str = _format_antimony_token(factor, name_map, expression=True)
        return f"({expr_str})"

    # Anything else - fallback to string representation
    return _format_antimony_token(factor, name_map, expression=True)


def gma_to_antimony(
    result: RecastResult, model_name: str = "recast", lifted_mode: str = "ode"
) -> str:
    """
    Format GMA equations to Antimony with clear labeling.
    Preserves all production/degradation terms exactly.

    Args:
        result: RecastResult to format
        model_name: Name for the output model
        lifted_mode: How to output lifted auxiliary variables (Y_1, Y_2, etc.)
            - "ode": Output as species with ODEs (default, may drift)
            - "assignment": Output as assignment rules (algebraically exact)
    """
    name_map = _build_name_sanitization_map(_collect_antimony_names(result))
    result.solver_requirement = classify_solver_requirement(result, lifted_mode=lifted_mode)

    lines: list[str] = []
    lines.append(f"model {model_name}()")
    lines.extend(_format_solver_metadata_lines(result.solver_requirement))
    lines.append("")

    # --- Identify lifted auxiliaries based on mode ---
    # Lifted auxiliaries have definitions in auxiliary_defs.
    # In "ode" mode: output them as species with ODEs
    # In "assignment" mode: output them as assignment rules (prevents manifold drift)
    # EXCEPTION: Clock variables (T := time) are always state variables (need T' = 1)
    lifted_aux_names: set[str] = set()
    if lifted_mode == "assignment" and result.auxiliary_defs:
        for aux, defn in result.auxiliary_defs.items():
            aux_name = aux.name if hasattr(aux, "name") else str(aux)
            # Check if this is a clock variable (definition is just 'time')
            is_clock = defn == sp.Symbol("time") or str(defn) == "time"
            if not is_clock:
                lifted_aux_names.add(aux_name)

    # --- Compartment declaration (for SBML compatibility) ---
    # Use original compartment name if available, otherwise default to "cell"
    if result.compartments:
        for comp_name, comp_size in result.compartments.items():
            comp_id = _format_antimony_token(comp_name, name_map)
            lines.append(f"compartment {comp_id} = {_format_antimony_number(comp_size)};")
        # Use first compartment name for species declarations
        default_compartment = _format_antimony_token(next(iter(result.compartments.keys())), name_map)
    else:
        lines.append("compartment cell = 1;")
        default_compartment = "cell"
    lines.append("")

    # --- Species declarations in compartment ---
    # Only declare species for variables that are NOT lifted auxiliaries (in assignment mode)
    all_gma_vars = sorted([eq.var for eq in result.gma_equations], key=lambda s: s.name)
    state_vars = [v for v in all_gma_vars if v.name not in lifted_aux_names]
    if state_vars:
        for v in state_vars:
            lines.append(f"species {_format_antimony_token(v, name_map)} in {default_compartment};")
        lines.append("")

    lines.append("// GMA (Generalized Mass Action) format")
    lines.append("// Multiple flux channels with different kinetic orders preserved exactly")

    # Add refusal reason if canonical S-system was not attempted
    if result.canonical_refusal_reason:
        lines.append("//")
        lines.append("// NOTE: Canonical S-system recast was not attempted because:")
        lines.append(f"//   {result.canonical_refusal_reason}")
        lines.append("//")
        lines.append(
            "// Using GMA format preserves exact dynamics with better numerical properties."
        )
    else:
        lines.append("// Cannot be reduced to canonical S-system form without loss of information")

    lines.append("")

    # --- auxiliary variable definitions ---
    # For comments: include ALL auxiliaries (including clock) for documentation
    # For assignment rules: filter out clock variables (they need ODEs, not assignment rules)
    all_aux_defs_for_comments = {}
    filtered_aux_defs_for_rules = {}
    if result.auxiliary_defs:
        for k, v in result.auxiliary_defs.items():
            if k.name == "dummy_const":
                continue  # Internal implementation detail - exclude from both
            # Check if this is a clock variable (definition is just 'time')
            is_clock = v == sp.Symbol("time") or str(v) == "time"
            all_aux_defs_for_comments[k] = v  # Include in comments for validator
            if not is_clock:
                filtered_aux_defs_for_rules[k] = v  # Exclude clock from assignment rules

    if lifted_mode == "assignment" and filtered_aux_defs_for_rules:
        # Output as assignment rules (algebraically exact, prevents drift)
        # EXCLUDE clock variables - they need ODEs, not assignment rules
        lines.append("// ========================================================================")
        lines.append("// LIFTED DENOMINATORS (assignment rules to prevent drift)")
        lines.append("// ========================================================================")
        for aux, defn in sorted(filtered_aux_defs_for_rules.items(), key=lambda kv: str(kv[0])):
            aux_id = _format_antimony_token(aux, name_map)
            defn_str = _format_antimony_token(defn, name_map, expression=True)
            lines.append(f"{aux_id} := {defn_str};")
        lines.append("")
    elif all_aux_defs_for_comments:
        # ODE mode: output definitions as comments for documentation
        # INCLUDE clock variables in comments for validator to recognize
        lines.append("// ========================================================================")
        lines.append("// AUXILIARY DEFINITIONS (for lifted variables)")
        lines.append("// ========================================================================")
        for aux, defn in sorted(all_aux_defs_for_comments.items(), key=lambda kv: str(kv[0])):
            lines.append(f"// {aux} := {defn}")
        lines.append("// ========================================================================")
        lines.append("")

    # --- Parameters (copied from original model) ---
    if result.params:
        lines.append("// Parameters (from original model)")
        for param_name in sorted(result.params.keys()):
            # Skip parameters that are actually assignment rules (they're computed, not constants)
            if result.assignment_rules and param_name in result.assignment_rules:
                continue
            param_val = result.params[param_name]
            lines.append(
                f"{_format_antimony_token(param_name, name_map)} = "
                f"{_format_antimony_number(param_val)};"
            )
        lines.append("")

    # --- Assignment rules from original model (time-dependent quantities) ---
    if result.assignment_rules:
        lines.append("// Assignment rules (from original model)")
        for var_name in sorted(result.assignment_rules.keys()):
            expr = result.assignment_rules[var_name]
            sanitized_expr = _format_antimony_token(expr, name_map, expression=True)
            lines.append(f"{_format_antimony_token(var_name, name_map)} := {sanitized_expr};")
        lines.append("")

    # Initial assignments - handle both Symbol and tuple keys
    # Skip variables that are assignment rules (they don't have ICs)
    def _init_sort_key(kv):
        k = kv[0]
        if hasattr(k, "name"):
            return k.name
        elif isinstance(k, tuple):
            return str(k)
        else:
            return str(k)

    for s, v in sorted(result.initials.items(), key=_init_sort_key):
        # Skip tuple keys (compartment info, const params)
        if not hasattr(s, "name"):
            continue
        # Skip variables that are assignment rules (they use := not =)
        if result.assignment_rules and s.name in result.assignment_rules:
            continue
        # Skip lifted auxiliaries (they're assignment rules, not species)
        if s.name in lifted_aux_names:
            continue
        lines.append(
            f"{_format_antimony_token(s, name_map)} = {_format_antimony_number(v)};"
        )

    lines.append("")

    # GMA ODEs with multiple terms per side
    # CRITICAL: Skip ODEs for lifted auxiliaries (they're assignment rules now)
    for eq in result.gma_equations:
        # Skip lifted auxiliaries - they're assignment rules, not species with ODEs
        if eq.var.name in lifted_aux_names:
            continue
        if result.assignment_rules and eq.var.name in result.assignment_rules:
            continue

        # Format production terms
        if eq.production:
            prod_strs = [product_to_antimony(c, e, name_map) for c, e in eq.production]
            production = " + ".join(prod_strs)
        else:
            production = "0"

        # Format degradation terms
        if eq.degradation:
            deg_strs = [product_to_antimony(c, e, name_map) for c, e in eq.degradation]
            degradation = " + ".join(deg_strs)
        else:
            degradation = "0"

        # Write ODE - output directly for GMA format (no transformation)
        # Pure constants like T' = 1 should be output as-is
        var_id = _format_antimony_token(eq.var, name_map)
        if degradation == "0":
            lines.append(f"{var_id}' = {production};")
        else:
            lines.append(f"{var_id}' = {production} - ({degradation});")

    lines.append("end")

    # Add @SIM metadata AFTER end (file-level metadata, not model content)
    sim_lines = _format_sim_metadata_lines(result)
    if sim_lines:
        lines.append("")
        lines.extend(sim_lines)

    # Convert ** to ^ for valid Antimony syntax
    return _sympy_to_antimony_syntax("\n".join(lines))


def ssystem_to_antimony(
    result, model_name: str = "recast", mode: str = "simplified", lifted_mode: str = "ode"
) -> str:
    """
    Format canonical S-system or GMA to Antimony based on result status.

    Args:
        result: RecastResult to format
        model_name: Name for the output model
        mode: Output mode ('simplified' or 'canonical')
            - 'simplified': Basic format with comments
            - 'canonical': Enhanced format with species declarations, observables, and detailed comments
        lifted_mode: How to output lifted auxiliary variables (Y_1, Y_2, etc.)
            - 'ode': Output as species with ODEs (default, may drift)
            - 'assignment': Output as assignment rules (algebraically exact)
    """
    # CRITICAL: Antimony identifiers cannot start with numbers, contain hyphens, or periods
    # Prefix with 'm_' if name starts with digit, replace invalid chars with underscores
    if model_name:
        # Replace periods with underscores (period is invalid in identifiers)
        model_name = model_name.replace(".", "_")
        # Replace hyphens with underscores (hyphen is subtraction in Antimony)
        model_name = model_name.replace("-", "_")
        # Prefix with 'm_' if name starts with digit
        if model_name[0].isdigit():
            model_name = f"m_{model_name}"
        # Sanitize reserved keywords (e.g., model_name="model" -> "model_var")
        model_name = _sanitize_antimony_name(model_name)

    # Check if recasting failed
    if result.status == RecastStatus.FAILED:
        return _failed_to_antimony(result, model_name)

    result.solver_requirement = classify_solver_requirement(result, lifted_mode=lifted_mode)

    # Check if this is GMA format
    if result.status == RecastStatus.GMA:
        return gma_to_antimony(result, model_name, lifted_mode=lifted_mode)

    # Route to appropriate formatter based on mode
    if mode == "canonical":
        return _ssystem_to_antimony_canonical(result, model_name)
    else:
        return _ssystem_to_antimony_simplified(result, model_name)


def _failed_to_antimony(result: RecastResult, model_name: str) -> str:
    """Format a failed recast result with error message."""
    lines: list[str] = []
    lines.append(f"model {model_name}()")
    lines.extend(_format_solver_metadata_lines(result.solver_requirement))
    lines.append("")
    lines.append("// ========================================================================")
    lines.append("// RECAST FAILED")
    lines.append("// ========================================================================")
    lines.append("//")

    if result.error_message:
        # Format error message as comments
        for line in result.error_message.split("\n"):
            lines.append(f"// {line}")
    else:
        lines.append("// Recasting failed for unknown reason.")

    lines.append("//")
    lines.append("// ========================================================================")
    lines.append("")

    # Include original initial conditions if available
    if result.initials:
        lines.append("// Original initial conditions:")
        for s, v in sorted(result.initials.items(), key=lambda kv: kv[0].name):
            lines.append(f"// {s.name} = {_format_antimony_number(v)}")
        lines.append("")

    lines.append("// No recast equations generated.")
    lines.append("")
    lines.append("end")
    return "\n".join(lines)


def _ssystem_to_antimony_simplified(result, model_name: str) -> str:
    """Format S-system in simplified mode with enhanced documentation and assignment rules."""
    name_map = _build_name_sanitization_map(_collect_antimony_names(result))

    lines: list[str] = []
    lines.append(f"model {model_name}()")
    lines.append("")

    # --- Compartment declaration (for SBML compatibility) ---
    # Use original compartment name if available, otherwise default to "cell"
    if result.compartments:
        for comp_name, comp_size in result.compartments.items():
            comp_id = _format_antimony_token(comp_name, name_map)
            lines.append(f"compartment {comp_id} = {_format_antimony_number(comp_size)};")
        # Use first compartment name for species declarations
        default_compartment = _format_antimony_token(next(iter(result.compartments.keys())), name_map)
    else:
        lines.append("compartment cell = 1;")
        default_compartment = "cell"
    lines.append("")

    # --- Species declarations ---
    # All variables with ODEs must be declared as species in the compartment
    all_state_vars = sorted(result.variables, key=lambda s: s.name)
    if all_state_vars:
        for v in all_state_vars:
            lines.append(f"species {_format_antimony_token(v, name_map)} in {default_compartment};")
        lines.append("")

    # --- Enhanced metadata header ---
    lines.append("// ========================================================================")
    lines.append("// RECAST METADATA")
    lines.append("// ========================================================================")
    lines.append(f"// Recast variables: {len(result.variables)}")
    lines.append(f"// Original variables: {len(result.factor_map)}")
    lines.append(f"// Parameters: {len(result.params)}")
    if result.auxiliary_defs:
        lines.append(f"// Auxiliary definitions: {len(result.auxiliary_defs)}")
    lines.append("// ========================================================================")
    lines.append("")

    # --- mapping: original → product of auxiliaries ---
    if result.factor_map:
        lines.append("// ========================================================================")
        lines.append("// VARIABLE MAPPING")
        lines.append("// ========================================================================")
        for orig in sorted(result.factor_map.keys(), key=lambda s: s.name):
            aux = result.factor_map[orig]
            rhs = "*".join(a.name for a in aux) if aux else "1"
            lines.append(f"// {orig.name} = {rhs}")
        lines.append("// ========================================================================")
        lines.append("")

    # --- auxiliary variable definitions ---
    if result.auxiliary_defs:
        lines.append("// ========================================================================")
        lines.append("// AUXILIARY DEFINITIONS (for lifted variables)")
        lines.append("// ========================================================================")
        for aux, defn in sorted(result.auxiliary_defs.items(), key=lambda kv: str(kv[0])):
            lines.append(f"// {aux} := {defn}")
        lines.append("// ========================================================================")
        lines.append("")

    # --- Parameters ---
    # IMPORTANT: Filter out original variable names - they're species, not parameters
    # The SBML parser may include "X = 10" as a parameter when X is really a species IC
    original_var_names = (
        {orig.name for orig in result.factor_map.keys()} if result.factor_map else set()
    )
    param_names_to_output = [n for n in sorted(result.params.keys()) if n not in original_var_names]

    if param_names_to_output:
        lines.append("// ========================================================================")
        lines.append("// PARAMETERS (copied from original)")
        lines.append("// ========================================================================")
        for param_name in param_names_to_output:
            param_val = result.params[param_name]
            param_id = _format_antimony_token(param_name, name_map)
            lines.append(f"{param_id} = {_format_antimony_number(param_val)};")
        lines.append("")

    # --- Assignment rules (time-dependent quantities) ---
    # These are variables that depend only on time, not state variables
    if result.assignment_rules:
        lines.append("// ========================================================================")
        lines.append("// ASSIGNMENT RULES (time-dependent quantities)")
        lines.append("// ========================================================================")
        for var_name in sorted(result.assignment_rules.keys()):
            expr = result.assignment_rules[var_name]
            var_id = _format_antimony_token(var_name, name_map)
            expr_str = _format_antimony_token(expr, name_map, expression=True)
            lines.append(f"{var_id} := {expr_str};")
        lines.append("")

    # --- Initial conditions for auxiliary variables ONLY ---
    # Note: We only output ICs for variables that are in result.variables (the recast auxiliaries).
    # Original variables are reconstructed via assignment rules and should NOT have ICs here.
    # CRITICAL: Variables with assignment rules (Z := f(t)) get their value from the rule,
    # so they MUST NOT have initial conditions (Antimony forbids this).
    lines.append("// ========================================================================")
    lines.append("// INITIAL CONDITIONS (auxiliary variables)")
    lines.append("// ========================================================================")
    # Check if any IC uses EPS_INIT (indicating zero approximation)
    uses_eps_init = any(
        abs(v - EPS_INIT) < 1e-12 for s, v in result.initials.items() if s in result.variables
    )
    if uses_eps_init:
        lines.append(f"// NOTE: Initial conditions near {EPS_INIT} are used to approximate zero")
        lines.append("//       This prevents numerical instability from negative exponents")
        lines.append("//       while maintaining dynamics qualitatively equivalent to zero ICs")

    # Build name-based lookup for state variables (handles symbol object mismatch)
    state_var_names = {v.name for v in result.variables}

    # Build set of assignment rule variable names (these cannot have ICs)
    assignment_rule_vars = set(result.assignment_rules.keys()) if result.assignment_rules else set()

    # Sort initials - handle both Symbol and tuple keys
    def _init_sort_key(kv):
        k = kv[0]
        if hasattr(k, "name"):
            return k.name
        elif isinstance(k, tuple):
            return str(k)
        else:
            return str(k)

    # Track which state variables have been output
    output_state_vars = set()

    for s, v in sorted(result.initials.items(), key=_init_sort_key):
        # Skip tuple keys (compartment info, const params) - only process Symbol keys
        if not hasattr(s, "name"):
            continue
        # Skip variables with assignment rules (their value comes from the rule)
        if s.name in assignment_rule_vars:
            continue
        # Output ICs for state variables
        # Use name-based matching to handle symbol object mismatch
        if s.name in state_var_names:
            # Check if we have a symbolic expression for this IC
            state_id = _format_antimony_token(s, name_map)
            if s in result.initial_exprs:
                # Use symbolic expression
                expr_str = _format_antimony_token(result.initial_exprs[s], name_map, expression=True)
                lines.append(f"{state_id} = {expr_str};")
            else:
                # Use numeric value
                lines.append(f"{state_id} = {_format_antimony_number(v)};")
            output_state_vars.add(s.name)

    # CRITICAL: SBML parser may put species ICs in params instead of initials
    # Check for any state variables whose ICs were not output from initials
    for var_name in sorted(state_var_names):
        if var_name not in output_state_vars and var_name not in assignment_rule_vars:
            # Check if IC is in params
            if var_name in result.params:
                var_id = _format_antimony_token(var_name, name_map)
                lines.append(f"{var_id} = {_format_antimony_number(result.params[var_name])};")

    lines.append("")

    # --- Assignment rules to reconstruct original variables ---
    # Only output assignment rules for non-identity mappings
    non_identity_mappings = []
    if result.factor_map:
        for orig in sorted(result.factor_map.keys(), key=lambda s: s.name):
            aux = result.factor_map[orig]
            # Skip identity mappings (where variable maps to itself)
            if len(aux) == 1 and aux[0] == orig:
                continue
            non_identity_mappings.append((orig, aux))

    if non_identity_mappings:
        lines.append("// ========================================================================")
        lines.append("// OBSERVABLE VARIABLES (reconstructed from auxiliaries)")
        lines.append("// ========================================================================")
        for orig, aux in non_identity_mappings:
            orig_id = _format_antimony_token(orig, name_map)
            if len(aux) > 1:
                # Multiple auxiliaries - product form
                rhs = " * ".join(_format_antimony_token(a, name_map) for a in aux)
                lines.append(f"{orig_id} := {rhs};")
            else:
                # Single auxiliary (but not identity)
                aux_id = _format_antimony_token(aux[0], name_map)
                lines.append(f"{orig_id} := {aux_id};")
        lines.append("")

    # --- S-system dynamics ---
    lines.append("// ========================================================================")
    lines.append("// S-SYSTEM DYNAMICS")
    lines.append("// ========================================================================")
    for eq in result.equations:
        if eq.var.name in assignment_rule_vars:
            continue

        g_exps = _expand_exps_through_factors(eq.growth[1], result.factor_map)
        h_exps = _expand_exps_through_factors(eq.decay[1], result.factor_map)

        # Check for pure constant terms (empty exponent dict)
        g_is_const = len(g_exps) == 0
        h_is_const = len(h_exps) == 0

        # For simplified mode, output constant terms directly
        if g_is_const and not h_is_const:
            # Pure constant production: X' = C - h(vars)
            g_coeff = eq.growth[0]
            # Format constant coefficient (numeric or symbolic)
            if isinstance(g_coeff, sp.Expr):
                g_coeff_simplified = sp.simplify(g_coeff)
                if g_coeff_simplified.is_Number:
                    g_str = _format_antimony_number(g_coeff_simplified)
                else:
                    # Symbolic constant (contains parameters)
                    g_str = _format_antimony_token(
                        g_coeff_simplified, name_map, expression=True
                    )
            else:
                g_str = _format_antimony_number(g_coeff)
            h = product_to_antimony(eq.decay[0], h_exps, name_map)
            var_id = _format_antimony_token(eq.var, name_map)
            lines.append(f"{var_id}' = {g_str} - {h};")
        elif h_is_const and not g_is_const:
            # Pure constant decay: X' = g(vars) - C
            h_coeff = eq.decay[0]
            # Format constant coefficient (numeric or symbolic)
            if isinstance(h_coeff, sp.Expr):
                h_coeff_simplified = sp.simplify(h_coeff)
                if h_coeff_simplified.is_Number:
                    h_str = _format_antimony_number(h_coeff_simplified)
                else:
                    # Symbolic constant (contains parameters)
                    h_str = _format_antimony_token(
                        h_coeff_simplified, name_map, expression=True
                    )
            else:
                h_str = _format_antimony_number(h_coeff)
            g = product_to_antimony(eq.growth[0], g_exps, name_map)
            var_id = _format_antimony_token(eq.var, name_map)
            lines.append(f"{var_id}' = {g} - {h_str};")
        elif g_is_const and h_is_const:
            # Both constants: X' = C1 - C2
            g_coeff = eq.growth[0]
            h_coeff = eq.decay[0]
            # Format both coefficients (numeric or symbolic)
            if isinstance(g_coeff, sp.Expr):
                g_coeff_simplified = sp.simplify(g_coeff)
                if g_coeff_simplified.is_Number:
                    g_str = _format_antimony_number(g_coeff_simplified)
                else:
                    g_str = _format_antimony_token(
                        g_coeff_simplified, name_map, expression=True
                    )
            else:
                g_str = _format_antimony_number(g_coeff)
            if isinstance(h_coeff, sp.Expr):
                h_coeff_simplified = sp.simplify(h_coeff)
                if h_coeff_simplified.is_Number:
                    h_str = _format_antimony_number(h_coeff_simplified)
                else:
                    h_str = _format_antimony_token(
                        h_coeff_simplified, name_map, expression=True
                    )
            else:
                h_str = _format_antimony_number(h_coeff)
            var_id = _format_antimony_token(eq.var, name_map)
            lines.append(f"{var_id}' = {g_str} - {h_str};")
        else:
            # Normal monomial form
            g = product_to_antimony(eq.growth[0], g_exps, name_map)
            h = product_to_antimony(eq.decay[0], h_exps, name_map)
            var_id = _format_antimony_token(eq.var, name_map)
            lines.append(f"{var_id}' = {g} - {h};")

    lines.append("end")

    # Add @SIM metadata AFTER end (file-level metadata, not model content)
    sim_lines = _format_sim_metadata_lines(result)
    if sim_lines:
        lines.append("")
        lines.extend(sim_lines)

    # Convert ** to ^ for valid Antimony syntax
    # NOTE: We do NOT apply global name sanitization here because it would corrupt
    # Antimony keywords. For example, "compartment compartment_var = 1" would become
    # "compartment_var compartment_var = 1" if we replaced all "compartment" occurrences.
    # Instead, sanitization is applied during line construction using sanitize() calls.
    output = _sympy_to_antimony_syntax("\n".join(lines))
    return output


def _ssystem_to_antimony_canonical(result, model_name: str) -> str:
    """
    Format S-system in canonical mode with enhanced annotations.

    Features:
    - Species declarations for all auxiliary variables
    - Observable variables showing original-to-auxiliary mappings
    - Detailed explanatory comments
    - Clean equation formatting
    """
    name_map = _build_name_sanitization_map(_collect_antimony_names(result))

    lines: list[str] = []

    # Model declaration with _SSystem suffix
    if not model_name.endswith("_SSystem") and not model_name.endswith("_SSystem_exact"):
        model_name = f"{model_name}_SSystem_exact"
    lines.append(f"model {model_name}()")
    lines.extend(["  " + line for line in _format_solver_metadata_lines(result.solver_requirement)])
    lines.append("")

    # Identify auxiliary and original variables
    aux_vars = list(result.variables)
    orig_vars = sorted(result.factor_map.keys(), key=lambda s: s.name)

    # --- mapping: original → product of auxiliaries ---
    if result.factor_map:
        lines.append("  // Mapping from original variables to canonical auxiliaries (product form)")
        for orig in orig_vars:
            aux = result.factor_map[orig]
            rhs = "*".join(a.name for a in aux) if aux else "1"
            lines.append(f"  // {orig.name} = {rhs}")
        lines.append("  // --- end mapping ---")
        lines.append("")

    # --- auxiliary variable definitions (for lifted variables) ---
    if result.auxiliary_defs:
        lines.append(
            "  // ========================================================================"
        )
        lines.append("  // AUXILIARY DEFINITIONS (for lifted variables)")
        lines.append(
            "  // ========================================================================"
        )
        for aux, defn in sorted(result.auxiliary_defs.items(), key=lambda kv: str(kv[0])):
            lines.append(f"  // {aux} := {defn}")
        lines.append(
            "  // ========================================================================"
        )
        lines.append("")

    # Species declarations for auxiliary variables
    if aux_vars:
        species_names = ", ".join([_format_antimony_token(v, name_map) for v in aux_vars])
        lines.append(f"  species {species_names};")
        lines.append("")

    # Parameter declarations (from result.params)
    # IMPORTANT: Filter out parameters that are assignment rules (they're computed, not constants)
    if result.params:
        lines.append("  // Parameters")
        for param_name in sorted(result.params.keys()):
            # Skip parameters that are actually assignment rules
            if result.assignment_rules and param_name in result.assignment_rules:
                continue
            param_val = result.params[param_name]
            param_id = _format_antimony_token(param_name, name_map)
            lines.append(f"  {param_id} = {_format_antimony_number(param_val)};")
        lines.append("")

    # --- Assignment rules from original model (time-dependent quantities) ---
    # These define quantities like mu := mu_max * S / (K_S + S + S^2/K_I)
    # CRITICAL: Must come AFTER observable definitions so referenced variables exist
    # But we need to define observables first - do that now with original names

    # Define original variables as observables with original names (not _obs suffix)
    # This is needed so assignment rules that reference them (like mu := f(S)) work
    # CRITICAL: Skip identity mappings (X = X) - these cause "Loop detected" errors
    if orig_vars:
        non_identity_mappings = []
        for orig in orig_vars:
            aux_list = result.factor_map[orig]
            # Skip identity mappings (where variable maps to itself)
            if len(aux_list) == 1 and aux_list[0] == orig:
                continue
            rhs = (
                " * ".join([_format_antimony_token(a, name_map) for a in aux_list])
                if aux_list
                else "1"
            )
            non_identity_mappings.append((orig, rhs))

        if non_identity_mappings:
            lines.append("  // Observable variables (reconstructed from auxiliaries)")
            for orig, rhs in non_identity_mappings:
                orig_id = _format_antimony_token(orig, name_map)
                lines.append(f"  {orig_id} := {rhs};")
            lines.append("")

    # Now output assignment rules
    if result.assignment_rules:
        lines.append("  // Assignment rules (from original model)")
        for var_name in sorted(result.assignment_rules.keys()):
            expr = result.assignment_rules[var_name]
            var_id = _format_antimony_token(var_name, name_map)
            expr_str = _format_antimony_token(expr, name_map, expression=True)
            lines.append(f"  {var_id} := {expr_str};")
        lines.append("")

    assignment_rule_vars = set(result.assignment_rules.keys()) if result.assignment_rules else set()

    # Add slack variable if needed (for pure decay OR pure growth terms)
    needs_slack = False
    for eq in result.equations:
        g_coeff = eq.growth[0]
        h_coeff = eq.decay[0]
        # Check if we have a pure decay (growth is 0) or pure growth (decay is 0)
        g_is_zero = (isinstance(g_coeff, (int, float)) and g_coeff == 0) or (
            isinstance(g_coeff, sp.Expr) and g_coeff == sp.Integer(0)
        )
        h_is_zero = (isinstance(h_coeff, (int, float)) and h_coeff == 0) or (
            isinstance(h_coeff, sp.Expr) and h_coeff == sp.Integer(0)
        )
        if g_is_zero or h_is_zero:
            needs_slack = True
            break

    if needs_slack:
        # Use user-specified EPS_SLACK if available, otherwise use module default
        eps_slack_value = result.eps_slack if result.eps_slack is not None else EPS_SLACK
        lines.append("  // Slack variable (keeps both coefficients >0)")
        lines.append(f"  epsilon = {_format_antimony_number(eps_slack_value)};")
        lines.append("")

    # Canonical S-system dynamics with clean formatting and slack variables
    lines.append("  // Canonical S-system dynamics (two monomials per ODE)")
    for eq in result.equations:
        if eq.var.name in assignment_rule_vars:
            continue

        g_exps = _expand_exps_through_factors(eq.growth[1], result.factor_map)
        h_exps = _expand_exps_through_factors(eq.decay[1], result.factor_map)
        g_coeff = eq.growth[0]
        h_coeff = eq.decay[0]
        var_id = _format_antimony_token(eq.var, name_map)

        # Check if growth or decay is zero (need slack variable)
        g_is_zero = (isinstance(g_coeff, (int, float)) and g_coeff == 0) or (
            isinstance(g_coeff, sp.Expr) and g_coeff == sp.Integer(0)
        )
        h_is_zero = (isinstance(h_coeff, (int, float)) and h_coeff == 0) or (
            isinstance(h_coeff, sp.Expr) and h_coeff == sp.Integer(0)
        )

        if g_is_zero and not h_is_zero:
            # Pure decay: X' = 0 - h  =>  X' = epsilon*monomial - (epsilon + h)*monomial
            # Use the decay exponents for both terms
            # If h_exps is empty (pure constant decay), output just coefficients
            if not h_exps:
                # Pure constant: X' = epsilon - (epsilon + h)
                if isinstance(h_coeff, sp.Expr):
                    combined = sp.Symbol("epsilon") + h_coeff
                else:
                    combined = sp.Symbol("epsilon") + sp.Float(h_coeff)
                combined_str = _format_antimony_token(combined, name_map, expression=True)
                lines.append(f"  {var_id}' = epsilon - ({combined_str});")
                continue
            g_str = product_to_antimony(sp.Symbol("epsilon"), h_exps, name_map)
            # Combine epsilon + h_coeff symbolically
            if isinstance(h_coeff, sp.Expr):
                combined_coeff = sp.Symbol("epsilon") + h_coeff
            else:
                combined_coeff = sp.Symbol("epsilon") + sp.Float(h_coeff)
            h_str = product_to_antimony(combined_coeff, h_exps, name_map)
            lines.append(f"  {var_id}' = {g_str} - {h_str};")
        elif h_is_zero and not g_is_zero:
            # Pure growth: X' = g - 0  =>  X' = (g + epsilon)*monomial - epsilon*monomial
            # Use the growth exponents for both terms
            # If g_exps is empty (pure constant growth), output just coefficients
            if not g_exps:
                # Pure constant: X' = (g + epsilon) - epsilon
                if isinstance(g_coeff, sp.Expr):
                    combined = g_coeff + sp.Symbol("epsilon")
                else:
                    combined = sp.Float(g_coeff) + sp.Symbol("epsilon")
                combined_str = _format_antimony_token(combined, name_map, expression=True)
                lines.append(f"  {var_id}' = ({combined_str}) - epsilon;")
                continue
            if isinstance(g_coeff, sp.Expr):
                combined_coeff = g_coeff + sp.Symbol("epsilon")
            else:
                combined_coeff = sp.Float(g_coeff) + sp.Symbol("epsilon")
            g_str = product_to_antimony(combined_coeff, g_exps, name_map)
            h_str = product_to_antimony(sp.Symbol("epsilon"), g_exps, name_map)
            lines.append(f"  {var_id}' = {g_str} - {h_str};")
        else:
            # Both terms present (or both zero) - use as-is
            g = product_to_antimony(g_coeff, g_exps, name_map)
            h = product_to_antimony(h_coeff, h_exps, name_map)
            lines.append(f"  {var_id}' = {g} - {h};")
    lines.append("")

    # Initial conditions
    lines.append("  // Initial conditions")
    for v in aux_vars:
        if v.name in assignment_rule_vars:
            continue
        if v in result.initials:
            # Check if we have a symbolic expression for this IC
            var_id = _format_antimony_token(v, name_map)
            if v in result.initial_exprs:
                # Use symbolic expression
                expr_str = _format_antimony_token(result.initial_exprs[v], name_map, expression=True)
                lines.append(f"  {var_id} = {expr_str};")
            else:
                # Use numeric value
                val = result.initials[v]
                lines.append(f"  {var_id} = {_format_antimony_number(val)};")

    # Add @SIM metadata if available
    sim_lines = _format_sim_metadata_lines(result)
    if sim_lines:
        lines.append("")
        lines.extend(["  " + line for line in sim_lines])  # Indent canonical mode

    lines.append("end")
    # Convert ** to ^ for valid Antimony syntax
    return _sympy_to_antimony_syntax("\n".join(lines))
