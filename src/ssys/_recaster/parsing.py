"""Antimony and SBML parsing plus symbolic-system construction."""

import sympy as sp

from ssys._recaster.common import arrow_pat, func_def_pat, prime_rule_pat
from ssys._recaster.sbml_helpers import (
    _apply_initial_assignments,
    _checked_sbml_model,
    _extract_sbml_function_templates,
    _iter_kinetic_law_local_parameters,
    _reaction_scope_name,
    _replace_formula_identifiers,
    _sanitize_sbml_identifier,
    _sympify_sbml_formula,
    _unique_identifier,
    _validate_sbml_identifier,
)
from ssys._recaster.sbml_helpers import (
    _evaluate_initial_assignment as _evaluate_initial_assignment,
)
from ssys._recaster.sbml_helpers import (
    _warn_or_raise_initial_assignment_error as _warn_or_raise_initial_assignment_error,
)
from ssys._recaster.templates import _expand_function_calls
from ssys.classification import classify_sym_system_solver_requirement
from ssys.metadata import _extract_sim_metadata, _extract_solver_requirement_metadata
from ssys.types import (
    ModelIR,
    Reaction,
    SBMLParseError,
    SolverRequirement,
    SymSystem,
)


def tokenize_species_side(side: str) -> list[tuple[int, str]]:
    parts = [p.strip() for p in side.split("+") if p.strip()]
    result = []
    for p in parts:
        p = p.strip()
        if p.startswith("$"):
            p = p[1:].strip()
        toks = p.split()
        if len(toks) == 1:
            coeff = 1
            name = toks[0]
        else:
            try:
                coeff = int(sp.sympify(toks[0]))
                name = toks[1]
            except (TypeError, ValueError, sp.SympifyError):
                coeff = 1
                name = p
        result.append((coeff, name))
    return result


def _numeric_param_subs(expr: sp.Expr, params: dict[str, float]) -> sp.Expr:
    """Replace parameter symbols in expr with their numeric values from params."""
    if not params:
        return expr
    # Build a mapping only for symbols actually used in expr
    subs = {s: sp.Float(params[s.name]) for s in expr.free_symbols if s.name in params}
    return sp.simplify(expr.subs(subs)) if subs else expr


def _antimony_to_sympy_syntax(expr_str: str) -> str:
    """Convert Antimony exponentiation syntax (^) to Python/SymPy syntax (**)."""
    # Simple string replacement is safe because ^ is XOR in Python (not exponentiation)
    # and Antimony uses ^ for exponentiation, not XOR
    return expr_str.replace("^", "**")


def _sympy_to_antimony_syntax(expr_str: str) -> str:
    """Convert Python/SymPy syntax to Antimony syntax.

    Conversions:
    - ** → ^ (exponentiation)
    - Abs() → abs() (SBML-compatible absolute value)
    """
    result = expr_str.replace("**", "^")
    result = result.replace("Abs(", "abs(")
    return result


def parse_antimony(text: str) -> ModelIR:
    ir = ModelIR()
    ir.antimony_text = text  # Cache original text for RoadRunner
    ir.raw_lines = [ln.rstrip() for ln in text.splitlines()]
    ir.solver_requirement = _extract_solver_requirement_metadata(text) or SolverRequirement.ODE_ONLY

    (
        ir.sim_t_start,
        ir.sim_t_end,
        ir.sim_n_steps,
        ir.eps_init,
        ir.eps_slack,
    ) = _extract_sim_metadata(text)

    # NOTE: Line continuation is NOT used for our simple parser.
    # Each line is treated as a separate statement.
    # Complex models should use parse_antimony_via_sbml() instead.

    for raw in ir.raw_lines:
        # strip inline comments
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("model ") or line.lower() == "end":
            continue

        # REACTIONS: keep the whole line (rate law is after ';')
        if ("->" in line) or ("<->" in line):
            s = line
            before_rate, rate_expr = s.split(";", 1)
            rate_expr = rate_expr.strip().rstrip(";").strip()
            if ":" in before_rate:
                rxn_name, stoich = before_rate.split(":", 1)
                rxn_name = rxn_name.strip()
                stoich = stoich.strip()
            else:
                rxn_name = None
                stoich = before_rate.strip()
            arrow = arrow_pat.search(stoich)
            if not arrow:
                continue
            lhs = stoich[: arrow.start()].strip()
            rhs = stoich[arrow.end() :].strip()
            lhs_list = tokenize_species_side(lhs) if lhs else []
            rhs_list = tokenize_species_side(rhs) if rhs else []
            for _, nm in lhs_list + rhs_list:
                ir.species.add(nm)
            # Convert Antimony ^ syntax to SymPy ** syntax for rate expression
            rate_expr = _antimony_to_sympy_syntax(rate_expr)
            ir.reactions.append(Reaction(rxn_name, lhs_list, rhs_list, rate_expr))
            continue

        # NON-REACTION LINES: may contain multiple ';'-separated statements
        for stmt in [seg.strip() for seg in line.split(";") if seg.strip()]:
            # explicit rate rule: S' = ...
            m = prime_rule_pat.match(stmt)
            if m:
                sp_name = m.group(1)
                expr = m.group(2).strip()
                if stmt.strip().startswith("$"):
                    ir.boundary.add(sp_name)
                ir.species.add(sp_name)
                # Convert Antimony ^ syntax to SymPy ** syntax for rate expression
                ir.explicit_rates[sp_name] = _antimony_to_sympy_syntax(expr)
                continue

            # parameter/initial assignment: X = 2.5  or  const X = 2.5
            if ("=" in stmt) and (":=" not in stmt):
                left, right = stmt.split("=", 1)
                left = left.strip()
                right = right.strip()

                if left in {"0", "0.0"}:
                    ir.algebraic_constraints.append(_antimony_to_sympy_syntax(right))
                    ir.solver_requirement = SolverRequirement.DAE_REQUIRED
                    continue

                # Handle 'const' keyword (strip it - const is just documentation)
                if left.startswith("const "):
                    left = left[6:].strip()

                # Store the expression string for later resolution
                if left.startswith("$"):
                    left = left[1:].strip()
                    ir.boundary.add(left)
                ir.param_exprs[left] = right  # Store expression for all assignments

                # Check if this is a non-trivial expression (not just a number)
                # Store as initial_expr if it contains functions, operators, or variables
                is_simple_number = False
                try:
                    # Try direct float conversion
                    float(right)
                    is_simple_number = True
                except (TypeError, ValueError):
                    # Not a simple number - it's an expression
                    pass

                if not is_simple_number:
                    # Store the expression string for symbolic preservation
                    ir.initial_exprs[left] = right

                # Try to evaluate immediately (will work for simple numeric constants)
                try:
                    val = float(sp.sympify(right))
                except (TypeError, ValueError, sp.SympifyError):
                    val = None
                ir.initial[left] = val if val is not None else 0.0
                continue

            # Assignment rules: name := expression
            # FIRST check if this is a function definition: name(params) := expr
            func_match = func_def_pat.match(stmt)
            if func_match:
                func_name = func_match.group(1)
                params_str = func_match.group(2).strip()
                body = _antimony_to_sympy_syntax(func_match.group(3).strip())
                # Parse parameter list
                params = [p.strip() for p in params_str.split(",") if p.strip()]
                # Store as function template
                ir.function_templates[func_name] = (params, body)
                continue

            if ":=" in stmt:
                left, right = stmt.split(":=", 1)
                left = left.strip()
                right = right.strip()
                ir.assignment_rules[left] = right
                continue

    # promote non-species initializations to parameters
    for nm, val in list(ir.initial.items()):
        if nm not in ir.species:
            ir.params[nm] = val

    # Resolve parameter dependencies
    _resolve_parameter_dependencies(ir)

    return ir


def _preprocess_antimony_text(text: str) -> str:
    """
    Preprocess Antimony text before passing to libantimony.

    Currently a no-op - we require input .ant files to use standard Antimony syntax:
    - Use backslash \\ for line continuation (NOT implicit continuation)
    - Use `function name(x) expr end` for function definitions (NOT `name(x) := expr`)

    This function exists as a hook for future preprocessing needs.
    """
    return text


def parse_antimony_via_sbml(
    antimony_text: str, *, warn_initial_assignment_failures: bool = False
) -> "SymSystem":
    """
    Parse Antimony text using the Antimony library (SBML-first approach).

    Pipeline: Antimony text → antimony lib → SBML string → libSBML → SymSystem

    This replaces the fragile hand-rolled parse_antimony() + build_sym_system()
    with the reference Antimony parser, ensuring all valid Antimony syntax is handled.

    Args:
        antimony_text: Antimony model text

    Returns:
        SymSystem ready for recasting

    Raises:
        ImportError: If antimony is not installed
        ValueError: If Antimony cannot be parsed
    """
    try:
        import antimony
    except ImportError:
        raise ImportError(
            "antimony is required for SBML-first parsing. Install with: pip install antimony"
        )

    # Extract @SIM metadata BEFORE conversion (comments are lost in SBML)
    t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(antimony_text)
    solver_requirement_metadata = _extract_solver_requirement_metadata(antimony_text)

    # CRITICAL: Preprocess to join multi-line statements
    # libantimony requires complete statements on single lines
    preprocessed_text = _preprocess_antimony_text(antimony_text)

    # Use antimony library to parse Antimony and convert to SBML
    try:
        # Clear any previous models
        antimony.clearPreviousLoads()

        # Load the preprocessed Antimony string
        result = antimony.loadAntimonyString(preprocessed_text)
        if result == -1:
            error_msg = antimony.getLastError()
            raise ValueError(f"Antimony parsing error: {error_msg}")

        # Get the module name (first/main module)
        module_name = antimony.getMainModuleName()
        if not module_name:
            raise ValueError("No module found in Antimony text")

        # Convert to SBML
        sbml_string = antimony.getSBMLString(module_name)
        if not sbml_string:
            error_msg = antimony.getLastError()
            raise ValueError(f"SBML conversion failed: {error_msg}")
    except Exception as e:
        if "Antimony parsing error" in str(e) or "SBML conversion" in str(e):
            raise
        raise ValueError(f"Antimony library failed: {e}")

    # Parse SBML string using existing infrastructure
    sym = parse_sbml_from_string(
        sbml_string, warn_initial_assignment_failures=warn_initial_assignment_failures
    )

    # Attach simulation metadata if found
    # These public attributes are used by the validator for trajectory tests
    sym.sim_t_start = t_start
    sym.sim_t_end = t_end
    sym.sim_n_steps = n_steps
    sym.eps_init = eps_init
    sym.eps_slack = eps_slack
    classified_requirement = classify_sym_system_solver_requirement(sym)
    sym.solver_requirement = (
        classified_requirement
        if classified_requirement == SolverRequirement.DAE_REQUIRED
        else solver_requirement_metadata or classified_requirement
    )

    # Cache original Antimony text for RoadRunner simulation
    sym.antimony_text = antimony_text

    return sym


def parse_sbml_from_string(
    sbml_string: str, *, warn_initial_assignment_failures: bool = False
) -> "SymSystem":
    """
    Parse an SBML string and return a SymSystem for recasting.

    This is the string-input variant of parse_sbml(), used by parse_antimony_via_sbml().

    Args:
        sbml_string: SBML model as string
        warn_initial_assignment_failures: If true, invalid InitialAssignments are warned
            and the parser keeps existing defaults. The default raises.

    Returns:
        SymSystem ready for recasting

    Raises:
        ImportError: If python-libsbml is not installed
        ValueError: If SBML cannot be parsed or has no model
    """
    try:
        import libsbml
    except ImportError:
        raise ImportError(
            "python-libsbml is required for SBML parsing. Install with: pip install python-libsbml"
        )

    # Read SBML from string
    doc = libsbml.readSBMLFromString(sbml_string)

    # Delegate to shared implementation
    return _parse_sbml_document(
        doc,
        source="<string>",
        warn_initial_assignment_failures=warn_initial_assignment_failures,
    )


def parse_sbml(sbml_path: str, *, warn_initial_assignment_failures: bool = False) -> "SymSystem":
    """
    Parse an SBML file and return a SymSystem for recasting.

    Uses libSBML to extract:
    - Species (state variables) and boundary species (constants)
    - Parameters with values
    - Reactions with stoichiometry and kinetic laws
    - Computes ODEs as dX/dt = sum(stoich * rate) for each species

    This bypasses the Antimony parser and works with complex SBML models
    that may have features incompatible with our simple Antimony parser.

    Args:
        sbml_path: Path to SBML file (.xml)
        warn_initial_assignment_failures: If true, invalid InitialAssignments are warned
            and the parser keeps existing defaults. The default raises.

    Returns:
        SymSystem ready for recasting

    Raises:
        ImportError: If python-libsbml is not installed
        ValueError: If SBML file cannot be parsed or has no model
    """
    try:
        import libsbml
    except ImportError:
        raise ImportError(
            "python-libsbml is required for SBML parsing. Install with: pip install python-libsbml"
        )

    # Read SBML file
    doc = libsbml.readSBML(sbml_path)

    # Delegate to shared implementation
    return _parse_sbml_document(
        doc,
        source=sbml_path,
        warn_initial_assignment_failures=warn_initial_assignment_failures,
    )


def _parse_sbml_document(
    doc, source: str = "<unknown>", *, warn_initial_assignment_failures: bool = False
) -> "SymSystem":
    """
    Shared implementation for parsing an SBML document.

    Args:
        doc: libsbml.SBMLDocument object
        source: Description of source for error messages
        warn_initial_assignment_failures: If true, invalid InitialAssignments are warned
            and the parser keeps existing defaults. The default raises.

    Returns:
        SymSystem ready for recasting
    """
    import libsbml

    model = _checked_sbml_model(doc, libsbml, source=source)
    function_templates = _extract_sbml_function_templates(model, libsbml, source=source)

    # ========================================================================
    # STEP 1: Extract species information
    # ========================================================================
    species_info: dict[str, dict] = {}
    boundary_species: set[str] = set()

    for i in range(model.getNumSpecies()):
        sp_obj = model.getSpecies(i)
        sid = sp_obj.getId()
        _validate_sbml_identifier(sid, kind="species", source=source)
        bc = sp_obj.getBoundaryCondition()

        # Get initial value (prefer amount over concentration)
        if sp_obj.isSetInitialAmount():
            init = sp_obj.getInitialAmount()
        elif sp_obj.isSetInitialConcentration():
            init = sp_obj.getInitialConcentration()
        else:
            init = 0.0

        species_info[sid] = {"bc": bc, "init": init}
        if bc:
            boundary_species.add(sid)

    # ========================================================================
    # STEP 2: Extract parameters (global and local)
    # ========================================================================
    params: dict[str, float] = {}

    # Global parameters
    for i in range(model.getNumParameters()):
        p = model.getParameter(i)
        pid = p.getId()
        _validate_sbml_identifier(pid, kind="parameter", source=source)
        if p.isSetValue():
            params[pid] = p.getValue()
        else:
            params[pid] = 0.0

    # Local parameters (from kinetic laws) are scoped to their reaction.
    # Same-named locals in different reactions must not overwrite each other.
    reaction_local_param_maps: dict[int, dict[str, str]] = {}
    used_param_ids = set(params) | set(species_info)
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl:
            scope = _reaction_scope_name(rxn, i)
            local_map: dict[str, str] = {}
            for lp in _iter_kinetic_law_local_parameters(kl):
                lpid = lp.getId()
                if not lpid:
                    continue
                scoped_id = _unique_identifier(
                    f"{scope}__{_sanitize_sbml_identifier(lpid, fallback='local')}",
                    used_param_ids,
                )
                used_param_ids.add(scoped_id)
                local_map[lpid] = scoped_id
                if lp.isSetValue():
                    params[scoped_id] = lp.getValue()
                else:
                    params[scoped_id] = 0.0
            reaction_local_param_maps[i] = local_map

    # ========================================================================
    # STEP 3: Extract compartments
    # ========================================================================
    compartments: dict[str, float] = {}
    for i in range(model.getNumCompartments()):
        c = model.getCompartment(i)
        cid = c.getId()
        _validate_sbml_identifier(cid, kind="compartment", source=source)
        if c.isSetSize():
            compartments[cid] = c.getSize()
        else:
            compartments[cid] = 1.0

    # ========================================================================
    # STEP 4: Build SymPy symbol dictionary
    # ========================================================================
    all_syms: dict[str, sp.Symbol] = {}

    # Time symbol
    all_syms["time"] = sp.Symbol("time", positive=True)
    all_syms["t"] = sp.Symbol("t", positive=True)  # Alias for time

    # Species symbols
    for sid in species_info:
        all_syms[sid] = sp.Symbol(sid, positive=True)

    # Parameter symbols
    for pid in params:
        all_syms[pid] = sp.Symbol(pid, positive=True)

    # Compartment symbols
    for cid in compartments:
        all_syms[cid] = sp.Symbol(cid, positive=True)

    # ========================================================================
    # STEP 5: Compute ODEs from reactions
    # ========================================================================
    # Initialize ODEs for floating species only (not boundary)
    odes: dict[sp.Symbol, sp.Expr] = {}
    for sid, info in species_info.items():
        if not info["bc"]:
            odes[all_syms[sid]] = sp.Integer(0)

    # Process each reaction
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()

        if not kl:
            continue

        # Get rate law as infix string
        math = kl.getMath()
        if math is None:
            continue

        formula_str = libsbml.formulaToString(math)
        formula_str = _replace_formula_identifiers(
            formula_str, reaction_local_param_maps.get(i, {})
        )
        formula_str = _expand_function_calls(formula_str, function_templates)

        # Parse rate expression
        rate_expr = _sympify_sbml_formula(
            formula_str,
            all_syms,
            source=source,
            kind="kinetic_law",
            reaction_id=rxn.getId() or None,
            reaction_name=rxn.getName() or None,
        )

        # Process reactants (subtract from ODE)
        for j in range(rxn.getNumReactants()):
            ref = rxn.getReactant(j)
            sid = ref.getSpecies()
            stoich = ref.getStoichiometry()

            if sid in species_info and not species_info[sid]["bc"]:
                var_sym = all_syms[sid]
                odes[var_sym] -= stoich * rate_expr

        # Process products (add to ODE)
        for j in range(rxn.getNumProducts()):
            ref = rxn.getProduct(j)
            sid = ref.getSpecies()
            stoich = ref.getStoichiometry()

            if sid in species_info and not species_info[sid]["bc"]:
                var_sym = all_syms[sid]
                odes[var_sym] += stoich * rate_expr

    # ========================================================================
    # STEP 6: Handle rate rules (explicit ODEs defined in SBML)
    # ========================================================================
    rate_rule_variables: set[str] = set()
    for i in range(model.getNumRules()):
        rule = model.getRule(i)

        if rule.getTypeCode() == libsbml.SBML_RATE_RULE:
            var_id = rule.getVariable()
            formula_str = libsbml.formulaToString(rule.getMath())
            formula_str = _expand_function_calls(formula_str, function_templates)

            if var_id in rate_rule_variables:
                raise SBMLParseError(
                    "ambiguous_model",
                    formula_str,
                    f"multiple rate rules for variable {var_id}",
                    source=source,
                    variable=var_id,
                )
            rate_rule_variables.add(var_id)

            if var_id not in all_syms:
                raise SBMLParseError(
                    "rate_rule",
                    formula_str,
                    f"unknown rule variable: {var_id}",
                    source=source,
                    variable=var_id,
                )

            rate_expr = _sympify_sbml_formula(
                formula_str,
                all_syms,
                source=source,
                kind="rate_rule",
                variable=var_id,
            )
            var_sym = all_syms[var_id]
            # Rate rules replace the reaction-based ODE
            odes[var_sym] = rate_expr

    # ========================================================================
    # STEP 6a: Handle assignment rules (V_1 := expression)
    # ========================================================================
    # Assignment rules define algebraic relationships, not ODEs
    # They are used for quantities that can be computed from other quantities
    assignment_rules: dict[str, str] = {}
    for i in range(model.getNumRules()):
        rule = model.getRule(i)

        if rule.getTypeCode() == libsbml.SBML_ASSIGNMENT_RULE:
            var_id = rule.getVariable()
            formula_str = libsbml.formulaToString(rule.getMath())
            formula_str = _expand_function_calls(formula_str, function_templates)
            # Store as string for SymSystem (will be parsed later if needed)
            assignment_rules[var_id] = formula_str

    # ========================================================================
    # STEP 6a.1: Handle algebraic rules (implicit constraints)
    # ========================================================================
    algebraic_constraints: list[str] = []
    for i in range(model.getNumRules()):
        rule = model.getRule(i)

        if rule.getTypeCode() == libsbml.SBML_ALGEBRAIC_RULE:
            formula_str = libsbml.formulaToString(rule.getMath())
            formula_str = _expand_function_calls(formula_str, function_templates)
            _sympify_sbml_formula(
                formula_str,
                all_syms,
                source=source,
                kind="algebraic_rule",
            )
            algebraic_constraints.append(formula_str)

    _apply_initial_assignments(
        model,
        libsbml,
        function_templates=function_templates,
        species_info=species_info,
        params=params,
        compartments=compartments,
        all_syms=all_syms,
        source=source,
        warn_initial_assignment_failures=warn_initial_assignment_failures,
    )

    # ========================================================================
    # STEP 7: Build SymSystem
    # ========================================================================
    # Variables (floating species with ODEs)
    vars_list = list(odes.keys())

    # Initial conditions
    initials: dict[sp.Symbol, float] = {}
    for sid, info in species_info.items():
        if sid in all_syms:
            initials[all_syms[sid]] = float(info["init"])

    # Add compartment values to params (they're effectively constants)
    for cid, cval in compartments.items():
        if cid not in params:
            params[cid] = cval

    # Simplify ODEs
    # CRITICAL: Do NOT use sp.simplify() here - it combines separate fractions
    # over a common denominator, which corrupts the term structure needed for
    # lift_rational_functions() to identify individual denominators.
    # Example: A/(1+x^4) + B/(1+a^2) becomes (A*(1+a^2) + B*(1+x^4))/((1+x^4)*(1+a^2))
    # This causes incorrect lifting where all terms get divided by the PRODUCT
    # of denominators instead of their individual denominators.
    # See: McMillen2002 regression bug (December 2025)
    simplified_odes = dict(odes)

    return SymSystem(
        vars=vars_list,
        params=params,
        odes=simplified_odes,
        initials=initials,
        assignment_rules=assignment_rules,
        algebraic_constraints=algebraic_constraints,
        compartments=compartments,
        solver_requirement=(
            SolverRequirement.DAE_REQUIRED
            if algebraic_constraints
            else (
                SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
                if assignment_rules
                else SolverRequirement.ODE_ONLY
            )
        ),
    )


def _resolve_parameter_dependencies(ir: ModelIR) -> None:
    """
    Resolve parameter dependencies by evaluating expressions iteratively.

    This handles cases like:
        N_A = 6.02e23
        V = 1e-12
        K_T = 1e6/(N_A*V)  # Depends on N_A and V

    Modifies ir.initial and ir.params in place.
    """
    max_iterations = 100  # Prevent infinite loops
    resolved = {}  # Track successfully resolved parameters

    # Start with parameters that evaluated successfully
    for name, val in ir.initial.items():
        if val != 0.0 or name not in ir.param_exprs:
            resolved[name] = val

    # Iteratively resolve dependencies
    for _iteration in range(max_iterations):
        made_progress = False

        for name, expr_str in ir.param_exprs.items():
            if name in resolved:
                continue  # Already resolved

            try:
                # Try to evaluate with currently resolved parameters
                expr = sp.sympify(expr_str)
                # Substitute known values
                for resolved_name, resolved_val in resolved.items():
                    expr = expr.subs(sp.Symbol(resolved_name), resolved_val)

                # Try to evaluate to a number
                val = float(expr)
                resolved[name] = val
                ir.initial[name] = val
                if name in ir.params:
                    ir.params[name] = val
                made_progress = True
            except (TypeError, ValueError):
                # Can't evaluate yet - dependencies not resolved
                continue

        if not made_progress:
            # No progress this iteration - done or stuck
            break


def build_sym_system(ir: ModelIR) -> SymSystem:
    var_syms: dict[str, sp.Symbol] = {
        nm: sp.symbols(nm, positive=True) for nm in sorted(ir.species)
    }
    param_syms: dict[str, sp.Symbol] = {}
    for nm, _val in ir.params.items():
        if nm in var_syms:
            continue
        param_syms[nm] = sp.symbols(nm, positive=True)

    # CRITICAL: Expand function template calls BEFORE sympifying
    # This handles Antimony function definitions like:
    #   M(x) := 1 + gam*h*x^(h-1)/(1+x^h)^2
    #   x1' = beta*(y1-x1)/M(x1)  →  x1' = beta*(y1-x1)/(1 + gam*h*x1^(h-1)/(1+x1^h)^2)
    expanded_explicit_rates: dict[str, str] = {}
    for nm, expr_str in ir.explicit_rates.items():
        expanded_explicit_rates[nm] = _expand_function_calls(expr_str, ir.function_templates)

    expanded_assignment_rules: dict[str, str] = {}
    for nm, expr_str in ir.assignment_rules.items():
        expanded_assignment_rules[nm] = _expand_function_calls(expr_str, ir.function_templates)

    # Parse assignment rules into symbolic expressions
    # Handle nested rules by expanding iteratively until stable
    assignment_exprs: dict[str, sp.Expr] = {}
    for name, expr_str in expanded_assignment_rules.items():
        assignment_exprs[name] = sp.sympify(expr_str, locals={**var_syms, **param_syms})

    # Expand nested assignment rules (rule A may reference rule B)
    max_iterations = 20
    for _ in range(max_iterations):
        changed = False
        for name, expr in list(assignment_exprs.items()):
            new_expr = expr
            for other_name, other_expr in assignment_exprs.items():
                if other_name != name:
                    new_expr = new_expr.subs(sp.Symbol(other_name), other_expr)
            if new_expr != assignment_exprs[name]:
                assignment_exprs[name] = new_expr
                changed = True
        if not changed:
            break

    odes: dict[sp.Symbol, sp.Expr] = {var_syms[nm]: sp.Integer(0) for nm in var_syms}
    for rxn in ir.reactions:
        rate_expr = _expand_function_calls(rxn.rate_expr, ir.function_templates)
        rate = sp.sympify(rate_expr, locals={**var_syms, **param_syms})
        # NOTE: Do NOT substitute assignment rules into rate expressions
        # This preserves the compact form with rule names (like k_23) instead of expanded expressions
        # Assignment rules are passed through to the final output unchanged
        lhs_sto = {nm: coeff for coeff, nm in rxn.lhs}
        rhs_sto = {nm: coeff for coeff, nm in rxn.rhs}
        all_sp = set(lhs_sto) | set(rhs_sto)
        for nm in all_sp:
            if nm not in var_syms:
                continue
            net = rhs_sto.get(nm, 0) - lhs_sto.get(nm, 0)
            if nm in ir.boundary:
                continue
            odes[var_syms[nm]] += sp.Integer(net) * rate
    for nm, expr in expanded_explicit_rates.items():
        if nm not in var_syms:
            continue
        rate = sp.sympify(expr, locals={**var_syms, **param_syms})
        # NOTE: Do NOT substitute assignment rules into explicit rate expressions
        # Assignment rules stay as compact names, passed through to output
        odes[var_syms[nm]] += rate
    initials: dict[sp.Symbol, float] = {}
    for nm, sym in var_syms.items():
        initials[sym] = float(ir.initial.get(nm, 0.0))
    for nm, sym in param_syms.items():
        initials[sym] = float(ir.initial.get(nm, ir.params.get(nm, 0.0)))

    # Propagate symbolic initial condition expressions
    initial_exprs: dict[sp.Symbol, str] = {}
    for name, expr_str in ir.initial_exprs.items():
        # Check if this is a species or parameter
        if name in var_syms:
            initial_exprs[var_syms[name]] = expr_str
        elif name in param_syms:
            initial_exprs[param_syms[name]] = expr_str

    solver_requirement = ir.solver_requirement
    if ir.algebraic_constraints:
        solver_requirement = SolverRequirement.DAE_REQUIRED
    elif expanded_assignment_rules and solver_requirement == SolverRequirement.ODE_ONLY:
        solver_requirement = SolverRequirement.ODE_WITH_ASSIGNMENT_RULES

    return SymSystem(
        vars=list(odes.keys()),
        params={k: float(v) for k, v in ir.params.items()},
        odes=odes,
        initials=initials,
        initial_exprs=initial_exprs,
        assignment_rules=expanded_assignment_rules,  # Pass through expanded legacy templates
        algebraic_constraints=list(ir.algebraic_constraints),
        compartments=ir.compartments,
        sim_t_start=ir.sim_t_start,
        sim_t_end=ir.sim_t_end,
        sim_n_steps=ir.sim_n_steps,
        eps_init=ir.eps_init,
        eps_slack=ir.eps_slack,
        antimony_text=ir.antimony_text,
        solver_requirement=solver_requirement,
    )
