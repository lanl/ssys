"""Antimony and SBML parsing plus symbolic-system construction."""

import re
from collections.abc import Callable

import sympy as sp

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
    SBMLParseError,
    SolverRequirement,
    SymSystem,
)


def _notify_progress(progress_callback: Callable[[str], None] | None, phase: str) -> None:
    """Best-effort parser progress hook for timeout diagnostics."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase)
    except Exception:
        return


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


def _preprocess_antimony_text(text: str) -> str:
    """
    Preprocess Antimony text before passing to libantimony.

    We generally require input .ant files to use standard Antimony syntax:
    - Use backslash \\ for line continuation (NOT implicit continuation)
    - Use `function name(x) expr end` for function definitions (NOT `name(x) := expr`)
    """
    # libAntimony emits this when exporting an SBML compartment literally named
    # "compartment". The source compartment has size 1, but the exported helper
    # parameter is left unset, which otherwise parses as 0 and zeroes all rates.
    return re.sub(r"(?m)^(\s*compartment_\s*=\s*);\s*$", r"\g<1>1;", text)


def parse_antimony_via_sbml(
    antimony_text: str,
    *,
    warn_initial_assignment_failures: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    progress_prefix: str = "parse_antimony_via_sbml",
) -> "SymSystem":
    """
    Parse Antimony text using the Antimony library (SBML-first approach).

    Pipeline: Antimony text → antimony lib → SBML string → libSBML → SymSystem

    Uses the reference Antimony parser (libAntimony), ensuring all valid Antimony
    syntax is handled.

    Args:
        antimony_text: Antimony model text
        warn_initial_assignment_failures: If true, invalid InitialAssignments are warned
            and the parser keeps existing defaults. The default raises.
        progress_callback: Optional callback for parser subphase timeout diagnostics.
        progress_prefix: Prefix used for emitted progress phase names.

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
    _notify_progress(progress_callback, f"{progress_prefix}_preprocess")
    preprocessed_text = _preprocess_antimony_text(antimony_text)

    # Use antimony library to parse Antimony and convert to SBML
    try:
        # Clear any previous models
        antimony.clearPreviousLoads()

        # Load the preprocessed Antimony string
        _notify_progress(progress_callback, f"{progress_prefix}_antimony_load")
        result = antimony.loadAntimonyString(preprocessed_text)
        if result == -1:
            error_msg = antimony.getLastError()
            raise ValueError(f"Antimony parsing error: {error_msg}")

        # Get the module name (first/main module)
        _notify_progress(progress_callback, f"{progress_prefix}_sbml_export")
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
        sbml_string,
        warn_initial_assignment_failures=warn_initial_assignment_failures,
        progress_callback=progress_callback,
        progress_prefix=f"{progress_prefix}_sym_system_parse",
    )

    # Attach simulation metadata if found
    # These public attributes are used by the validator for trajectory tests
    _notify_progress(progress_callback, f"{progress_prefix}_metadata_attach")
    sym.sim_t_start = t_start
    sym.sim_t_end = t_end
    sym.sim_n_steps = n_steps
    sym.eps_init = eps_init
    sym.eps_slack = eps_slack
    _notify_progress(progress_callback, f"{progress_prefix}_solver_requirement")
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
    sbml_string: str,
    *,
    warn_initial_assignment_failures: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    progress_prefix: str = "parse_sbml_from_string",
) -> "SymSystem":
    """
    Parse an SBML string and return a SymSystem for recasting.

    This is the string-input variant of parse_sbml(), used by parse_antimony_via_sbml().

    Args:
        sbml_string: SBML model as string
        warn_initial_assignment_failures: If true, invalid InitialAssignments are warned
            and the parser keeps existing defaults. The default raises.
        progress_callback: Optional callback for parser subphase timeout diagnostics.
        progress_prefix: Prefix used for emitted progress phase names.

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
    _notify_progress(progress_callback, f"{progress_prefix}_libsbml_read")
    doc = libsbml.readSBMLFromString(sbml_string)

    # Delegate to shared implementation
    _notify_progress(progress_callback, f"{progress_prefix}_sym_system_build")
    return _parse_sbml_document(
        doc,
        source="<string>",
        warn_initial_assignment_failures=warn_initial_assignment_failures,
        progress_callback=progress_callback,
        progress_prefix=f"{progress_prefix}_sym_system_build",
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
    doc,
    source: str = "<unknown>",
    *,
    warn_initial_assignment_failures: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    progress_prefix: str = "parse_sbml_document",
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

    _notify_progress(progress_callback, f"{progress_prefix}_checked_model")
    model = _checked_sbml_model(doc, libsbml, source=source)
    _notify_progress(progress_callback, f"{progress_prefix}_function_templates")
    function_templates = _extract_sbml_function_templates(model, libsbml, source=source)

    # ========================================================================
    # STEP 1: Extract species information
    # ========================================================================
    _notify_progress(progress_callback, f"{progress_prefix}_species")
    species_info: dict[str, dict] = {}
    boundary_species: set[str] = set()

    # An L3 Model may declare a default conversionFactor that applies to every
    # species without its own. STEP 5b multiplies each species' reaction-derived
    # amount rate by this factor (species' own factor takes precedence). No-op
    # when unset. See STEP 5b for the semantics (GH #232).
    model_conversion_factor = (
        model.getConversionFactor() if model.isSetConversionFactor() else None
    )

    for i in range(model.getNumSpecies()):
        sp_obj = model.getSpecies(i)
        sid = sp_obj.getId()
        _validate_sbml_identifier(sid, kind="species", source=source)
        bc = sp_obj.getBoundaryCondition()
        # A hasOnlySubstanceUnits=true species is denoted by its *amount* in every
        # rate law, rule, and initial value; the SBML default (false) means it is
        # denoted by its *concentration* (amount / compartment size). STEP 3a and
        # STEP 5a use these to reconcile initial values and scale ODEs by volume.
        hosu = sp_obj.getHasOnlySubstanceUnits()
        compartment = sp_obj.getCompartment()
        # A species' conversionFactor (a constant Parameter) scales its
        # reaction-derived amount rate in STEP 5b (GH #232). None falls back to the
        # model default recorded above.
        conversion_factor = (
            sp_obj.getConversionFactor() if sp_obj.isSetConversionFactor() else None
        )

        # Record the initial value together with the unit it was supplied in so
        # STEP 3a can convert it to the species-symbol convention once compartment
        # sizes are known.
        if sp_obj.isSetInitialAmount():
            init = sp_obj.getInitialAmount()
            init_kind = "amount"
        elif sp_obj.isSetInitialConcentration():
            init = sp_obj.getInitialConcentration()
            init_kind = "concentration"
        else:
            init = 0.0
            init_kind = "none"

        species_info[sid] = {
            "bc": bc,
            "init": init,
            "init_kind": init_kind,
            "hosu": hosu,
            "compartment": compartment,
            "conversion_factor": conversion_factor,
        }
        if bc:
            boundary_species.add(sid)

    # ========================================================================
    # STEP 2: Extract parameters (global and local)
    # ========================================================================
    _notify_progress(progress_callback, f"{progress_prefix}_parameters")
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
    _notify_progress(progress_callback, f"{progress_prefix}_compartments")
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
    # STEP 3a: Reconcile species initial values to the symbol's unit
    # ========================================================================
    # A hasOnlySubstanceUnits=false species symbol denotes concentration; a
    # hasOnlySubstanceUnits=true species symbol denotes amount. When the model
    # supplies the initial value in the other unit, convert it with the owning
    # compartment size so initial-assignment evaluation, ODE assembly, and the
    # reported initial conditions all speak the same unit as the symbol.
    # (No-op at compartment size 1, where amount == concentration. This uses the
    # declared compartment size, before any InitialAssignment resizes it.)
    for info in species_info.values():
        vol = compartments.get(info["compartment"], 1.0)
        if vol == 1.0:
            continue
        if info["init_kind"] == "amount" and not info["hosu"]:
            info["init"] = info["init"] / vol
        elif info["init_kind"] == "concentration" and info["hosu"]:
            info["init"] = info["init"] * vol

    # ========================================================================
    # STEP 4: Build SymPy symbol dictionary
    # ========================================================================
    _notify_progress(progress_callback, f"{progress_prefix}_symbols")
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
    _notify_progress(progress_callback, f"{progress_prefix}_reactions")
    # Initialize ODEs for floating species only (not boundary)
    odes: dict[sp.Symbol, sp.Expr] = {}
    for sid, info in species_info.items():
        if not info["bc"]:
            odes[all_syms[sid]] = sp.Integer(0)

    # A speciesReference's stoichiometry is folded to a constant coefficient. A
    # plain attribute is used directly; an L2 <stoichiometryMath> is constant-
    # folded over parameters and compartment sizes. Genuinely variable
    # stoichiometry (time/species/rule-driven) is rejected in _checked_sbml_model
    # before we reach here, so anything surviving must fold to a number (GH #237).
    stoich_numeric_values = {**params, **compartments}

    def _reference_stoichiometry(ref, rxn) -> float:
        if hasattr(ref, "isSetStoichiometryMath") and ref.isSetStoichiometryMath():
            stoich_math = ref.getStoichiometryMath()
            math = stoich_math.getMath() if stoich_math is not None else None
            formula_str = libsbml.formulaToString(math) if math is not None else None
            expr = _sympify_sbml_formula(
                formula_str,
                all_syms,
                source=source,
                kind="stoichiometry",
                reaction_id=rxn.getId() or None,
                reaction_name=rxn.getName() or None,
            )
            subs = {
                all_syms[name]: value
                for name, value in stoich_numeric_values.items()
                if name in all_syms
            }
            expr = expr.subs(subs)
            if expr.free_symbols:
                unresolved = ", ".join(sorted(s.name for s in expr.free_symbols))
                raise SBMLParseError(
                    "stoichiometry",
                    formula_str,
                    f"non-constant stoichiometry: {unresolved}",
                    source=source,
                    reaction_id=rxn.getId() or None,
                    reaction_name=rxn.getName() or None,
                )
            return float(expr)
        return float(ref.getStoichiometry())

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
            stoich = _reference_stoichiometry(ref, rxn)

            if sid in species_info and not species_info[sid]["bc"]:
                var_sym = all_syms[sid]
                odes[var_sym] -= stoich * rate_expr

        # Process products (add to ODE)
        for j in range(rxn.getNumProducts()):
            ref = rxn.getProduct(j)
            sid = ref.getSpecies()
            stoich = _reference_stoichiometry(ref, rxn)

            if sid in species_info and not species_info[sid]["bc"]:
                var_sym = all_syms[sid]
                odes[var_sym] += stoich * rate_expr

    # ------------------------------------------------------------------------
    # STEP 5a: Scale reaction-derived amount rates into the species' unit
    # ------------------------------------------------------------------------
    # The accumulation above is dAmount/dt = Σ stoich·kineticLaw, since an SBML
    # kinetic law is an extent (amount) rate. A hasOnlySubstanceUnits=false
    # species symbol denotes concentration, so its ODE is that amount rate
    # divided by the owning compartment size: d[S]/dt = (1/V)·Σ stoich·K.
    # hasOnlySubstanceUnits=true species are amounts already and are not scaled.
    # Division uses the compartment *symbol* (a positive parameter carrying V),
    # which cancels exactly in the common ``compartment·k·A·B`` idiom and tracks
    # the live value when the compartment has an assignment rule. Skipped at
    # V == 1 so unit-volume models keep their exact current form. This runs before
    # STEP 6 rate rules, which state d(symbol)/dt directly and are never scaled.
    for sid, info in species_info.items():
        if info["bc"] or info["hosu"]:
            continue
        if compartments.get(info["compartment"], 1.0) == 1.0:
            continue
        var_sym = all_syms.get(sid)
        comp_sym = all_syms.get(info["compartment"])
        if var_sym is None or comp_sym is None or var_sym not in odes:
            continue
        odes[var_sym] = odes[var_sym] / comp_sym

    # ------------------------------------------------------------------------
    # STEP 5b: Scale reaction-derived rates by each species' conversionFactor
    # ------------------------------------------------------------------------
    # An SBML L3 species may declare a conversionFactor (a constant Parameter),
    # or inherit the Model default, that scales how its amount changes per unit
    # reaction extent: d(amount_S)/dt = cf_S·Σ stoich·kineticLaw, without altering
    # how S appears in any rate law. Because cf_S is a per-species scalar it
    # factors out of the accumulated sum, so multiplying the whole reaction-
    # derived ODE by the conversionFactor *symbol* is exact — for both amount and
    # concentration species — and commutes with the STEP 5a volume division
    # ((cf/V)·Σ = (1/V)·cf·Σ). The symbol (a positive constant parameter) is used
    # so it stays visible to recasting, mirroring the compartment symbol. No-op
    # when neither the species nor the model sets a conversionFactor (GH #232).
    for sid, info in species_info.items():
        if info["bc"]:
            continue
        cf_id = info["conversion_factor"] or model_conversion_factor
        if cf_id is None:
            continue
        var_sym = all_syms.get(sid)
        if var_sym is None or var_sym not in odes:
            continue
        cf_sym = all_syms.get(cf_id)
        if cf_sym is None:
            raise SBMLParseError(
                "conversion_factor",
                None,
                f"conversionFactor '{cf_id}' is not a declared parameter",
                source=source,
                variable=cf_id,
            )
        odes[var_sym] = cf_sym * odes[var_sym]

    # ========================================================================
    # STEP 6: Handle rate rules (explicit ODEs defined in SBML)
    # ========================================================================
    _notify_progress(progress_callback, f"{progress_prefix}_rate_rules")
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
    _notify_progress(progress_callback, f"{progress_prefix}_assignment_rules")
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
    _notify_progress(progress_callback, f"{progress_prefix}_algebraic_rules")
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

    _notify_progress(progress_callback, f"{progress_prefix}_initial_assignments")
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
    _notify_progress(progress_callback, f"{progress_prefix}_build_system")
    # Variables are floating species with ODEs. SBML assignment-rule targets are
    # algebraic quantities, not independent ODE states, even when the target is a
    # species that appeared in the reaction-derived ODE table.
    assignment_rule_targets = set(assignment_rules)
    for target in assignment_rule_targets:
        target_sym = all_syms.get(target)
        if target_sym is not None:
            odes.pop(target_sym, None)
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
