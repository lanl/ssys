"""Antimony and SBML parsing plus symbolic-system construction."""

import re
from collections.abc import Callable

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


def _notify_progress(progress_callback: Callable[[str], None] | None, phase: str) -> None:
    """Best-effort parser progress hook for timeout diagnostics."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase)
    except Exception:
        return


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

    This replaces the fragile hand-rolled parse_antimony() + build_sym_system()
    with the reference Antimony parser, ensuring all valid Antimony syntax is handled.

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
