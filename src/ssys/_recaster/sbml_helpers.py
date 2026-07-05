"""SBML parsing helpers shared by the recaster parser."""

from __future__ import annotations

import keyword
import re
import warnings

import sympy as sp

from ssys._recaster.templates import _expand_function_calls, _parse_function_args
from ssys.types import SBMLParseError

_SYMPY_FUNCTION_NAMES = frozenset({
    "Abs",
    "Piecewise",
    "acos",
    "asin",
    "atan",
    "ceiling",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "max",
    "min",
    "piecewise",
    "pow",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
})
_SYMPY_CONSTANT_NAMES = frozenset({"E", "EulerGamma", "oo", "pi"})
_NUMERIC_LITERAL_PATTERN = re.compile(
    r"(?<![A-Za-z_0-9.])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


def _strip_numeric_literals(formula_text: str) -> str:
    """Remove numeric literals before identifier scanning."""
    return _NUMERIC_LITERAL_PATTERN.sub("", formula_text)


def _keyword_identifier_replacements(
    identifiers: set[str], all_syms: dict[str, sp.Symbol]
) -> dict[str, str]:
    """Return safe aliases for declared SBML identifiers that are Python keywords."""
    replacements: dict[str, str] = {}
    used = set(identifiers) | set(all_syms)
    for identifier in sorted(identifiers):
        if identifier not in all_syms or not keyword.iskeyword(identifier):
            continue
        alias = f"__sbml_{identifier}"
        while alias in used:
            alias = f"{alias}_"
        replacements[identifier] = alias
        used.add(alias)
    return replacements


def _sympify_sbml_formula(
    formula_str: str | None,
    all_syms: dict[str, sp.Symbol],
    *,
    source: str,
    kind: str,
    reaction_id: str | None = None,
    reaction_name: str | None = None,
    variable: str | None = None,
) -> sp.Expr:
    """Parse an SBML formula string as a numeric SymPy expression or raise with context."""
    if formula_str in (None, ""):
        raise SBMLParseError(
            kind,
            formula_str,
            "missing math formula",
            source=source,
            reaction_id=reaction_id,
            reaction_name=reaction_name,
            variable=variable,
        )
    assert isinstance(formula_str, str)
    formula_text = formula_str

    identifier_scan_text = _strip_numeric_literals(formula_text)
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", identifier_scan_text))
    function_calls = set(
        re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", identifier_scan_text)
    )
    unsupported_functions = sorted(function_calls - _SYMPY_FUNCTION_NAMES)
    if unsupported_functions:
        raise SBMLParseError(
            kind,
            formula_str,
            f"unsupported function(s): {', '.join(unsupported_functions)}",
            source=source,
            reaction_id=reaction_id,
            reaction_name=reaction_name,
            variable=variable,
        )

    unknown_identifiers = sorted(
        identifiers - set(all_syms) - _SYMPY_FUNCTION_NAMES - _SYMPY_CONSTANT_NAMES
    )
    if unknown_identifiers:
        raise SBMLParseError(
            kind,
            formula_str,
            f"unknown identifier(s): {', '.join(unknown_identifiers)}",
            source=source,
            reaction_id=reaction_id,
            reaction_name=reaction_name,
            variable=variable,
        )

    formula_for_sympify = formula_str
    sympify_locals = dict(all_syms)
    keyword_replacements = _keyword_identifier_replacements(identifiers, all_syms)
    if keyword_replacements:
        formula_for_sympify = _replace_formula_identifiers(
            formula_for_sympify,
            keyword_replacements,
        )
        for original_name, alias in keyword_replacements.items():
            sympify_locals[alias] = all_syms[original_name]

    try:
        expr = sp.sympify(formula_for_sympify, locals=sympify_locals)
    except Exception as exc:
        raise SBMLParseError(
            kind,
            formula_str,
            str(exc),
            source=source,
            reaction_id=reaction_id,
            reaction_name=reaction_name,
            variable=variable,
        )

    if not isinstance(expr, sp.Expr):
        raise SBMLParseError(
            kind,
            formula_str,
            f"expected numeric expression, got {type(expr).__name__}",
            source=source,
            reaction_id=reaction_id,
            reaction_name=reaction_name,
            variable=variable,
        )

    return expr


def _warn_or_raise_initial_assignment_error(
    error: SBMLParseError, warn_initial_assignment_failures: bool
) -> None:
    if warn_initial_assignment_failures:
        warnings.warn(str(error), RuntimeWarning, stacklevel=2)
        return
    raise error


def _reaction_scope_name(rxn, reaction_index: int) -> str:
    raw = rxn.getId() or rxn.getName() or f"reaction_{reaction_index + 1}"
    return _sanitize_sbml_identifier(raw, fallback=f"reaction_{reaction_index + 1}")


def _sanitize_sbml_identifier(value: str, *, fallback: str = "id") -> str:
    sanitized = re.sub(r"\W", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = fallback
    if not re.match(r"^[A-Za-z_]", sanitized):
        sanitized = f"_{sanitized}"
    return sanitized


def _unique_identifier(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    return f"{base}_{idx}"


def _iter_kinetic_law_local_parameters(kinetic_law) -> list:
    if hasattr(kinetic_law, "getNumLocalParameters") and kinetic_law.getNumLocalParameters() > 0:
        return [
            kinetic_law.getLocalParameter(i)
            for i in range(kinetic_law.getNumLocalParameters())
        ]
    return [kinetic_law.getParameter(i) for i in range(kinetic_law.getNumParameters())]


def _replace_formula_identifiers(formula_str: str, replacements: dict[str, str]) -> str:
    result = formula_str
    for old_name, new_name in replacements.items():
        if not old_name:
            continue
        pattern = r"(?<![A-Za-z_\d])" + re.escape(old_name) + r"(?![A-Za-z_\d])"
        result = re.sub(pattern, new_name, result)
    return result


def _parse_sbml_function_lambda(
    formula_str: str | None,
    *,
    function_id: str,
    source: str,
) -> tuple[list[str], str]:
    """Parse libSBML's lambda(...) string for a FunctionDefinition."""
    if formula_str in (None, ""):
        raise SBMLParseError(
            "function_definition",
            formula_str,
            f"missing math formula for function definition {function_id}",
            source=source,
            variable=function_id,
        )
    assert isinstance(formula_str, str)
    text = formula_str.strip()
    if not (text.startswith("lambda(") and text.endswith(")")):
        raise SBMLParseError(
            "function_definition",
            formula_str,
            f"function definition {function_id} is not a lambda expression",
            source=source,
            variable=function_id,
        )

    parts = _parse_function_args(text[len("lambda(") : -1])
    if len(parts) < 2:
        raise SBMLParseError(
            "function_definition",
            formula_str,
            f"function definition {function_id} must have parameter(s) and a body",
            source=source,
            variable=function_id,
        )

    params = parts[:-1]
    body = parts[-1]
    for param in params:
        _validate_sbml_identifier(param, kind="function parameter", source=source)
    return params, body


def _extract_sbml_function_templates(model, libsbml, *, source: str) -> dict[str, tuple[list[str], str]]:
    """Extract SBML FunctionDefinition lambdas as local expansion templates."""
    function_templates: dict[str, tuple[list[str], str]] = {}
    for i in range(model.getNumFunctionDefinitions()):
        function = model.getFunctionDefinition(i)
        function_id = function.getId()
        _validate_sbml_identifier(function_id, kind="function", source=source)
        formula_str = libsbml.formulaToString(function.getMath())
        function_templates[function_id] = _parse_sbml_function_lambda(
            formula_str,
            function_id=function_id,
            source=source,
        )
    return function_templates


def _evaluate_initial_assignment(
    formula_str: str | None,
    all_syms: dict[str, sp.Symbol],
    numeric_values: dict[str, float],
    *,
    source: str,
    variable: str,
    warn_initial_assignment_failures: bool,
    defer_unresolved: bool = False,
) -> float | None:
    try:
        init_expr = _sympify_sbml_formula(
            formula_str,
            all_syms,
            source=source,
            kind="initial_assignment",
            variable=variable,
        )
        substitutions = {
            all_syms[name]: value
            for name, value in numeric_values.items()
            if name != variable and name in all_syms
        }
        if substitutions:
            init_expr = init_expr.subs(substitutions)
        unresolved = sorted(symbol.name for symbol in init_expr.free_symbols)
        if unresolved:
            if defer_unresolved:
                return None
            joined = ", ".join(unresolved)
            raise ValueError(f"unresolved symbol(s): {joined}")
        return float(init_expr)
    except SBMLParseError as exc:
        _warn_or_raise_initial_assignment_error(exc, warn_initial_assignment_failures)
    except Exception as exc:
        error = SBMLParseError(
            "initial_assignment",
            formula_str,
            str(exc),
            source=source,
            variable=variable,
        )
        _warn_or_raise_initial_assignment_error(error, warn_initial_assignment_failures)

    return None


def _raise_unsupported_sbml_feature(feature: str, *, source: str, detail: str = "") -> None:
    message = f"unsupported SBML feature: {feature}"
    if detail:
        message = f"{message} ({detail})"
    raise SBMLParseError(
        "unsupported_feature",
        None,
        message,
        source=source,
    )


def _validate_sbml_identifier(identifier: str, *, kind: str, source: str) -> None:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
        return
    raise SBMLParseError(
        "invalid_identifier",
        identifier,
        f"invalid {kind} identifier: {identifier!r}",
        source=source,
        variable=identifier or None,
    )


def _sbml_rule_target_sets(model, libsbml):
    """Return (rate_rule_targets, assignment_rule_targets, assignment_rule_math)."""
    rate_targets: set[str] = set()
    assignment_targets: set[str] = set()
    assignment_math: dict[str, object] = {}
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        type_code = rule.getTypeCode()
        if type_code == libsbml.SBML_RATE_RULE:
            rate_targets.add(rule.getVariable())
        elif type_code == libsbml.SBML_ASSIGNMENT_RULE:
            var_id = rule.getVariable()
            assignment_targets.add(var_id)
            assignment_math[var_id] = rule.getMath()
    return rate_targets, assignment_targets, assignment_math


def _sbml_species_ids(model) -> set[str]:
    return {model.getSpecies(i).getId() for i in range(model.getNumSpecies())}


def _ast_references_time_varying(node, libsbml, *, species_ids: set[str], varying_ids: set[str]) -> bool:
    """True when an SBML math AST reads a quantity that changes in time.

    A leaf that is the ``time`` csymbol, a species, or a rule-driven identifier
    makes the expression non-constant. An expression over only numbers and
    constant parameters/compartments constant-folds and returns False.
    """
    if node is None:
        return False
    if node.getType() == libsbml.AST_NAME_TIME:
        return True
    name = node.getName()
    if name and (name in species_ids or name in varying_ids or name in {"time", "t"}):
        return True
    for i in range(node.getNumChildren()):
        if _ast_references_time_varying(
            node.getChild(i), libsbml, species_ids=species_ids, varying_ids=varying_ids
        ):
            return True
    return False


def _iter_reaction_species_references(rxn):
    for j in range(rxn.getNumReactants()):
        yield rxn.getReactant(j)
    for j in range(rxn.getNumProducts()):
        yield rxn.getProduct(j)


def _reject_variable_stoichiometry(model, libsbml, *, source: str) -> None:
    """Refuse reactions whose stoichiometry is not a compile-time constant.

    ssys folds each reactant/product to a constant coefficient and builds a
    power-law contribution ``stoich·rate``. A time-varying coefficient — an L2
    ``<stoichiometryMath>`` that reads time/a species/a rule-driven parameter, or
    an L3 speciesReference id that is a rule or initialAssignment target — is not
    power-law-recastable, so it is rejected before any artifact is produced rather
    than silently frozen at its load-time value (GH #237). Constant
    stoichiometryMath and plain constant coefficients fold in STEP 5 as usual.
    """
    rate_targets, assignment_targets, _ = _sbml_rule_target_sets(model, libsbml)
    ia_targets = {
        model.getInitialAssignment(i).getSymbol()
        for i in range(model.getNumInitialAssignments())
    }
    species_ids = _sbml_species_ids(model)
    varying_ids = rate_targets | assignment_targets
    rule_or_ia = rate_targets | assignment_targets | ia_targets
    for ri in range(model.getNumReactions()):
        rxn = model.getReaction(ri)
        rid = rxn.getId() or f"reaction_{ri + 1}"
        for ref in _iter_reaction_species_references(rxn):
            if hasattr(ref, "isSetStoichiometryMath") and ref.isSetStoichiometryMath():
                stoich_math = ref.getStoichiometryMath()
                math = stoich_math.getMath() if stoich_math is not None else None
                if math is None or _ast_references_time_varying(
                    math, libsbml, species_ids=species_ids, varying_ids=varying_ids
                ):
                    _raise_unsupported_sbml_feature(
                        "variable stoichiometry",
                        source=source,
                        detail=(
                            f"reaction '{rid}' species reference to "
                            f"'{ref.getSpecies()}' has a non-constant stoichiometryMath"
                        ),
                    )
            ref_id = ref.getId() if hasattr(ref, "getId") else ""
            if ref_id and ref_id in rule_or_ia:
                _raise_unsupported_sbml_feature(
                    "variable stoichiometry",
                    source=source,
                    detail=(
                        f"reaction '{rid}' species reference '{ref_id}' is a rule "
                        f"or initialAssignment target"
                    ),
                )


def _reject_time_varying_compartments(model, libsbml, *, source: str) -> None:
    """Refuse compartments whose volume changes in time under a concentration species.

    The volume scaling in STEP 5a divides a concentration species' amount rate by
    its compartment symbol, which is exact for a constant volume but omits the
    dilution term ``-[S]·(dV/dt)/V`` when the volume itself varies in time. A
    compartment that is a rate-rule target, or an assignment-rule target whose RHS
    does not constant-fold, therefore produces a silently wrong concentration ODE
    for any non-boundary ``hasOnlySubstanceUnits=false`` species it owns and is
    rejected here (GH #231). Constant volumes — including assignment rules that
    fold to a constant — are unaffected.
    """
    rate_targets, assignment_targets, assignment_math = _sbml_rule_target_sets(model, libsbml)
    species_ids = _sbml_species_ids(model)
    varying_ids = rate_targets | assignment_targets

    # Only compartments owning a concentration-valued state species carry the
    # missing dilution term; a time-varying volume over amount-only, boundary, or
    # constant species introduces no error and stays supported.
    concentration_compartments: set[str] = set()
    for i in range(model.getNumSpecies()):
        sp_obj = model.getSpecies(i)
        if sp_obj.getBoundaryCondition() or sp_obj.getConstant():
            continue
        if sp_obj.getHasOnlySubstanceUnits():
            continue
        concentration_compartments.add(sp_obj.getCompartment())

    for i in range(model.getNumCompartments()):
        cid = model.getCompartment(i).getId()
        if cid not in concentration_compartments:
            continue
        if cid in rate_targets:
            varies = True
        elif cid in assignment_targets:
            math = assignment_math.get(cid)
            varies = math is None or _ast_references_time_varying(
                math, libsbml, species_ids=species_ids, varying_ids=varying_ids
            )
        else:
            varies = False
        if varies:
            _raise_unsupported_sbml_feature(
                "time-varying compartment volume",
                source=source,
                detail=(
                    f"compartment '{cid}' size varies in time and owns a "
                    f"concentration-valued species; the dilution term is not modeled"
                ),
            )


def _checked_sbml_model(doc, libsbml, *, source: str):
    if doc.getNumErrors() > 0:
        errors = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                errors.append(err.getMessage())
        if errors:
            joined_errors = "; ".join(errors)
            lower_errors = joined_errors.lower()
            if "delay" in lower_errors:
                _raise_unsupported_sbml_feature(
                    "delays",
                    source=source,
                    detail=joined_errors,
                )
            if "syntax of 'id' attribute" in lower_errors or "sbml type 'sid'" in lower_errors:
                raise SBMLParseError(
                    "invalid_identifier",
                    None,
                    f"invalid SBML identifier: {joined_errors}",
                    source=source,
                )
            raise ValueError(f"SBML parsing errors in {source}: {joined_errors}")

    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model found in SBML source: {source}")

    if model.getNumEvents() > 0:
        _raise_unsupported_sbml_feature(
            "events",
            source=source,
            detail=f"found {model.getNumEvents()} event(s)",
        )

    if model.getNumConstraints() > 0:
        _raise_unsupported_sbml_feature(
            "constraints",
            source=source,
            detail=f"found {model.getNumConstraints()} constraint(s)",
        )

    _reject_variable_stoichiometry(model, libsbml, source=source)
    _reject_time_varying_compartments(model, libsbml, source=source)

    return model


def _apply_initial_assignments(
    model,
    libsbml,
    *,
    function_templates: dict[str, tuple[list[str], str]],
    species_info: dict[str, dict],
    params: dict[str, float],
    compartments: dict[str, float],
    all_syms: dict[str, sp.Symbol],
    source: str,
    warn_initial_assignment_failures: bool,
) -> None:
    initial_assignments: list[tuple[str, str]] = []
    for i in range(model.getNumInitialAssignments()):
        ia = model.getInitialAssignment(i)
        var_id = ia.getSymbol()
        formula_str = libsbml.formulaToString(ia.getMath())
        formula_str = _expand_function_calls(formula_str, function_templates)
        initial_assignments.append((var_id, formula_str))

    initial_assignment_symbols = {var_id for var_id, _formula in initial_assignments}
    numeric_initial_values: dict[str, float] = {
        sid: info["init"]
        for sid, info in species_info.items()
        if sid not in initial_assignment_symbols
    }
    numeric_initial_values.update(
        {
            pid: value
            for pid, value in params.items()
            if pid not in initial_assignment_symbols
        }
    )
    numeric_initial_values.update(
        {
            cid: value
            for cid, value in compartments.items()
            if cid not in initial_assignment_symbols
        }
    )

    def store_initial_assignment_value(var_id: str, init_value: float) -> None:
        if var_id in species_info:
            species_info[var_id]["init"] = init_value
        elif var_id in params:
            params[var_id] = init_value
        elif var_id in compartments:
            compartments[var_id] = init_value
        numeric_initial_values[var_id] = init_value

    pending_initial_assignments = initial_assignments
    while pending_initial_assignments:
        made_progress = False
        deferred: list[tuple[str, str]] = []

        for var_id, formula_str in pending_initial_assignments:
            if (
                var_id not in species_info
                and var_id not in params
                and var_id not in compartments
            ):
                continue

            init_value = _evaluate_initial_assignment(
                formula_str,
                all_syms,
                numeric_initial_values,
                source=source,
                variable=var_id,
                warn_initial_assignment_failures=warn_initial_assignment_failures,
                defer_unresolved=True,
            )
            if init_value is None:
                deferred.append((var_id, formula_str))
                continue

            store_initial_assignment_value(var_id, init_value)
            made_progress = True

        if not deferred:
            break

        if not made_progress:
            for var_id, formula_str in deferred:
                init_value = _evaluate_initial_assignment(
                    formula_str,
                    all_syms,
                    numeric_initial_values,
                    source=source,
                    variable=var_id,
                    warn_initial_assignment_failures=warn_initial_assignment_failures,
                )
                if init_value is not None:
                    store_initial_assignment_value(var_id, init_value)
            break

        pending_initial_assignments = deferred
