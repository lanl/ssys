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
