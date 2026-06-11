# mypy: ignore-errors
# ruff: noqa: F401, F403, F405, I001
"""Identifier sanitization and name collection helpers."""

from ssys._recaster.common import *

def _sanitize_antimony_name(name: str) -> str:
    """
    Sanitize a name to avoid Antimony reserved keyword conflicts.

    Models in BioModels commonly use variable names (compartment, species,
    parameter names) that conflict with Antimony reserved keywords. When ssys
    outputs recast models, these cause parsing errors. We fix this by appending
    '_var' suffix to conflicting names.

    The comparison is case-insensitive: "Compartment", "COMPARTMENT", and
    "compartment" are all sanitized to "Compartment_var", "COMPARTMENT_var", etc.

    Args:
        name: Original identifier name

    Returns:
        Sanitized name with '_var' suffix if it conflicts, otherwise unchanged

    Examples:
        >>> _sanitize_antimony_name("compartment")
        'compartment_var'
        >>> _sanitize_antimony_name("DNA")
        'DNA_var'
        >>> _sanitize_antimony_name("X")
        'X'
    """
    if name.lower() in {n.lower() for n in ANTIMONY_RESERVED_KEYWORDS}:
        return f"{name}_var"
    return name


def _build_name_sanitization_map(names: set[str]) -> dict[str, str]:
    """
    Build a mapping of original names to sanitized names.

    Only includes entries for names that need sanitization.

    Args:
        names: Set of all identifier names used in the model

    Returns:
        Dict mapping original_name -> sanitized_name for conflicting names
    """
    mapping = {}
    for name in names:
        sanitized = _sanitize_antimony_name(name)
        if sanitized != name:
            mapping[name] = sanitized
    return mapping


def _apply_name_sanitization(text: str, name_map: dict[str, str]) -> str:
    """
    Apply name sanitization to a text string (expression or identifier).

    Uses word-boundary matching to avoid partial replacements.

    Args:
        text: Original text
        name_map: Mapping of original_name -> sanitized_name

    Returns:
        Text with names sanitized
    """
    if not name_map:
        return text
    result = text
    for orig, sanitized in name_map.items():
        # Use word boundary regex to match whole identifiers only
        pattern = r"\b" + re.escape(orig) + r"\b"
        result = re.sub(pattern, sanitized, result)
    return result


def _format_antimony_token(
    value: object, name_map: dict[str, str] | None, *, expression: bool = False
) -> str:
    """
    Format an Antimony identifier or expression with the shared sanitizer.

    Comments may intentionally keep original model names for validator/notebook
    readability, but executable Antimony identifiers must pass through this
    helper so reserved names are rewritten consistently.
    """
    if expression:
        return _apply_name_sanitization(str(value), name_map or {})

    name = value.name if isinstance(value, sp.Symbol) else str(value)
    if name_map and name in name_map:
        return name_map[name]
    return name


def _collect_antimony_names(result: "RecastResult") -> set[str]:
    """Collect model identifiers that may need Antimony reserved-name sanitization."""
    names: set[str] = set()

    def add(value: object) -> None:
        if isinstance(value, sp.Symbol):
            names.add(value.name)
        else:
            names.add(str(value))

    def add_expr_symbols(value: object) -> None:
        if isinstance(value, sp.Expr):
            for sym in value.free_symbols:
                add(sym)
        else:
            for token in re.findall(r"\b[A-Za-z_]\w*\b", str(value)):
                names.add(token)

    for var in result.variables:
        add(var)
    for eq in result.equations:
        add(eq.var)
        add_expr_symbols(eq.growth[0])
        add_expr_symbols(eq.decay[0])
        for sym, exp in eq.growth[1].items():
            add(sym)
            add_expr_symbols(exp)
        for sym, exp in eq.decay[1].items():
            add(sym)
            add_expr_symbols(exp)
    for gma_eq in result.gma_equations:
        add(gma_eq.var)
        for coeff, exps in [*gma_eq.production, *gma_eq.degradation]:
            add_expr_symbols(coeff)
            for sym, exp in exps.items():
                add(sym)
                add_expr_symbols(exp)
    for orig, aux_list in result.factor_map.items():
        add(orig)
        for aux in aux_list:
            add(aux)
    for aux, defn in result.auxiliary_defs.items():
        add(aux)
        add_expr_symbols(defn)
    for symbol, expr in result.initial_exprs.items():
        add(symbol)
        add_expr_symbols(expr)
    for initial in result.initials.keys():
        if isinstance(initial, sp.Symbol):
            add(initial)
    for comp_name in result.compartments.keys():
        names.add(comp_name)
    for param_name in result.params.keys():
        names.add(param_name)
    for rule_name, rule_expr in result.assignment_rules.items():
        names.add(rule_name)
        add_expr_symbols(rule_expr)

    return names
