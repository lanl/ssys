# mypy: ignore-errors
# ruff: noqa: F401, F403, F405, I001
"""Antimony function-template expansion helpers."""

from ssys._recaster.common import *

def _expand_function_calls(
    expr_str: str, function_templates: dict[str, tuple[list[str], str]], max_depth: int = 10
) -> str:
    """
    Recursively expand function calls in an expression using function templates.

    Args:
        expr_str: Expression string potentially containing function calls
        function_templates: Dict mapping function name to (params_list, body_expr)
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Expression string with all function calls expanded

    Examples:
        Given templates:
            s(x) := x^h / (1 + x^h)
            f(x) := (1 - delta) * (1 - s(x)) + delta
            M(x) := 1 + gam * h * x^(h - 1) / (1 + x^h)^2

        _expand_function_calls("M(x1)", templates)
        → "1 + gam * h * x1^(h - 1) / (1 + x1^h)^2"

        _expand_function_calls("f(x3)", templates)
        → "(1 - delta) * (1 - x3^h / (1 + x3^h)) + delta"
    """
    if max_depth <= 0:
        return expr_str  # Prevent infinite recursion

    if not function_templates:
        return expr_str

    result = expr_str

    while max_depth > 0:
        max_depth -= 1

        call = _find_next_template_call(result, function_templates)
        if call is None:
            break

        start, end, func_name, args = call
        params, body = function_templates[func_name]

        # Perform substitution: replace each param with corresponding arg
        expanded = body
        for param, arg in zip(params, args, strict=False):
            # Use word boundary replacement to avoid partial matches
            # e.g., replacing 'x' shouldn't affect 'x1' or 'exp'
            expanded = _substitute_param(expanded, param, arg)

        # Wrap in parentheses to preserve operator precedence
        expanded = f"({expanded})"

        # Replace the function call with the expanded body
        result = result[:start] + expanded + result[end:]

    return result


def _find_next_template_call(
    expr_str: str, function_templates: dict[str, tuple[list[str], str]]
) -> tuple[int, int, str, list[str]] | None:
    """Find the next expandable template call, handling balanced parentheses."""
    for match in func_call_start_pat.finditer(expr_str):
        func_name = match.group(1)
        if func_name not in function_templates:
            continue

        paren_start = match.end() - 1
        paren_end = _find_matching_paren(expr_str, paren_start)
        if paren_end is None:
            continue

        args_str = expr_str[paren_start + 1 : paren_end]
        args = _parse_function_args(args_str)
        params, _body = function_templates[func_name]
        if len(args) != len(params):
            continue

        return match.start(), paren_end + 1, func_name, args

    return None


def _find_matching_paren(text: str, paren_start: int) -> int | None:
    """Return the index of the closing parenthesis matching paren_start."""
    depth = 0
    for idx in range(paren_start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _parse_function_args(args_str: str) -> list[str]:
    """
    Parse comma-separated function arguments, respecting nested parentheses.

    Example: "x1, (a + b), f(y, z)" → ["x1", "(a + b)", "f(y, z)"]
    """
    if not args_str.strip():
        return []

    args = []
    current_arg = ""
    paren_depth = 0

    for char in args_str:
        if char == "(":
            paren_depth += 1
            current_arg += char
        elif char == ")":
            paren_depth -= 1
            current_arg += char
        elif char == "," and paren_depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char

    # Don't forget the last argument
    if current_arg.strip():
        args.append(current_arg.strip())

    return args


def _substitute_param(body: str, param: str, arg: str) -> str:
    """
    Substitute a parameter with its argument in a function body.
    Uses word boundary matching to avoid partial substitutions.

    Example: _substitute_param("x^h / (1 + x^h)", "x", "x1")
             → "x1^h / (1 + x1^h)"

    This correctly handles:
    - "x" → "x1" (simple variable)
    - "x^h" → "x1^h" (variable with exponent)
    - Does NOT change "exp" when replacing "x"
    """
    replacement = _format_function_argument_for_substitution(arg)

    # Build a regex pattern that matches the parameter as a whole word
    # Word boundaries include: start/end of string, operators, parentheses, spaces
    # Use negative lookbehind and lookahead for alphanumerics and underscore
    pattern = r"(?<![A-Za-z_\d])" + re.escape(param) + r"(?![A-Za-z_\d])"
    return re.sub(pattern, replacement, body)


def _format_function_argument_for_substitution(arg: str) -> str:
    """Parenthesize non-atomic function arguments before template substitution."""
    stripped = arg.strip()
    if simple_identifier_pat.match(stripped) or simple_numeric_literal_pat.match(stripped):
        return stripped
    return f"({stripped})"


def expand_antimony_function_templates(text: str) -> str:
    """
    Expand legacy Antimony-style function templates in executable model text.

    This handles paper-style shorthand such as ``f(x) := x/(1+x)`` before text is
    passed to the reference Antimony parser, which does not accept that syntax.
    """
    function_templates: dict[str, tuple[list[str], str]] = {}
    non_template_lines: list[str] = []

    for raw_line in text.splitlines():
        code, comment_sep, comment = raw_line.partition("//")
        stmt = code.strip().rstrip(";").strip()
        func_match = func_def_pat.match(stmt)
        if func_match:
            func_name = func_match.group(1)
            params_str = func_match.group(2).strip()
            body = func_match.group(3).strip().rstrip(";").strip()
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            function_templates[func_name] = (params, body)
            if comment_sep:
                non_template_lines.append(comment_sep + comment)
            else:
                non_template_lines.append("")
            continue
        non_template_lines.append(raw_line)

    if not function_templates:
        return text

    expanded_lines: list[str] = []
    for raw_line in non_template_lines:
        code, comment_sep, comment = raw_line.partition("//")
        stripped = code.strip()
        lower = stripped.lower()
        if stripped and not lower.startswith("model ") and lower != "end":
            code = _expand_function_calls(code, function_templates)
        expanded_lines.append(code + (comment_sep + comment if comment_sep else ""))

    return "\n".join(expanded_lines)
