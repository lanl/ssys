import re
from dataclasses import dataclass, field
from enum import Enum

import sympy as sp

arrow_pat = re.compile(r"<->|->")
prime_rule_pat = re.compile(r"^\s*\$?([A-Za-z_]\w*)\s*'\s*=\s*(.+)$")
# Pattern to detect function definitions: name(param1, param2, ...) := expression
# Examples: s(x) := x^h / (1 + x^h)
#           f(x) := (1 - delta) * (1 - s(x)) + delta
#           M(x) := 1 + gam * h * x^(h - 1) / (1 + x^h)^2
func_def_pat = re.compile(r"^([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:=\s*(.+)$")
# Pattern to detect function calls in expressions: name(arg1, arg2, ...)
func_call_pat = re.compile(r"([A-Za-z_]\w*)\s*\(([^()]*)\)")

EPS_INIT = 1e-6


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
    changed = True

    while changed and max_depth > 0:
        changed = False
        max_depth -= 1

        # Find all function calls in the expression
        # We need to handle nested parentheses carefully
        # Use an iterative approach: find innermost function calls first

        for match in list(func_call_pat.finditer(result)):
            func_name = match.group(1)
            args_str = match.group(2).strip()

            # Check if this is a known function template
            if func_name not in function_templates:
                continue  # Skip built-in functions like sin, cos, exp, etc.

            params, body = function_templates[func_name]

            # Parse arguments (split by comma, handling nested parentheses)
            args = _parse_function_args(args_str)

            # Check argument count matches parameter count
            if len(args) != len(params):
                continue  # Skip if argument count doesn't match

            # Perform substitution: replace each param with corresponding arg
            expanded = body
            for param, arg in zip(params, args, strict=False):
                # Use word boundary replacement to avoid partial matches
                # e.g., replacing 'x' shouldn't affect 'x1' or 'exp'
                expanded = _substitute_param(expanded, param, arg)

            # Wrap in parentheses to preserve operator precedence
            expanded = f"({expanded})"

            # Replace the function call with the expanded body
            result = result[: match.start()] + expanded + result[match.end() :]
            changed = True
            break  # Start over after each substitution to handle nested calls

    return result


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
    # Build a regex pattern that matches the parameter as a whole word
    # Word boundaries include: start/end of string, operators, parentheses, spaces
    # Use negative lookbehind and lookahead for alphanumerics and underscore
    pattern = r"(?<![A-Za-z_\d])" + re.escape(param) + r"(?![A-Za-z_\d])"
    return re.sub(pattern, arg, body)


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
            except Exception:
                coeff = 1
                name = p
        result.append((coeff, name))
    return result


def _expand_exps_through_factors(exps, factor_map):
    """Return a new {sym: exp} dict where any original symbol present in factor_map
    is replaced by its factor variables, each receiving the same exponent."""
    new: dict[sp.Symbol, sp.Expr] = {}
    for s, e in exps.items():
        if s in factor_map:
            for v in factor_map[s]:
                new[v] = new.get(v, sp.sympify(0)) + e
        else:
            new[s] = new.get(s, sp.sympify(0)) + e
    return new


def _numeric_param_subs(expr: sp.Expr, params: dict[str, float]) -> sp.Expr:
    """Replace parameter symbols in expr with their numeric values from params."""
    if not params:
        return expr
    # Build a mapping only for symbols actually used in expr
    subs = {s: sp.Float(params[s.name]) for s in expr.free_symbols if s.name in params}
    return sp.simplify(expr.subs(subs)) if subs else expr


@dataclass
class Reaction:
    name: str | None
    lhs: list[tuple[int, str]]
    rhs: list[tuple[int, str]]
    rate_expr: str


@dataclass
class ModelIR:
    species: set[str] = field(default_factory=set)
    boundary: set[str] = field(default_factory=set)
    params: dict[str, float] = field(default_factory=dict)
    initial: dict[str, float] = field(default_factory=dict)
    reactions: list[Reaction] = field(default_factory=list)
    explicit_rates: dict[str, str] = field(default_factory=dict)
    assignment_rules: dict[str, str] = field(default_factory=dict)
    raw_lines: list[str] = field(default_factory=list)
    param_exprs: dict[str, str] = field(
        default_factory=dict
    )  # Store parameter expressions before evaluation
    initial_exprs: dict[str, str] = field(
        default_factory=dict
    )  # Store initial condition expressions
    antimony_text: str = ""  # Cache original Antimony text for RoadRunner
    # Compartment info: {compartment_name: size} and {species_name: compartment_name}
    compartments: dict[str, float] = field(default_factory=dict)  # compartment_name -> size
    species_compartment: dict[str, str] = field(
        default_factory=dict
    )  # species_name -> compartment_name
    # Simulation metadata (from @SIM comments)
    sim_t_start: float | None = None  # Simulation start time
    sim_t_end: float | None = None  # Simulation end time
    sim_n_steps: int | None = None  # Number of simulation steps
    eps_init: float | None = None  # User-specified EPS_INIT for zero IC replacement
    # Function templates: name -> (param_list, expression_body)
    # e.g., "M" -> (["x"], "1 + gam * h * x^(h-1) / (1 + x^h)^2")
    function_templates: dict[str, tuple[list[str], str]] = field(default_factory=dict)


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


def _format_sim_metadata_lines(result: "RecastResult") -> list[str]:
    """
    Format @SIM metadata as Antimony comment lines.

    Returns list of lines to append before 'end' in output.
    """
    lines = []
    
    # Build @SIM line with all available metadata
    sim_parts = []
    if result.sim_t_start is not None:
        sim_parts.append(f"T_START={result.sim_t_start:g}")
    if result.sim_t_end is not None:
        sim_parts.append(f"T_END={result.sim_t_end:g}")
    if result.sim_n_steps is not None:
        sim_parts.append(f"N_STEPS={result.sim_n_steps}")
    if result.eps_init is not None:
        sim_parts.append(f"EPS_INIT={result.eps_init:g}")
    
    if sim_parts:
        lines.append(f"// @SIM {' '.join(sim_parts)}")
        # Add note about zero IC replacement if eps_init was used
        if result.eps_init is not None:
            lines.append("// Note: Zero-valued initial conditions are replaced with EPS_INIT during recasting.")
    
    return lines


def parse_antimony(text: str) -> ModelIR:
    ir = ModelIR()
    ir.antimony_text = text  # Cache original text for RoadRunner
    ir.raw_lines = [ln.rstrip() for ln in text.splitlines()]

    # FIRST: Extract @SIM metadata from comments BEFORE any processing
    # This must happen before raw_lines is overwritten by joined_lines
    # Format: // @SIM T_START=0 T_END=100 N_STEPS=500
    import re

    sim_marker_pattern = re.compile(r"@SIM\b")
    key_value_pattern = re.compile(r"(\w+)\s*=\s*([0-9.eE+-]+)")

    for raw in ir.raw_lines:
        if "//" in raw:
            comment_part = raw.split("//", 1)[1]
            if sim_marker_pattern.search(comment_part):
                for match in key_value_pattern.finditer(comment_part):
                    key = match.group(1).upper()
                    value = match.group(2)
                    if key == "T_START":
                        try:
                            ir.sim_t_start = float(value)
                        except ValueError:
                            pass
                    elif key == "T_END":
                        try:
                            ir.sim_t_end = float(value)
                        except ValueError:
                            pass
                    elif key == "N_STEPS":
                        try:
                            ir.sim_n_steps = int(float(value))
                        except ValueError:
                            pass

    # NOTE: Line continuation is NOT used for our simple parser.
    # Each line is treated as a separate statement.
    # Complex models should use parse_antimony_via_sbml() instead.

    # First pass: extract @SIM metadata from comments
    # Format: // @SIM T_START=0 T_END=100 N_STEPS=500
    import re

    # Match @SIM marker
    sim_marker_pattern = re.compile(r"@SIM\b")
    # Match individual key=value pairs
    key_value_pattern = re.compile(r"(\w+)\s*=\s*([0-9.eE+-]+)")

    for raw in ir.raw_lines:
        if "//" in raw:
            comment_part = raw.split("//", 1)[1]
            # Check if this line has @SIM or @SIMTIME
            if sim_marker_pattern.search(comment_part):
                # Extract all key=value pairs from this line
                for match in key_value_pattern.finditer(comment_part):
                    key = match.group(1).upper()
                    value = match.group(2)
                    if key == "T_START":
                        try:
                            ir.sim_t_start = float(value)
                        except ValueError:
                            pass
                    elif key == "T_END":
                        try:
                            ir.sim_t_end = float(value)
                        except ValueError:
                            pass
                    elif key == "N_STEPS":
                        try:
                            ir.sim_n_steps = int(float(value))
                        except ValueError:
                            pass

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
                except Exception:
                    # Not a simple number - it's an expression
                    pass

                if not is_simple_number:
                    # Store the expression string for symbolic preservation
                    ir.initial_exprs[left] = right

                # Try to evaluate immediately (will work for simple numeric constants)
                try:
                    val = float(sp.sympify(right))
                except Exception:
                    val = None
                ir.initial[left] = val if val is not None else 0.0
                continue

            # Assignment rules: name := expression
            # FIRST check if this is a function definition: name(params) := expr
            func_match = func_def_pat.match(stmt)
            if func_match:
                func_name = func_match.group(1)
                params_str = func_match.group(2).strip()
                body = func_match.group(3).strip()
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


def _extract_sim_metadata(
    text: str,
) -> tuple[float | None, float | None, int | None, float | None]:
    """
    Extract @SIM metadata from Antimony comments.

    Format: // @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6

    Args:
        text: Antimony text to scan for @SIM metadata

    Returns:
        Tuple of (t_start, t_end, n_steps, eps_init), any of which may be None if not found
    """
    t_start = None
    t_end = None
    n_steps = None
    eps_init = None

    sim_marker_pattern = re.compile(r"@SIM\b")
    key_value_pattern = re.compile(r"(\w+)\s*=\s*([0-9.eE+-]+)")

    for line in text.splitlines():
        if "//" in line:
            comment_part = line.split("//", 1)[1]
            if sim_marker_pattern.search(comment_part):
                for match in key_value_pattern.finditer(comment_part):
                    key = match.group(1).upper()
                    value = match.group(2)
                    if key == "T_START":
                        try:
                            t_start = float(value)
                        except ValueError:
                            pass
                    elif key == "T_END":
                        try:
                            t_end = float(value)
                        except ValueError:
                            pass
                    elif key == "N_STEPS":
                        try:
                            n_steps = int(float(value))
                        except ValueError:
                            pass
                    elif key == "EPS_INIT":
                        try:
                            eps_init = float(value)
                        except ValueError:
                            pass

    return t_start, t_end, n_steps, eps_init


def _preprocess_antimony_text(text: str) -> str:
    """
    Preprocess Antimony text before passing to libantimony.

    Currently a no-op - we require input .ant files to use standard Antimony syntax:
    - Use backslash \\ for line continuation (NOT implicit continuation)
    - Use `function name(x) expr end` for function definitions (NOT `name(x) := expr`)

    This function exists as a hook for future preprocessing needs.
    """
    return text


def parse_antimony_via_sbml(antimony_text: str) -> "SymSystem":
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
    t_start, t_end, n_steps, eps_init = _extract_sim_metadata(antimony_text)

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
    sym = parse_sbml_from_string(sbml_string)

    # Attach simulation metadata if found
    # These public attributes are used by the validator for trajectory tests
    sym.sim_t_start = t_start
    sym.sim_t_end = t_end
    sym.sim_n_steps = n_steps
    sym.eps_init = eps_init

    # Cache original Antimony text for RoadRunner simulation
    sym.antimony_text = antimony_text

    return sym


def parse_sbml_from_string(sbml_string: str) -> "SymSystem":
    """
    Parse an SBML string and return a SymSystem for recasting.

    This is the string-input variant of parse_sbml(), used by parse_antimony_via_sbml().

    Args:
        sbml_string: SBML model as string

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
    return _parse_sbml_document(doc, source="<string>")


def parse_sbml(sbml_path: str) -> "SymSystem":
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
    return _parse_sbml_document(doc, source=sbml_path)


def _parse_sbml_document(doc, source: str = "<unknown>") -> "SymSystem":
    """
    Shared implementation for parsing an SBML document.

    Args:
        doc: libsbml.SBMLDocument object
        source: Description of source for error messages

    Returns:
        SymSystem ready for recasting
    """
    import libsbml

    # Check for errors
    if doc.getNumErrors() > 0:
        errors = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                errors.append(err.getMessage())
        if errors:
            raise ValueError(f"SBML parsing errors in {source}: {'; '.join(errors)}")

    model = doc.getModel()
    if model is None:
        raise ValueError(f"No model found in SBML source: {source}")

    # ========================================================================
    # STEP 1: Extract species information
    # ========================================================================
    species_info: dict[str, dict] = {}
    boundary_species: set[str] = set()

    for i in range(model.getNumSpecies()):
        sp_obj = model.getSpecies(i)
        sid = sp_obj.getId()
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
        if p.isSetValue():
            params[pid] = p.getValue()
        else:
            params[pid] = 0.0

    # Local parameters (from kinetic laws)
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl:
            for j in range(kl.getNumParameters()):
                lp = kl.getParameter(j)
                lpid = lp.getId()
                if lp.isSetValue():
                    params[lpid] = lp.getValue()
                else:
                    params[lpid] = 0.0

    # ========================================================================
    # STEP 3: Extract compartments
    # ========================================================================
    compartments: dict[str, float] = {}
    for i in range(model.getNumCompartments()):
        c = model.getCompartment(i)
        cid = c.getId()
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

        # Parse rate expression
        try:
            rate_expr = sp.sympify(formula_str, locals=all_syms)
        except Exception:
            # Skip reactions with unparseable rate laws
            continue

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
    for i in range(model.getNumRules()):
        rule = model.getRule(i)

        if rule.getTypeCode() == libsbml.SBML_RATE_RULE:
            var_id = rule.getVariable()
            formula_str = libsbml.formulaToString(rule.getMath())

            if var_id in all_syms:
                try:
                    rate_expr = sp.sympify(formula_str, locals=all_syms)
                    var_sym = all_syms[var_id]
                    # Rate rules replace the reaction-based ODE
                    odes[var_sym] = rate_expr
                except Exception:
                    pass

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
            # Store as string for SymSystem (will be parsed later if needed)
            assignment_rules[var_id] = formula_str

    # ========================================================================
    # STEP 6b: Handle InitialAssignments (parameter expressions for ICs)
    # ========================================================================
    # In SBML, InitialAssignments allow ICs to be set via formulas like I = I_b
    # or parameter expressions like gamma_rate = 1/7
    # We need to evaluate these formulas to get numeric initial conditions
    #
    # NOTE: When Antimony converts "gamma_rate = 1/7" to SBML, it creates:
    #   1. A parameter element (value may be unset or 0)
    #   2. An InitialAssignment that defines the actual value
    # So we MUST process InitialAssignments for parameters, not just species
    for i in range(model.getNumInitialAssignments()):
        ia = model.getInitialAssignment(i)
        var_id = ia.getSymbol()
        formula_str = libsbml.formulaToString(ia.getMath())

        if var_id in species_info:
            try:
                # Evaluate the formula with known parameters
                init_expr = sp.sympify(formula_str, locals=all_syms)
                # Substitute parameter values using correct symbols from all_syms
                for param_name, param_val in params.items():
                    if param_name in all_syms:
                        init_expr = init_expr.subs(all_syms[param_name], param_val)
                # Update the species initial value
                species_info[var_id]["init"] = float(init_expr)
            except Exception:
                pass  # Keep default if evaluation fails

        elif var_id in params:
            # Handle parameter expressions like gamma_rate = 1/7
            try:
                # Evaluate the formula
                init_expr = sp.sympify(formula_str, locals=all_syms)
                # Substitute known parameter values
                for param_name, param_val in params.items():
                    if param_name != var_id and param_name in all_syms:
                        init_expr = init_expr.subs(all_syms[param_name], param_val)
                # Update the parameter value
                params[var_id] = float(init_expr)
            except Exception:
                pass  # Keep default if evaluation fails

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
        compartments=compartments,
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


@dataclass
class SymSystem:
    vars: list[sp.Symbol]
    params: dict[str, float]
    odes: dict[sp.Symbol, sp.Expr]
    initials: dict[sp.Symbol, float]
    initial_exprs: dict[sp.Symbol, str] = field(default_factory=dict)  # Symbolic IC expressions
    assignment_rules: dict[str, str] = field(
        default_factory=dict
    )  # Assignment rules from original model
    compartments: dict[str, float] = field(default_factory=dict)  # compartment_name -> size
    # Optional metadata attributes (set by parse_antimony_via_sbml)
    sim_t_start: float | None = None  # Simulation start time from @SIM
    sim_t_end: float | None = None  # Simulation end time from @SIM
    sim_n_steps: int | None = None  # Number of simulation steps from @SIM
    eps_init: float | None = None  # User-specified EPS_INIT for zero IC replacement
    antimony_text: str = ""  # Cache original Antimony text for RoadRunner


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
    for name, expr_str in ir.assignment_rules.items():
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
        rate = sp.sympify(rxn.rate_expr, locals={**var_syms, **param_syms})
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

    return SymSystem(
        vars=list(odes.keys()),
        params={k: float(v) for k, v in ir.params.items()},
        odes=odes,
        initials=initials,
        initial_exprs=initial_exprs,
        assignment_rules=dict(ir.assignment_rules),  # Pass through from ModelIR
    )


from dataclasses import dataclass


def expand_to_terms(expr: sp.Expr) -> list[sp.Expr]:
    expr = sp.expand(expr)
    if expr.is_Add:
        return list(expr.args)
    else:
        return [expr]


@dataclass
class SSysEquation:
    var: sp.Symbol
    growth: tuple[sp.Expr, dict[sp.Symbol, float]]  # coefficient (symbolic), {sym: exponent}
    decay: tuple[sp.Expr, dict[sp.Symbol, float]]  # coefficient (symbolic), {sym: exponent}


@dataclass
class GMAEquation:
    """Generalized Mass Action equation with multiple production/degradation terms"""

    var: sp.Symbol
    production: list[tuple[sp.Expr, dict[sp.Symbol, float]]]  # [(coeff, {sym: exp}), ...]
    degradation: list[tuple[sp.Expr, dict[sp.Symbol, float]]]  # [(coeff, {sym: exp}), ...]


class RecastStatus(Enum):
    """Status of recasting operation"""

    CANONICAL_SSYSTEM = "canonical_ssystem"
    GMA = "gma"
    FAILED = "failed"


class SystemClass(Enum):
    """Classification of system form"""

    SSYSTEM = "S-system"  # 1-2 positive monomial terms per equation
    CANONICAL_SSYSTEM = "Canonical S-system"  # Exactly 2 positive terms (1 growth + 1 decay)
    GMA = "GMA"  # All monomials, constant coefficients
    GMA_TIME_VARYING = (
        "GMA with time-varying coefficients"  # Power-law structure, coefficients depend on clock T
    )
    GENERAL = "General"  # Contains non-monomial terms


@dataclass
class RecastResult:
    status: RecastStatus
    equations: list[SSysEquation]
    initials: dict[sp.Symbol, float]
    variables: list[sp.Symbol]
    factor_map: dict[sp.Symbol, list[sp.Symbol]] = field(default_factory=dict)
    gma_equations: list[GMAEquation] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    compartments: dict[str, float] = field(default_factory=dict)  # compartment_name -> size
    error_message: str | None = None
    blockers: dict[str, list[str]] = field(default_factory=dict)
    auxiliary_defs: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)  # Y_1 -> K_2 + X_1
    canonical_refusal_reason: str | None = None  # Why canonical S-system was refused
    initial_exprs: dict[sp.Symbol, str] = field(default_factory=dict)  # Symbolic IC expressions
    assignment_rules: dict[str, str] = field(
        default_factory=dict
    )  # Assignment rules from original model
    # Simulation metadata (propagated from input)
    sim_t_start: float | None = None
    sim_t_end: float | None = None
    sim_n_steps: int | None = None
    eps_init: float | None = None


def classify_system(sym: SymSystem) -> SystemClass:
    """
    Classify a SymSystem based on its structure.

    IMPORTANT: Expands assignment rules before classification to ensure
    hidden complexity (rational functions, time-dependence) is detected.

    Returns:
        SystemClass enum indicating the system type
    """
    # Build substitution map for assignment rules
    # This reveals hidden complexity like rational functions or time-dependence
    rule_subs = {}
    if sym.assignment_rules:
        # Build symbol map for parsing - include rule names as symbols
        all_syms = {s.name: s for s in sym.vars}
        for param_name in sym.params:
            all_syms[param_name] = sp.Symbol(param_name, positive=True)
        all_syms["time"] = sp.Symbol("time", positive=True)
        # Pre-create symbols for all rule names
        for rule_name in sym.assignment_rules:
            if rule_name not in all_syms:
                all_syms[rule_name] = sp.Symbol(rule_name, positive=True)
        
        # Parse assignment rules into sympy expressions
        for rule_name, rule_str in sym.assignment_rules.items():
            try:
                rule_expr = sp.sympify(rule_str, locals=all_syms)
                rule_sym = all_syms[rule_name]  # Use consistent symbol
                rule_subs[rule_sym] = rule_expr
            except Exception:
                pass
        
        # Expand nested rules (A may reference B)
        for _ in range(10):
            changed = False
            for rule_sym, rule_expr in list(rule_subs.items()):
                new_expr = rule_expr.subs(rule_subs)
                if new_expr != rule_expr:
                    rule_subs[rule_sym] = new_expr
                    changed = True
            if not changed:
                break
        
        # CRITICAL: Also create a name-based lookup for substitution
        # because ODEs may use different symbol objects with same name
        name_to_expr = {sym.name: expr for sym, expr in rule_subs.items()}

    is_canonical = True
    is_ssystem = True
    is_gma = True

    for _var, ode in sym.odes.items():
        # Expand assignment rules to reveal hidden structure
        if rule_subs:
            # First try direct substitution
            ode = ode.subs(rule_subs)
            # Also substitute by name (for symbol object mismatches)
            for sym_in_ode in ode.free_symbols:
                if sym_in_ode.name in name_to_expr:
                    ode = ode.subs(sym_in_ode, name_to_expr[sym_in_ode.name])
        terms = expand_to_terms(sp.expand(ode))

        # Separate into positive and negative monomial terms
        pos_monomials = []
        neg_monomials = []

        for term in terms:
            if term == 0:
                continue

            # Check if term is a monomial
            if not _is_term_monomial(term):
                # Has non-monomial terms - must be GENERAL
                return SystemClass.GENERAL

            # Determine sign
            sign = _get_coefficient_sign(term)
            if sign > 0:
                pos_monomials.append(term)
            else:
                neg_monomials.append(term)

        # Check canonical S-system: exactly 1 positive + 1 negative
        if len(pos_monomials) != 1 or len(neg_monomials) != 1:
            is_canonical = False

        # Check S-system: 1-2 total terms
        total_terms = len(pos_monomials) + len(neg_monomials)
        if total_terms < 1 or total_terms > 2:
            is_ssystem = False

        # Check GMA: may have multiple terms, but all must be monomials
        # (already verified above)

    # Return most specific classification
    if is_canonical:
        return SystemClass.CANONICAL_SSYSTEM
    elif is_ssystem:
        return SystemClass.SSYSTEM
    elif is_gma:
        return SystemClass.GMA
    else:
        return SystemClass.GENERAL


def classify_result(result: RecastResult, mode: str = "simplified") -> SystemClass:
    """
    Classify a RecastResult based on its output structure.

    Args:
        result: The RecastResult to classify
        mode: Output mode ('simplified' or 'canonical')
              In canonical mode, epsilon slack is added to zero coefficients,
              converting S-systems to Canonical S-systems

    Returns:
        SystemClass enum indicating the output type
    """
    # Check for time-varying coefficients (assignment rules that depend on clock T)
    # If assignment rules exist and reference the clock state T, this is GMA_TIME_VARYING
    has_time_varying_coeffs = False
    if result.assignment_rules:
        for _rule_name, rule_expr in result.assignment_rules.items():
            # Check if rule contains 'T' (clock state) as a standalone identifier
            # Use word boundary regex to match T but not words containing T (like "TIME")
            if re.search(r"\bT\b", str(rule_expr)):
                has_time_varying_coeffs = True
                break

    # If time-varying, return GMA_TIME_VARYING regardless of structure
    if has_time_varying_coeffs:
        return SystemClass.GMA_TIME_VARYING

    if result.status == RecastStatus.GMA:
        # GMA format - validate it's truly GMA, not just canonical
        has_multi_term = False
        for eq in result.gma_equations:
            # Check if production or degradation has multiple incompatible terms
            if len(eq.production) > 1:
                # Multiple production terms - check if they have same exponents
                if len(eq.production) > 1:
                    first_exps = eq.production[0][1]
                    for _, exps in eq.production[1:]:
                        if not _exponents_match(first_exps, exps):
                            has_multi_term = True
                            break

            if len(eq.degradation) > 1:
                # Multiple degradation terms - check if they have same exponents
                first_exps = eq.degradation[0][1]
                for _, exps in eq.degradation[1:]:
                    if not _exponents_match(first_exps, exps):
                        has_multi_term = True
                        break

            if has_multi_term:
                break

        if has_multi_term:
            return SystemClass.GMA

        # All terms are compatible - could be canonical
        # Check if each equation has exactly 1 production + 1 degradation
        for gma_eq2 in result.gma_equations:
            if len(gma_eq2.production) != 1 or len(gma_eq2.degradation) != 1:
                return SystemClass.GMA
        return SystemClass.CANONICAL_SSYSTEM

    elif result.status == RecastStatus.CANONICAL_SSYSTEM:
        # Count actual non-zero terms in each equation
        # An equation is canonical if it has exactly 2 non-zero terms (1 growth + 1 decay)
        # An equation is S-system if it has 1-2 non-zero terms
        #
        # IMPORTANT: In canonical mode, epsilon slack is added to zero coefficients,
        # converting equations like X' = 0 - h*X into X' = epsilon*X - (epsilon+h)*X
        # So we need to account for this transformation when classifying

        is_canonical = True
        is_ssystem = True

        for ssys_eq in result.equations:
            g_coeff = ssys_eq.growth[0]
            d_coeff = ssys_eq.decay[0]

            # Check if growth coefficient is non-zero
            g_nonzero = False
            if isinstance(g_coeff, (int, float)):
                g_nonzero = g_coeff != 0
            elif isinstance(g_coeff, sp.Expr):
                g_nonzero = g_coeff != sp.Integer(0)

            # Check if decay coefficient is non-zero
            d_nonzero = False
            if isinstance(d_coeff, (int, float)):
                d_nonzero = d_coeff != 0
            elif isinstance(d_coeff, sp.Expr):
                d_nonzero = d_coeff != sp.Integer(0)

            # In canonical mode, check if epsilon will be added
            # If either coefficient is zero, epsilon makes it canonical
            if mode == "canonical":
                # If we have 1 or 2 terms and one is zero, epsilon will make it canonical
                has_zero = (not g_nonzero) or (not d_nonzero)
                has_nonzero = g_nonzero or d_nonzero

                if has_zero and has_nonzero:
                    # This will become canonical with epsilon: e.g., 0 - h*X becomes epsilon*X - (epsilon+h)*X
                    # Count as 2 terms for canonical check
                    nonzero_count = 2
                else:
                    # Both nonzero or both zero - count actual terms
                    nonzero_count = (1 if g_nonzero else 0) + (1 if d_nonzero else 0)
            else:
                # Simplified mode: count actual non-zero terms
                nonzero_count = (1 if g_nonzero else 0) + (1 if d_nonzero else 0)

            # Check canonical: exactly 2 terms (actual or after epsilon)
            if nonzero_count != 2:
                is_canonical = False

            # Check S-system: 1-2 terms
            if nonzero_count < 1 or nonzero_count > 2:
                is_ssystem = False

        # Return most specific classification
        if is_canonical:
            return SystemClass.CANONICAL_SSYSTEM
        elif is_ssystem:
            return SystemClass.SSYSTEM
        else:
            return SystemClass.GMA

    else:
        return SystemClass.GENERAL


def _is_term_monomial(term: sp.Expr) -> bool:
    """
    Check if a term is a monomial (product of powers).

    A monomial is a product of:
    - Numeric constants
    - Symbols (can be parameters or state variables)
    - Powers with symbol bases (exponents can be numeric or symbolic)

    What makes it NON-monomial:
    - Functions (exp, sin, log, etc.)
    - Sums/differences in the base
    - Division by non-constant expressions
    """
    if term.is_Number:
        return True
    if isinstance(term, sp.Symbol):
        return True
    if isinstance(term, sp.Pow):
        base, exp = term.args
        # Base must be a symbol (state var or parameter)
        # Exponent can be numeric or symbolic (parameters are OK)
        return isinstance(base, sp.Symbol)
    if term.is_Mul:
        # All factors must be monomials
        for factor in term.args:
            if not _is_term_monomial(factor):
                return False
        return True
    return False


def _get_coefficient_sign(term: sp.Expr) -> int:
    """Get the sign of a term's coefficient. Returns 1 for positive, -1 for negative."""
    if term.is_Number:
        return 1 if float(term) >= 0 else -1
    if term.is_Mul:
        # Extract numeric coefficient
        coeff = 1.0
        for factor in term.args:
            if factor.is_Number:
                coeff *= float(factor)
        return 1 if coeff >= 0 else -1
    return 1  # Assume positive if no numeric coefficient


# --- CANONICALIZE AUX NAMES: must be placed below the dataclasses ---
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


def product_expr(coeff, exps: dict[sp.Symbol, float]) -> sp.Expr:
    """
    Build symbolic product expression from coefficient and exponents.
    coeff can be either float or sp.Expr (symbolic).
    """
    # Handle coefficient (numeric or symbolic)
    if isinstance(coeff, sp.Expr):
        expr = coeff
    else:
        # Numeric coefficient - convert to appropriate sympy type
        if isinstance(coeff, int) or (isinstance(coeff, float) and coeff == int(coeff)):
            expr = sp.Integer(int(coeff))
        else:
            expr = sp.Float(coeff)

    # Add power-law terms
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        if abs(e) < 1e-14:
            continue
        # Convert exponent to appropriate sympy type
        if isinstance(e, int) or (isinstance(e, float) and e == int(e)):
            exp_sym = sp.Integer(int(e))
        else:
            exp_sym = sp.Float(e)
        expr *= s**exp_sym

    return sp.simplify(expr)


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
            # Normalize the definition for comparison
            lifted_aux_exprs.add(sp.simplify(defn))

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
        const_denoms = set()  # denominators that depend only on constants
        dynamic_denoms = set()  # denominators that depend on state variables AND need lifting

        for denom in all_denoms:
            denom_vars = denom.free_symbols & state_vars
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
                for param_name, param_val in sym.params.items():
                    param_sym = sp.Symbol(param_name)
                    if param_sym in denom.free_symbols:
                        denom_val = denom_val.subs(param_sym, param_val)
                try:
                    recip_val = float(1.0 / denom_val)
                    # Replace 1/D with its numeric value
                    new_ode = new_ode.replace(denom ** (-1), sp.Float(recip_val))
                    # Handle other negative powers if present
                    for n in range(2, 6):
                        if denom ** (-n) in new_ode.atoms():
                            new_ode = new_ode.replace(denom ** (-n), sp.Float(recip_val**n))
                except Exception:
                    pass  # If evaluation fails, leave it as is
            new_odes[var] = new_ode

        # Now create auxiliary symbols only for dynamic denominators
        # Y = D (denominator itself, not reciprocal)
        # CRITICAL: Skip denominators that are already lifted composite auxiliaries
        # ALSO: Skip denominators that already have an auxiliary from previous iterations
        denom_to_aux: dict[sp.Expr, sp.Symbol] = {}

        # Build reverse lookup: normalized_denom -> existing auxiliary
        existing_denom_to_aux: dict[sp.Expr, sp.Symbol] = {}
        for aux, defn in all_aux_defs.items():
            defn_normalized = sp.simplify(defn)
            existing_denom_to_aux[defn_normalized] = aux

        for denom in sorted(dynamic_denoms, key=str):
            # Check if this denominator is already a lifted composite auxiliary
            # Normalize for comparison
            denom_normalized = sp.simplify(denom)

            if denom_normalized in lifted_aux_exprs:
                # This denominator is already a lifted auxiliary - SKIP
                # Use negative exponent directly (e.g., log(Z_1)^-1)
                continue

            # Check if we already have an auxiliary for this denominator (from previous iteration)
            if denom_normalized in existing_denom_to_aux:
                # Reuse existing auxiliary
                denom_to_aux[denom] = existing_denom_to_aux[denom_normalized]
                continue

            # Not a lifted auxiliary and no existing auxiliary - create new Y
            Y = sp.symbols(f"Y_{aux_counter}", positive=True)
            denom_to_aux[denom] = Y
            all_aux_defs[Y] = denom  # Accumulate definitions
            existing_denom_to_aux[denom_normalized] = Y  # Track for future denoms in this iteration
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
                    # Check if base matches this denominator (using simplify for robustness)
                    if sp.simplify(base - denom) == 0:
                        # Replace denom^exp with Y^exp
                        new_ode = new_ode.subs(atom, Y**exp)
            new_odes[var] = sp.simplify(new_ode)

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
            new_aux_odes[Y] = sp.simplify(Y_ode)

        # Combine original and auxiliary ODEs
        combined_odes = {**new_odes, **new_aux_odes}

        # Compute initial conditions for auxiliaries
        new_initials = dict(sym.initials)
        
        # Build name-based lookup for initial conditions to handle symbol object mismatch
        # (Different Symbol objects with same name won't match in dict lookup)
        initials_by_name = {str(k): v for k, v in sym.initials.items() if isinstance(k, sp.Symbol)}
        
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
            except Exception:
                Y_init = 1.0  # Fallback if evaluation fails
            new_initials[Y] = Y_init

        # Create new variable list: keep original vars, add Y auxiliaries
        new_vars = list(sym.vars) + list(denom_to_aux.values())

        # Update sym for next iteration
        sym = SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=combined_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,  # Propagate symbolic IC expressions
            assignment_rules=sym.assignment_rules,  # Preserve original assignment rules
            compartments=sym.compartments,  # Propagate compartments
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
                    new_terms.append(const_value * sp.Pow(dummy, 0, evaluate=False))
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
    if isinstance(func.func, type(sp.sin(sp.Symbol("x")).func)):
        # sin(x) ∈ [-1, 1] → add 2 → [1, 3]
        return True, 2.0
    if isinstance(func.func, type(sp.cos(sp.Symbol("x")).func)):
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
    X_at_0 = X
    for var in sym.vars:
        if var in X.free_symbols:
            X_at_0 = X_at_0.subs(var, sym.initials.get(var, 1.0))
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

        # Create updated system
        sym = SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=new_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,
            assignment_rules=new_assignment_rules,
            compartments=sym.compartments,  # Propagate compartments
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

    time_only_functions = set()
    state_dependent_functions = set()

    for func in all_functions:
        if _is_time_only_function(func, state_vars):
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
        if _is_time_only_function(sqrt_expr, state_vars):
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

    # Compute Z' using coupled derivatives for sin/cos
    new_aux_odes: dict[sp.Symbol, sp.Expr] = {}

    # Handle sin/cos pairs with coupled derivatives
    # CRITICAL: Create time symbol with positive=True to match SBML parser
    time_sym = sp.Symbol("time", positive=True)
    
    for arg, funcs_dict2 in sin_cos_pairs.items():
        sin_func = funcs_dict2.get("sin", sp.sin(arg))
        cos_func = funcs_dict2.get("cos", sp.cos(arg))
        Z_sin = func_to_aux[sin_func]
        Z_cos = func_to_aux[cos_func]

        # d/dt[sin(arg) + 2] = cos(arg) * d(arg)/dt = (Z_cos - 2) * d(arg)/dt
        # d/dt[cos(arg) + 2] = -sin(arg) * d(arg)/dt = -(Z_sin - 2) * d(arg)/dt = (2 - Z_sin) * d(arg)/dt

        # Compute d(arg)/dt using chain rule
        arg_prime = sp.Integer(0)
        for var in sym.vars:
            if var in arg.free_symbols:
                partial = sp.diff(arg, var)
                arg_prime += partial * new_odes[var]

        # Handle explicit time dependence: d(time)/dt = 1
        # CRITICAL: For sin(time), cos(time), the argument IS time, so arg' = 1
        if time_sym in arg.free_symbols:
            partial_t = sp.diff(arg, time_sym)
            arg_prime += partial_t  # d(time)/dt = 1

        # Z_sin' = (Z_cos - 2) * arg'
        Z_sin_ode = (Z_cos - 2) * arg_prime
        new_aux_odes[Z_sin] = sp.simplify(Z_sin_ode)

        # Z_cos' = (2 - Z_sin) * arg'
        Z_cos_ode = (2 - Z_sin) * arg_prime
        new_aux_odes[Z_cos] = sp.simplify(Z_cos_ode)

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

        # Compute d(base)/dt using chain rule
        base_prime = sp.Integer(0)
        for var in sym.vars:
            if var in base.free_symbols:
                partial = sp.diff(base, var)
                base_prime += partial * new_odes[var]

        # Handle time dependence (time_sym defined above at start of sin/cos loop)
        if time_sym in base.free_symbols:
            partial_t = sp.diff(base, time_sym)
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

        # Compute df/dt using chain rule: df/dt = sum_i (∂f/∂X_i) * dX_i/dt
        func_prime = sp.Integer(0)
        # CRITICAL FIX: Use all_vars_with_odes which includes ALL variables with ODEs
        # (original variables + sin/cos auxiliaries created earlier)
        for var in all_vars_with_odes:
            if var in func.free_symbols:
                partial = sp.diff(func, var)

                # Use the ODE for var (either from new_odes or new_aux_odes)
                var_ode = new_odes.get(var) or new_aux_odes.get(var)
                if var_ode is not None:
                    # Compute the chain rule term
                    term = partial * var_ode

                    # Replace composite functions with auxiliaries AFTER multiplication
                    # Use .subs() instead of .replace() to handle algebraic simplifications
                    # (e.g., exp(2*x) = exp(x)^2)
                    subs_map = {}
                    for other_func, other_Z in func_to_aux.items():
                        offset = func_to_offset[other_func]
                        if offset > 0:
                            subs_map[other_func] = other_Z - offset
                        else:
                            subs_map[other_func] = other_Z
                    term = term.subs(subs_map)

                    func_prime += term

        # Handle explicit time dependence: d(time)/dt = 1
        # (time_sym defined above at start of sin/cos loop)
        if time_sym in func.free_symbols:
            partial_t = sp.diff(func, time_sym)
            # Substitute auxiliaries in the time derivative term
            subs_map = {}
            for other_func, other_Z in func_to_aux.items():
                offset = func_to_offset[other_func]
                if offset > 0:
                    subs_map[other_func] = other_Z - offset
                else:
                    subs_map[other_func] = other_Z
            partial_t = partial_t.subs(subs_map)
            func_prime += partial_t

        # Store the computed ODE
        Z_ode = func_prime

        # CRITICAL: Final expansion and simplification pass
        # Expand products and collect like terms
        Z_ode = sp.expand(Z_ode)

        # Replace any remaining instances of composite functions with auxiliaries
        # Use .subs() instead of .replace() to handle algebraic simplifications
        subs_map = {}
        for other_func, other_Z in func_to_aux.items():
            offset = func_to_offset[other_func]
            if offset > 0:
                subs_map[other_func] = other_Z - offset
            else:
                subs_map[other_func] = other_Z
        Z_ode = Z_ode.subs(subs_map)

        Z_ode = sp.simplify(Z_ode)

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

        new_aux_odes[Z] = sp.simplify(Z_ode)

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
        new_ode = new_ode.subs(subs_map)

        # Also substitute sqrt(sum) expressions
        for sqrt_expr, Z in sqrt_to_aux.items():
            new_ode = new_ode.subs(sqrt_expr, Z)

        new_odes[var] = sp.simplify(new_ode)

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
        func_at_0 = func
        # CRITICAL: Substitute time=0 FIRST for time-only functions
        time_sym = sp.Symbol("time")
        func_at_0 = func_at_0.subs(time_sym, 0)
        # Then substitute state variables
        for var in sym.vars:
            if var in func_at_0.free_symbols:
                func_at_0 = func_at_0.subs(var, sym.initials.get(var, 1.0))
        # Then substitute parameters - use actual symbols from expression
        for param_sym in func_at_0.free_symbols:
            param_name = param_sym.name
            if param_name in sym.params:
                func_at_0 = func_at_0.subs(param_sym, sym.params[param_name])
        # Z(0) = f(X(0)) + offset
        offset = func_to_offset.get(func, 0.0)  # Use .get() to handle missing keys
        try:
            Z_init = float(func_at_0) + offset
        except Exception:
            Z_init = 1.0 + offset  # Fallback if evaluation fails
        new_initials[Z] = Z_init

    # Compute initial conditions for sqrt auxiliaries
    for sqrt_expr, Z in sqrt_to_aux.items():
        # Evaluate sqrt at t=0
        sqrt_at_0 = sqrt_expr
        # First substitute time=0
        time_sym = sp.Symbol("time")
        sqrt_at_0 = sqrt_at_0.subs(time_sym, 0)
        # Then substitute state variables
        for var in sym.vars:
            if var in sqrt_at_0.free_symbols:
                sqrt_at_0 = sqrt_at_0.subs(var, sym.initials.get(var, 1.0))
        # Then substitute parameters
        for param_sym in sqrt_at_0.free_symbols:
            param_name = param_sym.name
            if param_name in sym.params:
                sqrt_at_0 = sqrt_at_0.subs(param_sym, sym.params[param_name])
        try:
            Z_init = float(sqrt_at_0)
        except Exception:
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
            compartments=sym.compartments,  # Propagate compartments
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

    # Return augmented system and auxiliary definitions
    return (
        SymSystem(
            vars=new_vars,
            params=sym.params,
            odes=combined_odes,
            initials=new_initials,
            initial_exprs=sym.initial_exprs,  # Propagate symbolic IC expressions
            assignment_rules=assignment_rules,  # Time-only auxiliaries as assignment rules
            compartments=sym.compartments,  # Propagate compartments
        ),
        aux_to_func_with_offset,  # Dictionary mapping Z_i -> f(X) + offset
    )


def _exponents_match(exps1: dict[sp.Symbol, float], exps2: dict[sp.Symbol, float]) -> bool:
    """Check if two exponent patterns match (within tolerance).

    Handles both numeric and symbolic exponents.
    """
    all_vars = set(exps1.keys()) | set(exps2.keys())
    for var in all_vars:
        e1 = exps1.get(var, 0.0)
        e2 = exps2.get(var, 0.0)

        # Handle symbolic exponents
        if isinstance(e1, sp.Expr) or isinstance(e2, sp.Expr):
            # Convert to sympy if needed
            e1_sym = sp.sympify(e1)
            e2_sym = sp.sympify(e2)
            diff = sp.simplify(e1_sym - e2_sym)

            # Check if the difference is zero (either exactly or numerically)
            if diff == 0:
                continue
            if diff.is_number:
                try:
                    if abs(float(diff)) > 1e-10:
                        return False
                except (TypeError, ValueError):
                    return False  # Can't compare, assume different
            else:
                # Symbolic difference - not the same unless zero
                return False
        else:
            # Both numeric
            try:
                if abs(float(e1) - float(e2)) > 1e-10:
                    return False
            except (TypeError, ValueError):
                return False
    return True


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
        except Exception:
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

    # Handle constant terms: skip dummy variable in simplified mode for cleaner output
    # In simplified mode, constant terms like "t' = 1" are acceptable and valid
    # In canonical mode, we could add dummy if strict power-law form is required
    if mode == "canonical":
        # Canonical mode: use dummy variable for strict S-system form
        sym, dummy_aux_defs = add_dummy_for_constants(sym)
        all_auxiliary_defs.update(dummy_aux_defs)
    # else: simplified mode - leave constants as-is

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

    # Propagate simulation metadata from input SymSystem
    result.sim_t_start = sym.sim_t_start
    result.sim_t_end = sym.sim_t_end
    result.sim_n_steps = sym.sim_n_steps
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
        # (No lifted variables in this path)
        if False:  # This branch never executes - kept for structural consistency
            # Lifted vars already have power-law ODEs from the lifting process
            # No need for pool construction - just convert to growth/decay form
            rhs = sp.simplify(sym.odes[Xi])
            rhs = _numeric_param_subs(rhs, sym.params)
            terms = expand_to_terms(rhs)

            # Separate positive and negative terms for growth/decay
            growth_terms = []
            decay_terms = []
            for t in terms:
                if t == 0:
                    continue
                coeff, exps = term_to_coeff_exps(t)
                if coeff >= 0:
                    growth_terms.append((coeff, exps))
                else:
                    decay_terms.append((abs(coeff), exps))

            # Combine growth terms (sum coefficients, average exponents)
            if growth_terms:
                g_coeff = sum(c for c, _ in growth_terms)
                g_exps = {}
                for c, e in growth_terms:
                    weight = c / g_coeff
                    for s, exp in e.items():
                        g_exps[s] = g_exps.get(s, 0.0) + weight * exp
            else:
                g_coeff, g_exps = 0.0, {}

            # Combine decay terms (sum coefficients, average exponents)
            if decay_terms:
                d_coeff = sum(c for c, _ in decay_terms)
                d_exps = {}
                for c, e in decay_terms:
                    weight = c / d_coeff
                    for s, exp in e.items():
                        d_exps[s] = d_exps.get(s, 0.0) + weight * exp
            else:
                d_coeff, d_exps = 0.0, {}

            # Add the lifted variable itself with its equation
            new_variables.append(Xi)
            new_equations.append(SSysEquation(Xi, (g_coeff, g_exps), (d_coeff, d_exps)))

            # Do NOT add to factor_map - lifted vars are not reconstructed
            continue

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

    # 5) Detect which variables have negative exponents
    vars_with_neg_exp = set()
    for eq in new_equations:
        # Check growth exponents
        for var, exp in eq.growth[1].items():
            if isinstance(exp, (int, float)) and exp < 0:
                vars_with_neg_exp.add(var)
            elif isinstance(exp, sp.Expr) and exp.is_number and float(exp) < 0:
                vars_with_neg_exp.add(var)
        # Check decay exponents
        for var, exp in eq.decay[1].items():
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


def product_to_antimony(coeff, exps: dict[sp.Symbol, float]) -> str:
    """
    Format coefficient and exponents as Antimony expression string.
    coeff can be either float or sp.Expr (symbolic).
    Exponents can also be symbolic expressions.
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
                parts.append(f"({coeff_simplified})")
            else:
                # Break symbolic coefficient into factors for clean formatting
                coeff_factors = _format_symbolic_coeff(coeff_simplified)
                if coeff_factors:
                    parts.extend(coeff_factors)
    else:
        # Numeric coefficient
        if coeff == 0.0:
            return "0"
        elif coeff != 1.0 or has_dummy_const_zero:
            # Always show coefficient if we have dummy_const^0 (even if coeff=1)
            parts.append(f"{coeff:g}")

    # Add power-law terms
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        # Special case: dummy_const with exponent 0 should always be shown
        # This represents constant terms as C * dummy_const^0
        is_dummy_const = s.name == "dummy_const"

        # Handle both numeric and symbolic exponents
        if isinstance(e, sp.Expr):
            e_simplified = sp.simplify(e)
            if e_simplified == 0 and not is_dummy_const:
                continue
            # Check if it's effectively 1.0 (including sympy Float)
            elif e_simplified == 1 or (
                e_simplified.is_Number and abs(float(e_simplified) - 1.0) < 1e-10
            ):
                parts.append(s.name)
            else:
                # Format as integer if it's an integer value
                if e_simplified.is_Number:
                    e_val = float(e_simplified)
                    if abs(e_val - round(e_val)) < 1e-10:
                        parts.append(f"{s.name}^{int(round(e_val))}")
                    else:
                        parts.append(f"{s.name}^{e_simplified}")
                else:
                    # Symbolic exponent - add parentheses if it's a sum/difference
                    if e_simplified.is_Add:
                        parts.append(f"{s.name}^({e_simplified})")
                    else:
                        parts.append(f"{s.name}^{e_simplified}")
        else:
            # Numeric exponent
            # Always show dummy_const even with exponent 0
            if abs(e) < 1e-14 and not is_dummy_const:
                continue
            s_name = s.name if hasattr(s, "name") else str(s)
            if abs(e - 1.0) < 1e-14:
                parts.append(f"{s_name}")
            else:
                parts.append(f"{s_name}^{e:g}")

    # Special case: if parts is empty, return the coefficient string
    # This handles pure constants like "1" (coeff=1, exps={})
    if not parts:
        # If we have a non-zero coefficient but no variables, return coefficient as string
        if isinstance(coeff, sp.Expr):
            coeff_simplified = sp.simplify(coeff)
            if coeff_simplified == 0:
                return "0"
            else:
                return str(coeff_simplified)
        else:
            # Numeric coefficient
            if coeff == 0.0:
                return "0"
            else:
                return f"{coeff:g}"
    return "*".join(parts)


def _format_symbolic_coeff(coeff: sp.Expr) -> list[str]:
    """
    Format a symbolic coefficient cleanly by extracting factors.
    Returns list of string parts to be joined with '*'.
    """
    parts: list[str] = []

    # If it's a multiplication, extract factors
    if coeff.is_Mul:
        for factor in coeff.args:
            part = _format_factor(factor)
            if part:
                parts.append(part)
    else:
        # Single factor
        part = _format_factor(coeff)
        if part:
            parts.append(part)

    return parts


def _format_factor(factor: sp.Expr) -> str:
    """Format a single factor from a coefficient."""
    # Pure number
    if factor.is_Number:
        val = float(factor)
        if val == int(val):
            return str(int(val))
        else:
            return f"{val:g}"

    # Symbol (parameter)
    if isinstance(factor, sp.Symbol):
        return str(factor.name)

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
            base_str = base.name
        elif base.is_Number:
            base_str = f"{float(base):g}"
        else:
            base_str = f"({_format_factor(base)})"

        # Format exponent
        if exp.is_Number:
            exp_val = float(exp)
            if exp_val == int(exp_val):
                exp_str = str(int(exp_val))
            else:
                exp_str = f"{exp_val:g}"
        else:
            exp_str = str(exp)

        return f"{base_str}^{exp_str}"

    # Sum expression - MUST be wrapped in parentheses to preserve operator precedence
    # This is critical for expressions like H*(A + B + C)*X^-1 where the sum is a factor
    if factor.is_Add:
        return f"({factor})"

    # Anything else - fallback to string representation
    return str(factor)


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
    lines: list[str] = []
    lines.append(f"model {model_name}()")
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
            lines.append(f"compartment {comp_name} = {comp_size:g};")
        # Use first compartment name for species declarations
        default_compartment = next(iter(result.compartments.keys()))
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
            lines.append(f"species {v.name} in {default_compartment};")
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
            defn_str = _sympy_to_antimony_syntax(str(defn))
            lines.append(f"{aux} := {defn_str};")
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
            lines.append(f"{param_name} = {param_val:g};")
        lines.append("")

    # --- Assignment rules from original model (time-dependent quantities) ---
    if result.assignment_rules:
        lines.append("// Assignment rules (from original model)")
        for var_name in sorted(result.assignment_rules.keys()):
            expr = result.assignment_rules[var_name]
            lines.append(f"{var_name} := {expr};")
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
        lines.append(f"{s.name} = {float(v):g};")

    lines.append("")

    # GMA ODEs with multiple terms per side
    # CRITICAL: Skip ODEs for lifted auxiliaries (they're assignment rules now)
    for eq in result.gma_equations:
        # Skip lifted auxiliaries - they're assignment rules, not species with ODEs
        if eq.var.name in lifted_aux_names:
            continue

        # Format production terms
        if eq.production:
            prod_strs = [product_to_antimony(c, e) for c, e in eq.production]
            production = " + ".join(prod_strs)
        else:
            production = "0"

        # Format degradation terms
        if eq.degradation:
            deg_strs = [product_to_antimony(c, e) for c, e in eq.degradation]
            degradation = " + ".join(deg_strs)
        else:
            degradation = "0"

        # Write ODE - output directly for GMA format (no transformation)
        # Pure constants like T' = 1 should be output as-is
        if degradation == "0":
            lines.append(f"{eq.var.name}' = {production};")
        else:
            lines.append(f"{eq.var.name}' = {production} - ({degradation});")

    # Add @SIM metadata if available
    sim_lines = _format_sim_metadata_lines(result)
    if sim_lines:
        lines.append("")
        lines.extend(sim_lines)

    lines.append("end")
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

    # Check if recasting failed
    if result.status == RecastStatus.FAILED:
        return _failed_to_antimony(result, model_name)

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
            lines.append(f"// {s.name} = {float(v):g}")
        lines.append("")

    lines.append("// No recast equations generated.")
    lines.append("")
    lines.append("end")
    return "\n".join(lines)


def _ssystem_to_antimony_simplified(result, model_name: str) -> str:
    """Format S-system in simplified mode with enhanced documentation and assignment rules."""
    lines: list[str] = []
    lines.append(f"model {model_name}()")
    lines.append("")

    # --- Compartment declaration (for SBML compatibility) ---
    # Use original compartment name if available, otherwise default to "cell"
    if result.compartments:
        for comp_name, comp_size in result.compartments.items():
            lines.append(f"compartment {comp_name} = {comp_size:g};")
        # Use first compartment name for species declarations
        default_compartment = next(iter(result.compartments.keys()))
    else:
        lines.append("compartment cell = 1;")
        default_compartment = "cell"
    lines.append("")

    # --- Species declarations ---
    # All variables with ODEs must be declared as species in the compartment
    all_state_vars = sorted(result.variables, key=lambda s: s.name)
    if all_state_vars:
        for v in all_state_vars:
            lines.append(f"species {v.name} in {default_compartment};")
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
            lines.append(f"{param_name} = {param_val:g};")
        lines.append("")

    # --- Assignment rules (time-dependent quantities) ---
    # These are variables that depend only on time, not state variables
    if result.assignment_rules:
        lines.append("// ========================================================================")
        lines.append("// ASSIGNMENT RULES (time-dependent quantities)")
        lines.append("// ========================================================================")
        for var_name in sorted(result.assignment_rules.keys()):
            expr = result.assignment_rules[var_name]
            lines.append(f"{var_name} := {expr};")
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

    for s, v in sorted(result.initials.items(), key=_init_sort_key):
        # Skip tuple keys (compartment info, const params) - only process Symbol keys
        if not hasattr(s, "name"):
            continue
        # Skip variables with assignment rules (their value comes from the rule)
        if s.name in assignment_rule_vars:
            continue
        # Output ICs for ALL state variables (original + auxiliary), NOT parameters
        # Use name-based matching to handle symbol object mismatch
        if s.name in state_var_names and s.name not in result.params:
            # Check if we have a symbolic expression for this IC
            if s in result.initial_exprs:
                # Use symbolic expression
                lines.append(f"{s.name} = {result.initial_exprs[s]};")
            else:
                # Use numeric value
                lines.append(f"{s.name} = {float(v):g};")
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
            if len(aux) > 1:
                # Multiple auxiliaries - product form
                rhs = " * ".join(a.name for a in aux)
                lines.append(f"{orig.name} := {rhs};")
            else:
                # Single auxiliary (but not identity)
                lines.append(f"{orig.name} := {aux[0].name};")
        lines.append("")

    # --- S-system dynamics ---
    lines.append("// ========================================================================")
    lines.append("// S-SYSTEM DYNAMICS")
    lines.append("// ========================================================================")
    for eq in result.equations:
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
                    g_str = f"{float(g_coeff_simplified):g}"
                else:
                    # Symbolic constant (contains parameters)
                    g_str = str(g_coeff_simplified)
            else:
                g_str = f"{float(g_coeff):g}"
            h = product_to_antimony(eq.decay[0], h_exps)
            lines.append(f"{eq.var.name}' = {g_str} - {h};")
        elif h_is_const and not g_is_const:
            # Pure constant decay: X' = g(vars) - C
            h_coeff = eq.decay[0]
            # Format constant coefficient (numeric or symbolic)
            if isinstance(h_coeff, sp.Expr):
                h_coeff_simplified = sp.simplify(h_coeff)
                if h_coeff_simplified.is_Number:
                    h_str = f"{float(h_coeff_simplified):g}"
                else:
                    # Symbolic constant (contains parameters)
                    h_str = str(h_coeff_simplified)
            else:
                h_str = f"{float(h_coeff):g}"
            g = product_to_antimony(eq.growth[0], g_exps)
            lines.append(f"{eq.var.name}' = {g} - {h_str};")
        elif g_is_const and h_is_const:
            # Both constants: X' = C1 - C2
            g_coeff = eq.growth[0]
            h_coeff = eq.decay[0]
            # Format both coefficients (numeric or symbolic)
            if isinstance(g_coeff, sp.Expr):
                g_coeff_simplified = sp.simplify(g_coeff)
                if g_coeff_simplified.is_Number:
                    g_str = f"{float(g_coeff_simplified):g}"
                else:
                    g_str = str(g_coeff_simplified)
            else:
                g_str = f"{float(g_coeff):g}"
            if isinstance(h_coeff, sp.Expr):
                h_coeff_simplified = sp.simplify(h_coeff)
                if h_coeff_simplified.is_Number:
                    h_str = f"{float(h_coeff_simplified):g}"
                else:
                    h_str = str(h_coeff_simplified)
            else:
                h_str = f"{float(h_coeff):g}"
            lines.append(f"{eq.var.name}' = {g_str} - {h_str};")
        else:
            # Normal monomial form
            g = product_to_antimony(eq.growth[0], g_exps)
            h = product_to_antimony(eq.decay[0], h_exps)
            lines.append(f"{eq.var.name}' = {g} - {h};")

    # Add @SIM metadata if available
    sim_lines = _format_sim_metadata_lines(result)
    if sim_lines:
        lines.append("")
        lines.extend(sim_lines)

    lines.append("end")
    # Convert ** to ^ for valid Antimony syntax
    return _sympy_to_antimony_syntax("\n".join(lines))


def _ssystem_to_antimony_canonical(result, model_name: str) -> str:
    """
    Format S-system in canonical mode with enhanced annotations.

    Features:
    - Species declarations for all auxiliary variables
    - Observable variables showing original-to-auxiliary mappings
    - Detailed explanatory comments
    - Clean equation formatting
    """
    lines: list[str] = []

    # Model declaration with _SSystem suffix
    if not model_name.endswith("_SSystem") and not model_name.endswith("_SSystem_exact"):
        model_name = f"{model_name}_SSystem_exact"
    lines.append(f"model {model_name}()")
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
        species_names = ", ".join([v.name for v in aux_vars])
        lines.append(f"  species {species_names};")
        lines.append("")

    # Parameter declarations (from result.params)
    if result.params:
        lines.append("  // Parameters")
        for param_name in sorted(result.params.keys()):
            param_val = result.params[param_name]
            lines.append(f"  {param_name} = {param_val:g};")
        lines.append("")

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
        lines.append("  // Slack variable (keeps both coefficients >0)")
        lines.append("  epsilon = 1.0;")
        lines.append("")

    # Canonical S-system dynamics with clean formatting and slack variables
    lines.append("  // Canonical S-system dynamics (two monomials per ODE)")
    for eq in result.equations:
        g_exps = _expand_exps_through_factors(eq.growth[1], result.factor_map)
        h_exps = _expand_exps_through_factors(eq.decay[1], result.factor_map)
        g_coeff = eq.growth[0]
        h_coeff = eq.decay[0]

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
                lines.append(f"  {eq.var.name}' = epsilon - ({combined});")
                continue
            g_str = product_to_antimony(sp.Symbol("epsilon"), h_exps)
            # Combine epsilon + h_coeff symbolically
            if isinstance(h_coeff, sp.Expr):
                combined_coeff = sp.Symbol("epsilon") + h_coeff
            else:
                combined_coeff = sp.Symbol("epsilon") + sp.Float(h_coeff)
            h_str = product_to_antimony(combined_coeff, h_exps)
            lines.append(f"  {eq.var.name}' = {g_str} - {h_str};")
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
                lines.append(f"  {eq.var.name}' = ({combined}) - epsilon;")
                continue
            if isinstance(g_coeff, sp.Expr):
                combined_coeff = g_coeff + sp.Symbol("epsilon")
            else:
                combined_coeff = sp.Float(g_coeff) + sp.Symbol("epsilon")
            g_str = product_to_antimony(combined_coeff, g_exps)
            h_str = product_to_antimony(sp.Symbol("epsilon"), g_exps)
            lines.append(f"  {eq.var.name}' = {g_str} - {h_str};")
        else:
            # Both terms present (or both zero) - use as-is
            g = product_to_antimony(g_coeff, g_exps)
            h = product_to_antimony(h_coeff, h_exps)
            lines.append(f"  {eq.var.name}' = {g} - {h};")
    lines.append("")

    # Observable variables for original variables (if factorized)
    if orig_vars and any(len(result.factor_map[orig]) > 1 for orig in orig_vars):
        lines.append("  // Observable: original variable(s)")
        for orig in orig_vars:
            aux_list = result.factor_map[orig]
            if len(aux_list) > 1:  # Only show non-trivial mappings
                obs_name = f"{orig.name}_obs"
                rhs = " * ".join([a.name for a in aux_list])
                lines.append(f"  var {obs_name};")
                lines.append(f"  {obs_name} := {rhs};")
        lines.append("")

    # Initial conditions
    lines.append("  // Initial conditions")
    for v in aux_vars:
        if v in result.initials:
            # Check if we have a symbolic expression for this IC
            if v in result.initial_exprs:
                # Use symbolic expression
                lines.append(f"  {v.name} = {result.initial_exprs[v]};")
            else:
                # Use numeric value
                val = result.initials[v]
                lines.append(f"  {v.name} = {float(val):g};")

    # Add @SIM metadata if available
    sim_lines = _format_sim_metadata_lines(result)
    if sim_lines:
        lines.append("")
        lines.extend(["  " + line for line in sim_lines])  # Indent canonical mode

    lines.append("end")
    # Convert ** to ^ for valid Antimony syntax
    return _sympy_to_antimony_syntax("\n".join(lines))


def latex_odes(sym: "SymSystem") -> str:
    lines = []
    for v in sorted(sym.odes.keys(), key=lambda s: s.name):
        rhs = sp.simplify(sym.odes[v])
        lines.append(rf"\dot{{{sp.latex(v)}}} = {sp.latex(rhs)}")
    return r"\\begin{aligned}" + r"\\\\\n".join(lines) + r"\\end{aligned}"


def _latex_power_law(coeff, exps: dict[sp.Symbol, float]) -> str:
    """
    Format a power-law term as clean LaTeX.
    - Skip coefficients of 1
    - Skip exponents of 1
    - Display integers as integers (not floats)
    """
    parts = []

    # Handle coefficient
    if isinstance(coeff, sp.Expr):
        coeff_simplified = sp.simplify(coeff)
        if coeff_simplified == 0:
            return "0"
        elif coeff_simplified != 1:
            # Use sympy's latex for symbolic coefficients
            parts.append(sp.latex(coeff_simplified))
    else:
        # Numeric coefficient
        if coeff == 0:
            return "0"
        elif coeff != 1:
            # Display integers as integers
            if isinstance(coeff, int) or (isinstance(coeff, float) and coeff == int(coeff)):
                parts.append(str(int(coeff)))
            else:
                parts.append(f"{coeff:g}")

    # Handle power-law terms
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        if abs(e) < 1e-14:
            continue

        var_latex = sp.latex(s)

        # Skip exponent if it's 1
        if abs(e - 1.0) < 1e-14:
            parts.append(var_latex)
        else:
            # Display integer exponents as integers
            if isinstance(e, int) or (isinstance(e, float) and e == int(e)):
                parts.append(f"{var_latex}^{{{int(e)}}}")
            else:
                parts.append(f"{var_latex}^{{{e:g}}}")

    if not parts:
        return "0"

    # Join with space (LaTeX handles multiplication)
    return " ".join(parts)


def latex_ssys(result: "RecastResult") -> str:
    """
    Generate clean LaTeX representation of S-system equations.
    - Skip coefficients of 1
    - Skip exponents of 1
    - Display integers as integers
    - Apply slack variable transformation for canonical form (matches Antimony output)
    """
    lines = []
    for eq in result.equations:
        # Expand exponents through factor map
        g_exps = _expand_exps_through_factors(eq.growth[1], result.factor_map)
        h_exps = _expand_exps_through_factors(eq.decay[1], result.factor_map)
        g_coeff = eq.growth[0]
        h_coeff = eq.decay[0]

        # Check if growth or decay is zero (need slack variable transformation)
        g_is_zero = (isinstance(g_coeff, (int, float)) and g_coeff == 0) or (
            isinstance(g_coeff, sp.Expr) and g_coeff == sp.Integer(0)
        )
        h_is_zero = (isinstance(h_coeff, (int, float)) and h_coeff == 0) or (
            isinstance(h_coeff, sp.Expr) and h_coeff == sp.Integer(0)
        )

        if g_is_zero and not h_is_zero:
            # Pure decay: X' = 0 - h  =>  X' = epsilon*monomial - (epsilon + h)*monomial
            # Special case: pure constant decay (empty exponents)
            if not h_exps:
                if isinstance(h_coeff, sp.Expr):
                    combined = sp.Symbol("epsilon") + h_coeff
                else:
                    combined = sp.Symbol("epsilon") + sp.Float(h_coeff)
                g_latex = r"\epsilon"
                h_latex = sp.latex(combined)
            else:
                g_latex = _latex_power_law(sp.Symbol("epsilon"), h_exps)
                if isinstance(h_coeff, sp.Expr):
                    combined_coeff = sp.Symbol("epsilon") + h_coeff
                else:
                    combined_coeff = sp.Symbol("epsilon") + sp.Float(h_coeff)
                h_latex = _latex_power_law(combined_coeff, h_exps)
        elif h_is_zero and not g_is_zero:
            # Pure growth: X' = g - 0  =>  X' = (g + epsilon)*monomial - epsilon*monomial
            # Special case: pure constant growth (empty exponents)
            if not g_exps:
                if isinstance(g_coeff, sp.Expr):
                    combined = g_coeff + sp.Symbol("epsilon")
                else:
                    combined = sp.Float(g_coeff) + sp.Symbol("epsilon")
                g_latex = sp.latex(combined)
                h_latex = r"\epsilon"
            else:
                if isinstance(g_coeff, sp.Expr):
                    combined_coeff = g_coeff + sp.Symbol("epsilon")
                else:
                    combined_coeff = sp.Float(g_coeff) + sp.Symbol("epsilon")
                g_latex = _latex_power_law(combined_coeff, g_exps)
                h_latex = _latex_power_law(sp.Symbol("epsilon"), g_exps)
        else:
            # Both terms present (or both zero) - use as-is
            g_latex = _latex_power_law(g_coeff, g_exps)
            h_latex = _latex_power_law(h_coeff, h_exps)

        # Build equation
        var_latex = sp.latex(eq.var)
        lines.append(rf"\dot{{{var_latex}}} &= {g_latex} - {h_latex}")

    return r"\begin{aligned}" + "\\\\\n".join(lines) + r"\end{aligned}"
