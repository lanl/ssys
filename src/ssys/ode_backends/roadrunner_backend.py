"""
libRoadRunner ODE solver backend.
"""

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from ..recaster import ModelIR, expand_antimony_function_templates

_GAMMA_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_GAMMA_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _UnresolvedGammaArgument(ValueError):
    """Raised when a gamma argument is symbolic and must remain in the model."""


def simulate_with_roadrunner(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Simulate using libRoadRunner (CVODE integrator).

    Args:
        model_ir: Model intermediate representation
        t0: Start time
        t_end: End time
        n_points: Number of time points
        y0_override: Override initial conditions
        options: Solver options:
            - integrator: 'cvode' (default) or 'gillespie'
            - absolute_tolerance: float (default 1e-9)
            - relative_tolerance: float (default 1e-6)
            - maximum_num_steps: int (default 20000)
            - log_solver_details: bool (default False)

    Returns:
        Dictionary with simulation results
    """
    try:
        import roadrunner as rr
    except ImportError:
        raise ImportError("libRoadRunner not installed. Install with: pip install libroadrunner")

    try:
        import antimony
    except ImportError:
        raise ImportError(
            "Antimony not installed. Install with: pip install antimony (or pip install tellurium)"
        )

    if options is None:
        options = {}

    try:
        # Get or reconstruct Antimony text
        antimony_text = _get_antimony_text(model_ir)

        # Convert Antimony → SBML using antimony library
        # RoadRunner only understands SBML, not Antimony directly
        antimony.clearPreviousLoads()
        rc = antimony.loadAntimonyString(antimony_text)
        if rc < 0:
            raise RuntimeError(f"Antimony parse error: {antimony.getLastError()}")

        model_name = antimony.getMainModuleName()
        if not model_name:
            raise RuntimeError("Could not get Antimony module name")

        sbml_str = antimony.getSBMLString(model_name)
        if not sbml_str:
            raise RuntimeError(f"Antimony→SBML conversion failed: {antimony.getLastError()}")

        # Build RoadRunner model from SBML
        r = rr.RoadRunner(sbml_str)

        # Configure integrator
        integrator_name = options.get("integrator", "cvode")
        r.setIntegrator(integrator_name)

        if integrator_name == "cvode":
            r.integrator.absolute_tolerance = options.get("absolute_tolerance", 1e-9)
            r.integrator.relative_tolerance = options.get("relative_tolerance", 1e-6)
            r.integrator.maximum_num_steps = options.get("maximum_num_steps", 20000)

        # Set initial conditions
        initial_condition_warnings = _set_initial_conditions(r, model_ir, y0_override)

        # Run simulation
        result = r.simulate(t0, t_end, n_points)

        # Extract results
        # result is array with shape (n_points, 1 + n_species)
        # Column 0 is time, rest are species
        t = result[:, 0]
        y = result[:, 1:]

        # Get species names from result column headers
        # CRITICAL: This ensures state_names match the actual column order
        # in the simulation result (getFloatingSpeciesIds() may not match!)
        col_names = list(result.colnames)  # e.g., ['time', 'S', 'I', 'R']
        state_names = col_names[1:]  # Skip 'time' column

        # Strip brackets if present (RoadRunner sometimes uses [species])
        state_names = [name.strip("[]") if name.startswith("[") else name for name in state_names]

        # Get integrator statistics
        stats = {}
        if integrator_name == "cvode":
            try:
                stats = {
                    "n_steps": r.integrator.getNumSteps(),
                    "n_failed_steps": r.integrator.getNumErrTestFails(),
                    "last_time": t[-1],
                }
            except (AttributeError, RuntimeError, TypeError):
                # Some RoadRunner versions may not support these
                stats = {"last_time": t[-1]}
        if initial_condition_warnings:
            stats["initial_condition_warnings"] = initial_condition_warnings

        if options.get("log_solver_details", False):
            print("RoadRunner simulation completed:")
            print(f"  Integrator: {integrator_name}")
            print(f"  Steps: {stats.get('n_steps', 'N/A')}")
            print(f"  Time range: [{t[0]}, {t[-1]}]")

        return {
            "t": t,
            "y": y,
            "state_names": list(state_names),
            "success": True,
            "message": "",
            "integrator_stats": stats,
        }

    except Exception as e:
        # Simulation failed - return structured error
        return {
            "t": np.array([]),
            "y": np.array([]),
            "state_names": [],
            "success": False,
            "message": f"RoadRunner simulation failed: {str(e)}",
            "integrator_stats": {},
        }


def _expand_parametric_functions(text: str) -> str:
    """
    Expand user-defined parametric functions in Antimony text.

    Handles function definitions like:
        M(x) := 1 + gamma * x^2;
    And expands their usages like:
        x1' = beta / M(x1);
    To:
        x1' = beta / (1 + gamma * x1^2);

    This is NOT standard Antimony syntax, but is commonly used in
    mathematical biology papers for readability.

    Delegates to the parser-level implementation so legacy parsing and
    RoadRunner preprocessing preserve the same function-substitution semantics.
    """
    return expand_antimony_function_templates(text)


def _get_antimony_text(model_ir: ModelIR) -> str:
    """
    Get Antimony text from ModelIR.

    Prefers cached text if available, otherwise reconstructs.
    Applies fixes for Antimony/RoadRunner compatibility:
    - Numeric model names: "model 24_name" → "model m_24_name"
    - Multi-line equations: join continuation lines
    - gamma() function: compute at Python level (Antimony only has incomplete gamma)
    - Parametric functions: expand inline (e.g., M(x1) → full expression)
    """
    import re

    # Check if antimony_text was cached during parsing
    if hasattr(model_ir, "antimony_text") and model_ir.antimony_text:
        text = model_ir.antimony_text

        # CRITICAL: Expand parametric functions FIRST, before other fixes
        text = _expand_parametric_functions(text)

        # Fix 1: Numeric model names
        text = re.sub(r"^(model\s+)(\d)", r"\1m_\2", text, flags=re.MULTILINE)

        # Fix 2: Multi-line equations - join continuation lines
        # Handles: X3' = term1   // comment
        #          + term2       // comment
        #          - term3;      // comment
        # CRITICAL: Strip inline comments before joining, otherwise comments
        # corrupt the equation (e.g., "term1 // comment - term2" loses "- term2")

        def strip_inline_comment(s):
            """Remove inline comment (// ...) from a line, preserving semicolon."""
            # Find // that's not inside quotes
            idx = s.find("//")
            if idx >= 0:
                # Check for semicolon after the //
                has_semi = ";" in s[idx:]
                code_part = s[:idx].rstrip()
                if has_semi and not code_part.endswith(";"):
                    code_part += ";"
                return code_part
            return s

        lines = text.split("\n")
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if this is an ODE line without semicolon
            line_stripped = strip_inline_comment(line.rstrip())
            if "'" in line and "=" in line and not line_stripped.endswith(";"):
                # Collect continuation lines
                combined = line_stripped
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    # Check if next line starts with operator
                    if next_line and next_line[0] in ["+", "-"]:
                        next_stripped = strip_inline_comment(next_line)
                        combined += " " + next_stripped
                        i += 1
                        if next_stripped.rstrip().endswith(";"):
                            break
                    else:
                        break
                fixed_lines.append(combined)
            else:
                fixed_lines.append(line)
                i += 1
        text = "\n".join(fixed_lines)

        # Fix 3: gamma() function - compute complete gamma for numeric arguments.
        # Antimony's gamma() is incomplete gamma (needs 2+ args), while published
        # model formulas often use the complete gamma Γ(x). Keep symbolic gamma
        # calls intact, but reject malformed or unsafe numeric expressions.
        text = _replace_complete_gamma_calls(text)

        # Fix 4: Exponentiation syntax - convert ** to ^
        # Python/SymPy uses ** for exponentiation, Antimony uses ^
        text = text.replace("**", "^")

        return text

    # Otherwise, reconstruct from IR
    return _reconstruct_antimony(model_ir)


def _replace_complete_gamma_calls(text: str) -> str:
    """Replace numeric complete-gamma calls without evaluating arbitrary Python."""
    result: list[str] = []
    i = 0
    while i < len(text):
        if not text.startswith("gamma(", i) or (i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")):
            result.append(text[i])
            i += 1
            continue

        arg, close_idx, has_top_level_comma = _read_gamma_argument(text, i + len("gamma("))
        original = text[i : close_idx + 1]
        if has_top_level_comma:
            result.append(original)
            i = close_idx + 1
            continue

        try:
            arg_value = _evaluate_complete_gamma_argument(arg)
        except _UnresolvedGammaArgument:
            result.append(original)
        except ValueError as exc:
            raise ValueError(f"Malformed complete gamma expression {original!r}: {exc}")
        else:
            result.append(str(math.gamma(arg_value)))
        i = close_idx + 1

    return "".join(result)


def _read_gamma_argument(text: str, start_idx: int) -> tuple[str, int, bool]:
    """Read the parenthesized argument of a gamma call."""
    depth = 0
    has_top_level_comma = False
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return text[start_idx:idx], idx, has_top_level_comma
            depth -= 1
        elif char == "," and depth == 0:
            has_top_level_comma = True
    raise ValueError("missing closing ')' for gamma call")


def _evaluate_complete_gamma_argument(expr_str: str) -> float:
    """Evaluate a numeric gamma argument using a small arithmetic allowlist."""
    expr = expr_str.strip().replace("^", "**")
    if not expr:
        raise ValueError("empty argument")
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(str(exc)) from exc
    return _eval_gamma_ast(parsed.body)


def _eval_gamma_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError(f"unsupported literal {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        raise _UnresolvedGammaArgument(f"symbolic name {node.id!r}")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _GAMMA_UNARYOPS:
        return float(_GAMMA_UNARYOPS[type(node.op)](_eval_gamma_ast(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in _GAMMA_BINOPS:
        lhs = _eval_gamma_ast(node.left)
        rhs = _eval_gamma_ast(node.right)
        try:
            return float(_GAMMA_BINOPS[type(node.op)](lhs, rhs))
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
            raise ValueError("only sqrt() is allowed in complete gamma arguments")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("sqrt() accepts exactly one positional argument")
        value = _eval_gamma_ast(node.args[0])
        if value < 0:
            raise ValueError("sqrt() argument must be non-negative")
        return math.sqrt(value)

    raise ValueError(f"unsupported syntax {type(node).__name__}")


def _reconstruct_antimony(model_ir: ModelIR) -> str:
    """
    Reconstruct Antimony text from ModelIR.

    Note: This is a basic reconstruction. For production use,
    consider caching the original Antimony text in ModelIR.
    """
    lines = []
    lines.append("model recast_model()")
    lines.append("")

    # Species
    if model_ir.species:
        lines.append("  // Species")
        for species in model_ir.species:
            lines.append(f"  species {species};")
        lines.append("")

    # Parameters
    if model_ir.params:
        lines.append("  // Parameters")
        for param, value in model_ir.params.items():
            lines.append(f"  {param} = {value};")
        lines.append("")

    # Reactions
    if model_ir.reactions:
        lines.append("  // Reactions")
        for rxn in model_ir.reactions:
            # lhs and rhs are lists of (coeff, species_name) tuples
            lhs_str = " + ".join(name for _coeff, name in rxn.lhs) if rxn.lhs else ""
            rhs_str = " + ".join(name for _coeff, name in rxn.rhs) if rxn.rhs else ""
            arrow = "->" if not lhs_str else "-> " if not rhs_str else " -> "
            rxn_name = rxn.name if rxn.name else ""
            rxn_prefix = f"{rxn_name}: " if rxn_name else ""
            lines.append(f"  {rxn_prefix}{lhs_str}{arrow}{rhs_str}; {rxn.rate_expr};")
        lines.append("")

    # Explicit rate rules (ODEs)
    if model_ir.explicit_rates:
        lines.append("  // ODEs")
        for species, rate_expr in model_ir.explicit_rates.items():
            lines.append(f"  {species}' = {rate_expr};")
        lines.append("")

    lines.append("end")
    return "\n".join(lines)


def _set_initial_conditions(
    r, model_ir: ModelIR, y0_override: dict[str, float] | None
) -> list[dict[str, str]]:
    """
    Set initial conditions in RoadRunner model.

    Only sets values for actual floating species (state variables).
    Parameters and compartments are already embedded in the SBML model
    during Antimony→SBML conversion.

    Args:
        r: RoadRunner instance
        model_ir: Model IR with initial conditions
        y0_override: Optional overrides
    """
    # Reset to model-defined initial conditions
    r.resetToOrigin()

    # Get list of actual floating species from RoadRunner model
    warnings: list[dict[str, str]] = []

    try:
        floating_species = set(r.getFloatingSpeciesIds())
    except (AttributeError, RuntimeError, TypeError) as exc:
        # Fallback: empty set means we'll try all species
        warnings.append({"stage": "floating_species", "message": str(exc)})
        floating_species = None

    # Apply model_ir initials (only for floating species)
    if hasattr(model_ir, "initials") and model_ir.initials:
        for species, value in model_ir.initials.items():
            # Skip tuple keys (compartment metadata like (compartment, plasma))
            if isinstance(species, tuple):
                continue

            # Convert sympy Symbol to string if needed
            species_name = str(species) if hasattr(species, "name") else str(species)

            # Only set if it's a floating species (not a parameter)
            # If floating_species is None, try all non-tuple keys
            if floating_species is not None and species_name not in floating_species:
                continue

            try:
                # Use bracket notation for floating species
                r[f"[{species_name}]"] = value
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                warnings.append({
                    "stage": "model_initial",
                    "species": species_name,
                    "message": str(exc),
                })

    # Apply overrides (only for floating species)
    if y0_override:
        for species, value in y0_override.items():
            species_name = str(species) if hasattr(species, "name") else str(species)

            # Only set if it's a floating species
            if floating_species is not None and species_name not in floating_species:
                continue

            try:
                r[f"[{species_name}]"] = value
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                warnings.append({
                    "stage": "override_initial",
                    "species": species_name,
                    "message": str(exc),
                })

    return warnings
