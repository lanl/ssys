"""Mapping and auxiliary identity validation mixins."""

import keyword
import re
import time
from typing import Any

import sympy as sp

from ssys._recaster.names import _sanitize_antimony_name
from ssys._validator.common import (
    _canonicalize_expr_by_name,
    _cheap_zero_simplification,
    _simplify_identity_difference,
    _substitute_symbols_by_name,
)
from ssys._validator.report import EquivalenceTest, ValidationResult
from ssys._validator.state import ValidatorState

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMPY_FUNCTION_NAMES = frozenset({
    "Abs",
    "abs",
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
    "pi",
    "pow",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
})

_MODEL_MATH_LOCALS: dict[str, object] = {
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "pi": sp.pi,
}
AUXILIARY_SLOW_SIMPLIFY_MAX_OPS = 250
AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS = 64
AUXILIARY_CANDIDATE_GENERATION_MAX_OPS = 750
AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS = 60


def _is_external_clock_symbol_name(name: str) -> bool:
    """Return True for external clock symbols without treating state ``T`` as time."""
    return name.lower() == "time" or name == "t"


def _notify_progress(progress_callback, phase: str) -> None:
    """Best-effort progress hook for auxiliary identity subphases."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase)
    except Exception:
        return


def _has_risky_symbolic_power(expr: sp.Expr) -> bool:
    """Return True for power forms that commonly trigger expensive simplification."""
    for power in expr.atoms(sp.Pow):
        exponent = power.exp
        if exponent.free_symbols:
            return True
    return False


def _replace_expr_identifiers(expr: str, replacements: dict[str, str]) -> str:
    """Replace complete identifiers in expression text with safe aliases."""
    result = expr
    for old_name, new_name in replacements.items():
        if not old_name:
            continue
        pattern = r"(?<![A-Za-z_\d])" + re.escape(old_name) + r"(?![A-Za-z_\d])"
        result = re.sub(pattern, new_name, result)
    return result


def _keyword_identifier_replacements(
    identifiers: set[str],
    local_symbols: dict[str, sp.Symbol],
) -> dict[str, str]:
    """Return temporary aliases for model identifiers that are Python keywords."""
    replacements: dict[str, str] = {}
    used = set(identifiers) | set(local_symbols)
    for identifier in sorted(identifiers):
        if identifier not in local_symbols or not keyword.iskeyword(identifier):
            continue
        alias = f"__ssys_{identifier}"
        while alias in used:
            alias = f"{alias}_"
        replacements[identifier] = alias
        used.add(alias)
    return replacements


def _sympify_model_expression(
    expr: str,
    local_symbols: dict[str, sp.Symbol],
    *,
    positive_new_symbols: bool = False,
) -> sp.Expr:
    """Parse model expression text without confusing model ids for Python syntax."""
    identifiers = set(_IDENTIFIER_PATTERN.findall(expr))
    locals_for_sympify = dict(local_symbols)
    for name in identifiers:
        if name not in locals_for_sympify and name not in _SYMPY_FUNCTION_NAMES:
            locals_for_sympify[name] = (
                sp.Symbol(name, positive=True)
                if positive_new_symbols
                else sp.Symbol(name)
            )
    for name, helper in _MODEL_MATH_LOCALS.items():
        locals_for_sympify.setdefault(name, helper)

    replacements = _keyword_identifier_replacements(identifiers, locals_for_sympify)
    expr_for_sympify = (
        _replace_expr_identifiers(expr, replacements) if replacements else expr
    )
    for original_name, alias in replacements.items():
        locals_for_sympify[alias] = locals_for_sympify[original_name]

    return sp.sympify(expr_for_sympify, locals=locals_for_sympify)


def _passes_relative_numeric_zero_check(
    expr: sp.Expr,
    *,
    samples: int = 3,
    atol: float = 1e-8,
    rtol: float = 1e-12,
) -> bool:
    """Return True when deterministic samples show only roundoff-scale residuals."""
    symbols = sorted(expr.free_symbols, key=str)
    if not symbols:
        return False
    terms = sp.Add.make_args(expr)
    for sample_index in range(samples):
        offset = 0.25 * (sample_index + 1)
        substitutions = {
            symbol: (symbol_index + 1) * 0.13 + offset
            for symbol_index, symbol in enumerate(symbols)
        }
        try:
            value = complex(expr.evalf(30, subs=substitutions))
            scale = sum(
                abs(complex(term.evalf(30, subs=substitutions)))
                for term in terms
            )
        except (TypeError, ValueError, ArithmeticError, OverflowError):
            return False
        if not (
            abs(value.real) < float("inf")
            and abs(value.imag) < float("inf")
            and scale < float("inf")
        ):
            return False
        if abs(value) > atol and abs(value) / (scale or 1.0) > rtol:
            return False
    return True


class AuxiliaryIdentityComplexityError(RuntimeError):
    """Raised when an auxiliary identity needs bounded complexity classification."""

    def __init__(
        self,
        *,
        context: str,
        operation_count: int,
        free_symbol_count: int,
        residual: sp.Expr,
        candidate_index: int | None = None,
        active_subphase: str = "simplification",
        max_operation_count: int | None = None,
        max_free_symbol_count: int | None = None,
        risky_symbolic_power_guard: bool = False,
        elapsed_seconds: float | None = None,
    ) -> None:
        candidate_text = (
            f" candidate {candidate_index}" if candidate_index is not None else ""
        )
        super().__init__(
            f"{context}{candidate_text} residual exceeded bounded auxiliary "
            f"equivalence checks "
            f"({operation_count} operations, {free_symbol_count} free symbols)"
        )
        self.context = context
        self.operation_count = operation_count
        self.free_symbol_count = free_symbol_count
        self.residual = residual
        self.candidate_index = candidate_index
        self.active_subphase = active_subphase
        self.max_operation_count = max_operation_count
        self.max_free_symbol_count = max_free_symbol_count
        self.operation_threshold = max_operation_count
        self.free_symbol_threshold = max_free_symbol_count
        self.risky_symbolic_power_guard = risky_symbolic_power_guard
        self.elapsed_seconds = elapsed_seconds


def _auxiliary_complexity_metadata(
    error: AuxiliaryIdentityComplexityError,
) -> dict[str, object]:
    """Return stable metadata for bounded auxiliary-equivalence failures."""
    return {
        "context": error.context,
        "candidate_index": error.candidate_index,
        "active_subphase": error.active_subphase,
        "operation_count": error.operation_count,
        "free_symbol_count": error.free_symbol_count,
        "operation_threshold": error.operation_threshold,
        "free_symbol_threshold": error.free_symbol_threshold,
        "max_operation_count": error.max_operation_count,
        "max_free_symbol_count": error.max_free_symbol_count,
        "risky_symbolic_power_guard": error.risky_symbolic_power_guard,
        "elapsed_seconds": error.elapsed_seconds,
        "residual": str(error.residual),
    }


class MappingValidationMixin(ValidatorState):
    def _sanitized_identity_mapping(
        self, orig_var: sp.Symbol, recast_symbols_by_name: dict[str, sp.Symbol]
    ) -> sp.Symbol:
        """Return the recast symbol matching a sanitized identity name, if present."""
        var_name = str(orig_var)
        candidate_bases = [var_name]
        if var_name.endswith("_") and len(var_name) > 1:
            candidate_bases.append(var_name[:-1])

        seen: set[str] = set()
        for candidate_base in candidate_bases:
            sanitized = _sanitize_antimony_name(candidate_base)
            if sanitized == candidate_base or sanitized in seen:
                continue
            seen.add(sanitized)
            if sanitized in recast_symbols_by_name:
                return recast_symbols_by_name[sanitized]

        return orig_var

    def _extract_mapping_from_comments(self, recast_text: str) -> dict:
        """
        Extract variable mapping from recast file comments.

        Looks for patterns like:
        // VARIABLE MAPPING (new format)
        // ========
        // S = X_1
        // ========

        Or old format:
        // Mapping from original variables...
        // S = X_1
        // --- end mapping ---
        """

        mapping = {}
        in_mapping = False
        seen_first_separator = False

        for line in recast_text.split("\n"):
            line = line.strip()

            # Start of mapping section (both formats)
            if "VARIABLE MAPPING" in line or "Mapping from original variables" in line:
                in_mapping = True
                seen_first_separator = False
                continue

            # Handle separators in new format
            if in_mapping and "========" in line:
                if not seen_first_separator:
                    # This is the opening separator, skip it
                    seen_first_separator = True
                    continue
                else:
                    # This is the closing separator, end mapping section
                    break

            # End of mapping section (old format)
            if in_mapping and "--- end mapping ---" in line:
                break

            # Parse mapping line: // VAR = EXPR
            if in_mapping and line.startswith("//"):
                content = line[2:].strip()
                # Must have '=' but not start with '=' (skip separator lines)
                if "=" in content and not content.startswith("="):
                    parts = content.split("=", 1)
                    if len(parts) == 2:
                        orig_var = parts[0].strip()
                        expr_str = parts[1].strip()

                        # Convert to sympy symbols
                        orig_sym = sp.Symbol(orig_var)

                        # Parse expression (handles products like X_1*X_2)
                        try:
                            expr = _sympify_model_expression(
                                expr_str,
                                {},
                                positive_new_symbols=True,
                            )
                            mapping[orig_sym] = expr
                        except (TypeError, ValueError, sp.SympifyError):
                            # If parsing fails, treat as single symbol
                            mapping[orig_sym] = sp.Symbol(expr_str)

        return mapping

    def _merge_assignment_rules_as_auxiliaries(self):
        """
        Merge actual Antimony assignment rules into auxiliary_defs.

        When using lifted_mode='assignment', lifted auxiliaries are output as actual
        Antimony assignment rules (e.g., `Y_1 := a^2 + 1;`) rather than comment definitions.
        These rules are already parsed into recast_ir.assignment_rules, but we need to
        also add them to auxiliary_defs for the symbolic validation to substitute them.

        IMPORTANT: Only merge rules that look like lifted auxiliaries (Y_1, Y_2, etc.)
        and don't correspond to original model assignment rules.
        """
        # Get original assignment rule names to avoid treating them as auxiliaries
        orig_rule_names = set(self.orig_ir.assignment_rules.keys())

        # Get recast assignment rules
        recast_rules = self.recast_ir.assignment_rules

        # Get state variables (we don't want to treat state vars as auxiliaries)
        state_vars = {str(v) for v in self.recast_odes.keys()}

        # Pattern for lifted auxiliary names: Y_N or similar
        lifted_aux_pattern = re.compile(r"^[YZ]_\d+$")

        for rule_name, rule_expr_str in recast_rules.items():
            # Skip if this is an original assignment rule
            if rule_name in orig_rule_names:
                continue

            # Skip if this is a state variable
            if rule_name in state_vars:
                continue

            # Only process rules that look like lifted auxiliaries
            # OR any rule that's not in original (could be lifted denominator)
            if lifted_aux_pattern.match(rule_name) or rule_name not in orig_rule_names:
                # Parse the rule expression to sympy
                try:
                    # Build local dict for sympify (include state vars and params)
                    local_dict = {}
                    for var in self.recast_odes.keys():
                        local_dict[str(var)] = sp.Symbol(str(var), positive=True)
                    for param_name in self.recast_ir.params:
                        local_dict[param_name] = sp.Symbol(param_name, positive=True)

                    # Also extract identifiers from the rule itself
                    # Parse expression
                    aux_sym = sp.Symbol(rule_name, positive=True)
                    aux_expr = _sympify_model_expression(
                        rule_expr_str,
                        local_dict,
                        positive_new_symbols=True,
                    )

                    # Add to auxiliary_defs if not already there
                    if aux_sym not in self.auxiliary_defs:
                        self.auxiliary_defs[aux_sym] = aux_expr

                except (TypeError, ValueError, sp.SympifyError):
                    # Skip rules that fail to parse
                    continue

    def _extract_refusal_reason(self, recast_text: str) -> str | None:
        """
        Extract canonical S-system refusal reason from GMA output comments.

        Looks for pattern like:
        // NOTE: Canonical S-system recast was not attempted because:
        //   <reason>
        """
        lines = recast_text.split("\n")
        for i, line in enumerate(lines):
            if "Canonical S-system recast was not attempted because:" in line:
                # Get next line which should contain the reason
                if i + 1 < len(lines):
                    reason_line = lines[i + 1].strip()
                    if reason_line.startswith("//"):
                        reason = reason_line[2:].strip()
                        return reason
        return None

    def _extract_auxiliary_definitions(self, recast_text: str) -> dict[sp.Symbol, sp.Expr]:
        """
        Extract auxiliary variable definitions from recast file comments.

        Looks for patterns like:
        // AUXILIARY DEFINITIONS (for lifted variables)
        // ========================================================================
        // Z_1 := exp(X*k)
        // ========================================================================
        """
        aux_defs = {}
        parse_errors: list[dict[str, Any]] = []
        in_aux_section = False
        seen_first_separator = False

        for line in recast_text.split("\n"):
            line = line.strip()

            # Start of auxiliary definitions section (both old and new formats)
            if "AUXILIARY DEFINITIONS" in line or "Auxiliary variable definitions" in line:
                in_aux_section = True
                seen_first_separator = False
                continue

            # End of auxiliary definitions section (old format)
            if in_aux_section and "--- end auxiliary definitions ---" in line:
                break

            # Handle separators
            if in_aux_section and "========" in line:
                if not seen_first_separator:
                    # Opening separator, skip it
                    seen_first_separator = True
                    continue
                else:
                    # Closing separator, end section
                    break

            # Parse auxiliary definition line (after we've seen the opening separator)
            if in_aux_section and seen_first_separator and line.startswith("//"):
                content = line[2:].strip()

                # Skip separator lines
                if content.startswith("="):
                    continue

                # Check if it's a definition line (has :=)
                if ":=" in content:
                    parts = content.split(":=", 1)
                    if len(parts) == 2:
                        aux_var = parts[0].strip()
                        expr_str = parts[1].strip()

                        # Convert to sympy - handle underscores by pre-creating symbols
                        try:
                            aux_sym = sp.Symbol(aux_var)
                            expr = _sympify_model_expression(expr_str, {})
                            aux_defs[aux_sym] = expr
                        except (TypeError, ValueError, SyntaxError, sp.SympifyError) as exc:
                            parse_errors.append({
                                "auxiliary": aux_var,
                                "expression": expr_str,
                                "exception": str(exc),
                            })
                            continue

        self.auxiliary_definition_parse_errors = parse_errors
        return aux_defs

    def _infer_auxiliary_definitions(self):
        """
        Infer definitions of auxiliary variables by pattern matching.

        Strategy:
        1. Check if Y' = X' pattern exists (Y = X + constant)
        2. Match auxiliaries with denominators based on usage context
        """
        # Identify auxiliary variables
        mapped_vars = set()
        mapped_var_names = set()
        factor_map_key_names = set()
        if self.factor_map:
            factor_map_key_names = {str(key) for key in self.factor_map}
            for val in self.factor_map.values():
                if isinstance(val, sp.Symbol):
                    mapped_vars.add(val)
                    mapped_var_names.add(str(val))
                elif hasattr(val, "free_symbols"):
                    mapped_vars.update(val.free_symbols)
                    mapped_var_names.update(str(sym) for sym in val.free_symbols)

        orig_vars = set(self.orig_odes.keys())
        recast_vars = set(self.recast_odes.keys())
        potential_auxiliaries = {
            var
            for var in recast_vars - orig_vars - mapped_vars
            if str(var) not in mapped_var_names and str(var) not in factor_map_key_names
        }

        # Pattern 1: Check if Y' matches any original variable derivative
        # If Y' = X', then Y = X + constant, find the constant from denominators
        for aux in potential_auxiliaries:
            if aux in self.factor_map or str(aux) in factor_map_key_names:
                continue

            aux_ode = self.recast_odes[aux]

            # Check if Y' equals some original variable's recast ODE
            for orig_var in orig_vars:
                if orig_var in self.recast_odes:
                    if aux_ode.equals(self.recast_odes[orig_var]):
                        # Y' = X', so Y = X + C
                        # Find C by checking denominators
                        orig_ode = self.orig_odes[orig_var]
                        denoms = self._find_denominators(orig_ode)

                        # Look for denominator of form (C + X)
                        for denom in denoms:
                            # Check if denom = constant + orig_var
                            if denom.is_Add and orig_var in denom.free_symbols:
                                # This is likely Y = denom
                                self.factor_map[aux] = denom
                                break
                        break

    def _has_negative_exponent(self, expr: sp.Expr, symbol: sp.Symbol) -> bool:
        """Check if symbol appears with negative exponent in expression."""
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Pow):
                base, exp = term.args
                if base == symbol and exp.is_number and float(exp) < 0:
                    return True
        return False

    def _find_denominators(self, expr: sp.Expr) -> list[sp.Expr]:
        """Find denominator expressions in a symbolic expression."""
        denominators = []

        def visit(e):
            if isinstance(e, sp.Pow):
                base, exp = e.args
                # Negative exponent indicates division
                if exp.is_number and float(exp) < 0:
                    if not isinstance(base, sp.Symbol):  # Non-trivial denominator
                        denominators.append(base)
            elif isinstance(e, (sp.Add, sp.Mul)):
                for arg in e.args:
                    visit(arg)

        visit(expr)
        return denominators

    def _build_mapping(self):
        """
        Build symbolic mapping Φ(Z) that reconstructs original variables.

        For variables with factor_map entries: X = X1 * X2 * ...
        For unmapped variables: X = X (identity)

        Also attempts to infer auxiliary variable definitions.
        """
        # First, try to infer auxiliary definitions
        self._infer_auxiliary_definitions()

        self.mapping = {}  # X_orig -> expression in Z_recast

        # Get all original and recast variables
        orig_vars = set(self.orig_odes.keys())
        recast_vars = set(self.recast_odes.keys())

        # Build symbol mapping from names to actual recast symbols
        recast_symbols_by_name = {str(v): v for v in recast_vars}

        if self.factor_map:
            # Use provided/extracted factor map
            # Match symbols by name and substitute with actual recast symbols
            factor_map_by_name = {str(k): v for k, v in self.factor_map.items()}

            for orig_var in orig_vars:
                var_name = str(orig_var)
                if var_name in factor_map_by_name:
                    expr = factor_map_by_name[var_name]
                    if isinstance(expr, list):
                        # List of auxiliaries - compute product
                        product = sp.prod(expr)
                        self.mapping[orig_var] = product
                    else:
                        # Substitute symbols in expression with actual recast symbols
                        # This ensures Jacobian computation works correctly
                        substitutions = {}
                        for sym in expr.free_symbols:
                            sym_name = str(sym)
                            if sym_name in recast_symbols_by_name:
                                substitutions[sym] = recast_symbols_by_name[sym_name]

                        if substitutions:
                            expr = expr.subs(substitutions)

                        self.mapping[orig_var] = expr
                else:
                    # No mapping found, assume identity or sanitizer-renamed identity.
                    self.mapping[orig_var] = self._sanitized_identity_mapping(
                        orig_var, recast_symbols_by_name
                    )
        else:
            # Assume identity mapping (recast uses same variable names), allowing
            # for recast output names sanitized away from Antimony built-ins.
            for var in orig_vars:
                self.mapping[var] = self._sanitized_identity_mapping(
                    var, recast_symbols_by_name
                )

        # Add auxiliary variable mappings to the full mapping
        # For Jacobian computation, we need Φ to map ALL recast variables
        # IMPORTANT: Match by name, not by object identity, since symbols
        # from comments may be different objects than symbols from ODEs
        recast_only_vars = recast_vars - orig_vars
        factor_map_by_name = {str(k): v for k, v in self.factor_map.items()}

        for aux_var in recast_only_vars:
            aux_name = str(aux_var)
            if aux_name in factor_map_by_name:
                # We have a definition for this auxiliary
                expr = factor_map_by_name[aux_name]
                # Substitute symbols with actual recast symbols
                substitutions = {}
                for sym in expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name in recast_symbols_by_name:
                        substitutions[sym] = recast_symbols_by_name[sym_name]
                if substitutions:
                    expr = expr.subs(substitutions)
                self.mapping[aux_var] = expr
            else:
                # No definition, assume identity
                self.mapping[aux_var] = aux_var

        # Build list of all recast state variables
        self.recast_state_vars = list(self.recast_odes.keys())

    def _expand_assignment_rules_in_odes(self, odes, model_ir):
        """
        Substitute assignment rules into ODEs for numerical validation.

        Assignment rules like J_1 := f(X, params) are kept symbolic in the ODEs
        for compact output, but must be expanded for lambdify to work correctly
        during numerical validation.

        Also expands assignment rules iteratively to handle nested rules like:
          Ca_ER := f(Ca)
          J_1 := g(Ca_ER, Ca)  # J_1 depends on Ca_ER which depends on Ca

        Args:
            odes: Dict of ODEs to expand
            model_ir: IR containing assignment_rules and params
        """
        assignment_rules = dict(model_ir.assignment_rules)
        if not assignment_rules:
            return odes  # No expansion needed, return original

        # Get state variables from ODEs
        state_vars = list(odes.keys())
        local_dict = {str(var): var for var in state_vars}
        for param_name in model_ir.params:
            local_dict[param_name] = sp.Symbol(param_name, positive=True)
        # Also add other rule names as symbols (for nested rules)
        for rule_name in assignment_rules:
            if rule_name not in local_dict:
                local_dict[rule_name] = sp.Symbol(rule_name, positive=True)

        # Parse assignment rules to sympy expressions
        parsed_rules = {}
        for rule_name, rule_expr_str in assignment_rules.items():
            try:
                parsed_rules[rule_name] = sp.sympify(rule_expr_str, locals=local_dict)
            except (TypeError, ValueError, sp.SympifyError):
                # Skip rules that fail to parse
                continue

        # Iteratively expand nested rules (max 10 iterations)
        for _ in range(10):
            changed = False
            for rule_name, rule_expr in list(parsed_rules.items()):
                substitutions = {
                    sym: parsed_rules[str(sym)]
                    for sym in rule_expr.free_symbols
                    if str(sym) != rule_name and str(sym) in parsed_rules
                }
                if not substitutions:
                    continue
                new_expr = rule_expr.xreplace(substitutions)
                if new_expr != parsed_rules[rule_name]:
                    parsed_rules[rule_name] = new_expr
                    changed = True
            if not changed:
                break

        # Now substitute expanded rules into ODEs (return new dict, don't modify original)
        expanded_odes = {}
        for var, ode in odes.items():
            substitutions = {
                sym: parsed_rules[str(sym)]
                for sym in ode.free_symbols
                if str(sym) in parsed_rules
            }
            if substitutions:
                expanded_odes[var] = ode.xreplace(substitutions)
            else:
                expanded_odes[var] = ode

        return expanded_odes

    def _canonicalize_symbols(self):
        """
        Unify all symbols across original and recast models by name.

        This fixes the symbol identity bug where K_S from original and K_S from recast
        are different Symbol objects, preventing simplification of (K_S - K_S) to 0.

        After canonicalization, both models share the same Symbol objects for parameters
        and variables, enabling proper symbolic simplification.
        """
        # Build canonical symbol table from all parameters and species
        all_params = set(self.orig_ir.params.keys()) | set(self.recast_ir.params.keys())
        all_species = {str(v) for v in self.orig_odes.keys()} | {
            str(v) for v in self.recast_odes.keys()
        }

        # CRITICAL: Also collect ALL free symbols from ALL expressions
        # This ensures auxiliary variables (Z_1, etc.) that appear in ODEs are canonicalized
        # even if they weren't already in the keys or params
        all_expr_symbols: set[str] = set()
        for ode in self.orig_odes.values():
            all_expr_symbols.update(s.name for s in ode.free_symbols)
        for ode in self.recast_odes.values():
            all_expr_symbols.update(s.name for s in ode.free_symbols)
        for expr in self.mapping.values():
            all_expr_symbols.update(s.name for s in expr.free_symbols)
        for expr in self.auxiliary_defs.values():
            all_expr_symbols.update(s.name for s in expr.free_symbols)

        canon = {}
        for name in all_params | all_species | all_expr_symbols:
            canon[name] = sp.Symbol(name, positive=True)

        # Store canonical symbols for later use (e.g., in numerical validation)
        self.canonical_symbols = canon

        def rename_expr(expr):
            return _canonicalize_expr_by_name(expr, canon)

        # Canonicalize original ODEs
        self.orig_odes = {
            canon.get(str(v), v): rename_expr(expr) for v, expr in self.orig_odes.items()
        }

        # Canonicalize recast ODEs
        self.recast_odes = {
            canon.get(str(v), v): rename_expr(expr) for v, expr in self.recast_odes.items()
        }

        # Canonicalize mapping Φ
        self.mapping = {canon.get(str(k), k): rename_expr(expr) for k, expr in self.mapping.items()}

        # Canonicalize auxiliary definitions
        self.auxiliary_defs = {
            canon.get(str(k), k): rename_expr(expr) for k, expr in self.auxiliary_defs.items()
        }

        # Update recast_state_vars list with canonical symbols
        self.recast_state_vars = [canon.get(str(v), v) for v in self.recast_state_vars]

        # Update initials dict with canonical symbols
        self.orig_system.initials = {
            canon.get(str(k), k): v for k, v in self.orig_system.initials.items()
        }
        self.recast_system.initials = {
            canon.get(str(k), k): v for k, v in self.recast_system.initials.items()
        }

    def _canonical_expr(self, expr: sp.Expr) -> sp.Expr:
        """Canonicalize expression symbols using this validator's symbol table."""
        return _canonicalize_expr_by_name(expr, self.canonical_symbols)

    def _parse_expr_with_canonical_symbols(self, expr: str | sp.Expr) -> sp.Expr:
        """Parse an expression using canonical symbols for every identifier."""
        if isinstance(expr, sp.Expr):
            return self._canonical_expr(expr)

        local_dict = dict(self.canonical_symbols)
        parsed = _sympify_model_expression(
            expr,
            local_dict,
            positive_new_symbols=True,
        )
        return self._canonical_expr(parsed)

    def _finite_recast_parameter_substitutions(self) -> dict[sp.Symbol, sp.Expr]:
        """Return finite concrete parameter values for model-specific identities."""
        substitutions: dict[sp.Symbol, sp.Expr] = {}
        for name, value in self.recast_ir.params.items():
            symbol = self.canonical_symbols.get(name, sp.Symbol(name, positive=True))
            try:
                value_expr = sp.sympify(value)
            except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
                continue
            if value_expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                continue
            if value_expr.is_finite is False:
                continue
            substitutions[symbol] = value_expr
        return substitutions

    def _expressions_equivalent(
        self,
        lhs: sp.Expr,
        rhs: sp.Expr,
        extra_subs: dict[sp.Symbol, sp.Expr] | None = None,
        *,
        context: str = "auxiliary_identity",
    ) -> tuple[bool, sp.Expr]:
        """Compare expressions after canonical symbol and auxiliary substitutions."""
        started_at = time.perf_counter()

        def raise_complexity(
            *,
            operation_count: int,
            free_symbol_count: int,
            residual: sp.Expr,
            candidate_index: int | None,
            active_subphase: str,
            max_operation_count: int | None = AUXILIARY_SLOW_SIMPLIFY_MAX_OPS,
            max_free_symbol_count: int | None = AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS,
            risky_symbolic_power_guard: bool = False,
        ) -> None:
            raise AuxiliaryIdentityComplexityError(
                context=context,
                operation_count=operation_count,
                free_symbol_count=free_symbol_count,
                residual=residual,
                candidate_index=candidate_index,
                active_subphase=active_subphase,
                max_operation_count=max_operation_count,
                max_free_symbol_count=max_free_symbol_count,
                risky_symbolic_power_guard=risky_symbolic_power_guard,
                elapsed_seconds=round(time.perf_counter() - started_at, 6),
            )

        progress_callback = getattr(self, "progress_callback", None)
        _notify_progress(progress_callback, f"auxiliaries_equivalence:{context}")
        lhs = self._canonical_expr(lhs)
        rhs = self._canonical_expr(rhs)
        clock_vars = self._clock_variable_by_name()
        if clock_vars:
            lhs = _substitute_symbols_by_name(lhs, clock_vars)
            rhs = _substitute_symbols_by_name(rhs, clock_vars)
        diff = lhs - rhs

        if extra_subs:
            diff = diff.subs(extra_subs)

        _notify_progress(
            progress_callback,
            f"auxiliaries_simplify:{context}:candidate_0",
        )
        operation_count = int(sp.count_ops(diff))
        free_symbol_count = len(diff.free_symbols)
        has_float = bool(diff.atoms(sp.Float))
        has_risky_power = _has_risky_symbolic_power(diff)
        if has_float and (
            has_risky_power or operation_count > AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS
        ):
            if _passes_relative_numeric_zero_check(diff):
                return True, sp.Integer(0)
            raise_complexity(
                operation_count=operation_count,
                free_symbol_count=free_symbol_count,
                residual=diff,
                candidate_index=0,
                active_subphase="float_simplification_preflight",
                max_operation_count=AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS,
                risky_symbolic_power_guard=has_risky_power,
            )
        cheap = _cheap_zero_simplification(diff)
        if cheap == 0:
            return True, cheap
        if (
            operation_count > 200
            and has_float
            and _passes_relative_numeric_zero_check(diff)
        ):
            return True, sp.Integer(0)
        if operation_count > AUXILIARY_CANDIDATE_GENERATION_MAX_OPS:
            raise_complexity(
                operation_count=operation_count,
                free_symbol_count=free_symbol_count,
                residual=diff,
                candidate_index=0,
                active_subphase="candidate_generation",
                max_operation_count=AUXILIARY_CANDIDATE_GENERATION_MAX_OPS,
                max_free_symbol_count=None,
            )

        aux_to_def: dict[sp.Symbol, sp.Expr] = {}
        def_to_aux: dict[sp.Expr, sp.Symbol] = {}
        for aux, defn in self.auxiliary_defs.items():
            aux_c = self._canonical_expr(aux)
            defn_c = self._canonical_expr(defn)
            if self._is_clock_definition(defn_c):
                continue
            if clock_vars:
                defn_c = _substitute_symbols_by_name(defn_c, clock_vars)
            aux_to_def[aux_c] = defn_c
            def_to_aux[defn_c] = aux_c

        candidates = [diff]
        if aux_to_def:
            candidates.append(diff.subs(aux_to_def))
        if def_to_aux:
            candidates.append(diff.subs(def_to_aux))
        if aux_to_def and def_to_aux:
            candidates.append(diff.subs(aux_to_def).subs(def_to_aux))

        ordered_candidates: list[sp.Expr] = []
        param_subs = self._finite_recast_parameter_substitutions()
        if param_subs:
            for candidate in candidates:
                try:
                    substituted = candidate.subs(param_subs)
                except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
                    continue
                if substituted.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    continue
                ordered_candidates.append(substituted)
        ordered_candidates.extend(candidates)

        best = ordered_candidates[0]
        for index, candidate in enumerate(ordered_candidates):
            _notify_progress(
                progress_callback,
                f"auxiliaries_simplify:{context}:candidate_{index}",
            )
            operation_count = int(sp.count_ops(candidate))
            free_symbol_count = len(candidate.free_symbols)
            has_float = bool(candidate.atoms(sp.Float))
            has_risky_power = _has_risky_symbolic_power(candidate)
            if has_float and (
                has_risky_power or operation_count > AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS
            ):
                if _passes_relative_numeric_zero_check(candidate):
                    return True, sp.Integer(0)
                raise_complexity(
                    operation_count=operation_count,
                    free_symbol_count=free_symbol_count,
                    residual=candidate,
                    candidate_index=index,
                    active_subphase="float_simplification_preflight",
                    max_operation_count=AUXILIARY_FLOAT_SIMPLIFY_MAX_OPS,
                    risky_symbolic_power_guard=has_risky_power,
                )
            if operation_count > AUXILIARY_SLOW_SIMPLIFY_MAX_OPS:
                raise_complexity(
                    operation_count=operation_count,
                    free_symbol_count=free_symbol_count,
                    residual=candidate,
                    candidate_index=index,
                    active_subphase="cheap_simplification",
                )
            if free_symbol_count > AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS:
                raise_complexity(
                    operation_count=operation_count,
                    free_symbol_count=free_symbol_count,
                    residual=candidate,
                    candidate_index=index,
                    active_subphase="cheap_simplification",
                )
            cheap = _cheap_zero_simplification(candidate)
            if cheap == 0:
                return True, cheap
            if (
                operation_count > 200
                and has_float
                and _passes_relative_numeric_zero_check(candidate)
            ):
                return True, sp.Integer(0)

        for index, candidate in enumerate(ordered_candidates):
            _notify_progress(
                progress_callback,
                f"auxiliaries_simplify:{context}:candidate_{index}",
            )
            operation_count = int(sp.count_ops(candidate))
            free_symbol_count = len(candidate.free_symbols)
            has_float = bool(candidate.atoms(sp.Float))
            if (
                operation_count > AUXILIARY_SLOW_SIMPLIFY_MAX_OPS
                or free_symbol_count > AUXILIARY_SLOW_SIMPLIFY_MAX_FREE_SYMBOLS
            ):
                raise_complexity(
                    operation_count=operation_count,
                    free_symbol_count=free_symbol_count,
                    residual=candidate,
                    candidate_index=index,
                    active_subphase="simplification",
                    risky_symbolic_power_guard=True,
                )
            if has_float and _has_risky_symbolic_power(candidate):
                raise_complexity(
                    operation_count=operation_count,
                    free_symbol_count=free_symbol_count,
                    residual=candidate,
                    candidate_index=index,
                    active_subphase="simplification",
                )
            simplified = _simplify_identity_difference(candidate)
            if simplified == 0:
                return True, sp.Integer(0)
            best = simplified
            if (
                sp.count_ops(candidate) > 100
                and _has_risky_symbolic_power(candidate)
            ):
                raise_complexity(
                    operation_count=int(sp.count_ops(candidate)),
                    free_symbol_count=free_symbol_count,
                    residual=simplified,
                    candidate_index=index,
                    active_subphase="simplification",
                    risky_symbolic_power_guard=True,
                )

        return False, best

    def _is_clock_definition(self, expr: sp.Expr) -> bool:
        """Return True when an auxiliary definition is the integration clock."""
        return isinstance(expr, sp.Symbol) and _is_external_clock_symbol_name(str(expr))

    def _clock_variable_by_name(self) -> dict[str, sp.Symbol]:
        """Map clock source names such as time/t to the recast clock state symbol."""
        clocks = {}
        recast_vars_by_name = {str(v): v for v in self.recast_state_vars}
        for aux, defn in self.auxiliary_defs.items():
            defn_c = self._canonical_expr(defn)
            if self._is_clock_definition(defn_c):
                aux_name = str(aux)
                if aux_name in recast_vars_by_name:
                    clocks[str(defn_c).lower()] = recast_vars_by_name[aux_name]
        return clocks

    def check_mapping_complete(self) -> EquivalenceTest:
        """Verify every original state has an explicit or valid identity mapping."""
        recast_var_names = {str(v) for v in self.recast_state_vars}
        assignment_rule_names = set(self.recast_ir.assignment_rules.keys())
        missing = []

        for orig_var in sorted(self.orig_odes.keys(), key=str):
            orig_name = str(orig_var)
            mapped_expr = self.mapping.get(orig_var)
            if mapped_expr is None:
                missing.append(orig_name)
                continue

            identity_ok = (
                mapped_expr == orig_var
                and (orig_name in recast_var_names or orig_name in assignment_rule_names)
            )
            explicit_ok = mapped_expr != orig_var
            if not identity_ok and not explicit_ok:
                missing.append(orig_name)

        if missing:
            return EquivalenceTest(
                name="mapping_completeness",
                result=ValidationResult.FAIL,
                details=(
                    "Missing reconstruction mapping for original variables: "
                    + ", ".join(missing)
                ),
                metadata={"missing_variables": missing},
            )

        return EquivalenceTest(
            name="mapping_completeness",
            result=ValidationResult.PASS,
            details="Every original state has an explicit or valid identity mapping",
        )

    def _assignment_identity_tests(self) -> list[EquivalenceTest]:
        """Validate assignment-rule auxiliaries and observable reconstruction rules."""
        tests: list[EquivalenceTest] = []
        orig_vars_by_name = {str(v): v for v in self.orig_odes}
        mapping_by_name = {str(k): v for k, v in self.mapping.items()}
        aux_defs_by_name = {str(k): v for k, v in self.auxiliary_defs.items()}

        for rule_name, rule_expr in sorted(self.recast_ir.assignment_rules.items()):
            expected = None
            kind = None
            if rule_name in orig_vars_by_name and rule_name in mapping_by_name:
                expected = mapping_by_name[rule_name]
                kind = "observable_mapping"
            elif rule_name in aux_defs_by_name:
                expected = aux_defs_by_name[rule_name]
                kind = "assignment_auxiliary"

            if expected is None:
                continue

            try:
                _notify_progress(
                    getattr(self, "progress_callback", None),
                    f"auxiliaries_assignment_identity:{rule_name}",
                )
                actual_expr = self._parse_expr_with_canonical_symbols(rule_expr)
                context = f"assignment_identity:{kind}:{rule_name}"
                equivalent, residual = self._expressions_equivalent(
                    actual_expr,
                    expected,
                    context=context,
                )
                result = ValidationResult.PASS if equivalent else ValidationResult.FAIL
                reason = None
                details = (
                    f"{rule_name} assignment matches {kind} definition"
                    if equivalent
                    else f"{rule_name} assignment residual: {residual}"
                )
                metadata = {"rule": rule_name, "kind": kind}
            except AuxiliaryIdentityComplexityError as e:
                result = ValidationResult.INCONCLUSIVE
                reason = "auxiliary_complexity"
                details = str(e)
                metadata = {
                    "rule": rule_name,
                    "kind": kind,
                    **_auxiliary_complexity_metadata(e),
                }
            except Exception as e:
                result = ValidationResult.INCONCLUSIVE
                reason = None
                details = f"Could not parse {rule_name} assignment rule {rule_expr!r}: {e}"
                metadata = {"rule": rule_name, "kind": kind}

            tests.append(
                EquivalenceTest(
                    name=f"{kind}:{rule_name}",
                    result=result,
                    details=details,
                    metadata=metadata,
                    reason=reason,
                )
            )

        return tests

    def _ode_auxiliary_identity_tests(self) -> list[EquivalenceTest]:
        """Validate ODE-carried auxiliaries against their declared definitions."""
        tests: list[EquivalenceTest] = []
        recast_odes_by_name = {str(k): (k, v) for k, v in self.recast_odes.items()}
        orig_odes_by_name = {str(k): v for k, v in self.orig_odes_expanded.items()}
        clock_vars = self._clock_variable_by_name()

        for aux, defn in sorted(self.auxiliary_defs.items(), key=lambda kv: str(kv[0])):
            aux_name = str(aux)
            _notify_progress(
                getattr(self, "progress_callback", None),
                f"auxiliaries_ode_identity:{aux_name}",
            )
            if aux_name in self.recast_ir.assignment_rules:
                continue
            if aux_name not in recast_odes_by_name:
                if aux_name not in {str(v) for v in self.orig_odes.keys()}:
                    tests.append(
                        EquivalenceTest(
                            name=f"ode_auxiliary_identity:{aux_name}",
                            result=ValidationResult.INCONCLUSIVE,
                            details=f"{aux_name} has a definition but no ODE or assignment rule",
                        )
                    )
                continue

            _aux_var, actual_ode = recast_odes_by_name[aux_name]
            try:
                _notify_progress(
                    getattr(self, "progress_callback", None),
                    f"auxiliaries_ode_identity_build_expected:{aux_name}",
                )
                defn_c = self._canonical_expr(defn)
                if self._is_clock_definition(defn_c):
                    expected_ode = sp.Integer(1)
                else:
                    if clock_vars:
                        defn_c = _substitute_symbols_by_name(defn_c, clock_vars)
                    expected_ode = sp.Integer(0)
                    unresolved_dynamic_symbols = []
                    for sym in sorted(defn_c.free_symbols, key=str):
                        sym_name = str(sym)
                        if _is_external_clock_symbol_name(sym_name):
                            expected_ode += sp.diff(defn_c, sym)
                        elif sym_name in recast_odes_by_name:
                            _state_var, state_ode = recast_odes_by_name[sym_name]
                            expected_ode += sp.diff(defn_c, sym) * state_ode
                        elif sym_name in orig_odes_by_name:
                            ode_at_mapping = orig_odes_by_name[sym_name].subs(self.mapping)
                            if clock_vars:
                                ode_at_mapping = _substitute_symbols_by_name(
                                    ode_at_mapping, clock_vars
                                )
                            expected_ode += sp.diff(defn_c, sym) * ode_at_mapping
                        elif sym_name in self.recast_ir.params:
                            continue
                        else:
                            unresolved_dynamic_symbols.append(sym_name)

                    if unresolved_dynamic_symbols:
                        tests.append(
                            EquivalenceTest(
                                name=f"ode_auxiliary_identity:{aux_name}",
                                result=ValidationResult.INCONCLUSIVE,
                                details=(
                                    f"Could not resolve dynamics for symbols in {aux_name}: "
                                    + ", ".join(unresolved_dynamic_symbols)
                                ),
                                metadata={"unresolved_symbols": unresolved_dynamic_symbols},
                            )
                        )
                        continue

                equivalent, residual = self._expressions_equivalent(
                    actual_ode,
                    expected_ode,
                    context=f"ode_auxiliary_identity:{aux_name}",
                )
                tests.append(
                    EquivalenceTest(
                        name=f"ode_auxiliary_identity:{aux_name}",
                        result=ValidationResult.PASS if equivalent else ValidationResult.FAIL,
                        details=(
                            f"{aux_name} ODE matches derivative of its definition"
                            if equivalent
                            else f"{aux_name} ODE identity residual: {residual}"
                        ),
                        metadata={"auxiliary": aux_name},
                    )
                )
            except AuxiliaryIdentityComplexityError as e:
                tests.append(
                    EquivalenceTest(
                        name=f"ode_auxiliary_identity:{aux_name}",
                        result=ValidationResult.INCONCLUSIVE,
                        reason="auxiliary_complexity",
                        details=str(e),
                        metadata={
                            "auxiliary": aux_name,
                            **_auxiliary_complexity_metadata(e),
                        },
                    )
                )
            except Exception as e:
                tests.append(
                    EquivalenceTest(
                        name=f"ode_auxiliary_identity:{aux_name}",
                        result=ValidationResult.INCONCLUSIVE,
                        details=f"Could not validate {aux_name} ODE identity: {e}",
                    )
                )

        return tests

    def check_auxiliary_identities(self) -> list[EquivalenceTest]:
        """Validate lifted auxiliaries and observable assignments against definitions."""
        tests = self._assignment_identity_tests()
        tests.extend(self._ode_auxiliary_identity_tests())
        return tests
