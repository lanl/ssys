"""
Validation framework for verifying S-system and GMA recasts.

This module provides automated validation of recasting correctness through:
- Symbolic equivalence testing (Jacobian chain rule)
- Numerical pointwise validation with sampling
- Trajectory comparison via simulation
- Auxiliary identity verification
- Structural classification verification
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp
from sympy import Matrix, lambdify

from .recaster import (
    SystemClass,
    build_sym_system,
    classify_system,
    parse_antimony,
    parse_antimony_via_sbml,
)


class ValidationResult(Enum):
    """Validation test outcomes."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    NOT_ATTEMPTED = "not_attempted"


def _is_dev_mode() -> bool:
    """
    Detect if we're running in development mode.

    Development mode is indicated by having pytest installed (from [dev] extras).
    In dev mode, we run both JAX and non-JAX numerical tests for debugging.
    In production mode, we run JAX if available, else non-JAX.
    """
    try:
        import pytest

        return True
    except ImportError:
        return False


@dataclass
class EquivalenceTest:
    """Results from a single equivalence test."""

    name: str
    result: ValidationResult
    max_error: float | None = None
    mean_error: float | None = None
    details: str = ""
    counterexamples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report for an original/recast pair."""

    original_file: str
    recast_file: str

    # Structural classification
    original_class: SystemClass
    recast_class: SystemClass
    expected_class: SystemClass | None = None
    canonical_refusal_reason: str | None = None

    # Test results
    symbolic_test: EquivalenceTest | None = None
    numerical_test: EquivalenceTest | None = None
    trajectory_test: EquivalenceTest | None = None
    auxiliary_tests: list[EquivalenceTest] = field(default_factory=list)

    # Overall verdict
    overall_pass: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""

        def test_to_dict(test: EquivalenceTest | None) -> dict | None:
            if test is None:
                return None
            return {
                "name": test.name,
                "result": test.result.value,
                "max_error": test.max_error,
                "mean_error": test.mean_error,
                "details": test.details,
                "counterexamples": test.counterexamples[:5],  # Limit to first 5
            }

        return {
            "original_file": self.original_file,
            "recast_file": self.recast_file,
            "classification": {
                "original": self.original_class.value,
                "recast": self.recast_class.value,
                "expected": self.expected_class.value if self.expected_class else None,
                "canonical_refusal_reason": self.canonical_refusal_reason,
            },
            "tests": {
                "symbolic": test_to_dict(self.symbolic_test),
                "numerical": test_to_dict(self.numerical_test),
                "trajectory": test_to_dict(self.trajectory_test),
                "auxiliaries": [test_to_dict(t) for t in self.auxiliary_tests],
            },
            "overall_pass": self.overall_pass,
            "summary": self.summary,
        }


class RecastValidator:
    """
    Validates that a recast model is mathematically equivalent to the original.

    Uses multiple validation strategies:
    1. Symbolic equivalence via Jacobian chain rule
    2. Numerical pointwise testing with random sampling
    3. Trajectory comparison via ODE simulation
    4. Auxiliary identity verification for lifted functions
    """

    def __init__(
        self,
        original_file: str,
        recast_file: str,
        factor_map: dict[sp.Symbol, list[sp.Symbol]] | None = None,
        mode: str = "simplified",
        parser: str = "legacy",
    ):
        """
        Initialize validator.

        Args:
            original_file: Path to original Antimony file
            recast_file: Path to recast Antimony file
            factor_map: Mapping from original to auxiliary variables (X -> [X1, X2, ...])
            mode: Recast mode ('simplified' or 'canonical')
            parser: Parser to use for Antimony files ('legacy' or 'sbml')
        """
        self.original_file = original_file
        self.recast_file = recast_file
        self.mode = mode
        self.parser = parser

        # Read recast file to extract mapping comments
        recast_text = open(recast_file).read()

        # Read original file text
        orig_text = open(original_file).read()

        # Parse both models using the specified parser
        if parser == "sbml":
            # SBML-first parser (reference Antimony implementation)
            self.orig_system = parse_antimony_via_sbml(orig_text)
            self.recast_system = parse_antimony_via_sbml(recast_text)
        else:
            # Legacy parser
            orig_ir = parse_antimony(orig_text)
            self.orig_system = build_sym_system(orig_ir)
            # Attach original Antimony text for RoadRunner simulation
            # Note: roadrunner_backend checks for 'antimony_text' attribute
            self.orig_system.antimony_text = orig_text

            recast_ir = parse_antimony(recast_text)
            self.recast_system = build_sym_system(recast_ir)
            self.recast_system.antimony_text = recast_text

        # Create aliases for backward compatibility with code using orig_ir/recast_ir
        # SymSystem has the same key attributes: params, assignment_rules
        # Add compatibility attributes for ModelIR interface
        self.orig_ir = self.orig_system
        self.recast_ir = self.recast_system

        # Add 'initial' alias for 'initials' (ModelIR uses 'initial', SymSystem uses 'initials')
        if not hasattr(self.orig_ir, "initial"):
            # Convert initials dict to have string keys for ModelIR compatibility
            self.orig_ir.initial = {str(k): v for k, v in self.orig_system.initials.items()}  # type: ignore[attr-defined]
        if not hasattr(self.recast_ir, "initial"):
            self.recast_ir.initial = {str(k): v for k, v in self.recast_system.initials.items()}  # type: ignore[attr-defined]

        # Add @SIM metadata compatibility (SymSystem doesn't have these by default)
        if not hasattr(self.orig_ir, "sim_t_start"):
            self.orig_ir.sim_t_start = None
        if not hasattr(self.orig_ir, "sim_t_end"):
            self.orig_ir.sim_t_end = None
        if not hasattr(self.orig_ir, "sim_n_steps"):
            self.orig_ir.sim_n_steps = None

        # Add 'species' alias for 'vars' (ModelIR uses 'species', SymSystem uses 'vars')
        if not hasattr(self.orig_ir, "species"):
            self.orig_ir.species = [str(v) for v in self.orig_system.vars]  # type: ignore[attr-defined]
        if not hasattr(self.recast_ir, "species"):
            self.recast_ir.species = [str(v) for v in self.recast_system.vars]  # type: ignore[attr-defined]

        # Add 'reactions' attribute (SymSystem uses ODEs directly, no reactions)
        if not hasattr(self.orig_ir, "reactions"):
            self.orig_ir.reactions = []  # type: ignore[attr-defined]
        if not hasattr(self.recast_ir, "reactions"):
            self.recast_ir.reactions = []  # type: ignore[attr-defined]

        # Add 'explicit_rates' alias for 'odes' (for roadrunner backend)
        # Convert Python ** to Antimony ^ for exponentiation
        if not hasattr(self.orig_ir, "explicit_rates"):
            self.orig_ir.explicit_rates = {  # type: ignore[attr-defined]
                str(k): str(v).replace("**", "^") for k, v in self.orig_system.odes.items()
            }
        if not hasattr(self.recast_ir, "explicit_rates"):
            self.recast_ir.explicit_rates = {  # type: ignore[attr-defined]
                str(k): str(v).replace("**", "^") for k, v in self.recast_system.odes.items()
            }

        # Extract ODE dictionaries
        self.orig_odes = self.orig_system.odes
        self.recast_odes = self.recast_system.odes

        # Extract mapping from comments if not provided
        if factor_map is None:
            self.factor_map = self._extract_mapping_from_comments(recast_text)
        else:
            self.factor_map = factor_map

        # Extract auxiliary definitions from comments
        self.auxiliary_defs = self._extract_auxiliary_definitions(recast_text)

        # CRITICAL FIX: Also extract auxiliary definitions from ACTUAL assignment rules
        # (not just comment definitions). This handles lifted_mode='assignment' output
        # where Y_1 := a^2 + 1 is an actual Antimony statement, not a comment.
        self._merge_assignment_rules_as_auxiliaries()

        # Merge auxiliary definitions into factor_map for use in validation
        self.factor_map.update(self.auxiliary_defs)

        # Extract assignment rules from recast IR (needed for numerical validation)
        # These are expressions like J_1 := c_1 * (v_1 * p_open + v_2) * (Ca_ER - Ca)
        self.assignment_rules = dict(self.recast_ir.assignment_rules)

        # Build mapping function Φ: Z -> X
        self._build_mapping()

        # Classify systems
        self.orig_class = classify_system(self.orig_system)
        self.recast_class = classify_system(self.recast_system)

        # Extract refusal reason if present (for GMA outputs)
        self.canonical_refusal_reason = self._extract_refusal_reason(recast_text)

        # Canonicalize all symbols to fix symbol identity bug (K_S_orig vs K_S_recast)
        # This ensures that K_S - K_S simplifies to 0 in symbolic validation
        self._canonicalize_symbols()

        # Expand assignment rules for numerical validation (keep original for symbolic)
        # Assignment rules like J_1 := f(X, params) are symbolic in the ODEs
        # We expand them into a separate dict for lambdify to work correctly
        self.recast_odes_expanded = self._expand_assignment_rules_in_odes(
            self.recast_odes, self.recast_ir
        )

        # Also expand original ODEs - they may also use assignment rules
        # Use assignment rules from ORIGINAL model
        self.orig_assignment_rules = dict(self.orig_ir.assignment_rules)
        self.orig_odes_expanded = self._expand_assignment_rules_in_odes(
            self.orig_odes, self.orig_ir
        )

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
                            expr = sp.sympify(expr_str)
                            mapping[orig_sym] = expr
                        except Exception:
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
        import re

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
                    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", rule_expr_str)
                    sympy_functions = {
                        "exp",
                        "log",
                        "sin",
                        "cos",
                        "tan",
                        "sqrt",
                        "sinh",
                        "cosh",
                        "tanh",
                        "asin",
                        "acos",
                        "atan",
                    }
                    for name in identifiers:
                        if name not in local_dict and name not in sympy_functions:
                            local_dict[name] = sp.Symbol(name, positive=True)

                    # Parse expression
                    aux_sym = sp.Symbol(rule_name, positive=True)
                    aux_expr = sp.sympify(rule_expr_str, locals=local_dict)

                    # Add to auxiliary_defs if not already there
                    if aux_sym not in self.auxiliary_defs:
                        self.auxiliary_defs[aux_sym] = aux_expr

                except Exception:
                    # Skip rules that fail to parse
                    pass

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
                            import re

                            aux_sym = sp.Symbol(aux_var)

                            # Extract all identifiers from expression (including those with underscores)
                            identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr_str)

                            # Create symbols only for identifiers that aren't functions
                            # Common sympy functions to exclude
                            sympy_functions = {
                                "exp",
                                "log",
                                "sin",
                                "cos",
                                "tan",
                                "sqrt",
                                "sinh",
                                "cosh",
                                "tanh",
                                "asin",
                                "acos",
                                "atan",
                            }
                            local_dict = {
                                name: sp.Symbol(name)
                                for name in identifiers
                                if name not in sympy_functions
                            }

                            # Parse expression with pre-created symbols
                            expr = sp.sympify(expr_str, locals=local_dict)
                            aux_defs[aux_sym] = expr
                        except Exception:
                            # If parsing fails, skip this definition
                            pass

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
        if self.factor_map:
            for val in self.factor_map.values():
                if isinstance(val, sp.Symbol):
                    mapped_vars.add(val)
                elif hasattr(val, "free_symbols"):
                    mapped_vars.update(val.free_symbols)

        orig_vars = set(self.orig_odes.keys())
        recast_vars = set(self.recast_odes.keys())
        potential_auxiliaries = recast_vars - orig_vars - mapped_vars

        # Pattern 1: Check if Y' matches any original variable derivative
        # If Y' = X', then Y = X + constant, find the constant from denominators
        for aux in potential_auxiliaries:
            if aux in self.factor_map:
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
                    # No mapping found, assume identity
                    self.mapping[orig_var] = orig_var
        else:
            # Assume identity mapping (recast uses same variable names)
            for var in orig_vars:
                self.mapping[var] = var

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

        # Parse assignment rules to sympy expressions
        parsed_rules = {}
        for rule_name, rule_expr_str in assignment_rules.items():
            try:
                # Build local dict for sympify
                local_dict = {}
                for var in state_vars:
                    local_dict[str(var)] = var
                for param_name in model_ir.params:
                    local_dict[param_name] = sp.Symbol(param_name, positive=True)
                # Also add other rule names as symbols (for nested rules)
                for other_rule in assignment_rules:
                    if other_rule not in local_dict:
                        local_dict[other_rule] = sp.Symbol(other_rule, positive=True)

                parsed_rules[rule_name] = sp.sympify(rule_expr_str, locals=local_dict)
            except Exception:
                # Skip rules that fail to parse
                continue

        # Iteratively expand nested rules (max 10 iterations)
        for _ in range(10):
            changed = False
            for rule_name, rule_expr in list(parsed_rules.items()):
                new_expr = rule_expr
                for other_name, other_expr in parsed_rules.items():
                    if other_name != rule_name:
                        # Substitute by matching symbol name
                        for sym in new_expr.free_symbols:
                            if str(sym) == other_name:
                                new_expr = new_expr.subs(sym, other_expr)
                if new_expr != parsed_rules[rule_name]:
                    parsed_rules[rule_name] = new_expr
                    changed = True
            if not changed:
                break

        # Now substitute expanded rules into ODEs (return new dict, don't modify original)
        expanded_odes = {}
        for var, ode in odes.items():
            new_ode = ode
            for rule_name, rule_expr in parsed_rules.items():
                # Find matching symbol by name in the ODE
                for sym in new_ode.free_symbols:
                    if str(sym) == rule_name:
                        new_ode = new_ode.subs(sym, rule_expr)
            expanded_odes[var] = sp.simplify(new_ode)

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

        # Helper to rename all symbols in an expression
        def rename_expr(expr):
            # Handle sympy constants (I=imaginary, E=Euler) that match variable names
            # These have empty free_symbols, so normal substitution won't catch them
            if not expr.free_symbols:
                expr_str = str(expr)
                if expr_str in canon:
                    return canon[expr_str]
            # Handle compound expressions by substituting all symbols
            subs = {s: canon[s.name] for s in expr.free_symbols if s.name in canon}
            return expr.subs(subs) if subs else expr

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

    def check_symbolic_equivalence(self, timeout: float = 30.0) -> EquivalenceTest:
        """
        Check symbolic equivalence using Jacobian chain rule with constraint substitution.

        For lifted systems with auxiliary variables (e.g., Y_1 := K_S + S + S²/K_I),
        substitutes auxiliary definitions into recast ODEs before comparison.

        Tests if: J_Φ(Z) · f_recast(Z)|_constraints = f_orig(Φ(Z))

        Where:
        - Φ(Z) maps recast states to original states
        - J_Φ is the Jacobian matrix ∂Φ/∂Z
        - f_recast is the recast ODE RHS with auxiliaries substituted
        - f_orig is the original ODE RHS

        Args:
            timeout: Maximum time in seconds for simplification

        Returns:
            EquivalenceTest with symbolic validation results
        """
        try:
            # Build Φ as a vector
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            Phi_vector = Matrix([self.mapping[v] for v in orig_vars_ordered])

            # Build Z state vector
            Z_vector = Matrix(self.recast_state_vars)

            # Compute Jacobian J_Φ = ∂Φ/∂Z
            J_Phi = Phi_vector.jacobian(Z_vector)

            # Build f_recast as vector (no substitution yet)
            f_recast = Matrix([self.recast_odes[v] for v in self.recast_state_vars])

            # Compute J_Φ · f_recast
            lhs = J_Phi * f_recast

            # Build f_orig(Φ(Z)) by substituting Φ into original ODEs
            f_orig_at_Phi = Matrix(
                [self.orig_odes[v].subs(self.mapping) for v in orig_vars_ordered]
            )

            # CRITICAL: For time-dependent models, substitute time → T in f_orig_at_Phi
            # The recast model uses clock variable T (with T' = 1, T(0) = 0) instead of 'time'
            # Auxiliary Y_1 = T + 1 corresponds to original denominator (time + 1)
            # Without this substitution, we get mixed T/time terms that don't cancel
            #
            # KEY FIX: We must find the ACTUAL 'time' symbol object that appears in
            # f_orig_at_Phi.free_symbols, not a different Symbol object with the same name.
            # SymPy's .subs() matches by object identity, not by name.

            # Step 1: Find clock variable T in recast_state_vars
            clock_var = None
            for rv in self.recast_state_vars:
                if str(rv) == "T":
                    clock_var = rv
                    break

            # Step 2: If clock exists, find 'time' symbol in f_orig_at_Phi and substitute
            if clock_var is not None:
                # Collect all free symbols from f_orig_at_Phi (it's a Matrix)
                all_free_symbols = set()
                for component in f_orig_at_Phi:
                    all_free_symbols.update(component.free_symbols)

                # Find the actual 'time' symbol by name
                for sym in all_free_symbols:
                    if str(sym).lower() == "time":
                        f_orig_at_Phi = f_orig_at_Phi.subs(sym, clock_var)
                        break

            # Compute difference Δ = J_Φ · f_recast - f_orig(Φ(Z))
            Delta = lhs - f_orig_at_Phi

            # Build substitution dict for auxiliary variables BEFORE simplification
            # Match symbols by name to handle different symbol objects
            orig_var_names = {str(v) for v in orig_vars_ordered}
            recast_vars_by_name = {str(v): v for v in self.recast_state_vars}

            # Extract actual parameter symbols from recast ODEs
            param_symbols_by_name = {}
            for ode in self.recast_odes.values():
                for sym in ode.free_symbols:
                    sym_name = str(sym)
                    if sym_name in self.recast_ir.params:
                        param_symbols_by_name[sym_name] = sym

            aux_subs = {}
            for aux_sym, aux_def in self.auxiliary_defs.items():
                aux_name = str(aux_sym)
                # Only substitute if this is truly an auxiliary (not an original variable)
                # SKIP clock variables (T := time) - they don't need substitution in Delta
                # because we already substituted time→T in f_orig_at_Phi
                if aux_name not in orig_var_names and aux_name in recast_vars_by_name:
                    actual_aux_sym = recast_vars_by_name[aux_name]

                    # Check if this is a clock variable (definition is just 'time')
                    # Skip it - we handle time→T substitution separately above
                    if isinstance(aux_def, sp.Symbol) and str(aux_def).lower() == "time":
                        continue

                    # Substitute symbols in definition to match recast ODE symbols
                    # This includes both state variables AND parameters
                    substituted_def = aux_def

                    # Build a substitution dict for all symbols in the auxiliary definition
                    symbol_subs = {}
                    for def_sym in aux_def.free_symbols:
                        def_sym_name = str(def_sym)

                        # Check if it's a state variable
                        if def_sym_name in recast_vars_by_name:
                            symbol_subs[def_sym] = recast_vars_by_name[def_sym_name]
                        # Check if it's a parameter - use actual symbol from ODEs
                        elif def_sym_name in param_symbols_by_name:
                            symbol_subs[def_sym] = param_symbols_by_name[def_sym_name]

                    # Apply all substitutions at once
                    if symbol_subs:
                        substituted_def = substituted_def.subs(symbol_subs)

                    aux_subs[actual_aux_sym] = substituted_def

            # Apply auxiliary substitutions to Delta
            if aux_subs:
                Delta = Delta.subs(aux_subs)

            # Try to simplify each component
            simplified_components = []
            for i, component in enumerate(Delta):
                try:
                    # Normalize floating-point exponents to rational form
                    substituted = component

                    # Normalize floating-point exponents to rational form
                    # This converts 0.666666... back to 2/3, etc.
                    substituted = sp.nsimplify(substituted, rational=True, tolerance=1e-10)

                    # Now simplify with multiple aggressive strategies
                    simp = sp.simplify(substituted)

                    # Try additional simplification if not zero
                    if simp != 0:
                        # Strategy 1: Cancel and expand
                        simp = sp.cancel(sp.expand(simp))

                    if simp != 0:
                        # Strategy 2: Rational simplification (better for complex fractions)
                        simp = sp.ratsimp(simp)

                    if simp != 0:
                        # Strategy 3: Factor then simplify
                        try:
                            simp = sp.simplify(sp.factor(simp))
                        except Exception:
                            pass  # Factor may fail on some expressions

                    if simp != 0:
                        # Strategy 4: Expand and collect like terms
                        simp = sp.simplify(sp.expand(simp))

                    if simp != 0:
                        # Strategy 5: Try simplifying numerator separately for rational expressions
                        numer, denom = simp.as_numer_denom()
                        if numer != 1:  # If there's actually a numerator
                            # Expand and collect terms in numerator
                            numer_expanded = sp.expand(numer)
                            # Try to simplify the expanded numerator
                            numer_simp = sp.simplify(numer_expanded)
                            if numer_simp == 0:
                                simp = 0

                    if simp != 0:
                        # Strategy 6: Detect pattern (a - a) or (-a + a) in numerator
                        # This catches cases like (K_S - K_S) that sympy misses
                        numer, denom = simp.as_numer_denom()
                        if numer.is_Add:
                            # Expand numerator and group by structure
                            numer_expanded = sp.expand(numer)
                            # Check if all terms cancel
                            if numer_expanded == 0:
                                simp = 0
                            else:
                                # Try collecting terms - sometimes helps detect cancellation
                                collected = sp.collect(numer_expanded, numer_expanded.free_symbols)
                                if collected == 0:
                                    simp = 0

                    if simp != 0:
                        # Strategy 7: Check if numerator has a zero factor
                        # Pattern: a*b*0*c = 0, even if buried in products
                        numer, denom = simp.as_numer_denom()
                        if numer.is_Mul:
                            # Check each factor - if any is zero, the whole thing is zero
                            for factor in numer.args:
                                if factor == 0:
                                    simp = 0
                                    break
                                # Check if factor is an Add that simplifies to zero
                                if factor.is_Add:
                                    factor_simp = sp.simplify(factor)
                                    if factor_simp == 0:
                                        simp = 0
                                        break
                                    # Also try expanding it
                                    factor_expanded = sp.expand(factor)
                                    if factor_expanded == 0:
                                        simp = 0
                                        break

                    simplified_components.append(simp)
                except Exception as e:
                    return EquivalenceTest(
                        name="symbolic_equivalence",
                        result=ValidationResult.TIMEOUT,
                        details=f"Simplification timeout/error on component {i}: {e}",
                    )

            # Check if all components are zero
            all_zero = all(comp == 0 for comp in simplified_components)

            if all_zero:
                return EquivalenceTest(
                    name="symbolic_equivalence",
                    result=ValidationResult.PASS,
                    details="Symbolic proof: Δ(Z) ≡ 0 (exact equivalence)",
                )
            else:
                # Find non-zero components
                non_zero_indices = [i for i, c in enumerate(simplified_components) if c != 0]
                details = f"Non-zero components: {non_zero_indices}\n"
                details += (
                    f"Components: {[str(simplified_components[i]) for i in non_zero_indices[:3]]}"
                )

                return EquivalenceTest(
                    name="symbolic_equivalence", result=ValidationResult.FAIL, details=details
                )

        except Exception as e:
            return EquivalenceTest(
                name="symbolic_equivalence",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during symbolic test: {str(e)}",
            )

    def check_numerical_pointwise_jax(
        self,
        n_samples: int = 1000,
        domain_min: float = 0.01,
        domain_max: float = 10.0,
        threshold: float = 1e-5,
    ) -> EquivalenceTest:
        """
        Check numerical equivalence using JAX automatic differentiation.

        Computes J_Φ(Z) · f_recast(Z) and compares with f_orig(Φ(Z)) using
        JAX's autodiff for the Jacobian (no symbolic computation).

        Args:
            n_samples: Number of random sample points
            domain_min: Minimum value for each variable (log-uniform sampling)
            domain_max: Maximum value for each variable
            threshold: Error threshold for pass/fail

        Returns:
            EquivalenceTest with numerical validation results
        """
        try:
            import jax
            import jax.numpy as jnp
            from jax import jacfwd
        except ImportError:
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details="JAX not available. Install with: pip install jax jaxlib",
            )

        try:
            # Get ordered variables
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars

            # Extract parameter values
            param_values = self.recast_ir.params
            # CRITICAL: Filter out any param names that match state variable names
            # This prevents "duplicate argument 't'" errors when 't' is both a state var and param
            recast_var_names = {str(v) for v in recast_vars_ordered}
            param_names = sorted([p for p in param_values.keys() if p not in recast_var_names])
            param_vals_array = jnp.array([param_values[name] for name in param_names])

            # Build mapping functions Φ(Z) using lambdify for JAX
            # Φ maps recast variables to original variables
            phi_funcs = []
            for orig_var in orig_vars_ordered:
                mapping_expr = self.mapping[orig_var]
                # Create function that takes Z values and returns mapped value
                func = lambdify(
                    recast_vars_ordered + [sp.Symbol(p) for p in param_names],
                    mapping_expr,
                    modules="jax",
                )
                phi_funcs.append(func)

            # Build f_orig functions (original ODEs after mapping substitution)
            f_orig_funcs = []
            for orig_var in orig_vars_ordered:
                ode_expr = self.orig_odes[orig_var].subs(self.mapping)
                func = lambdify(
                    recast_vars_ordered + [sp.Symbol(p) for p in param_names],
                    ode_expr,
                    modules="jax",
                )
                f_orig_funcs.append(func)

            # Build f_recast functions (use expanded ODEs with assignment rules substituted)
            f_recast_funcs = []
            for recast_var in recast_vars_ordered:
                ode_expr = self.recast_odes_expanded[recast_var]
                func = lambdify(
                    recast_vars_ordered + [sp.Symbol(p) for p in param_names],
                    ode_expr,
                    modules="jax",
                )
                f_recast_funcs.append(func)

            # Define Φ as a JAX-compatible vector function
            def phi_vec(z_vals):
                """Φ: R^n_recast -> R^n_orig"""
                combined = jnp.concatenate([z_vals, param_vals_array])
                return jnp.array([func(*combined) for func in phi_funcs])

            # Define f_orig(Φ(Z)) as a JAX-compatible function
            def f_orig_at_phi(z_vals):
                """f_orig(Φ(Z))"""
                combined = jnp.concatenate([z_vals, param_vals_array])
                return jnp.array([func(*combined) for func in f_orig_funcs])

            # Define f_recast as a JAX-compatible function
            def f_recast_vec(z_vals):
                """f_recast(Z)"""
                combined = jnp.concatenate([z_vals, param_vals_array])
                return jnp.array([func(*combined) for func in f_recast_funcs])

            # Compute Jacobian of Φ using JAX autodiff
            jac_phi = jacfwd(phi_vec)

            errors: list[float] = []
            counterexamples: list[dict[str, Any]] = []

            # Sample points in log-uniform distribution
            np.random.seed(42)
            log_min = np.log(domain_min)
            log_max = np.log(domain_max)

            # Identify which recast variables are auxiliaries vs. original/independent
            orig_var_names = {str(v) for v in orig_vars_ordered}
            independent_vars: list[tuple[int, Any]] = []
            auxiliary_vars: list[tuple[int, Any]] = []
            auxiliary_var_indices: list[int] = []

            # Also identify clock variables (T := time) - these should be sampled, not computed
            # A clock variable is one whose definition IS exactly 'time' or 't'
            # (not just involves time, like sin(t) + 2)
            clock_vars = set()
            for aux_name, aux_def in self.auxiliary_defs.items():
                # Check if auxiliary definition IS just 'time' or 't' (identity)
                # This catches T := time but NOT Z_1 := sin(t) + 2
                if isinstance(aux_def, sp.Symbol):
                    if str(aux_def).lower() in ["time", "t"]:
                        clock_vars.add(str(aux_name))

            for i, var in enumerate(recast_vars_ordered):
                var_name = str(var)
                if var_name in orig_var_names:
                    # This is an original variable - sample independently
                    independent_vars.append((i, var))
                elif var_name in clock_vars:
                    # This is a clock variable (T := time) - sample as time
                    independent_vars.append((i, var))
                elif var_name in {str(k) for k in self.auxiliary_defs.keys()}:
                    # This is a lifted auxiliary - compute from definition
                    auxiliary_vars.append((i, var))
                    auxiliary_var_indices.append(i)
                else:
                    # Pool auxiliary from factorization - sample independently
                    independent_vars.append((i, var))

            for _ in range(n_samples):
                # Initialize array for all recast variables
                Z_sample = np.zeros(len(recast_vars_ordered))

                # Sample independent variables (originals + pool auxiliaries)
                for idx, var in independent_vars:
                    Z_sample[idx] = np.exp(np.random.uniform(log_min, log_max))

                # Compute auxiliary variables from their definitions
                for idx, var in auxiliary_vars:
                    var_name = str(var)
                    # Use the canonical symbol (var) directly as the key
                    aux_def = self.auxiliary_defs[var]

                    # Evaluate auxiliary definition
                    # Need to substitute current values of independent variables
                    # Match symbols by name to handle symbol object mismatches
                    subs_dict = {}

                    # Substitute independent variable values
                    for indep_idx, indep_var in independent_vars:
                        indep_var_name = str(indep_var)
                        # Find matching symbol in aux_def by name
                        for sym in aux_def.free_symbols:
                            if str(sym) == indep_var_name:
                                subs_dict[sym] = Z_sample[indep_idx]
                                break

                    # Substitute parameter values
                    for param_name, param_val in param_values.items():
                        # Find matching symbol in aux_def by name
                        for sym in aux_def.free_symbols:
                            if str(sym) == param_name:
                                subs_dict[sym] = param_val
                                break

                    # Evaluate - use evalf() then convert to float
                    try:
                        aux_val = float(aux_def.subs(subs_dict).evalf())
                    except Exception:
                        # Fallback: if still symbolic, use lambdify
                        aux_func = lambdify(list(aux_def.free_symbols), aux_def, modules="numpy")
                        aux_val = float(
                            aux_func(*[subs_dict.get(s, 1.0) for s in aux_def.free_symbols])
                        )

                    Z_sample[idx] = aux_val

                Z_jax = jnp.array(Z_sample)

                # Compute J_Φ(Z) using JAX autodiff
                J_at_Z = jac_phi(Z_jax)

                # Compute f_recast(Z)
                f_recast_at_Z = f_recast_vec(Z_jax)

                # Compute f_orig(Φ(Z))
                f_orig_at_Phi_Z = f_orig_at_phi(Z_jax)

                # Compute J_Φ(Z) · f_recast(Z)
                lhs = J_at_Z @ f_recast_at_Z

                # Compute error
                diff = lhs - f_orig_at_Phi_Z
                abs_error = float(jnp.max(jnp.abs(diff)))

                # Relative error
                scale = 1.0 + float(jnp.max(jnp.abs(f_orig_at_Phi_Z)))
                rel_error = abs_error / scale

                errors.append(rel_error)

                if rel_error > threshold and len(counterexamples) < 5:
                    counterexamples.append(
                        {
                            "Z": Z_sample.tolist(),
                            "error": float(rel_error),
                            "abs_error": float(abs_error),
                            "lhs": [float(x) for x in lhs],
                            "rhs": [float(x) for x in f_orig_at_Phi_Z],
                            "diff": [float(x) for x in diff],
                        }
                    )

            max_error = max(errors)
            mean_error = np.mean(errors)

            if max_error < threshold:
                return EquivalenceTest(
                    name="numerical_pointwise_jax",
                    result=ValidationResult.PASS,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"JAX autodiff: Passed with {n_samples} samples. Max error: {max_error:.2e}",
                )
            else:
                return EquivalenceTest(
                    name="numerical_pointwise_jax",
                    result=ValidationResult.FAIL,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"JAX autodiff: Failed - max error {max_error:.2e} > threshold {threshold:.2e}",
                    counterexamples=counterexamples,
                )

        except Exception as e:
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during JAX numerical test: {str(e)}",
            )

    def check_numerical_pointwise(
        self,
        n_samples: int = 1000,
        domain_min: float = 0.01,
        domain_max: float = 10.0,
        threshold: float = 1e-6,
    ) -> EquivalenceTest:
        """
        Check numerical equivalence at random sample points.

        Computes J_Φ(Z) · f_recast(Z) and compares with f_orig(Φ(Z)) using
        symbolic Jacobian (falls back to JAX if available).

        Args:
            n_samples: Number of random sample points
            domain_min: Minimum value for each variable (log-uniform sampling)
            domain_max: Maximum value for each variable
            threshold: Error threshold for pass/fail

        Returns:
            EquivalenceTest with numerical validation results
        """
        try:
            # Build symbolic Jacobian and lambdify it for speed
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars

            # Extract parameter values from recast IR
            param_values = self.recast_ir.params
            # CRITICAL: Filter out any param names that match state variable names
            # This prevents "duplicate argument 't'" errors when 't' is both a state var and param
            recast_var_names = {str(v) for v in recast_vars_ordered}
            filtered_param_names = sorted(
                [p for p in param_values.keys() if p not in recast_var_names]
            )
            # Use canonical symbols instead of creating new ones
            param_symbols = [self.canonical_symbols[name] for name in filtered_param_names]
            param_vals_ordered = [param_values[str(sym)] for sym in param_symbols]

            # Check if 'time' symbol appears in any ODE (time-dependent models)
            # Collect all free symbols from EXPANDED ODEs (assignment rules may contain 'time')
            all_ode_symbols = set()
            for ode in self.orig_odes_expanded.values():
                all_ode_symbols.update(ode.free_symbols)
            for ode in self.recast_odes_expanded.values():
                all_ode_symbols.update(ode.free_symbols)

            # Check for time symbol - ONLY 'time' (Antimony reserved keyword)
            # NOT 't' which may be a user-defined state variable (e.g., in 16_normal.ant)
            time_symbol = None
            for sym in all_ode_symbols:
                if str(sym).lower() == "time":
                    time_symbol = sym
                    break

            # If time symbol found, ensure it's in canonical symbols
            if time_symbol is not None and str(time_symbol) not in self.canonical_symbols:
                self.canonical_symbols[str(time_symbol)] = time_symbol

            # Build Φ as a vector
            Phi_vector = Matrix([self.mapping[v] for v in orig_vars_ordered])
            Z_vector = Matrix(recast_vars_ordered)

            # Compute symbolic Jacobian J_Φ = ∂Φ/∂Z
            J_Phi = Phi_vector.jacobian(Z_vector)

            # Lambdify each element of Jacobian with both state vars and params
            # ALSO include time symbol if present (for time-dependent models)
            # But ONLY if it's not already a state variable (e.g., 't' in cos_growth)
            # CRITICAL: Check by NAME, not object identity (different Symbol objects may exist)
            n_orig = len(orig_vars_ordered)
            n_recast = len(recast_vars_ordered)
            all_symbols = list(recast_vars_ordered) + param_symbols
            if time_symbol is not None and str(time_symbol) not in recast_var_names:
                all_symbols = all_symbols + [time_symbol]
            J_Phi_funcs = []
            for i in range(n_orig):
                row_funcs = []
                for j in range(n_recast):
                    elem_func = lambdify(all_symbols, J_Phi[i, j], modules="numpy")
                    row_funcs.append(elem_func)
                J_Phi_funcs.append(row_funcs)

            # Lambdify recast ODEs with params (use expanded ODEs with assignment rules substituted)
            f_recast_funcs = []
            for var in recast_vars_ordered:
                f_recast_funcs.append(
                    lambdify(all_symbols, self.recast_odes_expanded[var], modules="numpy")
                )

            # Lambdify original ODEs with mapping substituted (use expanded ODEs)
            # Also substitute time → T for time-dependent models (original uses 'time', recast uses 'T')
            clock_subs = {}
            if time_symbol is not None:
                # Find clock variable in recast (T := time)
                clock_var = None
                for aux_name, aux_def in self.auxiliary_defs.items():
                    if hasattr(aux_def, "free_symbols"):
                        aux_def_syms = {str(s).lower() for s in aux_def.free_symbols}
                        if aux_def_syms <= {"time", "t"}:
                            # This is a clock variable - find it in recast_vars
                            for rv in recast_vars_ordered:
                                if str(rv) == str(aux_name):
                                    clock_var = rv
                                    break
                            break

                # Find time symbol in ANY original ODE (not just the first)
                if clock_var is not None:
                    for var, ode in self.orig_odes_expanded.items():
                        for sym in ode.free_symbols:
                            if str(sym).lower() == "time":
                                clock_subs[sym] = clock_var
                                break
                        if clock_subs:
                            break

            f_orig_at_Phi_funcs = []
            for var in orig_vars_ordered:
                # Substitute mapping into original ODE (use expanded to substitute assignment rules)
                ode_at_phi = self.orig_odes_expanded[var].subs(self.mapping)
                # Also substitute time → T for time-dependent models
                if clock_subs:
                    ode_at_phi = ode_at_phi.subs(clock_subs)
                f_orig_at_Phi_funcs.append(lambdify(all_symbols, ode_at_phi, modules="numpy"))

            errors: list[float] = []
            counterexamples: list[dict[str, Any]] = []

            # Sample points in log-uniform distribution (positive orthant)
            np.random.seed(42)  # Reproducibility
            log_min = np.log(domain_min)
            log_max = np.log(domain_max)

            # Identify which recast variables are auxiliaries vs. original/independent
            orig_var_names = {str(v) for v in orig_vars_ordered}
            independent_vars = []
            auxiliary_vars = []
            auxiliary_var_indices = []

            # Also identify clock variables (T := time) - these should be sampled, not computed
            # A clock variable is one whose definition IS exactly 'time' or 't'
            # (not just involves time, like sin(t) + 2)
            clock_vars = set()
            for aux_name, aux_def in self.auxiliary_defs.items():
                # Check if auxiliary definition IS just 'time' or 't' (identity)
                # This catches T := time but NOT Z_1 := sin(t) + 2
                if isinstance(aux_def, sp.Symbol):
                    if str(aux_def).lower() in ["time", "t"]:
                        clock_vars.add(str(aux_name))

            for i, var in enumerate(recast_vars_ordered):
                var_name = str(var)
                if var_name in orig_var_names:
                    # This is an original variable - sample independently
                    independent_vars.append((i, var))
                elif var_name in clock_vars:
                    # This is a clock variable (T := time) - sample as time
                    independent_vars.append((i, var))
                elif var_name in {str(k) for k in self.auxiliary_defs.keys()}:
                    # This is a lifted auxiliary - compute from definition
                    auxiliary_vars.append((i, var))
                    auxiliary_var_indices.append(i)
                else:
                    # Pool auxiliary from factorization - sample independently
                    independent_vars.append((i, var))

            for _ in range(n_samples):
                # Initialize array for all recast variables
                Z_sample = np.zeros(len(recast_vars_ordered))

                # Sample time FIRST if time-dependent (needed for auxiliary evaluation)
                t_sample = None
                if time_symbol is not None:
                    # Sample time in [0.1, 100] range (log-uniform)
                    t_sample = np.exp(np.random.uniform(np.log(0.1), np.log(100.0)))

                # Sample independent variables (originals + pool auxiliaries)
                for idx, var in independent_vars:
                    Z_sample[idx] = np.exp(np.random.uniform(log_min, log_max))

                # Compute auxiliary variables from their definitions
                for idx, var in auxiliary_vars:
                    var_name = str(var)
                    # Use the canonical symbol (var) directly as the key
                    aux_def = self.auxiliary_defs[var]

                    # Evaluate auxiliary definition
                    # Need to substitute current values of independent variables
                    # Match symbols by name to handle symbol object mismatches
                    subs_dict = {}

                    # Substitute independent variable values
                    for indep_idx, indep_var in independent_vars:
                        indep_var_name = str(indep_var)
                        # Find matching symbol in aux_def by name
                        for sym in aux_def.free_symbols:
                            if str(sym) == indep_var_name:
                                subs_dict[sym] = Z_sample[indep_idx]
                                break

                    # Substitute parameter values
                    for param_name, param_val in param_values.items():
                        # Find matching symbol in aux_def by name
                        for sym in aux_def.free_symbols:
                            if str(sym) == param_name:
                                subs_dict[sym] = param_val
                                break

                    # Substitute time value if auxiliary depends on time
                    # BUT only if it's not already substituted as a state variable
                    # (e.g., t in cos_growth is a state variable, not integration time)
                    if t_sample is not None:
                        for sym in aux_def.free_symbols:
                            sym_name = str(sym).lower()
                            if sym_name == "time" or sym_name == "t":
                                # Check if this symbol was already substituted as state variable
                                if sym not in subs_dict:
                                    subs_dict[sym] = t_sample
                                break

                    # Evaluate - use evalf() then convert to float
                    try:
                        aux_val = float(aux_def.subs(subs_dict).evalf())
                    except Exception:
                        # Fallback: if still symbolic, use lambdify
                        aux_func = lambdify(list(aux_def.free_symbols), aux_def, modules="numpy")
                        aux_val = float(
                            aux_func(*[subs_dict.get(s, 1.0) for s in aux_def.free_symbols])
                        )

                    Z_sample[idx] = aux_val

                # Combine state variables and parameters for evaluation
                # Also include sampled time value if time-dependent AND time is not a state var
                # CRITICAL: Check by NAME, not object identity (different Symbol objects may exist)
                all_vals = tuple(Z_sample) + tuple(param_vals_ordered)
                if time_symbol is not None and str(time_symbol) not in recast_var_names:
                    all_vals = all_vals + (t_sample,)

                # Evaluate J_Φ(Z) element by element - returns a matrix
                J_at_Z = np.zeros((n_orig, n_recast))
                for i in range(n_orig):
                    for j in range(n_recast):
                        result = J_Phi_funcs[i][j](*all_vals)
                        # Handle both numeric and symbolic results
                        if hasattr(result, "evalf"):
                            # It's still symbolic, evaluate it
                            J_at_Z[i, j] = float(result.evalf())
                        else:
                            J_at_Z[i, j] = float(result)

                # Evaluate f_recast(Z) - returns a vector
                f_recast_at_Z = np.array([f(*all_vals) for f in f_recast_funcs], dtype=float)

                # Evaluate f_orig(Φ(Z)) - returns a vector
                f_orig_at_Phi_Z = np.array([f(*all_vals) for f in f_orig_at_Phi_funcs], dtype=float)

                # Compute J_Φ(Z) · f_recast(Z)
                lhs = J_at_Z @ f_recast_at_Z

                # Compute error: ||J_Φ · f_recast - f_orig(Φ)||
                diff = lhs - f_orig_at_Phi_Z
                abs_error = np.max(np.abs(diff))

                # Relative error
                scale = 1.0 + np.max(np.abs(f_orig_at_Phi_Z))
                rel_error = abs_error / scale

                errors.append(rel_error)

                if rel_error > threshold and len(counterexamples) < 5:
                    counterexamples.append(
                        {
                            "Z": Z_sample.tolist(),
                            "error": float(rel_error),
                            "abs_error": float(abs_error),
                            "lhs": lhs.tolist(),
                            "rhs": f_orig_at_Phi_Z.tolist(),
                            "diff": diff.tolist(),
                        }
                    )

            max_error = max(errors)
            mean_error = np.mean(errors)

            if max_error < threshold:
                return EquivalenceTest(
                    name="numerical_pointwise",
                    result=ValidationResult.PASS,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"Passed with {n_samples} samples. Max error: {max_error:.2e}",
                )
            else:
                return EquivalenceTest(
                    name="numerical_pointwise",
                    result=ValidationResult.FAIL,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"Failed: max error {max_error:.2e} > threshold {threshold:.2e}",
                    counterexamples=counterexamples,
                )

        except Exception as e:
            return EquivalenceTest(
                name="numerical_pointwise",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during numerical test: {str(e)}",
            )

    def check_trajectory_comparison(
        self, t_end: float = 1.0, n_points: int = 100, threshold: float = 3.0e-2
    ) -> EquivalenceTest:
        """
        Compare simulation trajectories between original and reconstructed recast.

        Steps:
        1. Simulate original model → X_orig(t)
        2. Simulate recast model → Z_recast(t)
        3. Reconstruct original vars: X_recast(t) = Φ(Z_recast(t))
        4. Compare X_orig vs X_recast using scaled relative error

        Error metric: |X_orig - X_recast| / (1 + max(|X_orig|, |X_recast|))
        This is bounded in [0, 1] and scale-invariant.

        Solver priority: roadrunner → scipy LSODA → rk4

        Uses @SIM metadata from original Antimony file when available:
        - T_START: simulation start time (default 0.0)
        - T_END: simulation end time (default t_end parameter)
        - N_STEPS: number of time points (default n_points parameter)

        Args:
            t_end: Default end time for simulation if @SIM not present
            n_points: Default number of time points if @SIM not present
            threshold: Error threshold for pass/fail (default 5%)

        Returns:
            EquivalenceTest with trajectory validation results
        """
        try:
            # Use @SIM metadata from original model if available
            t_end_use = self.orig_ir.sim_t_end if self.orig_ir.sim_t_end is not None else t_end
            # N_STEPS is the number of time intervals (steps), so we need N_STEPS+1 output points
            # This matches notebook_helpers.py: np.linspace(t0, t1, n_steps+1)
            n_points_use = (
                (self.orig_ir.sim_n_steps + 1) if self.orig_ir.sim_n_steps is not None else n_points
            )

            # Get initial conditions from models
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars

            # Get parameter values
            param_values = dict(self.recast_ir.params)

            # Build ODE functions for both systems
            list(recast_vars_ordered)
            [self.canonical_symbols[name] for name in sorted(param_values.keys())]

            # Check for time symbol
            all_ode_symbols = set()
            for ode in self.orig_odes.values():
                all_ode_symbols.update(ode.free_symbols)
            for ode in self.recast_odes.values():
                all_ode_symbols.update(ode.free_symbols)

            time_symbol = None
            for sym in all_ode_symbols:
                if str(sym).lower() == "time" or str(sym) == "t":
                    time_symbol = sym
                    break

            # Step 1: Simulate original model
            orig_result = self._simulate_model(
                self.orig_ir,
                self.orig_odes,
                orig_vars_ordered,
                t_end_use,
                n_points_use,
                param_values,
                time_symbol,
                "original",
            )

            if not orig_result["success"]:
                return EquivalenceTest(
                    name="trajectory_comparison",
                    result=ValidationResult.NOT_ATTEMPTED,
                    details=f"Original simulation failed: {orig_result['message']}",
                )

            # Step 2: Simulate recast model
            # Need to compute initial conditions for auxiliary variables
            recast_y0 = self._compute_recast_initial_conditions(
                recast_vars_ordered, orig_vars_ordered, param_values
            )

            recast_result = self._simulate_model(
                self.recast_ir,
                self.recast_odes,
                recast_vars_ordered,
                t_end_use,
                n_points_use,
                param_values,
                time_symbol,
                "recast",
                y0_override=recast_y0,
            )

            if not recast_result["success"]:
                return EquivalenceTest(
                    name="trajectory_comparison",
                    result=ValidationResult.NOT_ATTEMPTED,
                    details=f"Recast simulation failed: {recast_result['message']}",
                )

            # Step 3: Reconstruct original variables from recast
            t_orig = orig_result["t"]
            X_orig = orig_result["y"]  # shape: (n_points, n_orig_vars)

            t_recast = recast_result["t"]
            Z_recast = recast_result["y"]  # shape: (n_points, n_recast_vars)

            # Interpolate if time grids differ
            if len(t_orig) != len(t_recast) or not np.allclose(t_orig, t_recast):
                # Use common time grid
                from scipy.interpolate import interp1d

                t_common = t_orig
                Z_interp = np.zeros((len(t_common), len(recast_vars_ordered)))
                for j, _var in enumerate(recast_vars_ordered):
                    f = interp1d(t_recast, Z_recast[:, j], kind="linear", fill_value="extrapolate")
                    Z_interp[:, j] = f(t_common)
                Z_recast = Z_interp
            else:
                t_common = t_orig

            # Build mapping functions to reconstruct original from recast
            X_reconstructed = self._reconstruct_from_recast(
                Z_recast,
                recast_vars_ordered,
                orig_vars_ordered,
                param_values,
                t_common,
                time_symbol,
            )

            # Step 4: Compute scaled relative error
            # Normalize by characteristic scale (peak value over trajectory)
            # This matches notebook_helpers.py for consistent error reporting
            # error(t) = |X_orig - X_recast| / scale, where scale = max(peak_orig, peak_recast)
            scale = np.maximum(
                np.max(np.abs(X_orig), axis=0), np.max(np.abs(X_reconstructed), axis=0)
            )
            scale = np.maximum(scale, 1e-10)  # Floor to avoid division by zero
            errors = np.abs(X_orig - X_reconstructed) / scale[np.newaxis, :]

            max_error = float(np.max(errors))
            mean_error = float(np.mean(errors))

            # Find worst time point and variable
            worst_idx = np.unravel_index(np.argmax(errors), errors.shape)
            worst_t = t_common[worst_idx[0]]
            worst_var = str(orig_vars_ordered[worst_idx[1]])

            if max_error < threshold:
                return EquivalenceTest(
                    name="trajectory_comparison",
                    result=ValidationResult.PASS,
                    max_error=max_error,
                    mean_error=mean_error,
                    details=f"Trajectories match. Max scaled error: {max_error:.2e} at t={worst_t:.2g} ({worst_var})",
                )
            else:
                return EquivalenceTest(
                    name="trajectory_comparison",
                    result=ValidationResult.FAIL,
                    max_error=max_error,
                    mean_error=mean_error,
                    details=f"Trajectories diverge. Max scaled error: {max_error:.2e} at t={worst_t:.2g} ({worst_var})",
                    counterexamples=[
                        {
                            "t": float(worst_t),
                            "variable": worst_var,
                            "X_orig": float(X_orig[worst_idx]),
                            "X_recast": float(X_reconstructed[worst_idx]),
                            "error": float(errors[worst_idx]),
                        }
                    ],
                )

        except Exception as e:
            import traceback

            return EquivalenceTest(
                name="trajectory_comparison",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during trajectory test: {str(e)}\n{traceback.format_exc()}",
            )

    def _simulate_model(
        self,
        model_ir,
        odes,
        vars_ordered,
        t_end,
        n_points,
        param_values,
        time_symbol,
        model_name,
        y0_override=None,
    ):
        """
        Simulate a model using libRoadRunner.
        """
        return self._simulate_with_roadrunner(
            model_ir,
            odes,
            vars_ordered,
            t_end,
            n_points,
            param_values,
            time_symbol,
            model_name,
            y0_override,
        )

    def _simulate_with_roadrunner(
        self,
        model_ir,
        odes,
        vars_ordered,
        t_end,
        n_points,
        param_values,
        time_symbol,
        model_name,
        y0_override=None,
    ):
        """Simulate using RoadRunner (CVODE)."""
        try:
            from .ode_backends.roadrunner_backend import simulate_with_roadrunner

            result = simulate_with_roadrunner(
                model_ir, 0.0, t_end, n_points, y0_override=y0_override
            )

            if result["success"]:
                # Reorder columns to match vars_ordered
                state_names = result["state_names"]
                y_reordered = np.zeros((len(result["t"]), len(vars_ordered)))

                for i, var in enumerate(vars_ordered):
                    var_name = str(var)
                    if var_name in state_names:
                        j = state_names.index(var_name)
                        y_reordered[:, i] = result["y"][:, j]
                    else:
                        y_reordered[:, i] = 0.0

                return {"success": True, "t": result["t"], "y": y_reordered, "message": ""}
            else:
                return {
                    "success": False,
                    "t": np.array([]),
                    "y": np.array([]),
                    "message": f"RoadRunner simulation failed: {result.get('message', 'Unknown error')}",
                }
        except Exception as e:
            return {
                "success": False,
                "t": np.array([]),
                "y": np.array([]),
                "message": f"RoadRunner error: {str(e)}",
            }


    def _compute_recast_initial_conditions(self, recast_vars, orig_vars, param_values):
        """
        Compute initial conditions for recast model.

        Priority order:
        1. Explicit initial conditions from recast file (recast_ir.initial)
        2. For original variables: use original model ICs
        3. For auxiliaries: compute from definitions
        4. Fallback: 1.0

        This ensures clock variables (T=0) are initialized correctly from the recast file.
        """
        y0 = {}
        orig_var_names = {str(v) for v in orig_vars}

        # Get original initial values
        orig_initials: dict[str, float] = {}
        for var in orig_vars:
            var_name = str(var)
            if var_name in self.orig_ir.initial:  # type: ignore[attr-defined]
                orig_initials[var_name] = self.orig_ir.initial[var_name]  # type: ignore[attr-defined]
            else:
                orig_initials[var_name] = 1.0  # Default

        for var in recast_vars:
            var_name = str(var)

            # PRIORITY 1: Check if recast file has explicit IC for this variable
            # This handles clock variables (T=0) and any other explicit ICs
            if var_name in self.recast_ir.initial:  # type: ignore[attr-defined]
                y0[var_name] = self.recast_ir.initial[var_name]  # type: ignore[attr-defined]
            elif var_name in orig_var_names:
                # PRIORITY 2: Original variable - use original IC
                y0[var_name] = orig_initials.get(var_name, 1.0)
            elif var in self.auxiliary_defs:
                # PRIORITY 3: Auxiliary variable - compute from definition
                aux_def = self.auxiliary_defs[var]
                subs_dict = {}

                # Substitute original variable initial values
                for sym in aux_def.free_symbols:
                    sym_name = str(sym)
                    if sym_name in orig_initials:
                        subs_dict[sym] = orig_initials[sym_name]
                    elif sym_name in param_values:
                        subs_dict[sym] = param_values[sym_name]

                try:
                    y0[var_name] = float(aux_def.subs(subs_dict).evalf())
                except Exception:
                    y0[var_name] = 1.0  # Fallback
            else:
                # PRIORITY 4: Unknown variable - default
                y0[var_name] = 1.0

        return y0

    def _reconstruct_from_recast(
        self, Z_recast, recast_vars, orig_vars, param_values, t_array, time_symbol
    ):
        """
        Reconstruct original variables from recast simulation.

        Applies mapping Φ(Z) to get X values.

        Uses index-based multiplication (like notebook_helpers.py) instead of
        lambdify to avoid symbol identity mismatch issues.
        """
        n_points = len(t_array)
        n_orig = len(orig_vars)
        X_reconstructed = np.zeros((n_points, n_orig))

        # Build reconstruction functions
        recast_var_names = [str(v) for v in recast_vars]

        # Build name-to-index mapping for fast lookup
        name_to_idx = {name: idx for idx, name in enumerate(recast_var_names)}

        for i, orig_var in enumerate(orig_vars):
            mapping_expr = self.mapping[orig_var]

            # If mapping is identity, just copy
            if mapping_expr == orig_var:
                var_name = str(orig_var)
                if var_name in name_to_idx:
                    j = name_to_idx[var_name]
                    X_reconstructed[:, i] = Z_recast[:, j]
                continue

            # Check if mapping is a simple product of variables (common case)
            # This handles X = Z_1 * Z_2 * Z_3 type mappings without lambdify
            if mapping_expr.is_Mul:
                # Extract factors - check if all are symbols or powers of symbols
                all_symbols = True
                factor_indices: list[int] = []
                factor_exponents: list[float] = []

                for factor in mapping_expr.args:
                    if isinstance(factor, sp.Symbol):
                        factor_name = str(factor)
                        if factor_name in name_to_idx:
                            factor_indices.append(name_to_idx[factor_name])
                            factor_exponents.append(1)
                        else:
                            all_symbols = False
                            break
                    elif isinstance(factor, sp.Pow):
                        base, exp = factor.args
                        if isinstance(base, sp.Symbol) and str(base) in name_to_idx:
                            factor_indices.append(name_to_idx[str(base)])
                            factor_exponents.append(float(exp))
                        else:
                            all_symbols = False
                            break
                    elif factor.is_number:
                        # Numeric coefficient - will be handled by lambdify fallback
                        all_symbols = False
                        break
                    else:
                        all_symbols = False
                        break

                if all_symbols and factor_indices:
                    # Fast path: compute product directly using indices
                    prod = np.ones(n_points)
                    for idx, exp in zip(factor_indices, factor_exponents, strict=False):
                        prod *= Z_recast[:, idx] ** exp
                    X_reconstructed[:, i] = prod
                    continue

            # Check if mapping is a single symbol
            if isinstance(mapping_expr, sp.Symbol):
                sym_name = str(mapping_expr)
                if sym_name in name_to_idx:
                    X_reconstructed[:, i] = Z_recast[:, name_to_idx[sym_name]]
                    continue

            # Fallback: use lambdify for complex expressions
            # Substitute symbols in mapping_expr with canonical recast_vars symbols
            # to ensure name matching works correctly
            subs_dict = {}
            for sym in mapping_expr.free_symbols:
                sym_name = str(sym)
                if sym_name in name_to_idx:
                    # Find the actual symbol object from recast_vars
                    actual_sym = recast_vars[name_to_idx[sym_name]]
                    if sym is not actual_sym:
                        subs_dict[sym] = actual_sym

            if subs_dict:
                mapping_expr_fixed = mapping_expr.subs(subs_dict)
            else:
                mapping_expr_fixed = mapping_expr

            func = lambdify(
                list(recast_vars) + [sp.Symbol(k) for k in sorted(param_values.keys())],
                mapping_expr_fixed,
                modules="numpy",
            )

            param_vals = [param_values[k] for k in sorted(param_values.keys())]

            for t_idx in range(n_points):
                args = list(Z_recast[t_idx, :]) + param_vals
                X_reconstructed[t_idx, i] = float(func(*args))

        return X_reconstructed

    def validate(
        self,
        run_symbolic: bool = True,
        run_numerical: bool = True,
        run_trajectory: bool = True,
        use_jax: bool = False,
    ) -> ValidationReport:
        """
        Run full validation suite.

        Args:
            run_symbolic: Run symbolic equivalence test
            run_numerical: Run numerical pointwise test
            run_trajectory: Run trajectory comparison test
            use_jax: Use JAX autodiff for numerical validation (faster, no symbolic)

        Returns:
            ValidationReport with all test results
        """
        report = ValidationReport(
            original_file=self.original_file,
            recast_file=self.recast_file,
            original_class=self.orig_class,
            recast_class=self.recast_class,
            canonical_refusal_reason=self.canonical_refusal_reason,
        )

        # Run tests
        if run_symbolic:
            report.symbolic_test = self.check_symbolic_equivalence()

        if run_numerical:
            if use_jax:
                report.numerical_test = self.check_numerical_pointwise_jax()
            else:
                report.numerical_test = self.check_numerical_pointwise()

        if run_trajectory:
            report.trajectory_test = self.check_trajectory_comparison()

        # Determine overall pass/fail
        # Only REQUESTED tests must pass - if a test wasn't run, don't require it
        # If any requested test fails or times out, overall validation fails

        symbolic_pass = (
            report.symbolic_test is not None
            and report.symbolic_test.result == ValidationResult.PASS
        )
        numerical_pass = (
            report.numerical_test is not None
            and report.numerical_test.result == ValidationResult.PASS
        )
        trajectory_pass = (
            report.trajectory_test is not None
            and report.trajectory_test.result == ValidationResult.PASS
        )

        # Only require tests that were actually run
        # If a test was run, it must pass. If not run, don't count it.
        required_tests = []
        if run_symbolic:
            required_tests.append(symbolic_pass)
        if run_numerical:
            required_tests.append(numerical_pass)
        if run_trajectory:
            required_tests.append(trajectory_pass)
        
        # All REQUESTED tests must pass for overall pass
        report.overall_pass = all(required_tests) if required_tests else False

        # Generate summary
        if report.overall_pass:
            report.summary = "✓ Validation PASSED: Recast is mathematically equivalent"
        else:
            # Check if any test actually failed (vs. not attempted)
            tests = [report.symbolic_test, report.numerical_test, report.trajectory_test]
            any_fail = any(t is not None and t.result == ValidationResult.FAIL for t in tests)

            if any_fail:
                report.summary = "✗ Validation FAILED: Recast differs from original"
            else:
                report.summary = "? Validation INCONCLUSIVE: Not all tests passed or were attempted"

        return report


def validate_recast_pair(
    original_file: str,
    recast_file: str,
    factor_map: dict | None = None,
    mode: str = "simplified",
    output_json: str | None = None,
    parser: str = "legacy",
    run_symbolic: bool = True,
    run_numerical: bool = True,
    run_trajectory: bool = True,
    use_jax: bool = False,
) -> ValidationReport:
    """
    Convenience function to validate a recast.

    Args:
        original_file: Path to original Antimony file
        recast_file: Path to recast Antimony file
        factor_map: Optional factor map
        mode: Recast mode
        output_json: Optional path to save JSON report
        parser: Parser for Antimony files ('legacy' or 'sbml')
        run_symbolic: Run symbolic equivalence test
        run_numerical: Run numerical pointwise test
        run_trajectory: Run trajectory comparison test
        use_jax: Use JAX autodiff for numerical validation

    Returns:
        ValidationReport
    """
    validator = RecastValidator(original_file, recast_file, factor_map, mode, parser)
    report = validator.validate(run_symbolic, run_numerical, run_trajectory, use_jax)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report
