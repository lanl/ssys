# mypy: ignore-errors
# ruff: noqa: F401, F403, F405, I001
"""Mapping and auxiliary identity validation mixins."""

from ssys._validator.common import *
from ssys._validator.report import EquivalenceTest, ValidationResult

class MappingValidationMixin:
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
                        "pow",
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
                                "pow",
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
                        except (TypeError, ValueError, sp.SympifyError):
                            # If parsing fails, skip this definition
                            continue

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
            except (TypeError, ValueError, sp.SympifyError):
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

        import re

        sympy_functions = {
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "sqrt",
            "pow",
            "sinh",
            "cosh",
            "tanh",
            "asin",
            "acos",
            "atan",
        }
        local_dict = dict(self.canonical_symbols)
        for name in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr):
            if name not in local_dict and name not in sympy_functions:
                local_dict[name] = sp.Symbol(name, positive=True)
        return self._canonical_expr(sp.sympify(expr, locals=local_dict))

    def _expressions_equivalent(
        self,
        lhs: sp.Expr,
        rhs: sp.Expr,
        extra_subs: dict[sp.Symbol, sp.Expr] | None = None,
    ) -> tuple[bool, sp.Expr]:
        """Compare expressions after canonical symbol and auxiliary substitutions."""
        lhs = self._canonical_expr(lhs)
        rhs = self._canonical_expr(rhs)
        clock_vars = self._clock_variable_by_name()
        if clock_vars:
            lhs = _substitute_symbols_by_name(lhs, clock_vars)
            rhs = _substitute_symbols_by_name(rhs, clock_vars)
        diff = lhs - rhs

        if extra_subs:
            diff = diff.subs(extra_subs)

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

        best = candidates[0]
        for candidate in candidates:
            simplified = _simplify_identity_difference(candidate)
            if simplified == 0:
                return True, sp.Integer(0)
            best = simplified

        return False, best

    def _is_clock_definition(self, expr: sp.Expr) -> bool:
        """Return True when an auxiliary definition is the integration clock."""
        return isinstance(expr, sp.Symbol) and str(expr).lower() in {"time", "t"}

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
                actual_expr = self._parse_expr_with_canonical_symbols(rule_expr)
                equivalent, residual = self._expressions_equivalent(actual_expr, expected)
                result = ValidationResult.PASS if equivalent else ValidationResult.FAIL
                details = (
                    f"{rule_name} assignment matches {kind} definition"
                    if equivalent
                    else f"{rule_name} assignment residual: {residual}"
                )
            except Exception as e:
                result = ValidationResult.INCONCLUSIVE
                details = f"Could not parse {rule_name} assignment rule {rule_expr!r}: {e}"

            tests.append(
                EquivalenceTest(
                    name=f"{kind}:{rule_name}",
                    result=result,
                    details=details,
                    metadata={"rule": rule_name, "kind": kind},
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
                        sym_name_lower = sym_name.lower()
                        if sym_name_lower in {"time", "t"}:
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

                equivalent, residual = self._expressions_equivalent(actual_ode, expected_ode)
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
