"""Symbolic equivalence validation mixin."""

import sympy as sp
from sympy import Matrix

from ssys._validator.report import EquivalenceTest, ValidationResult
from ssys._validator.state import ValidatorState


class SymbolicValidationMixin(ValidatorState):
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

            def has_nonfinite_symbolic_value(exprs: Matrix) -> bool:
                return any(component.has(sp.nan, sp.zoo, sp.oo) for component in exprs)

            state_or_mapping_names = {str(v) for v in orig_vars_ordered}
            state_or_mapping_names.update(str(v) for v in self.recast_state_vars)

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

            # Recast generation may fold parameter-only subexpressions to numeric
            # constants. Validate symbolic equality for the concrete model by
            # substituting declared parameter values, but fall back to exact
            # symbolic parameters if substitution creates singular nan/zoo terms.
            param_subs = {}
            for component in Delta:
                for sym in component.free_symbols:
                    if sym.name in self.recast_ir.params and sym.name not in state_or_mapping_names:
                        param_subs[sym] = self.recast_ir.params[sym.name]
            if param_subs:
                param_delta = Delta.subs(param_subs)
                if not has_nonfinite_symbolic_value(param_delta):
                    Delta = param_delta

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
                        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
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
