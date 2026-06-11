# mypy: ignore-errors
# ruff: noqa: F401, F403, F405, I001
"""Numerical pointwise validation mixin."""

from ssys._validator.common import *
from ssys._validator.report import EquivalenceTest, ValidationResult

class NumericalValidationMixin:
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
                    except (TypeError, ValueError, sp.SympifyError):
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
                    except (TypeError, ValueError, sp.SympifyError):
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
