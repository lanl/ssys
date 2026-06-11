# mypy: ignore-errors
# ruff: noqa: F401, F403, F405, I001
"""Trajectory and algebraic-manifold validation mixin."""

from ssys._validator.common import *
from ssys._validator.report import EquivalenceTest, ValidationResult

class TrajectoryValidationMixin:
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
            trajectory_metadata = {
                "original_solver_requirement": self.orig_solver_requirement.value,
                "recast_solver_requirement": self.recast_solver_requirement.value,
                "original_backend": orig_result.get("backend"),
                "recast_backend": None,
            }

            if not orig_result["success"]:
                return EquivalenceTest(
                    name="trajectory_comparison",
                    result=(
                        ValidationResult.UNSUPPORTED
                        if orig_result.get("unsupported_solver_requirement")
                        else ValidationResult.NOT_ATTEMPTED
                    ),
                    details=f"Original simulation failed: {orig_result['message']}",
                    metadata=trajectory_metadata,
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
            trajectory_metadata["recast_backend"] = recast_result.get("backend")
            trajectory_metadata["algebraic_residuals"] = recast_result.get(
                "algebraic_residuals", {}
            )

            if not recast_result["success"]:
                return EquivalenceTest(
                    name="trajectory_comparison",
                    result=(
                        ValidationResult.UNSUPPORTED
                        if recast_result.get("unsupported_solver_requirement")
                        else ValidationResult.NOT_ATTEMPTED
                    ),
                    details=f"Recast simulation failed: {recast_result['message']}",
                    metadata=trajectory_metadata,
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
                    metadata=trajectory_metadata,
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
                    metadata=trajectory_metadata,
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
        Simulate a model using the solver backend selected for its requirement.
        """
        from ssys.ode_backends import simulate_model

        requirement = (
            self.orig_solver_requirement
            if model_name == "original"
            else self.recast_solver_requirement
        )
        options: dict[str, Any] = {
            "relative_tolerance": 1e-10,
            "absolute_tolerance": 1e-12,
            "maximum_num_steps": 200000,
            "max_num_steps": 200000,
        }
        if model_name == "recast":
            options["auxiliary_defs"] = dict(self.auxiliary_defs)
            if requirement == SolverRequirement.DAE_REQUIRED:
                options["repair_consistent_initial_conditions"] = True

        result = simulate_model(
            model_ir,
            0.0,
            t_end,
            n_points,
            y0_override=y0_override,
            options=options,
            solver_requirement=requirement,
        )

        if result["success"]:
            state_names = result["state_names"]
            y_reordered = np.zeros((len(result["t"]), len(vars_ordered)))

            for i, var in enumerate(vars_ordered):
                var_name = str(var)
                if var_name in state_names:
                    j = state_names.index(var_name)
                    y_reordered[:, i] = result["y"][:, j]
                else:
                    y_reordered[:, i] = 0.0

            return {
                "success": True,
                "t": result["t"],
                "y": y_reordered,
                "message": "",
                "backend": result.get("backend", "unknown"),
                "solver_requirement": result.get("solver_requirement", requirement.value),
                "unsupported_solver_requirement": False,
                "algebraic_residuals": result.get("algebraic_residuals", {}),
            }

        return {
            "success": False,
            "t": np.array([]),
            "y": np.array([]),
            "message": result.get("message", "Unknown simulation failure"),
            "backend": result.get("backend", "unknown"),
            "solver_requirement": result.get("solver_requirement", requirement.value),
            "unsupported_solver_requirement": result.get("unsupported_solver_requirement", False),
            "algebraic_residuals": result.get("algebraic_residuals", {}),
        }

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
            from ssys.ode_backends.roadrunner_backend import simulate_with_roadrunner

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
                except (TypeError, ValueError, sp.SympifyError):
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

    def _algebraic_definitions_for_residuals(self) -> dict[str, sp.Expr]:
        """Collect explicit algebraic definitions whose residuals should be monitored."""
        definitions: dict[str, sp.Expr] = {}
        for aux, defn in self.auxiliary_defs.items():
            defn_c = self._canonical_expr(defn)
            if self._is_clock_definition(defn_c):
                continue
            definitions[str(aux)] = defn_c

        for rule_name, rule_expr in self.recast_ir.assignment_rules.items():
            if rule_name in definitions:
                continue
            try:
                definitions[rule_name] = self._parse_expr_with_canonical_symbols(rule_expr)
            except (TypeError, ValueError, sp.SympifyError):
                continue

        return definitions

    def _evaluate_expr_on_recast_trajectory(
        self,
        expr: sp.Expr,
        Z_recast: np.ndarray,
        recast_vars: list[sp.Symbol],
        param_values: dict[str, float],
        t_array: np.ndarray,
    ) -> np.ndarray:
        expr = self._canonical_expr(expr)
        recast_var_names = [str(v) for v in recast_vars]
        name_to_idx = {name: idx for idx, name in enumerate(recast_var_names)}
        symbols = sorted(expr.free_symbols, key=lambda sym: sym.name)

        args: list[np.ndarray | float] = []
        for sym in symbols:
            name = sym.name
            if name in name_to_idx:
                args.append(Z_recast[:, name_to_idx[name]])
            elif name in param_values:
                args.append(float(param_values[name]))
            elif name.lower() == "time":
                args.append(t_array)
            elif name == "t" and name not in name_to_idx:
                args.append(t_array)
            else:
                raise ValueError(f"missing value for symbol {name!r}")

        if not symbols:
            return np.full_like(t_array, float(expr), dtype=float)

        func = lambdify(symbols, expr, modules="numpy")
        values = np.asarray(func(*args), dtype=float)
        if values.shape == ():
            return np.full_like(t_array, float(values), dtype=float)
        return values

    def _compute_algebraic_residual_norms(
        self,
        Z_recast: np.ndarray,
        recast_vars: list[sp.Symbol],
        param_values: dict[str, float],
        t_array: np.ndarray,
    ) -> tuple[dict[str, dict[str, float | bool]], list[dict[str, Any]]]:
        residuals: dict[str, dict[str, float | bool]] = {}
        errors: list[dict[str, Any]] = []
        recast_var_names = [str(v) for v in recast_vars]
        name_to_idx = {name: idx for idx, name in enumerate(recast_var_names)}
        assignment_rule_names = set(self.recast_ir.assignment_rules.keys())

        for name, defn in self._algebraic_definitions_for_residuals().items():
            if name not in name_to_idx:
                residuals[name] = {
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "max_scaled": 0.0,
                    "mean_scaled": 0.0,
                    "enforced_by_assignment_rule": name in assignment_rule_names,
                }
                continue

            try:
                expected = self._evaluate_expr_on_recast_trajectory(
                    defn, Z_recast, recast_vars, param_values, t_array
                )
                actual = Z_recast[:, name_to_idx[name]]
                residual = np.asarray(actual - expected, dtype=float)
                scale = np.maximum.reduce(
                    [
                        np.ones_like(residual, dtype=float),
                        np.abs(actual),
                        np.abs(expected),
                    ]
                )
                scaled_residual = np.abs(residual) / scale
                residuals[name] = {
                    "max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
                    "mean_abs": float(np.mean(np.abs(residual))) if residual.size else 0.0,
                    "max_scaled": (
                        float(np.max(scaled_residual)) if scaled_residual.size else 0.0
                    ),
                    "mean_scaled": (
                        float(np.mean(scaled_residual)) if scaled_residual.size else 0.0
                    ),
                    "enforced_by_assignment_rule": False,
                }
            except Exception as exc:
                errors.append({"constraint": name, "error": str(exc)})

        for idx, constraint in enumerate(getattr(self.recast_ir, "algebraic_constraints", []) or []):
            name = f"algebraic_constraint:{idx + 1}"
            try:
                expr = self._parse_expr_with_canonical_symbols(constraint)
                residual = self._evaluate_expr_on_recast_trajectory(
                    expr, Z_recast, recast_vars, param_values, t_array
                )
                residuals[name] = {
                    "max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
                    "mean_abs": float(np.mean(np.abs(residual))) if residual.size else 0.0,
                    "max_scaled": float(np.max(np.abs(residual))) if residual.size else 0.0,
                    "mean_scaled": float(np.mean(np.abs(residual))) if residual.size else 0.0,
                    "enforced_by_assignment_rule": False,
                }
            except Exception as exc:
                errors.append({"constraint": name, "error": str(exc)})

        return residuals, errors

    def check_algebraic_manifold_preservation(
        self,
        t_end: float = 1.0,
        n_points: int = 100,
        threshold: float = 1e-8,
    ) -> EquivalenceTest | None:
        """
        Validate algebraic-manifold residuals over a recast trajectory.

        The threshold is the maximum absolute residual allowed for each explicit
        auxiliary definition or algebraic constraint. Assignment-rule-only
        quantities that are not differential states are recorded as exactly
        enforced and do not force DAE simulation.
        """
        if not self._algebraic_definitions_for_residuals() and not getattr(
            self.recast_ir, "algebraic_constraints", []
        ):
            return None

        t_end_use = self.orig_ir.sim_t_end if self.orig_ir.sim_t_end is not None else t_end
        n_points_use = (
            (self.orig_ir.sim_n_steps + 1) if self.orig_ir.sim_n_steps is not None else n_points
        )
        orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
        recast_vars_ordered = self.recast_state_vars
        param_values = dict(self.recast_ir.params)
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
            None,
            "recast",
            y0_override=recast_y0,
        )

        metadata = {
            "threshold": threshold,
            "solver_requirement": self.recast_solver_requirement.value,
            "backend": recast_result.get("backend"),
            "residual_norms": recast_result.get("algebraic_residuals", {}),
        }

        if not recast_result["success"]:
            return EquivalenceTest(
                name="algebraic_manifold_residuals",
                result=(
                    ValidationResult.UNSUPPORTED
                    if recast_result.get("unsupported_solver_requirement")
                    else ValidationResult.NOT_ATTEMPTED
                ),
                details=f"Recast simulation failed for residual check: {recast_result['message']}",
                metadata=metadata,
            )

        residuals, errors = self._compute_algebraic_residual_norms(
            recast_result["y"],
            recast_vars_ordered,
            dict(self.recast_ir.params),
            recast_result["t"],
        )
        metadata["residual_norms"] = residuals
        if errors:
            metadata["errors"] = errors
            return EquivalenceTest(
                name="algebraic_manifold_residuals",
                result=ValidationResult.INCONCLUSIVE,
                details=f"Could not evaluate {len(errors)} algebraic residual(s)",
                metadata=metadata,
            )

        max_residual = max((float(item["max_abs"]) for item in residuals.values()), default=0.0)
        max_scaled_residual = max(
            (
                float(item.get("max_scaled", item["max_abs"]))
                for item in residuals.values()
            ),
            default=0.0,
        )
        max_effective_residual = max(
            (
                min(float(item["max_abs"]), float(item.get("max_scaled", item["max_abs"])))
                for item in residuals.values()
            ),
            default=0.0,
        )
        mean_residual = (
            float(np.mean([float(item["mean_abs"]) for item in residuals.values()]))
            if residuals
            else 0.0
        )

        if max_effective_residual <= threshold:
            return EquivalenceTest(
                name="algebraic_manifold_residuals",
                result=ValidationResult.PASS,
                max_error=max_effective_residual,
                mean_error=mean_residual,
                details=(
                    f"Algebraic manifold residuals within threshold {threshold:.1e}; "
                    f"max absolute residual {max_residual:.2e}, "
                    f"max scaled residual {max_scaled_residual:.2e}"
                ),
                metadata=metadata,
            )

        worst_name, worst = max(
            residuals.items(),
            key=lambda item: min(
                float(item[1]["max_abs"]),
                float(item[1].get("max_scaled", item[1]["max_abs"])),
            ),
        )
        worst_effective = min(
            float(worst["max_abs"]),
            float(worst.get("max_scaled", worst["max_abs"])),
        )
        return EquivalenceTest(
            name="algebraic_manifold_residuals",
            result=ValidationResult.FAIL,
            max_error=max_effective_residual,
            mean_error=mean_residual,
            details=(
                f"Algebraic manifold residual {worst_effective:.2e} exceeds "
                f"threshold {threshold:.1e} for {worst_name}"
            ),
            counterexamples=[
                {
                    "constraint": worst_name,
                    "max_abs": float(worst["max_abs"]),
                    "max_scaled": float(worst.get("max_scaled", worst["max_abs"])),
                    "threshold": threshold,
                }
            ],
            metadata=metadata,
        )
