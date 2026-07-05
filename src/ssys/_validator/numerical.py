"""Numerical pointwise validation mixin."""

import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp
from sympy import Matrix, lambdify

from ssys._recaster.names import _sanitize_antimony_name
from ssys._validator.report import EquivalenceTest, ValidationResult
from ssys._validator.state import ValidatorState

NUMERICAL_SAMPLE_SEED = 42
DEFAULT_TIME_DOMAIN = (0.1, 100.0)
NUMERICAL_SAMPLE_EVALUATION_TIMEOUT_SECONDS = 45.0
NUMERICAL_SAMPLE_EXPRESSION_MAX_OPS = 500
_UNSUPPORTED_NUMERICAL_FUNCTIONS = frozenset({
    "ceiling",
    "eq",
    "floor",
    "geq",
    "gt",
    "leq",
    "lt",
    "neq",
    "piecewise",
})


def _is_clock_symbol_name(name: str) -> bool:
    """Return True for external clock symbols, excluding state-style ``T``."""
    return name.lower() == "time" or name == "t"


class NumericalDiagnosticError(Exception):
    """Structured failure raised before a numerical comparison can run."""

    def __init__(self, message: str, *, reason: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.reason = reason
        self.metadata = metadata


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _metadata_value(value: object) -> float | str | None:
    number = _finite_float(value)
    if number is not None:
        return number
    if value is None:
        return None
    return str(value)


def _exception_reason(message: str) -> str | None:
    lowered = message.lower()
    if "unsupported numerical function" in lowered:
        return "unsupported_feature"
    if "unresolved numerical parameter" in lowered:
        return "unresolved_parameter"
    if "unresolved numerical symbol" in lowered:
        return "unresolved_symbol"
    if "invalid numerical sampling domain" in lowered:
        return "invalid_sampling_domain"
    if "non-finite value" in lowered:
        return "nonfinite_sample"
    if "division by zero" in lowered or "cannot be raised to a negative power" in lowered:
        return "singular_sample"
    return None


def _scope_normalized_identifier(name: str) -> str:
    """Collapse scoping punctuation variants used by Antimony/libSBML paths."""
    return re.sub(r"_+", "_", name)


def _unsupported_function_names(expr: sp.Expr) -> list[str]:
    names: set[str] = set()
    for applied in expr.atoms(sp.Function):
        name = applied.func.__name__
        if name.lower() in _UNSUPPORTED_NUMERICAL_FUNCTIONS:
            names.add(name)
    return sorted(names)


def _is_time_symbol_expr(expr: sp.Expr) -> bool:
    return isinstance(expr, sp.Symbol) and _is_clock_symbol_name(str(expr))


def _notify_progress(
    progress_callback: Callable[[str], None] | None,
    phase: str,
) -> None:
    """Best-effort numerical progress hook for timeout diagnostics."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase)
    except Exception:
        return


def _auxiliary_first_expressions(
    expressions: list[tuple[str, str, sp.Expr]],
) -> list[tuple[str, str, sp.Expr]]:
    """Prefer auxiliary diagnostics when the same unsupported feature appears broadly."""
    return sorted(expressions, key=lambda item: item[0] != "auxiliary")


class NumericalValidationMixin(ValidatorState):
    def _raw_parameter_values_by_name(self) -> dict[str, Any]:
        """Return original and recast parameter bindings keyed by symbol name."""
        values: dict[str, Any] = {}
        for model in (getattr(self, "orig_ir", None), getattr(self, "recast_ir", None)):
            if model is None:
                continue
            raw = getattr(model, "params", None)
            if not isinstance(raw, dict):
                continue
            for name, value in raw.items():
                values[str(name)] = value
        return values

    def _numerical_expression_symbol_names(self) -> set[str]:
        """Collect symbol names that may need parameter values for validation."""
        names: set[str] = set()
        expression_groups = (
            getattr(self, "orig_odes_expanded", {}),
            getattr(self, "recast_odes_expanded", {}),
            getattr(self, "mapping", {}),
            getattr(self, "auxiliary_defs", {}),
        )
        for group in expression_groups:
            if not isinstance(group, dict):
                continue
            for expr in group.values():
                if isinstance(expr, sp.Expr):
                    names.update(str(sym) for sym in expr.free_symbols)
        return names

    def _numerical_parameter_values(
        self,
        recast_var_names: set[str],
        clock_symbol_names: set[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """
        Build parameter bindings from both models and safe scoped-name aliases.

        BioModels validation compares expressions parsed through two paths: the
        original SBML-to-Antimony text and the generated recast artifact. Local
        parameters may be scoped with slightly different punctuation in those
        paths, for example ``reaction1_vi`` versus ``reaction1__vi``. Numerical
        validation needs values for every symbol name that survives into either
        expression set.
        """
        clock_symbol_names = {"time"} if clock_symbol_names is None else clock_symbol_names
        param_values = {}
        for name, value in self._raw_parameter_values_by_name().items():
            if name in recast_var_names:
                continue
            if name.lower() == "time" or name in clock_symbol_names:
                continue
            param_values[name] = value

        by_normalized: dict[str, list[str]] = {}
        for name in param_values:
            by_normalized.setdefault(_scope_normalized_identifier(name), []).append(name)

        by_sanitized_alias: dict[str, list[str]] = {}
        for name in param_values:
            if not name.endswith("_var"):
                continue
            base_name = name.removesuffix("_var")
            if _sanitize_antimony_name(base_name) != name:
                continue
            by_sanitized_alias.setdefault(base_name, []).append(name)
            by_sanitized_alias.setdefault(f"{base_name}_", []).append(name)

        aliases: dict[str, str] = {}
        for symbol_name in sorted(self._numerical_expression_symbol_names()):
            if symbol_name in param_values or symbol_name in recast_var_names:
                continue
            if _is_clock_symbol_name(symbol_name):
                continue
            candidates = by_sanitized_alias.get(symbol_name, [])
            if len(candidates) == 1:
                source = candidates[0]
                param_values[symbol_name] = param_values[source]
                aliases[symbol_name] = source
                continue
            candidates = by_normalized.get(_scope_normalized_identifier(symbol_name), [])
            if len(candidates) != 1:
                continue
            source = candidates[0]
            param_values[symbol_name] = param_values[source]
            aliases[symbol_name] = source

        for name in param_values:
            if str(name) not in self.canonical_symbols:
                self.canonical_symbols[str(name)] = sp.Symbol(str(name), positive=True)

        return param_values, aliases

    def _numerical_expression_symbols(self) -> set[sp.Symbol]:
        """Collect free symbols from numerical expressions and assignment rules."""
        symbols: set[sp.Symbol] = set()
        expression_groups = (
            getattr(self, "orig_odes_expanded", {}),
            getattr(self, "recast_odes_expanded", {}),
            getattr(self, "mapping", {}),
            getattr(self, "auxiliary_defs", {}),
        )
        for group in expression_groups:
            if not isinstance(group, dict):
                continue
            for expr in group.values():
                if isinstance(expr, sp.Expr):
                    symbols.update(expr.free_symbols)

        for rule_expr in dict(getattr(self.recast_ir, "assignment_rules", {}) or {}).values():
            try:
                parsed = self._parse_numerical_expression(rule_expr)
            except Exception:
                continue
            symbols.update(parsed.free_symbols)
        return symbols

    def _numerical_time_symbols(
        self,
        symbols: set[sp.Symbol],
        recast_var_names: set[str],
    ) -> list[sp.Symbol]:
        """Return external clock symbols that should share one sampled value."""
        explicit_time_symbols = [
            sym
            for sym in symbols
            if str(sym).lower() == "time" and str(sym) not in recast_var_names
        ]
        lower_t_symbols = [
            sym
            for sym in symbols
            if str(sym) == "t" and str(sym) not in recast_var_names
        ]
        time_symbols = explicit_time_symbols
        if lower_t_symbols and (
            explicit_time_symbols or "t" not in self._raw_parameter_values_by_name()
        ):
            time_symbols.extend(lower_t_symbols)
        return sorted(
            time_symbols,
            key=lambda sym: (
                str(sym) != "time",
                str(sym).lower() != "time",
                str(sym),
            ),
        )

    def _raise_for_unsupported_numerical_functions(
        self,
        expressions: list[tuple[str, str, sp.Expr]],
    ) -> None:
        for side, label, expr in expressions:
            unsupported = _unsupported_function_names(expr)
            if not unsupported:
                continue
            raise NumericalDiagnosticError(
                "Unsupported numerical function(s): " + ", ".join(unsupported),
                reason="unsupported_feature",
                metadata={
                    "side": side,
                    "expression_label": label,
                    "expression": str(expr),
                    "unsupported_functions": unsupported,
                },
            )

    def _raise_for_numerical_expression_complexity(
        self,
        *,
        side: str,
        label: str,
        expr: sp.Expr,
        max_ops: int | None,
    ) -> None:
        if max_ops is None:
            return
        expression_ops = int(sp.count_ops(expr))
        free_symbol_count = len(expr.free_symbols)
        if expression_ops <= max_ops:
            return
        raise NumericalDiagnosticError(
            "Numerical expression exceeds complexity budget",
            reason="numerical_complexity",
            metadata={
                "side": side,
                "reason": "numerical_complexity",
                "phase": "numerical_preflight",
                "active_subphase": "expression_complexity",
                "expression_label": label,
                "expression": str(expr),
                "expression_ops": expression_ops,
                "max_expression_ops": max_ops,
                "free_symbol_count": free_symbol_count,
            },
        )

    def _raise_for_unresolved_numerical_symbols(
        self,
        expressions: list[tuple[str, str, sp.Expr]],
        available_symbol_names: set[str],
        recast_var_names: set[str],
    ) -> None:
        for side, label, expr in expressions:
            missing = sorted(
                str(sym)
                for sym in expr.free_symbols
                if str(sym) not in available_symbol_names
            )
            if not missing:
                continue
            reason = (
                "unresolved_parameter"
                if all(
                    name not in recast_var_names and not _is_clock_symbol_name(name)
                    for name in missing
                )
                else "unresolved_symbol"
            )
            noun = "parameter" if reason == "unresolved_parameter" else "symbol"
            raise NumericalDiagnosticError(
                f"Unresolved numerical {noun}(s): " + ", ".join(missing),
                reason=reason,
                metadata={
                    "side": side,
                    "expression_label": label,
                    "expression": str(expr),
                    "unresolved_symbols": missing,
                },
            )

    def _record_numerical_diagnostic(
        self,
        sampling_metadata: dict[str, Any],
        diagnostic: NumericalDiagnosticError,
    ) -> dict[str, Any]:
        metadata = dict(sampling_metadata)
        metadata.setdefault("diagnostics", []).append(diagnostic.metadata)
        return metadata

    def _raise_for_auxiliary_definition_parse_errors(self) -> None:
        errors = list(getattr(self, "auxiliary_definition_parse_errors", []) or [])
        if not errors:
            return
        auxiliaries = [str(error.get("auxiliary")) for error in errors]
        raise NumericalDiagnosticError(
            "Unparsed auxiliary definition(s): " + ", ".join(auxiliaries),
            reason="unsupported_feature",
            metadata={
                "side": "auxiliary",
                "reason": "auxiliary_definition_parse_failed",
                "unparsed_auxiliary_definitions": errors,
            },
        )

    def _recast_assignment_rule_state_definitions(
        self,
        recast_vars_ordered: list[sp.Symbol],
    ) -> dict[sp.Symbol, sp.Expr]:
        """Return recast state variables that are actually assignment-rule values."""
        recast_vars_by_name = {str(var): var for var in recast_vars_ordered}
        definitions: dict[sp.Symbol, sp.Expr] = {}
        for rule_name, rule_expr in dict(
            getattr(self.recast_ir, "assignment_rules", {}) or {}
        ).items():
            recast_var = recast_vars_by_name.get(str(rule_name))
            if recast_var is None:
                continue
            try:
                definitions[recast_var] = self._parse_numerical_expression(rule_expr)
            except Exception as exc:
                raise NumericalDiagnosticError(
                    f"Unsupported recast assignment rule {rule_name!r}: {exc}",
                    reason="unsupported_feature",
                    metadata={
                        "side": "assignment",
                        "expression_label": str(rule_name),
                        "expression": str(rule_expr),
                        "exception": str(exc),
                    },
                ) from exc
        return definitions

    def _canonicalize_numerical_expression(self, expr: sp.Expr) -> sp.Expr:
        canonicalize = getattr(self, "_canonical_expr", None)
        if canonicalize is not None:
            try:
                return canonicalize(expr)
            except NotImplementedError:
                pass
        return expr

    def _parse_numerical_expression(self, expr: str | sp.Expr) -> sp.Expr:
        parse = getattr(self, "_parse_expr_with_canonical_symbols", None)
        if parse is not None:
            try:
                return parse(expr)
            except NotImplementedError:
                pass
        if isinstance(expr, sp.Expr):
            return self._canonicalize_numerical_expression(expr)
        return self._canonicalize_numerical_expression(
            sp.sympify(expr, locals=getattr(self, "canonical_symbols", {}))
        )

    def _non_state_assignment_rule_substitutions(
        self,
        recast_vars_ordered: list[sp.Symbol],
    ) -> dict[sp.Symbol, sp.Expr]:
        """Return assignment rules that should be inlined into computed definitions."""
        recast_var_names = {str(var) for var in recast_vars_ordered}
        substitutions: dict[sp.Symbol, sp.Expr] = {}
        for rule_name, rule_expr in dict(
            getattr(self.recast_ir, "assignment_rules", {}) or {}
        ).items():
            if str(rule_name) in recast_var_names:
                continue
            try:
                rule_symbol = self.canonical_symbols.get(
                    str(rule_name),
                    sp.Symbol(str(rule_name), positive=True),
                )
                substitutions[rule_symbol] = self._parse_numerical_expression(rule_expr)
            except Exception:
                continue
        return substitutions

    def _expand_numerical_assignment_rules(
        self,
        expr: sp.Expr,
        substitutions: dict[sp.Symbol, sp.Expr],
    ) -> sp.Expr:
        expanded = expr
        for _ in range(len(substitutions)):
            new_expanded = expanded.subs(substitutions)
            if new_expanded == expanded:
                break
            expanded = new_expanded
        return self._canonicalize_numerical_expression(expanded)

    def _computed_recast_variable_definitions(
        self,
        recast_vars_ordered: list[sp.Symbol],
    ) -> dict[sp.Symbol, sp.Expr]:
        """Return recast variables that must be computed before numerical checks."""
        recast_vars_by_name = {str(var): var for var in recast_vars_ordered}
        definitions: dict[sp.Symbol, sp.Expr] = {}

        for aux, aux_def in getattr(self, "auxiliary_defs", {}).items():
            recast_var = recast_vars_by_name.get(str(aux))
            if recast_var is not None:
                definitions[recast_var] = self._canonicalize_numerical_expression(aux_def)

        definitions.update(
            self._recast_assignment_rule_state_definitions(recast_vars_ordered)
        )
        substitutions = self._non_state_assignment_rule_substitutions(
            recast_vars_ordered
        )
        if substitutions:
            definitions = {
                var: self._expand_numerical_assignment_rules(expr, substitutions)
                for var, expr in definitions.items()
            }
        return definitions

    def _evaluate_computed_recast_variables(
        self,
        Z_sample: np.ndarray,
        computed_vars: list[tuple[int, sp.Symbol, sp.Expr]],
        independent_vars: list[tuple[int, sp.Symbol]],
        recast_vars_ordered: list[sp.Symbol],
        param_values: dict[str, Any],
        *,
        t_sample: float | None,
    ) -> None:
        """Evaluate computed recast variables in dependency order for one sample."""
        known_values: dict[str, float] = {
            str(var): float(Z_sample[idx]) for idx, var in independent_vars
        }
        parameter_values = {
            str(name): value for name, value in param_values.items()
        }
        pending = {str(var): (idx, var, expr) for idx, var, expr in computed_vars}

        while pending:
            progressed = False
            for var_name, (idx, _var, expr) in list(pending.items()):
                subs_dict: dict[sp.Symbol, Any] = {}
                unresolved: set[str] = set()
                for sym in expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name in known_values:
                        subs_dict[sym] = known_values[sym_name]
                    elif sym_name in parameter_values:
                        subs_dict[sym] = parameter_values[sym_name]
                    elif _is_clock_symbol_name(sym_name) and t_sample is not None:
                        subs_dict[sym] = t_sample
                    else:
                        unresolved.add(sym_name)

                if unresolved & set(pending):
                    continue

                aux_expr = expr.subs(subs_dict).evalf()
                remaining = sorted(str(sym) for sym in getattr(aux_expr, "free_symbols", []))
                if remaining:
                    reason = (
                        "unresolved_parameter"
                        if all(
                            name not in {str(var) for var in recast_vars_ordered}
                            and not _is_clock_symbol_name(name)
                            for name in remaining
                        )
                        else "unresolved_symbol"
                    )
                    noun = "parameter" if reason == "unresolved_parameter" else "symbol"
                    raise NumericalDiagnosticError(
                        f"Unresolved numerical {noun}(s): " + ", ".join(remaining),
                        reason=reason,
                        metadata={
                            "side": "auxiliary",
                            "expression_label": var_name,
                            "expression": str(expr),
                            "evaluation_expression": str(aux_expr),
                            "unresolved_symbols": remaining,
                        },
                    )

                value = float(aux_expr)
                Z_sample[idx] = value
                known_values[var_name] = value
                del pending[var_name]
                progressed = True

            if not progressed:
                var_name, (_idx, _var, expr) = next(iter(pending.items()))
                unresolved = {
                    str(sym)
                    for sym in expr.free_symbols
                    if str(sym) not in known_values
                    and str(sym) not in parameter_values
                    and not (_is_clock_symbol_name(str(sym)) and t_sample is not None)
                }
                raise NumericalDiagnosticError(
                    "Unresolved numerical symbol(s): " + ", ".join(sorted(unresolved)),
                    reason="unresolved_symbol",
                    metadata={
                        "side": "auxiliary",
                        "expression_label": var_name,
                        "expression": str(expr),
                        "unresolved_symbols": sorted(unresolved),
                    },
                )

    def _initial_values_by_name(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for model in (getattr(self, "orig_ir", None), getattr(self, "recast_ir", None)):
            if model is None:
                continue
            for attr in ("initial", "initials"):
                raw = getattr(model, attr, None)
                if not isinstance(raw, dict):
                    continue
                for name, value in raw.items():
                    number = _finite_float(value)
                    if number is not None:
                        values[str(name)] = number
        return values

    def _simulation_time_range(self) -> dict[str, float | str]:
        for model in (getattr(self, "orig_ir", None), getattr(self, "recast_ir", None)):
            if model is None:
                continue
            start = _finite_float(getattr(model, "sim_t_start", None))
            end = _finite_float(getattr(model, "sim_t_end", None))
            if end is None:
                continue
            lower = max(start if start is not None and start > 0 else 1.0e-12, 1.0e-12)
            upper = end
            if upper > lower:
                return {"min": lower, "max": upper, "source": "simulation_metadata"}
        return {
            "min": DEFAULT_TIME_DOMAIN[0],
            "max": DEFAULT_TIME_DOMAIN[1],
            "source": "default_time",
        }

    def _numerical_sampling_metadata(
        self,
        recast_vars: list[sp.Symbol],
        param_values: dict[str, Any],
        *,
        n_samples: int,
        domain_min: float,
        domain_max: float,
        threshold: float,
        include_time: bool,
    ) -> dict[str, Any]:
        domain_min_float = _finite_float(domain_min)
        domain_max_float = _finite_float(domain_max)
        if (
            domain_min_float is None
            or domain_max_float is None
            or domain_min_float <= 0
            or domain_max_float <= domain_min_float
        ):
            raise ValueError(
                "Invalid numerical sampling domain: expected finite "
                "0 < domain_min < domain_max"
            )

        initials = self._initial_values_by_name()
        variable_ranges: dict[str, dict[str, float | str]] = {}
        for var in recast_vars:
            var_name = str(var)
            initial = initials.get(var_name)
            if initial is not None and initial > 0:
                lower: float = max(
                    min(domain_min_float, initial / 10.0),
                    float(np.finfo(float).tiny),
                )
                upper: float = max(domain_max_float, initial * 10.0)
                source = "positive_initial"
            else:
                lower = domain_min_float
                upper = domain_max_float
                source = "default_positive"
            if upper <= lower:
                raise ValueError(
                    f"Invalid numerical sampling domain for {var_name}: "
                    f"min {lower} must be less than max {upper}"
                )
            variable_ranges[var_name] = {"min": lower, "max": upper, "source": source}
            if initial is not None:
                variable_ranges[var_name]["initial"] = initial

        metadata: dict[str, Any] = {
            "sample_seed": NUMERICAL_SAMPLE_SEED,
            "n_samples": n_samples,
            "threshold": threshold,
            "domain_defaults": {"min": domain_min_float, "max": domain_max_float},
            "sampling": {"state_variables": variable_ranges},
            "parameter_values": {
                str(name): _metadata_value(value) for name, value in param_values.items()
            },
        }
        if include_time:
            metadata["sampling"]["time"] = self._simulation_time_range()
        return metadata

    def _sample_log_uniform(
        self,
        rng: np.random.Generator,
        sampling_metadata: dict[str, Any],
        variable: sp.Symbol,
    ) -> float:
        variable_range = sampling_metadata["sampling"]["state_variables"][str(variable)]
        return float(
            np.exp(rng.uniform(np.log(variable_range["min"]), np.log(variable_range["max"])))
        )

    def check_numerical_pointwise_jax(
        self,
        n_samples: int = 1000,
        domain_min: float = 0.01,
        domain_max: float = 10.0,
        threshold: float = 1e-5,
        progress_callback: Callable[[str], None] | None = None,
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
            import jax.numpy as jnp
            from jax import jacfwd
        except ImportError:
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details="JAX not available. Install with: pip install jax jaxlib",
            )

        sampling_metadata: dict[str, Any] = {}
        try:
            _notify_progress(progress_callback, "numerical_jax_setup")
            # Get ordered variables
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars

            # CRITICAL: Filter out any param names that match state variable names
            # This prevents "duplicate argument 't'" errors when 't' is both a state var and param
            recast_var_names = {str(v) for v in recast_vars_ordered}
            param_values, parameter_aliases = self._numerical_parameter_values(
                recast_var_names
            )
            param_names = sorted([p for p in param_values.keys() if p not in recast_var_names])
            param_vals_array = jnp.array([param_values[name] for name in param_names])
            sampling_metadata = self._numerical_sampling_metadata(
                recast_vars_ordered,
                param_values,
                n_samples=n_samples,
                domain_min=domain_min,
                domain_max=domain_max,
                threshold=threshold,
                include_time=False,
            )
            if parameter_aliases:
                sampling_metadata["parameter_aliases"] = parameter_aliases

            param_symbols = [self.canonical_symbols[name] for name in param_names]

            # Build mapping functions Φ(Z) using lambdify for JAX
            # Φ maps recast variables to original variables
            _notify_progress(progress_callback, "numerical_jax_lambdify_mapping")
            phi_funcs = []
            preflight_expressions: list[tuple[str, str, sp.Expr]] = []
            for orig_var in orig_vars_ordered:
                mapping_expr = self.mapping[orig_var]
                preflight_expressions.append(("mapping", str(orig_var), mapping_expr))
                # Create function that takes Z values and returns mapped value
                func = lambdify(
                    recast_vars_ordered + param_symbols,
                    mapping_expr,
                    modules="jax",
                )
                phi_funcs.append(func)

            # Build f_orig functions (original ODEs after mapping substitution)
            _notify_progress(progress_callback, "numerical_jax_lambdify_original_odes")
            f_orig_funcs = []
            for orig_var in orig_vars_ordered:
                ode_expr = self.orig_odes_expanded[orig_var].subs(self.mapping)
                preflight_expressions.append(("original", str(orig_var), ode_expr))
                func = lambdify(
                    recast_vars_ordered + param_symbols,
                    ode_expr,
                    modules="jax",
                )
                f_orig_funcs.append(func)

            # Build f_recast functions (use expanded ODEs with assignment rules substituted)
            _notify_progress(progress_callback, "numerical_jax_lambdify_recast_odes")
            f_recast_funcs = []
            for recast_var in recast_vars_ordered:
                ode_expr = self.recast_odes_expanded[recast_var]
                preflight_expressions.append(("recast", str(recast_var), ode_expr))
                func = lambdify(
                    recast_vars_ordered + param_symbols,
                    ode_expr,
                    modules="jax",
                )
                f_recast_funcs.append(func)
            computed_definitions = self._computed_recast_variable_definitions(
                recast_vars_ordered
            )
            for aux, aux_def in computed_definitions.items():
                if not _is_time_symbol_expr(aux_def):
                    preflight_expressions.append(("auxiliary", str(aux), aux_def))
            preflight_expressions = _auxiliary_first_expressions(preflight_expressions)
            all_symbol_names = {str(sym) for sym in recast_vars_ordered + param_symbols}
            self._raise_for_unsupported_numerical_functions(preflight_expressions)
            self._raise_for_unresolved_numerical_symbols(
                preflight_expressions,
                all_symbol_names,
                recast_var_names,
            )
            _notify_progress(progress_callback, "numerical_jax_autodiff_setup")
            self._raise_for_auxiliary_definition_parse_errors()

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

            # Sample points in log-uniform distribution.
            _notify_progress(progress_callback, "numerical_jax_sample_evaluation")
            rng = np.random.default_rng(NUMERICAL_SAMPLE_SEED)

            # Identify which recast variables are auxiliaries vs. original/independent
            orig_var_names = {str(v) for v in orig_vars_ordered}
            independent_vars: list[tuple[int, Any]] = []
            auxiliary_vars: list[tuple[int, sp.Symbol, sp.Expr]] = []

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
                elif var in computed_definitions:
                    # This is a lifted auxiliary - compute from definition
                    auxiliary_vars.append((i, var, computed_definitions[var]))
                else:
                    # Pool auxiliary from factorization - sample independently
                    independent_vars.append((i, var))

            for _ in range(n_samples):
                # Initialize array for all recast variables
                Z_sample = np.zeros(len(recast_vars_ordered))

                # Sample independent variables (originals + pool auxiliaries)
                for idx, var in independent_vars:
                    Z_sample[idx] = self._sample_log_uniform(
                        rng, sampling_metadata, var
                    )

                self._evaluate_computed_recast_variables(
                    Z_sample,
                    auxiliary_vars,
                    independent_vars,
                    recast_vars_ordered,
                    param_values,
                    t_sample=None,
                )

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
                with np.errstate(invalid="ignore", over="ignore"):
                    diff = lhs - f_orig_at_Phi_Z
                if not all(
                    np.all(np.isfinite(np.asarray(values, dtype=float)))
                    for values in (lhs, f_orig_at_Phi_Z, diff)
                ):
                    raise ValueError("Non-finite value encountered during JAX numerical test")
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
            _notify_progress(progress_callback, "numerical_jax_result_aggregation")

            if max_error < threshold:
                return EquivalenceTest(
                    name="numerical_pointwise_jax",
                    result=ValidationResult.PASS,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"JAX autodiff: Passed with {n_samples} samples. Max error: {max_error:.2e}",
                    metadata=sampling_metadata,
                )
            else:
                return EquivalenceTest(
                    name="numerical_pointwise_jax",
                    result=ValidationResult.FAIL,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"JAX autodiff: Failed - max error {max_error:.2e} > threshold {threshold:.2e}",
                    counterexamples=counterexamples,
                    metadata=sampling_metadata,
                )

        except NumericalDiagnosticError as e:
            message = str(e)
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during JAX numerical test: {message}",
                metadata=self._record_numerical_diagnostic(sampling_metadata, e),
                reason=e.reason,
            )
        except Exception as e:
            message = str(e)
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during JAX numerical test: {message}",
                metadata=sampling_metadata,
                reason=_exception_reason(message),
            )

    def check_numerical_pointwise(
        self,
        n_samples: int = 1000,
        domain_min: float = 0.01,
        domain_max: float = 10.0,
        threshold: float = 1e-6,
        sample_evaluation_timeout: float | None = (
            NUMERICAL_SAMPLE_EVALUATION_TIMEOUT_SECONDS
        ),
        expression_complexity_limit: int | None = NUMERICAL_SAMPLE_EXPRESSION_MAX_OPS,
        progress_callback: Callable[[str], None] | None = None,
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
        sampling_metadata: dict[str, Any] = {}
        try:
            _notify_progress(progress_callback, "numerical_setup")
            # Build symbolic Jacobian and lambdify it for speed
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars

            # CRITICAL: Filter out any param names that match state variable names
            # This prevents "duplicate argument 't'" errors when 't' is both a state var and param
            recast_var_names = {str(v) for v in recast_vars_ordered}

            _notify_progress(progress_callback, "numerical_collect_symbols")
            # Check if 'time' symbol appears in any ODE (time-dependent models)
            # Collect all free symbols from EXPANDED ODEs (assignment rules may contain 'time')
            all_ode_symbols = self._numerical_expression_symbols()
            time_symbols = self._numerical_time_symbols(
                all_ode_symbols,
                recast_var_names,
            )
            time_symbol = time_symbols[0] if time_symbols else None
            time_subs = {
                sym: time_symbol
                for sym in time_symbols
                if time_symbol is not None and sym != time_symbol
            }

            # If time symbol found, ensure it's in canonical symbols
            if time_symbol is not None and str(time_symbol) not in self.canonical_symbols:
                self.canonical_symbols[str(time_symbol)] = time_symbol

            _notify_progress(progress_callback, "numerical_parameters")
            param_values, parameter_aliases = self._numerical_parameter_values(
                recast_var_names,
                {str(sym) for sym in time_symbols},
            )
            filtered_param_names = sorted(
                [p for p in param_values.keys() if p not in recast_var_names]
            )
            # Use canonical symbols instead of creating new ones
            param_symbols = [self.canonical_symbols[name] for name in filtered_param_names]
            param_vals_ordered = [param_values[str(sym)] for sym in param_symbols]

            _notify_progress(progress_callback, "numerical_sampling_metadata")
            sampling_metadata = self._numerical_sampling_metadata(
                recast_vars_ordered,
                param_values,
                n_samples=n_samples,
                domain_min=domain_min,
                domain_max=domain_max,
                threshold=threshold,
                include_time=time_symbol is not None,
            )
            if parameter_aliases:
                sampling_metadata["parameter_aliases"] = parameter_aliases

            # Build Φ as a vector
            _notify_progress(progress_callback, "numerical_jacobian")
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
            all_symbol_names = {str(sym) for sym in all_symbols}
            preflight_expressions: list[tuple[str, str, sp.Expr]] = [
                ("mapping", str(var), self.mapping[var].subs(time_subs))
                for var in orig_vars_ordered
            ]
            _notify_progress(progress_callback, "numerical_lambdify_jacobian")
            J_Phi_funcs = []
            for i in range(n_orig):
                for j in range(n_recast):
                    elem = J_Phi[i, j]
                    if elem == 0:
                        continue
                    self._raise_for_numerical_expression_complexity(
                        side="jacobian",
                        label=f"J_phi[{i},{j}]",
                        expr=elem,
                        max_ops=expression_complexity_limit,
                    )
                    elem_func = lambdify(all_symbols, elem, modules="numpy")
                    J_Phi_funcs.append((i, j, elem_func))

            # Lambdify recast ODEs with params (use expanded ODEs with assignment rules substituted)
            _notify_progress(progress_callback, "numerical_lambdify_recast_odes")
            f_recast_funcs = []
            for var in recast_vars_ordered:
                recast_ode = self.recast_odes_expanded[var].subs(time_subs)
                preflight_expressions.append(("recast", str(var), recast_ode))
                self._raise_for_numerical_expression_complexity(
                    side="recast",
                    label=str(var),
                    expr=recast_ode,
                    max_ops=expression_complexity_limit,
                )
                f_recast_funcs.append(
                    lambdify(all_symbols, recast_ode, modules="numpy")
                )

            # Lambdify original ODEs with mapping substituted (use expanded ODEs)
            # Also substitute time → T for time-dependent models (original uses 'time', recast uses 'T')
            _notify_progress(progress_callback, "numerical_lambdify_original_odes")
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

            f_orig_at_Phi_funcs = []
            for var in orig_vars_ordered:
                # Substitute mapping into original ODE (use expanded to substitute assignment rules)
                ode_at_phi = self.orig_odes_expanded[var].subs(self.mapping)
                # Also substitute time → T for time-dependent models
                if clock_subs:
                    ode_at_phi = ode_at_phi.subs(clock_subs)
                elif time_subs:
                    ode_at_phi = ode_at_phi.subs(time_subs)
                preflight_expressions.append(("original", str(var), ode_at_phi))
                self._raise_for_numerical_expression_complexity(
                    side="original",
                    label=str(var),
                    expr=ode_at_phi,
                    max_ops=expression_complexity_limit,
                )
                f_orig_at_Phi_funcs.append(lambdify(all_symbols, ode_at_phi, modules="numpy"))
            _notify_progress(progress_callback, "numerical_computed_definitions")
            computed_definitions = self._computed_recast_variable_definitions(
                recast_vars_ordered
            )
            if time_subs:
                computed_definitions = {
                    var: expr.subs(time_subs) for var, expr in computed_definitions.items()
                }
            for aux, aux_def in computed_definitions.items():
                if not _is_time_symbol_expr(aux_def):
                    preflight_expressions.append(("auxiliary", str(aux), aux_def))
                    self._raise_for_numerical_expression_complexity(
                        side="auxiliary",
                        label=str(aux),
                        expr=aux_def,
                        max_ops=expression_complexity_limit,
                    )
            _notify_progress(progress_callback, "numerical_preflight")
            self._raise_for_auxiliary_definition_parse_errors()
            preflight_expressions = _auxiliary_first_expressions(preflight_expressions)
            self._raise_for_unsupported_numerical_functions(preflight_expressions)
            self._raise_for_unresolved_numerical_symbols(
                preflight_expressions,
                all_symbol_names,
                recast_var_names,
            )

            errors: list[float] = []
            counterexamples: list[dict[str, Any]] = []

            # Sample points in log-uniform distribution (positive orthant).
            _notify_progress(progress_callback, "numerical_sample_evaluation")
            sample_evaluation_started_at = time.monotonic()
            sample_progress_stride = max(1, min(50, n_samples))
            active_subphase = "sample_setup"
            active_label: str | None = None

            def notify_sample_subphase(
                sample_index: int,
                subphase: str,
                label: str | None = None,
            ) -> None:
                nonlocal active_subphase, active_label
                active_subphase = subphase
                active_label = label
                if (
                    sample_index == 0
                    or sample_index == n_samples - 1
                    or sample_index % sample_progress_stride == 0
                ):
                    phase = f"numerical_sample_evaluation:{subphase}"
                    if label:
                        phase = f"{phase}:{label}"
                    _notify_progress(progress_callback, phase)

            def check_sample_evaluation_budget(
                sample_index: int,
                samples_completed: int,
            ) -> None:
                if sample_evaluation_timeout is None:
                    return
                elapsed = time.monotonic() - sample_evaluation_started_at
                # Use a strict comparison so a zero-second budget always trips,
                # even where time.monotonic() has coarse resolution (Windows)
                # and reports elapsed == 0.0 at the first check. Fail-closed at
                # the exact tie matches the complexity-budget philosophy.
                if elapsed < sample_evaluation_timeout:
                    return
                raise NumericalDiagnosticError(
                    "Numerical sample evaluation exceeded complexity budget",
                    reason="numerical_complexity",
                    metadata={
                        "side": "numerical",
                        "reason": "numerical_complexity",
                        "phase": "numerical_sample_evaluation",
                        "active_subphase": active_subphase,
                        "active_expression_label": active_label,
                        "sample_index": sample_index,
                        "samples_completed": samples_completed,
                        "n_samples": n_samples,
                        "elapsed_seconds": round(elapsed, 6),
                        "limit_seconds": sample_evaluation_timeout,
                    },
                )

            rng = np.random.default_rng(NUMERICAL_SAMPLE_SEED)

            # Identify which recast variables are auxiliaries vs. original/independent
            orig_var_names = {str(v) for v in orig_vars_ordered}
            independent_vars = []
            auxiliary_vars = []

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
                elif var in computed_definitions:
                    # This is a lifted auxiliary - compute from definition
                    auxiliary_vars.append((i, var, computed_definitions[var]))
                else:
                    # Pool auxiliary from factorization - sample independently
                    independent_vars.append((i, var))

            for sample_index in range(n_samples):
                notify_sample_subphase(sample_index, "sample_generation")
                # Initialize array for all recast variables
                Z_sample = np.zeros(len(recast_vars_ordered))

                # Sample time FIRST if time-dependent (needed for auxiliary evaluation)
                t_sample = None
                if time_symbol is not None:
                    time_range = sampling_metadata["sampling"]["time"]
                    t_sample = float(
                        np.exp(
                            rng.uniform(np.log(time_range["min"]), np.log(time_range["max"]))
                        )
                    )

                # Sample independent variables (originals + pool auxiliaries)
                for idx, var in independent_vars:
                    Z_sample[idx] = self._sample_log_uniform(
                        rng, sampling_metadata, var
                    )
                check_sample_evaluation_budget(sample_index, sample_index)

                notify_sample_subphase(sample_index, "computed_variables")
                self._evaluate_computed_recast_variables(
                    Z_sample,
                    auxiliary_vars,
                    independent_vars,
                    recast_vars_ordered,
                    param_values,
                    t_sample=t_sample,
                )
                check_sample_evaluation_budget(sample_index, sample_index)

                # Combine state variables and parameters for evaluation
                # Also include sampled time value if time-dependent AND time is not a state var
                # CRITICAL: Check by NAME, not object identity (different Symbol objects may exist)
                all_vals = tuple(Z_sample) + tuple(param_vals_ordered)
                if time_symbol is not None and str(time_symbol) not in recast_var_names:
                    all_vals = all_vals + (t_sample,)

                # Evaluate J_Φ(Z) element by element - returns a matrix
                J_at_Z = np.zeros((n_orig, n_recast))
                for i, j, elem_func in J_Phi_funcs:
                    notify_sample_subphase(
                        sample_index,
                        "jacobian_evaluation",
                        f"J_phi[{i},{j}]",
                    )
                    result = elem_func(*all_vals)
                    # Handle both numeric and symbolic results
                    if hasattr(result, "evalf"):
                        # It's still symbolic, evaluate it
                        J_at_Z[i, j] = float(result.evalf())
                    else:
                        J_at_Z[i, j] = float(result)
                    check_sample_evaluation_budget(sample_index, sample_index)

                # Evaluate f_recast(Z) - returns a vector
                f_recast_values = []
                for var, func in zip(recast_vars_ordered, f_recast_funcs):
                    notify_sample_subphase(
                        sample_index,
                        "recast_ode_evaluation",
                        str(var),
                    )
                    f_recast_values.append(func(*all_vals))
                    check_sample_evaluation_budget(sample_index, sample_index)
                f_recast_at_Z = np.array(f_recast_values, dtype=float)

                # Evaluate f_orig(Φ(Z)) - returns a vector
                f_orig_values = []
                for var, func in zip(orig_vars_ordered, f_orig_at_Phi_funcs):
                    notify_sample_subphase(
                        sample_index,
                        "original_ode_evaluation",
                        str(var),
                    )
                    f_orig_values.append(func(*all_vals))
                    check_sample_evaluation_budget(sample_index, sample_index)
                f_orig_at_Phi_Z = np.array(f_orig_values, dtype=float)

                notify_sample_subphase(sample_index, "error_evaluation")
                # Compute J_Φ(Z) · f_recast(Z)
                lhs = J_at_Z @ f_recast_at_Z

                # Compute error: ||J_Φ · f_recast - f_orig(Φ)||
                with np.errstate(invalid="ignore", over="ignore"):
                    diff = lhs - f_orig_at_Phi_Z
                if not all(
                    np.all(np.isfinite(np.asarray(values, dtype=float)))
                    for values in (lhs, f_orig_at_Phi_Z, diff)
                ):
                    raise ValueError("Non-finite value encountered during numerical test")
                abs_error = np.max(np.abs(diff))

                # Relative error
                scale = 1.0 + np.max(np.abs(f_orig_at_Phi_Z))
                rel_error = abs_error / scale

                errors.append(rel_error)
                check_sample_evaluation_budget(sample_index, sample_index + 1)

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
            _notify_progress(progress_callback, "numerical_result_aggregation")

            if max_error < threshold:
                return EquivalenceTest(
                    name="numerical_pointwise",
                    result=ValidationResult.PASS,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"Passed with {n_samples} samples. Max error: {max_error:.2e}",
                    metadata=sampling_metadata,
                )
            else:
                return EquivalenceTest(
                    name="numerical_pointwise",
                    result=ValidationResult.FAIL,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"Failed: max error {max_error:.2e} > threshold {threshold:.2e}",
                    counterexamples=counterexamples,
                    metadata=sampling_metadata,
                )

        except NumericalDiagnosticError as e:
            message = str(e)
            return EquivalenceTest(
                name="numerical_pointwise",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during numerical test: {message}",
                metadata=self._record_numerical_diagnostic(sampling_metadata, e),
                reason=e.reason,
            )
        except Exception as e:
            message = str(e)
            return EquivalenceTest(
                name="numerical_pointwise",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during numerical test: {message}",
                metadata=sampling_metadata,
                reason=_exception_reason(message),
            )
