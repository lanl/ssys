"""Public validation orchestration built from focused validation mixins."""

import json
import re
from collections.abc import Callable

import sympy as sp

from ssys._validator.mapping import MappingValidationMixin
from ssys._validator.numerical import NumericalValidationMixin
from ssys._validator.report import (
    EquivalenceTest,
    ValidationProfile,
    ValidationProfileSpec,
    ValidationReport,
    ValidationResult,
    _test_passed,
    custom_validation_profile,
    resolve_validation_profile,
)
from ssys._validator.serialization import validate_generated_output_roundtrip
from ssys._validator.symbolic import SymbolicValidationMixin
from ssys._validator.trajectory import TrajectoryValidationMixin
from ssys.classification import (
    classify_sym_system_solver_requirement,
    classify_system,
)
from ssys.parsing import parse_antimony_via_sbml
from ssys.types import SBMLParseError, SolverRequirement, SystemClass

_UNSUPPORTED_VALIDATION_PARSER_FUNCTIONS = frozenset({
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


def _required_test_names_for_profile(profile: ValidationProfileSpec) -> list[str]:
    """Return report test groups required by the selected profile."""
    tests = ["generated_output", "parser", "mapping"]
    if profile.run_symbolic:
        tests.append("symbolic")
    if profile.run_numerical:
        tests.append("numerical")
    if profile.run_trajectory:
        tests.extend(["trajectory", "algebraic_residuals"])
    if profile.run_auxiliaries:
        tests.append("auxiliaries")
    return tests


def _profile_excluded_equivalence_test(
    name: str,
    profile: ValidationProfileSpec,
) -> EquivalenceTest:
    """Represent a profile-excluded check explicitly in serialized reports."""
    return EquivalenceTest(
        name=name,
        result=ValidationResult.NOT_ATTEMPTED,
        reason="profile_excluded",
        details=f"Check is not part of validation profile {profile.name!r}",
        metadata={"validation_profile": profile.name, "required": False},
    )


def _parser_blocked_equivalence_test(
    name: str,
    parser: str,
    exception: Exception,
) -> EquivalenceTest:
    """Represent a check that could not run because validator parsing failed."""
    return EquivalenceTest(
        name=name,
        result=ValidationResult.NOT_ATTEMPTED,
        reason="parser_failed",
        details=f"Check could not run because validator parsing failed: {exception}",
        metadata={"parser": parser, "blocked_by": "validator_parser", "required": True},
    )


def _unsupported_parser_equivalence_test(
    name: str,
    parser: str,
    exception: Exception,
    unsupported_features: list[str],
) -> EquivalenceTest:
    """Represent a check blocked by unsupported parser-level validation semantics."""
    return EquivalenceTest(
        name=name,
        result=ValidationResult.NOT_ATTEMPTED,
        reason="unsupported_feature",
        details=(
            "Check could not run because validator parsing encountered unsupported "
            f"validation feature(s): {', '.join(unsupported_features)}. "
            f"Parser diagnostic: {exception}"
        ),
        metadata={
            "parser": parser,
            "blocked_by": "validator_parser",
            "required": True,
            "unsupported_features": unsupported_features,
            "exception": str(exception),
        },
    )


def _unsupported_parser_features(exception: Exception) -> list[str]:
    """Return unsupported validation features named by parser diagnostics."""
    features: set[str] = set()
    formula = getattr(exception, "formula", None)
    if isinstance(formula, str):
        for function_name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", formula):
            normalized = function_name.lower()
            if normalized in _UNSUPPORTED_VALIDATION_PARSER_FUNCTIONS:
                features.add(normalized)

    message = getattr(exception, "message", str(exception))
    match = re.search(r"unsupported function\(s\):\s*([^:]+)$", str(message))
    if match:
        for name in match.group(1).split(","):
            normalized = name.strip().lower()
            if normalized:
                features.add(normalized)

    if isinstance(exception, SBMLParseError) and exception.kind == "unsupported_feature":
        features.add("sbml")

    return sorted(features)


def _notify_progress(
    progress_callback: Callable[[str], None] | None,
    phase: str,
) -> None:
    """Best-effort validation progress hook for external timeout wrappers."""
    if progress_callback is None:
        return
    try:
        progress_callback(phase)
    except Exception:
        return


class RecastValidator(
    MappingValidationMixin,
    SymbolicValidationMixin,
    NumericalValidationMixin,
    TrajectoryValidationMixin,
):
    def __init__(
        self,
        original_file: str,
        recast_file: str,
        factor_map: dict[sp.Symbol, list[sp.Symbol]] | None = None,
        mode: str = "simplified",
        parser: str = "sbml",
        progress_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialize validator.

        Args:
            original_file: Path to original Antimony file
            recast_file: Path to recast Antimony file
            factor_map: Mapping from original to auxiliary variables (X -> [X1, X2, ...])
            mode: Recast mode ('simplified' or 'canonical')
            parser: Parser to use for Antimony files ('legacy' or 'sbml')
            progress_callback: Optional callback for parser subphase telemetry
        """
        self.original_file = original_file
        self.recast_file = recast_file
        self.mode = mode
        self.parser = parser
        self.expected_class = self._expected_class_for_mode(mode)
        self.progress_callback = progress_callback

        # Read recast file to extract mapping comments
        _notify_progress(progress_callback, "validator_parser_recast_read")
        recast_text = open(recast_file).read()
        self.recast_text = recast_text

        # Read original file text
        _notify_progress(progress_callback, "validator_parser_original_read")
        orig_text = open(original_file).read()

        # Parse both models with the SBML-first parser (reference Antimony impl)
        _notify_progress(progress_callback, "validator_parser_original_sbml")
        self.orig_system = parse_antimony_via_sbml(
            orig_text,
            progress_callback=progress_callback,
            progress_prefix="validator_parser_original_sbml",
        )
        _notify_progress(progress_callback, "validator_parser_recast_sbml")
        self.recast_system = parse_antimony_via_sbml(
            recast_text,
            progress_callback=progress_callback,
            progress_prefix="validator_parser_recast_sbml",
        )

        # Create aliases for backward compatibility with code using orig_ir/recast_ir
        # SymSystem has the same key attributes: params, assignment_rules
        # Add compatibility attributes for ModelIR interface
        _notify_progress(progress_callback, "validator_parser_compatibility_aliases")
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
        _notify_progress(progress_callback, "validator_parser_solver_requirement")
        self.orig_odes = self.orig_system.odes
        self.recast_odes = self.recast_system.odes
        self.orig_solver_requirement = classify_sym_system_solver_requirement(self.orig_system)
        self.recast_solver_requirement = classify_sym_system_solver_requirement(
            self.recast_system
        )

        # Extract mapping from comments if not provided
        _notify_progress(progress_callback, "validator_parser_mapping_comments")
        if factor_map is None:
            self.factor_map = self._extract_mapping_from_comments(recast_text)
        else:
            self.factor_map = factor_map

        # Extract auxiliary definitions from comments
        _notify_progress(progress_callback, "validator_parser_auxiliary_definitions")
        self.auxiliary_defs = self._extract_auxiliary_definitions(recast_text)

        # CRITICAL FIX: Also extract auxiliary definitions from ACTUAL assignment rules
        # (not just comment definitions). This handles lifted_mode='assignment' output
        # where Y_1 := a^2 + 1 is an actual Antimony statement, not a comment.
        self._merge_assignment_rules_as_auxiliaries()
        self._refine_recast_solver_requirement_from_auxiliaries()

        # Merge auxiliary definitions into factor_map for use in validation
        self.factor_map.update(self.auxiliary_defs)

        # Extract assignment rules from recast IR (needed for numerical validation)
        # These are expressions like J_1 := c_1 * (v_1 * p_open + v_2) * (Ca_ER - Ca)
        _notify_progress(progress_callback, "validator_parser_assignment_rules")
        self.assignment_rules = dict(self.recast_ir.assignment_rules)

        # Build mapping function Φ: Z -> X
        _notify_progress(progress_callback, "validator_parser_build_mapping")
        self._build_mapping()

        # Classify systems
        _notify_progress(progress_callback, "validator_parser_classify_systems")
        self.orig_class = classify_system(
            self.orig_system,
            progress_callback=progress_callback,
            progress_prefix="validator_parser_classify_systems:original",
        )
        self.recast_class = classify_system(
            self.recast_system,
            progress_callback=progress_callback,
            progress_prefix="validator_parser_classify_systems:recast",
        )

        # Extract refusal reason if present (for GMA outputs)
        self.canonical_refusal_reason = self._extract_refusal_reason(recast_text)

        # Canonicalize all symbols to fix symbol identity bug (K_S_orig vs K_S_recast)
        # This ensures that K_S - K_S simplifies to 0 in symbolic validation
        _notify_progress(progress_callback, "validator_parser_canonicalize_symbols")
        self._canonicalize_symbols()

        # Expand assignment rules for numerical validation (keep original for symbolic)
        # Assignment rules like J_1 := f(X, params) are symbolic in the ODEs
        # We expand them into a separate dict for lambdify to work correctly
        _notify_progress(progress_callback, "validator_parser_expand_recast_assignment_rules")
        self.recast_odes_expanded = self._expand_assignment_rules_in_odes(
            self.recast_odes, self.recast_ir
        )

        # Also expand original ODEs - they may also use assignment rules
        # Use assignment rules from ORIGINAL model
        _notify_progress(progress_callback, "validator_parser_expand_original_assignment_rules")
        self.orig_assignment_rules = dict(self.orig_ir.assignment_rules)
        self.orig_odes_expanded = self._expand_assignment_rules_in_odes(
            self.orig_odes, self.orig_ir
        )

    def _expected_class_for_mode(self, mode: str) -> SystemClass | None:
        """Return the selected target class implied by the recast mode."""
        if mode == "canonical":
            return SystemClass.CANONICAL_SSYSTEM
        if mode == "simplified":
            return SystemClass.SSYSTEM
        if mode == "gma":
            return SystemClass.GMA
        return None

    def _resolve_profile(
        self,
        profile: ValidationProfile | ValidationProfileSpec | str | None,
        *,
        run_symbolic: bool,
        run_numerical: bool,
        run_trajectory: bool,
        run_auxiliaries: bool,
    ) -> ValidationProfileSpec:
        """Resolve named profiles while preserving legacy boolean compatibility."""
        profile_spec = resolve_validation_profile(profile)
        if profile_spec is not None:
            return profile_spec
        if run_symbolic and run_numerical and run_trajectory and run_auxiliaries:
            strict = resolve_validation_profile(ValidationProfile.STRICT)
            assert strict is not None
            return strict
        return custom_validation_profile(
            run_symbolic=run_symbolic,
            run_numerical=run_numerical,
            run_trajectory=run_trajectory,
            run_auxiliaries=run_auxiliaries,
        )

    def _required_test_names(self, profile: ValidationProfileSpec) -> list[str]:
        """Return report test groups required by the selected profile."""
        return _required_test_names_for_profile(profile)

    def _profile_excluded_test(
        self,
        name: str,
        profile: ValidationProfileSpec,
    ) -> EquivalenceTest:
        """Represent a profile-excluded check explicitly in serialized reports."""
        return _profile_excluded_equivalence_test(name, profile)

    def _not_applicable_pass_test(self, name: str, details: str) -> EquivalenceTest:
        """Represent an applicable check family with no model-specific cases."""
        return EquivalenceTest(
            name=name,
            result=ValidationResult.PASS,
            details=details,
            metadata={"not_applicable": True},
        )

    def _refine_recast_solver_requirement_from_auxiliaries(self) -> None:
        """Use auxiliary definition comments to classify manifold-constrained recasts."""
        if self.recast_solver_requirement == SolverRequirement.DAE_REQUIRED:
            return
        state_names = {str(var) for var in self.recast_odes.keys()}
        assignment_rule_names = set(self.recast_ir.assignment_rules.keys())
        for aux, defn in self.auxiliary_defs.items():
            aux_name = str(aux)
            if aux_name in assignment_rule_names:
                continue
            if aux_name not in state_names:
                continue
            if self._is_clock_definition(defn):
                continue
            if any(sym.name in state_names for sym in defn.free_symbols):
                self.recast_solver_requirement = SolverRequirement.DAE_REQUIRED
                self.recast_system.solver_requirement = SolverRequirement.DAE_REQUIRED
                return

    def validate(
        self,
        run_symbolic: bool = True,
        run_numerical: bool = True,
        run_trajectory: bool = True,
        use_jax: bool = False,
        run_auxiliaries: bool = True,
        algebraic_residual_threshold: float = 1e-8,
        profile: ValidationProfile | ValidationProfileSpec | str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ValidationReport:
        """
        Run full validation suite.

        Args:
            run_symbolic: Run symbolic equivalence test
            run_numerical: Run numerical pointwise test
            run_trajectory: Run trajectory comparison test
            use_jax: Use JAX autodiff for numerical validation (faster, no symbolic)
            run_auxiliaries: Run auxiliary identity validation
            algebraic_residual_threshold: Maximum absolute algebraic residual over trajectory
            profile: Named validation profile. When set, profile flags override
                the legacy run_* boolean flags.

        Returns:
            ValidationReport with all test results
        """
        profile_spec = self._resolve_profile(
            profile,
            run_symbolic=run_symbolic,
            run_numerical=run_numerical,
            run_trajectory=run_trajectory,
            run_auxiliaries=run_auxiliaries,
        )
        run_symbolic = profile_spec.run_symbolic
        run_numerical = profile_spec.run_numerical
        run_trajectory = profile_spec.run_trajectory
        run_auxiliaries = profile_spec.run_auxiliaries

        _notify_progress(progress_callback, "generated_output_roundtrip")
        generated_output_test = validate_generated_output_roundtrip(
            self.recast_file, self.recast_text
        )
        _notify_progress(progress_callback, "mapping")
        mapping_test = self.check_mapping_complete()

        report = ValidationReport(
            original_file=self.original_file,
            recast_file=self.recast_file,
            original_class=self.orig_class,
            recast_class=self.recast_class,
            expected_class=self.expected_class,
            canonical_refusal_reason=self.canonical_refusal_reason,
            original_solver_requirement=self.orig_solver_requirement,
            recast_solver_requirement=self.recast_solver_requirement,
            validation_profile=profile_spec.name,
            validation_profile_description=profile_spec.description,
            required_tests=self._required_test_names(profile_spec),
            generated_output_test=generated_output_test,
            parser_test=EquivalenceTest(
                name="validator_parser",
                result=ValidationResult.PASS,
                details=f"Original and recast parsed with {self.parser} parser",
                metadata={
                    "parser": self.parser,
                    "original_solver_requirement": self.orig_solver_requirement.value,
                    "recast_solver_requirement": self.recast_solver_requirement.value,
                },
            ),
            mapping_test=mapping_test,
        )

        # Run tests
        if run_symbolic:
            _notify_progress(progress_callback, "symbolic")
            report.symbolic_test = self.check_symbolic_equivalence()
        else:
            report.symbolic_test = self._profile_excluded_test(
                "symbolic_equivalence", profile_spec
            )

        if run_numerical:
            _notify_progress(
                progress_callback,
                "numerical_jax" if use_jax else "numerical",
            )
            if use_jax:
                report.numerical_test = self.check_numerical_pointwise_jax(
                    progress_callback=progress_callback
                )
            else:
                report.numerical_test = self.check_numerical_pointwise(
                    progress_callback=progress_callback
                )
        else:
            report.numerical_test = self._profile_excluded_test(
                "numerical_pointwise", profile_spec
            )

        if run_trajectory:
            _notify_progress(progress_callback, "trajectory")
            report.trajectory_test = self.check_trajectory_comparison()
            _notify_progress(progress_callback, "algebraic_residuals")
            report.algebraic_residual_test = self.check_algebraic_manifold_preservation(
                threshold=algebraic_residual_threshold
            )
            if report.algebraic_residual_test is None:
                report.algebraic_residual_test = self._not_applicable_pass_test(
                    "algebraic_manifold_residuals",
                    "No algebraic definitions or constraints require residual checking",
                )
        else:
            report.trajectory_test = self._profile_excluded_test(
                "trajectory_comparison", profile_spec
            )
            report.algebraic_residual_test = self._profile_excluded_test(
                "algebraic_manifold_residuals", profile_spec
            )

        if run_auxiliaries:
            _notify_progress(progress_callback, "auxiliaries")
            report.auxiliary_tests = self.check_auxiliary_identities()
            if not report.auxiliary_tests:
                report.auxiliary_tests = [
                    self._not_applicable_pass_test(
                        "auxiliary_identities",
                        "No lifted auxiliary or observable assignment identities to validate",
                    )
                ]
        else:
            report.auxiliary_tests = [
                self._profile_excluded_test("auxiliary_identities", profile_spec)
            ]

        required_tests: list[EquivalenceTest | None] = [
            report.generated_output_test,
            report.parser_test,
            report.mapping_test,
        ]
        if run_symbolic:
            required_tests.append(report.symbolic_test)
        if run_numerical:
            required_tests.append(report.numerical_test)
        if run_trajectory:
            required_tests.append(report.trajectory_test)
            if report.algebraic_residual_test is not None:
                required_tests.append(report.algebraic_residual_test)
        if run_auxiliaries:
            required_tests.extend(report.auxiliary_tests)

        _notify_progress(progress_callback, "report_summary")
        report.overall_pass = all(_test_passed(test) for test in required_tests)
        report.overall_result = self._overall_result(required_tests)

        # Generate summary
        if report.overall_pass:
            report.summary = "Validation PASSED: recast roundtrips and required checks passed"
        else:
            if report.overall_result == ValidationResult.FAIL:
                report.summary = "Validation FAILED: at least one required check failed"
            elif report.overall_result == ValidationResult.UNSUPPORTED:
                report.summary = "Validation UNSUPPORTED: a required backend is unavailable"
            elif report.overall_result == ValidationResult.TIMEOUT:
                report.summary = "Validation TIMEOUT: a required check exceeded its limit"
            elif report.overall_result == ValidationResult.NOT_ATTEMPTED:
                report.summary = "Validation NOT ATTEMPTED: a required check was skipped"
            else:
                report.summary = "Validation INCONCLUSIVE: required checks did not all pass"

        return report

    def _overall_result(self, required_tests: list[EquivalenceTest | None]) -> ValidationResult:
        """Reduce required test statuses to one fail-closed report status."""
        if required_tests and all(_test_passed(test) for test in required_tests):
            return ValidationResult.PASS

        results = [test.result for test in required_tests if test is not None]
        if any(result == ValidationResult.FAIL for result in results):
            return ValidationResult.FAIL
        if any(result == ValidationResult.TIMEOUT for result in results):
            return ValidationResult.TIMEOUT
        if any(result == ValidationResult.UNSUPPORTED for result in results):
            return ValidationResult.UNSUPPORTED
        if any(result == ValidationResult.NOT_ATTEMPTED for result in results):
            return ValidationResult.NOT_ATTEMPTED
        return ValidationResult.INCONCLUSIVE

def validate_recast_pair(
    original_file: str,
    recast_file: str,
    factor_map: dict | None = None,
    mode: str = "simplified",
    output_json: str | None = None,
    parser: str = "sbml",
    run_symbolic: bool = True,
    run_numerical: bool = True,
    run_trajectory: bool = True,
    use_jax: bool = False,
    run_auxiliaries: bool = True,
    algebraic_residual_threshold: float = 1e-8,
    profile: ValidationProfile | ValidationProfileSpec | str | None = None,
    progress_callback: Callable[[str], None] | None = None,
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
        run_auxiliaries: Run auxiliary identity validation
        algebraic_residual_threshold: Maximum absolute residual for algebraic manifolds
        profile: Named validation profile. When set, profile flags override
            the legacy run_* boolean flags.

    Returns:
        ValidationReport
    """
    _notify_progress(progress_callback, "generated_output_roundtrip")
    generated_output_test = validate_generated_output_roundtrip(recast_file)
    try:
        _notify_progress(progress_callback, "validator_parser")
        validator_kwargs = {}
        if progress_callback is not None:
            validator_kwargs["progress_callback"] = progress_callback
        validator = RecastValidator(
            original_file,
            recast_file,
            factor_map,
            mode,
            parser,
            **validator_kwargs,
        )
        report = validator.validate(
            run_symbolic,
            run_numerical,
            run_trajectory,
            use_jax,
            run_auxiliaries,
            algebraic_residual_threshold,
            profile,
            progress_callback,
        )
    except Exception as e:
        parser_exception = e
        expected_class = None
        if mode == "canonical":
            expected_class = SystemClass.CANONICAL_SSYSTEM
        elif mode == "simplified":
            expected_class = SystemClass.SSYSTEM
        elif mode == "gma":
            expected_class = SystemClass.GMA

        unsupported_features = _unsupported_parser_features(parser_exception)
        if unsupported_features:
            parser_test = EquivalenceTest(
                name="validator_parser",
                result=ValidationResult.UNSUPPORTED,
                reason="unsupported_feature",
                details=(
                    f"Validator parser encountered unsupported validation feature(s) "
                    f"with {parser} parser: {', '.join(unsupported_features)}. "
                    f"Parser diagnostic: {parser_exception}"
                ),
                metadata={
                    "parser": parser,
                    "exception": str(parser_exception),
                    "unsupported_features": unsupported_features,
                },
            )
        else:
            parser_test = EquivalenceTest(
                name="validator_parser",
                result=ValidationResult.FAIL,
                details=f"Validator parser failed with {parser} parser: {parser_exception}",
                metadata={"parser": parser, "exception": str(parser_exception)},
            )
        failure_profile = resolve_validation_profile(profile)
        if failure_profile is None:
            if run_symbolic and run_numerical and run_trajectory and run_auxiliaries:
                failure_profile = resolve_validation_profile(ValidationProfile.STRICT)
                assert failure_profile is not None
            else:
                failure_profile = custom_validation_profile(
                    run_symbolic=run_symbolic,
                    run_numerical=run_numerical,
                    run_trajectory=run_trajectory,
                    run_auxiliaries=run_auxiliaries,
                )
        def blocked_test(name: str) -> EquivalenceTest:
            if unsupported_features:
                return _unsupported_parser_equivalence_test(
                    name,
                    parser,
                    parser_exception,
                    unsupported_features,
                )
            return _parser_blocked_equivalence_test(name, parser, parser_exception)

        mapping_test = blocked_test("mapping_completeness")
        symbolic_test = (
            blocked_test("symbolic_equivalence")
            if failure_profile.run_symbolic
            else _profile_excluded_equivalence_test("symbolic_equivalence", failure_profile)
        )
        numerical_test = (
            blocked_test("numerical_pointwise")
            if failure_profile.run_numerical
            else _profile_excluded_equivalence_test("numerical_pointwise", failure_profile)
        )
        trajectory_test = (
            blocked_test("trajectory_comparison")
            if failure_profile.run_trajectory
            else _profile_excluded_equivalence_test("trajectory_comparison", failure_profile)
        )
        algebraic_residual_test = (
            blocked_test("algebraic_manifold_residuals")
            if failure_profile.run_trajectory
            else _profile_excluded_equivalence_test(
                "algebraic_manifold_residuals", failure_profile
            )
        )
        auxiliary_tests = [
            (
                blocked_test("auxiliary_identities")
                if failure_profile.run_auxiliaries
                else _profile_excluded_equivalence_test("auxiliary_identities", failure_profile)
            )
        ]
        overall_result = (
            ValidationResult.UNSUPPORTED
            if unsupported_features
            else ValidationResult.FAIL
        )
        summary = (
            "Validation UNSUPPORTED: validator parser encountered unsupported "
            "validation feature(s)"
            if unsupported_features
            else "Validation FAILED: validator parser failed"
        )
        report = ValidationReport(
            original_file=original_file,
            recast_file=recast_file,
            original_class=None,
            recast_class=None,
            expected_class=expected_class,
            validation_profile=failure_profile.name,
            validation_profile_description=failure_profile.description,
            required_tests=_required_test_names_for_profile(failure_profile),
            generated_output_test=generated_output_test,
            parser_test=parser_test,
            mapping_test=mapping_test,
            symbolic_test=symbolic_test,
            numerical_test=numerical_test,
            trajectory_test=trajectory_test,
            algebraic_residual_test=algebraic_residual_test,
            auxiliary_tests=auxiliary_tests,
            overall_pass=False,
            overall_result=overall_result,
            summary=summary,
        )

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report
