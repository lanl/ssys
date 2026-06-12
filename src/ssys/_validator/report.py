"""Validation result and report data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ssys.types import SolverRequirement, SystemClass

VALIDATION_REPORT_SCHEMA_VERSION = "1.0"


class ValidationResult(Enum):
    """Validation test outcomes."""

    PASS = "pass"
    FAIL = "failed"
    TIMEOUT = "timeout"
    NOT_ATTEMPTED = "not_attempted"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


PASSING_RESULTS = frozenset({ValidationResult.PASS})


class ValidationProfile(Enum):
    """Named validation profiles for user-facing validation workflows."""

    STRICT = "strict"
    STRUCTURAL = "structural"
    SYMBOLIC = "symbolic"
    NUMERICAL = "numerical"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class ValidationProfileSpec:
    """Resolved validation profile behavior."""

    name: str
    description: str
    run_symbolic: bool
    run_numerical: bool
    run_trajectory: bool
    run_auxiliaries: bool


VALIDATION_PROFILES: dict[ValidationProfile, ValidationProfileSpec] = {
    ValidationProfile.STRICT: ValidationProfileSpec(
        name=ValidationProfile.STRICT.value,
        description=(
            "Release-grade validation: generated output, parser, mapping, symbolic, "
            "numerical, trajectory, algebraic residual, and auxiliary identity checks."
        ),
        run_symbolic=True,
        run_numerical=True,
        run_trajectory=True,
        run_auxiliaries=True,
    ),
    ValidationProfile.STRUCTURAL: ValidationProfileSpec(
        name=ValidationProfile.STRUCTURAL.value,
        description=(
            "Fast structural smoke: generated output, parser, and reconstruction mapping."
        ),
        run_symbolic=False,
        run_numerical=False,
        run_trajectory=False,
        run_auxiliaries=False,
    ),
    ValidationProfile.SYMBOLIC: ValidationProfileSpec(
        name=ValidationProfile.SYMBOLIC.value,
        description=(
            "Symbolic proof profile: generated output, parser, mapping, symbolic "
            "equivalence, and auxiliary identity checks."
        ),
        run_symbolic=True,
        run_numerical=False,
        run_trajectory=False,
        run_auxiliaries=True,
    ),
    ValidationProfile.NUMERICAL: ValidationProfileSpec(
        name=ValidationProfile.NUMERICAL.value,
        description=(
            "Pointwise numerical support profile: generated output, parser, mapping, "
            "numerical equivalence, and auxiliary identity checks."
        ),
        run_symbolic=False,
        run_numerical=True,
        run_trajectory=False,
        run_auxiliaries=True,
    ),
    ValidationProfile.TRAJECTORY: ValidationProfileSpec(
        name=ValidationProfile.TRAJECTORY.value,
        description=(
            "Trajectory support profile: generated output, parser, mapping, trajectory, "
            "algebraic residual, and auxiliary identity checks."
        ),
        run_symbolic=False,
        run_numerical=False,
        run_trajectory=True,
        run_auxiliaries=True,
    ),
}


def validation_profile_choices() -> list[str]:
    """Return stable profile names for CLI argument choices."""
    return [profile.value for profile in ValidationProfile]


def resolve_validation_profile(
    profile: ValidationProfile | ValidationProfileSpec | str | None,
) -> ValidationProfileSpec | None:
    """Resolve a user-facing profile value to a profile spec."""
    if profile is None:
        return None
    if isinstance(profile, ValidationProfileSpec):
        return profile
    if isinstance(profile, ValidationProfile):
        return VALIDATION_PROFILES[profile]
    try:
        normalized = ValidationProfile(str(profile).lower().replace("_", "-"))
    except ValueError as exc:
        choices = ", ".join(validation_profile_choices())
        raise ValueError(f"Unknown validation profile {profile!r}; expected one of: {choices}") from exc
    return VALIDATION_PROFILES[normalized]


def custom_validation_profile(
    *,
    run_symbolic: bool,
    run_numerical: bool,
    run_trajectory: bool,
    run_auxiliaries: bool,
) -> ValidationProfileSpec:
    """Create a compatibility profile from legacy boolean flags."""
    return ValidationProfileSpec(
        name="custom",
        description="Compatibility profile derived from explicit validation boolean flags.",
        run_symbolic=run_symbolic,
        run_numerical=run_numerical,
        run_trajectory=run_trajectory,
        run_auxiliaries=run_auxiliaries,
    )


NON_PASS_REASON_DEFAULTS = {
    ValidationResult.FAIL: "failed",
    ValidationResult.TIMEOUT: "timeout",
    ValidationResult.NOT_ATTEMPTED: "not_attempted",
    ValidationResult.UNSUPPORTED: "unsupported",
    ValidationResult.INCONCLUSIVE: "inconclusive",
}


def _test_passed(test: "EquivalenceTest | None") -> bool:
    return test is not None and test.result in PASSING_RESULTS


@dataclass
class EquivalenceTest:
    """Results from a single equivalence test."""

    name: str
    result: ValidationResult
    max_error: float | None = None
    mean_error: float | None = None
    details: str = ""
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        """Ensure non-pass outcomes carry a machine-readable reason code."""
        if self.result != ValidationResult.PASS and self.reason is None:
            self.reason = NON_PASS_REASON_DEFAULTS[self.result]


@dataclass
class ValidationReport:
    """Complete validation report for an original/recast pair."""

    original_file: str
    recast_file: str

    # Structural classification
    original_class: SystemClass | None
    recast_class: SystemClass | None
    expected_class: SystemClass | None = None
    canonical_refusal_reason: str | None = None
    original_solver_requirement: SolverRequirement | None = None
    recast_solver_requirement: SolverRequirement | None = None

    # Test results
    generated_output_test: EquivalenceTest | None = None
    parser_test: EquivalenceTest | None = None
    mapping_test: EquivalenceTest | None = None
    symbolic_test: EquivalenceTest | None = None
    numerical_test: EquivalenceTest | None = None
    trajectory_test: EquivalenceTest | None = None
    algebraic_residual_test: EquivalenceTest | None = None
    auxiliary_tests: list[EquivalenceTest] = field(default_factory=list)

    # Overall verdict
    overall_pass: bool = False
    overall_result: ValidationResult = ValidationResult.INCONCLUSIVE
    summary: str = ""
    validation_profile: str = "custom"
    validation_profile_description: str = ""
    required_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""

        def test_to_dict(test: EquivalenceTest | None) -> dict | None:
            if test is None:
                return None
            return {
                "name": test.name,
                "result": test.result.value,
                "reason": test.reason,
                "max_error": test.max_error,
                "mean_error": test.mean_error,
                "details": test.details,
                "counterexamples": test.counterexamples[:5],  # Limit to first 5
                "metadata": test.metadata,
            }

        return {
            "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
            "original_file": self.original_file,
            "recast_file": self.recast_file,
            "validation_profile": {
                "name": self.validation_profile,
                "description": self.validation_profile_description,
                "required_tests": self.required_tests,
            },
            "classification": {
                "original": self.original_class.value if self.original_class else None,
                "recast": self.recast_class.value if self.recast_class else None,
                "expected": self.expected_class.value if self.expected_class else None,
                "canonical_refusal_reason": self.canonical_refusal_reason,
            },
            "solver": {
                "original_requirement": (
                    self.original_solver_requirement.value
                    if self.original_solver_requirement
                    else None
                ),
                "recast_requirement": (
                    self.recast_solver_requirement.value
                    if self.recast_solver_requirement
                    else None
                ),
            },
            "tests": {
                "generated_output": test_to_dict(self.generated_output_test),
                "parser": test_to_dict(self.parser_test),
                "mapping": test_to_dict(self.mapping_test),
                "symbolic": test_to_dict(self.symbolic_test),
                "numerical": test_to_dict(self.numerical_test),
                "trajectory": test_to_dict(self.trajectory_test),
                "algebraic_residuals": test_to_dict(self.algebraic_residual_test),
                "auxiliaries": [test_to_dict(t) for t in self.auxiliary_tests],
            },
            "overall_pass": self.overall_pass,
            "overall_result": self.overall_result.value,
            "summary": self.summary,
        }
