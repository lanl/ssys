"""Validation result and report data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ssys.types import SolverRequirement, SystemClass


class ValidationResult(Enum):
    """Validation test outcomes."""

    PASS = "pass"
    FAIL = "failed"
    TIMEOUT = "timeout"
    NOT_ATTEMPTED = "not_attempted"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


PASSING_RESULTS = frozenset({ValidationResult.PASS})


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
                "metadata": test.metadata,
            }

        return {
            "original_file": self.original_file,
            "recast_file": self.recast_file,
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
