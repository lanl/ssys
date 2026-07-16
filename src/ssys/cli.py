#!/usr/bin/env python3
"""Command-line interface for ssys recasting tool."""

import argparse
import json
import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import ssys
from ssys import recast_to_ssystem, ssystem_to_antimony
from ssys.metadata import _extract_sim_metadata
from ssys.parsing import parse_antimony_via_sbml
from ssys.validator import (
    EquivalenceTest,
    ValidationReport,
    ValidationResult,
    validation_profile_choices,
)

CaseRecord = tuple[str, str, str, str | None]
FailureRecord = tuple[str, str]

TROUBLESHOOTING_HINT = "See README.md#troubleshooting for local failure guidance."


def _ensure_utf8_streams() -> None:
    """Force stdout/stderr to UTF-8 so Unicode output survives on any locale.

    ssys prints characters like ``✓``, Greek letters, and arrows. On Windows the
    console (or a captured pipe) defaults to the locale code page (cp1252), which
    raises UnicodeEncodeError on those characters. Reconfiguring to UTF-8 keeps
    the CLI working regardless of platform locale.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def _print_troubleshooting_hint() -> None:
    print(TROUBLESHOOTING_HINT, file=sys.stderr)


def _build_validation_crash_report(
    ant_path: str,
    out_path: str,
    validation_error: str,
    validation_profile: str,
) -> ValidationReport:
    from ssys._validator.core import _required_test_names_for_profile
    from ssys._validator.report import resolve_validation_profile

    profile_spec = resolve_validation_profile(validation_profile)
    assert profile_spec is not None

    def blocked_test(name: str, *, required: bool) -> EquivalenceTest:
        if required:
            return EquivalenceTest(
                name=name,
                result=ValidationResult.NOT_ATTEMPTED,
                reason="validation_crashed",
                details=f"Check could not run because validation crashed: {validation_error}",
                metadata={
                    "validation_profile": profile_spec.name,
                    "blocked_by": "validation_crash",
                    "required": True,
                },
            )
        return EquivalenceTest(
            name=name,
            result=ValidationResult.NOT_ATTEMPTED,
            reason="profile_excluded",
            details=f"Check is not part of validation profile {profile_spec.name!r}",
            metadata={"validation_profile": profile_spec.name, "required": False},
        )

    return ValidationReport(
        original_file=ant_path,
        recast_file=out_path,
        original_class=None,
        recast_class=None,
        validation_profile=profile_spec.name,
        validation_profile_description=profile_spec.description,
        required_tests=_required_test_names_for_profile(profile_spec),
        generated_output_test=blocked_test("generated_output_roundtrip", required=True),
        parser_test=EquivalenceTest(
            name="validator_crash",
            result=ValidationResult.FAIL,
            details=validation_error,
            metadata={"validation_profile": profile_spec.name},
        ),
        mapping_test=blocked_test("mapping_completeness", required=True),
        symbolic_test=blocked_test("symbolic_equivalence", required=profile_spec.run_symbolic),
        numerical_test=blocked_test("numerical_pointwise", required=profile_spec.run_numerical),
        trajectory_test=blocked_test(
            "trajectory_comparison", required=profile_spec.run_trajectory
        ),
        algebraic_residual_test=blocked_test(
            "algebraic_manifold_residuals", required=profile_spec.run_trajectory
        ),
        auxiliary_tests=[
            blocked_test("auxiliary_identities", required=profile_spec.run_auxiliaries)
        ],
        overall_pass=False,
        overall_result=ValidationResult.FAIL,
        summary=f"Validation crashed: {validation_error}",
    )


def read_manifest(path: str) -> list[str]:
    """Read a manifest file and return list of absolute paths to .ant files.

    Relative paths in the manifest are resolved relative to the manifest
    file's directory, not the current working directory.
    """
    manifest_dir = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            # Resolve relative paths from manifest's directory
            if os.path.isabs(s):
                items.append(s)
            else:
                items.append(os.path.join(manifest_dir, s))
    return items


def recast_file(
    ant_path: str,
    out_dir: str,
    mode: str = "simplified",
    validate: bool = False,
    validation_profile: str = "strict",
) -> tuple[str, str, str, str | None]:
    """
    Recast a single Antimony file to S-system form.

    Args:
        ant_path: Path to input Antimony file
        out_dir: Output directory for recast file
        mode: Output mode ('simplified' or 'canonical')
        validate: Whether to run validation tests
        validation_profile: Named validation profile to use when validate=True

    Returns:
        Tuple of (model_name, input_path, output_path, validation_json_path)
        validation_json_path is None if validate=False
    """
    name = os.path.splitext(os.path.basename(ant_path))[0]

    txt = open(ant_path).read()

    # Extract @SIM metadata FIRST (before any parsing)
    t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(txt)

    # SBML-first: libAntimony parses Antimony → SBML → libSBML extracts ODEs
    sym = parse_antimony_via_sbml(txt)

    # Set @SIM metadata on sym so recast_to_ssystem can propagate it
    # This ensures the recast output includes proper simulation parameters
    sym.sim_t_start = t_start
    sym.sim_t_end = t_end
    sym.sim_n_steps = n_steps
    sym.eps_init = eps_init
    sym.eps_slack = eps_slack

    rec = recast_to_ssystem(sym, mode=mode)
    out_text = ssystem_to_antimony(rec, model_name=f"{name}_recast", mode=mode)

    # NOTE: @SIM metadata is now handled by ssystem_to_antimony() in recaster.py
    # It outputs the @SIM line inside the model block (before 'end'), which is cleaner
    # and includes EPS_INIT information when relevant.

    out_path = os.path.join(out_dir, f"{name}_recast.ant")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    validation_json_path = None
    if validate:
        from ssys.validator import validate_recast_pair

        validation_json_path = os.path.join(out_dir, f"{name}_validation.json")
        try:
            report = validate_recast_pair(
                ant_path,
                out_path,
                mode=mode,
                output_json=validation_json_path,
                profile=validation_profile,
            )
            if not report.overall_pass:
                print(f"Validation failed for {name}: {report.summary}", file=sys.stderr)
        except Exception as e:
            validation_error = str(e)
            failure_report = _build_validation_crash_report(
                ant_path,
                out_path,
                validation_error,
                validation_profile,
            )
            try:
                with open(validation_json_path, "w", encoding="utf-8") as f:
                    json.dump(failure_report.to_dict(), f, indent=2)
            except OSError as write_error:
                print(
                    f"Validation crashed for {name}, and writing the failure report failed: "
                    f"{write_error}",
                    file=sys.stderr,
                )
                validation_json_path = None
            print(f"Validation crashed for {name}: {e}", file=sys.stderr)

    return name, ant_path, out_path, validation_json_path


def build_notebook(
    cases: list[CaseRecord],
    out_dir: str,
    mode: str = "simplified",
) -> str:
    """
    Build a Jupyter notebook that reports on all recast models.

    Args:
        cases: List of (name, input_path, output_path, validation_path)
        out_dir: Output directory for notebook
        mode: Output mode ('simplified' or 'canonical')

    Returns:
        Path to generated notebook
    """
    nb = new_notebook()
    nb.cells.append(new_markdown_cell("# ODE → S-System Recaster Report\nGenerated by ssys CLI"))

    # Simple import cell - imports from the notebook_helpers module
    helpers = """import os
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Code

import ssys
from ssys.notebook_helpers import load_and_report
"""
    nb.cells.append(new_code_cell(helpers))

    # Configuration cell - user-adjustable simulation parameters
    config_cell = '''# ============================================================
# SIMULATION SETTINGS (edit these values to adjust simulations)
# ============================================================
T_END = 1.0       # End time for simulations (default: 1.0)
N_STEPS = 100     # Number of time steps (default: 100)
'''
    nb.cells.append(new_code_cell(config_cell))

    # One code cell per case
    for name, ant_path, recast_path, validation_path in cases:
        nb.cells.append(new_markdown_cell(f"## {name}"))
        # Make paths relative to notebook location (basename only)
        recast_basename = os.path.basename(recast_path)
        validation_basename = os.path.basename(validation_path) if validation_path else None

        call = (
            f"load_and_report({repr(ant_path)}, "
            f"{repr(recast_basename)}, T=T_END, steps=N_STEPS, "
            f"mode={repr(mode)}, "
            f"validation_json={repr(validation_basename)})"
        )
        nb.cells.append(new_code_cell(call))

    out_nb = os.path.join(out_dir, "recast_report.ipynb")
    with open(out_nb, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return out_nb


def _load_manifest_or_exit(manifest: str) -> list[str]:
    try:
        ant_files = read_manifest(manifest)
    except OSError as e:
        print(f"Error: Could not read manifest {manifest!r}: {e}", file=sys.stderr)
        _print_troubleshooting_hint()
        sys.exit(1)

    if not ant_files:
        print("Error: Manifest contained no .ant files.", file=sys.stderr)
        _print_troubleshooting_hint()
        sys.exit(1)
    return ant_files


def _recast_manifest_models(args: argparse.Namespace, ant_files: list[str]) -> list[CaseRecord]:
    print(f"Processing {len(ant_files)} model(s)...")
    cases: list[CaseRecord] = []
    processing_failures: list[FailureRecord] = []
    for ant in ant_files:
        try:
            cases.append(
                recast_file(
                    ant,
                    args.outdir,
                    mode=args.mode,
                    validate=args.validate,
                    validation_profile=args.validation_profile,
                )
            )
        except Exception as e:
            processing_failures.append((ant, str(e)))
            print(f"Recast failed for {ant}: {e}", file=sys.stderr)

    if not cases:
        print("Error: No models were successfully recast.", file=sys.stderr)
        _print_troubleshooting_hint()
        sys.exit(1)

    if processing_failures:
        print("Recast failures:", file=sys.stderr)
        for name, summary in processing_failures:
            print(f"  {name}: {summary}", file=sys.stderr)
        _print_troubleshooting_hint()
        sys.exit(1)
    return cases


def _validation_summary(cases: list[CaseRecord]) -> tuple[int, list[FailureRecord]]:
    validated = 0
    validation_failures: list[FailureRecord] = []
    for name, _, _, vpath in cases:
        if not vpath or not os.path.exists(vpath):
            validation_failures.append((name, "validation report missing"))
            continue
        try:
            with open(vpath) as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            validation_failures.append((name, f"validation report unreadable: {e}"))
            continue
        if report.get("overall_pass"):
            validated += 1
        else:
            summary = report.get("summary", "validation did not pass")
            validation_failures.append((name, summary))
    return validated, validation_failures


def _report_validation_summary(
    cases: list[CaseRecord],
    validation_profile: str,
) -> list[FailureRecord]:
    validated, validation_failures = _validation_summary(cases)
    if validation_profile == "strict":
        print(f"✓ Validated {validated}/{len(cases)} models")
    else:
        print(f"✓ Validation profile '{validation_profile}' passed {validated}/{len(cases)} models")
    return validation_failures


def _exit_on_validation_failures(
    validation_failures: list[FailureRecord],
    *,
    allow_validation_failures: bool,
) -> None:
    if not validation_failures:
        return
    print("Validation failures:", file=sys.stderr)
    for name, summary in validation_failures:
        print(f"  {name}: {summary}", file=sys.stderr)
    _print_troubleshooting_hint()
    if not allow_validation_failures:
        sys.exit(1)


def main():
    """Main CLI entry point."""
    _ensure_utf8_streams()
    parser = argparse.ArgumentParser(
        description="Recast Antimony models to canonical S-system form.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input trust boundary: ssys treats Antimony and SBML model files as trusted "
            "scientific inputs, not safe untrusted uploads. Do not expose this CLI directly "
            "to arbitrary user-submitted model text in multi-tenant environments."
        ),
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Plain-text file: one .ant path per line.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for recast .ant files and the notebook.",
    )
    parser.add_argument(
        "--mode",
        choices=["simplified", "canonical"],
        default="simplified",
        help="Output mode: 'simplified' (default) or 'canonical' (strict 2-term S-system)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on each recast (symbolic and numerical tests)",
    )
    parser.add_argument(
        "--validation-profile",
        choices=validation_profile_choices(),
        default="strict",
        help=(
            "Validation profile to run with --validate. 'strict' is required for "
            "release-grade validated claims; partial profiles are diagnostic."
        ),
    )
    parser.add_argument(
        "--allow-validation-failures",
        action="store_true",
        help=(
            "Keep best-effort batch behavior when --validate reports failures. "
            "By default validation failures make the CLI exit nonzero."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {ssys.__version__}",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ant_files = _load_manifest_or_exit(args.manifest)
    cases = _recast_manifest_models(args, ant_files)

    try:
        nb_path = build_notebook(cases, args.outdir, mode=args.mode)
    except Exception as e:
        print(f"Notebook generation failed: {e}", file=sys.stderr)
        _print_troubleshooting_hint()
        sys.exit(1)

    print(f"✓ Recast complete. Notebook written: {nb_path}")

    if args.validate:
        validation_failures = _report_validation_summary(cases, args.validation_profile)
        _exit_on_validation_failures(
            validation_failures,
            allow_validation_failures=args.allow_validation_failures,
        )


if __name__ == "__main__":
    main()
