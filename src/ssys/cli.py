#!/usr/bin/env python3
"""Command-line interface for ssys recasting tool."""

import argparse
import json
import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import ssys
from ssys import build_sym_system, parse_antimony, recast_to_ssystem, ssystem_to_antimony
from ssys.recaster import _extract_sim_metadata, parse_antimony_via_sbml


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
    parser: str = "legacy",
) -> tuple[str, str, str, str | None]:
    """
    Recast a single Antimony file to S-system form.

    Args:
        ant_path: Path to input Antimony file
        out_dir: Output directory for recast file
        mode: Output mode ('simplified' or 'canonical')
        validate: Whether to run validation tests
        parser: Parser to use ('legacy' or 'sbml')
            - 'legacy': Hand-rolled regex parser (current behavior)
            - 'sbml': RoadRunner → SBML → libSBML (reference Antimony parser)

    Returns:
        Tuple of (model_name, input_path, output_path, validation_json_path)
        validation_json_path is None if validate=False
    """
    name = os.path.splitext(os.path.basename(ant_path))[0]
    txt = open(ant_path).read()

    # Extract @SIM metadata FIRST (before any parsing)
    # This works for both parser modes
    t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(txt)

    # Parse based on selected parser
    if parser == "sbml":
        # SBML-first: RoadRunner parses Antimony → SBML → libSBML extracts ODEs
        sym = parse_antimony_via_sbml(txt)
    else:
        # Legacy: Hand-rolled regex parser
        ir = parse_antimony(txt)
        sym = build_sym_system(ir)

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
    with open(out_path, "w") as f:
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
                parser=parser,
            )
            if not report.overall_pass:
                print(f"Validation failed for {name}: {report.summary}", file=sys.stderr)
        except Exception as e:
            failure_report = {
                "original_file": ant_path,
                "recast_file": out_path,
                "classification": {
                    "original": None,
                    "recast": None,
                    "expected": None,
                    "canonical_refusal_reason": None,
                },
                "tests": {
                    "validation": {
                        "name": "validation",
                        "result": "failed",
                        "details": str(e),
                    }
                },
                "overall_pass": False,
                "overall_result": "failed",
                "summary": f"Validation crashed: {e}",
            }
            try:
                with open(validation_json_path, "w") as f:
                    json.dump(failure_report, f, indent=2)
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
    cases: list[tuple[str, str, str, str | None]],
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
    with open(out_nb, "w") as f:
        nbformat.write(nb, f)
    return out_nb


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Recast Antimony models to canonical S-system form.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--allow-validation-failures",
        action="store_true",
        help=(
            "Keep best-effort batch behavior when --validate reports failures. "
            "By default validation failures make the CLI exit nonzero."
        ),
    )
    parser.add_argument(
        "--parser",
        choices=["legacy", "sbml"],
        default="sbml",
        help="Antimony parser: 'sbml' (RoadRunner reference, default) or 'legacy' (regex)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {ssys.__version__}",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ant_files = read_manifest(args.manifest)
    if not ant_files:
        print("Error: Manifest contained no .ant files.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(ant_files)} model(s)...")
    cases = [
        recast_file(
            ant,
            args.outdir,
            mode=args.mode,
            validate=args.validate,
            parser=args.parser,
        )
        for ant in ant_files
    ]

    validation_failures = []
    if args.validate:
        # Count validation results - check if validation PASSED, not just file exists
        validated = 0
        for name, _, _, vpath in cases:
            if vpath and os.path.exists(vpath):
                try:
                    with open(vpath) as f:
                        report = json.load(f)
                        if report.get("overall_pass"):
                            validated += 1
                        else:
                            summary = report.get("summary", "validation did not pass")
                            validation_failures.append((name, summary))
                except (OSError, json.JSONDecodeError) as e:
                    validation_failures.append((name, f"validation report unreadable: {e}"))
            else:
                validation_failures.append((name, "validation report missing"))
        print(f"✓ Validated {validated}/{len(cases)} models")

    nb_path = build_notebook(cases, args.outdir, mode=args.mode)
    print(f"✓ Recast complete. Notebook written: {nb_path}")

    if args.validate and validation_failures:
        print("Validation failures:", file=sys.stderr)
        for name, summary in validation_failures:
            print(f"  {name}: {summary}", file=sys.stderr)
        if not args.allow_validation_failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
