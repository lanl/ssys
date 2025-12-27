#!/usr/bin/env python3
"""Command-line interface for ssys recasting tool."""

import argparse
import os
import sys
from typing import Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import ssys
from ssys import parse_antimony, build_sym_system, recast_to_ssystem, ssystem_to_antimony
from ssys.recaster import parse_antimony_via_sbml, _extract_sim_metadata


def read_manifest(path: str) -> list[str]:
    """Read a manifest file and return list of absolute paths to .ant files.
    
    Relative paths in the manifest are resolved relative to the manifest
    file's directory, not the current working directory.
    """
    manifest_dir = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, "r") as f:
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


def recast_file(ant_path: str, out_dir: str, mode: str = "simplified", 
                validate: bool = False, solver: str = "roadrunner",
                parser: str = "legacy") -> tuple[str, str, str, Optional[str]]:
    """
    Recast a single Antimony file to S-system form.

    Args:
        ant_path: Path to input Antimony file
        out_dir: Output directory for recast file
        mode: Output mode ('simplified' or 'canonical')
        validate: Whether to run validation tests
        solver: ODE solver for validation ('roadrunner' or 'rk4')
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
    t_start, t_end, n_steps = _extract_sim_metadata(txt)
    
    # Parse based on selected parser
    if parser == "sbml":
        # SBML-first: RoadRunner parses Antimony → SBML → libSBML extracts ODEs
        sym = parse_antimony_via_sbml(txt)
        # Get @SIM metadata from attached attributes (if available)
        if hasattr(sym, '_sim_t_start') and sym._sim_t_start is not None:
            t_start = sym._sim_t_start
        if hasattr(sym, '_sim_t_end') and sym._sim_t_end is not None:
            t_end = sym._sim_t_end
        if hasattr(sym, '_sim_n_steps') and sym._sim_n_steps is not None:
            n_steps = sym._sim_n_steps
    else:
        # Legacy: Hand-rolled regex parser
        ir = parse_antimony(txt)
        sym = build_sym_system(ir)
        # Get @SIM metadata from ModelIR
        if ir.sim_t_start is not None:
            t_start = ir.sim_t_start
        if ir.sim_t_end is not None:
            t_end = ir.sim_t_end
        if ir.sim_n_steps is not None:
            n_steps = ir.sim_n_steps
    
    rec = recast_to_ssystem(sym, mode=mode)
    out_text = ssystem_to_antimony(rec, model_name=f"{name}_recast", mode=mode)
    
    # Propagate @SIM metadata from input to recast output
    # This ensures notebook simulations use the same time parameters for both
    sim_parts = []
    if t_start is not None:
        sim_parts.append(f"T_START={t_start:g}")
    if t_end is not None:
        sim_parts.append(f"T_END={t_end:g}")
    if n_steps is not None:
        sim_parts.append(f"N_STEPS={n_steps}")
    if sim_parts:
        sim_line = "// @SIM " + " ".join(sim_parts) + "\n"
        out_text = out_text.rstrip() + "\n" + sim_line

    out_path = os.path.join(out_dir, f"{name}_recast.ant")
    with open(out_path, "w") as f:
        f.write(out_text)

    validation_json_path = None
    if validate:
        from ssys.validator import validate_recast_pair
        validation_json_path = os.path.join(out_dir, f"{name}_validation.json")
        try:
            validate_recast_pair(ant_path, out_path, mode=mode,
                               output_json=validation_json_path,
                               solver=solver,
                               parser=parser)
        except Exception as e:
            print(f"Warning: Validation failed for {name}: {e}", file=sys.stderr)
            validation_json_path = None

    return name, ant_path, out_path, validation_json_path


def build_notebook(cases: list[tuple[str, str, str, Optional[str]]], 
                    out_dir: str, mode: str = "simplified",
                    solver: str = "roadrunner") -> str:
    """
    Build a Jupyter notebook that reports on all recast models.

    Args:
        cases: List of (name, input_path, output_path, validation_path)
        out_dir: Output directory for notebook
        mode: Output mode ('simplified' or 'canonical')
        solver: ODE solver to use ('roadrunner' or 'rk4')

    Returns:
        Path to generated notebook
    """
    nb = new_notebook()
    nb.cells.append(
        new_markdown_cell("# ODE → S-System Recaster Report\n"
                         "Generated by ssys CLI")
    )

    # Simple import cell - imports from the notebook_helpers module
    helpers = '''import os
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Code

import ssys
from ssys.notebook_helpers import load_and_report
'''
    nb.cells.append(new_code_cell(helpers))

    # Configuration cell - user-adjustable simulation parameters
    config_cell = f'''# ============================================================
# SIMULATION SETTINGS (edit these values to adjust simulations)
# ============================================================
T_END = 20.0      # End time for simulations
N_STEPS = 400     # Number of time steps
SOLVER = "{solver}"  # ODE solver: "roadrunner" or "rk4"
'''
    nb.cells.append(new_code_cell(config_cell))

    # One code cell per case
    for name, ant_path, recast_path, validation_path in cases:
        nb.cells.append(new_markdown_cell(f"## {name}"))
        # Make paths relative to notebook location (basename only)
        recast_basename = os.path.basename(recast_path)
        validation_basename = (os.path.basename(validation_path) 
                              if validation_path else None)
        
        call = (f"load_and_report({repr(ant_path)}, "
                f"{repr(recast_basename)}, T=T_END, steps=N_STEPS, "
                f"mode={repr(mode)}, "
                f"validation_json={repr(validation_basename)}, "
                f"solver=SOLVER)")
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
        help="Output mode: 'simplified' (default, current behavior) or 'canonical' (strict 2-term S-system form)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on each recast (symbolic and numerical tests)",
    )
    parser.add_argument(
        "--solver",
        choices=["roadrunner", "rk4"],
        default="roadrunner",
        help="ODE solver: 'roadrunner' (CVODE, default) or 'rk4'",
    )
    parser.add_argument(
        "--parser",
        choices=["legacy", "sbml"],
        default="sbml",
        help="Antimony parser: 'sbml' (RoadRunner reference parser, default) or 'legacy' (regex)",
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
    cases = [recast_file(ant, args.outdir, mode=args.mode, 
                        validate=args.validate, solver=args.solver,
                        parser=args.parser)
             for ant in ant_files]
    
    if args.validate:
        # Count validation results - check if validation PASSED, not just file exists
        import json
        validated = 0
        for _, _, _, vpath in cases:
            if vpath and os.path.exists(vpath):
                try:
                    with open(vpath) as f:
                        report = json.load(f)
                        if report.get('overall_pass'):
                            validated += 1
                except (json.JSONDecodeError, IOError):
                    pass  # Skip malformed/unreadable files
        print(f"✓ Validated {validated}/{len(cases)} models")

    nb_path = build_notebook(cases, args.outdir, mode=args.mode, 
                             solver=args.solver)
    print(f"✓ Recast complete. Notebook written: {nb_path}")


if __name__ == "__main__":
    main()
