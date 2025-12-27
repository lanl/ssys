#!/usr/bin/env python3
"""
Run ssys-recaster on model directories.

This script processes all models in a specified directory (via models.manifest),
recasts them to S-system form, validates the recastings, and generates
Jupyter notebooks for inspection.

Usage:
    python recast_models.py test_models1                 # default: simplified mode
    python recast_models.py test_models2 --mode canonical
    python recast_models.py test_models3 --both
    python recast_models.py pathological_models          # known problematic cases
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_mode(input_dir: str, mode: str, outdir: str, 
             solver: str = "roadrunner", parser: str = "sbml") -> int:
    """Run ssys-recaster for a specific mode."""
    manifest = f"{input_dir}/models.manifest"
    
    if not Path(manifest).exists():
        print(f"Error: Manifest not found: {manifest}")
        return 1
    
    cmd = [
        sys.executable, "-m", "ssys.cli",
        "--manifest", manifest,
        "--outdir", outdir,
        "--mode", mode,
        "--solver", solver,
        "--parser", parser,
        "--validate"
    ]
    
    print(f"Running ssys-recaster in {mode} mode...")
    print(f"Input directory: {input_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {outdir}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run ssys-recaster on test model directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python recast_models.py test_models1                    # Use simplified mode (default)
    python recast_models.py test_models2 --mode canonical   # Use canonical mode
    python recast_models.py test_models3 --both             # Run both modes (for comparison)
    python recast_models.py pathological_models             # Known problematic cases
    
Model directories:
    test_models1/        Original test models (basic examples)
    test_models2/        Literature recasting examples from publications
    test_models3/        Systems biology models from BioModels/literature
    pathological_models/ Models with stiffness or manifold drift issues
        """
    )
    parser.add_argument(
        "input",
        help="Input directory containing models and models.manifest file"
    )
    parser.add_argument(
        "--mode",
        choices=["simplified", "canonical"],
        default="simplified",
        help="Recasting mode: 'simplified' (default) or 'canonical'"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both simplified and canonical modes (generates two output directories)"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: out_<input_dir>)"
    )
    parser.add_argument(
        "--solver",
        choices=["roadrunner", "rk4"],
        default="roadrunner",
        help="ODE solver: 'roadrunner' (CVODE, default) or 'rk4'"
    )
    parser.add_argument(
        "--parser",
        choices=["sbml", "legacy"],
        default="legacy",
        help="Antimony parser: 'legacy' (regex, default) or 'sbml' (reference parser)"
    )
    args = parser.parse_args()
    
    # Normalize input path
    input_dir = args.input.rstrip("/")
    base_name = Path(input_dir).name
    
    if args.both:
        # Run both modes
        print("=" * 60)
        print("RUNNING BOTH MODES")
        print("=" * 60)
        print()
        
        # Simplified mode
        print("-" * 60)
        print("SIMPLIFIED MODE")
        print("-" * 60)
        ret1 = run_mode(input_dir, "simplified", 
                        f"out_{base_name}_simplified", args.solver, args.parser)
        
        print()
        print("-" * 60)
        print("CANONICAL MODE")
        print("-" * 60)
        ret2 = run_mode(input_dir, "canonical", 
                        f"out_{base_name}_canonical", args.solver, args.parser)
        
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Simplified mode: {'SUCCESS' if ret1 == 0 else 'FAILED'} (exit code {ret1})")
        print(f"Canonical mode:  {'SUCCESS' if ret2 == 0 else 'FAILED'} (exit code {ret2})")
        print()
        print("Output directories:")
        print(f"  - out_{base_name}_simplified/")
        print(f"  - out_{base_name}_canonical/")
        print()
        print("Notebooks:")
        print(f"  - out_{base_name}_simplified/recast_report.ipynb")
        print(f"  - out_{base_name}_canonical/recast_report.ipynb")
        
        # Return non-zero if either failed
        sys.exit(max(ret1, ret2))
    else:
        # Single mode
        outdir = args.outdir or f"out_{base_name}"
        ret = run_mode(input_dir, args.mode, outdir, args.solver, args.parser)
        if ret == 0:
            print(f"\nNotebook: {outdir}/recast_report.ipynb")
        sys.exit(ret)


if __name__ == "__main__":
    main()
