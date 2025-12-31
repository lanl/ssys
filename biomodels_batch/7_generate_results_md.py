#!/usr/bin/env python3
"""
Generate RESULTS.md from benchmark data.

Reads CSV files and generates markdown tables for RESULTS.md,
emphasizing transformation success stories (General→GMA, →S-system).

Usage:
    python 7_generate_results_md.py           # Print to stdout
    python 7_generate_results_md.py --write   # Update RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402


def count_files(directory: Path, pattern: str) -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def load_manifest() -> Optional[pd.DataFrame]:
    """Load validated models manifest."""
    manifest_path = Path(config.RESULTS_DIR) / "validated" / "manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return None


def load_results() -> Optional[pd.DataFrame]:
    """Load batch recast results."""
    results_path = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


def load_summary() -> Optional[dict]:
    """Load summary JSON if available."""
    summary_path = Path(config.RESULTS_DIR) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def generate_pipeline_table() -> str:
    """Generate pipeline summary table."""
    sbml_count = count_files(Path(config.SBML_DOWNLOADS_DIR), "*.xml")
    candidates_count = count_files(Path(config.SBML_CANDIDATES_DIR), "*.xml")
    transforms_count = count_files(Path(config.RECASTS_DIR), "*.ant")
    validation_count = count_files(Path(config.VALIDATION_DIR), "*_validation.json")
    validated_count = count_files(Path(config.RESULTS_DIR) / "validated", "*.ant")
    
    # Load results for failure breakdown
    results_df = load_results()
    if results_df is not None and not results_df.empty:
        timeouts = len(results_df[results_df["error"].str.contains("Timeout", na=False)])
        other_errors = len(results_df[results_df["error"].notna()]) - timeouts
    else:
        timeouts = 0
        other_errors = 0
    
    filtered_out = sbml_count - candidates_count if sbml_count > candidates_count else 0
    
    lines = [
        "## Pipeline Summary",
        "",
        "| Stage | Count | Notes |",
        "|-------|-------|-------|",
        f"| BioModels ODE models | 1,644 | Matched fetch query |",
        f"| SBML downloads | {sbml_count:,} | Successfully downloaded |",
        f"| Filtered out | {filtered_out:,} | No ODEs, events, delays, etc. |",
        f"| **Transformation candidates** | **{candidates_count:,}** | Passed heuristic filters |",
        f"| Successful transformations | {transforms_count:,} | {100*transforms_count/candidates_count:.1f}% success rate |" if candidates_count > 0 else f"| Successful transformations | {transforms_count:,} | - |",
    ]
    
    if timeouts > 0 or other_errors > 0:
        lines.append(f"| Failed (timeout) | {timeouts:,} | >60s processing time |")
        lines.append(f"| Failed (error) | {other_errors:,} | Various parsing/transformation errors |")
    
    if validation_count > 0:
        lines.append(f"| Validation reports | {validation_count:,} | Numerical validation attempted |")
    
    if validated_count > 0:
        lines.append(f"| **Validated models** | **{validated_count:,}** | Numerical validation passed |")
    
    return "\n".join(lines)


def generate_transformation_table(manifest_df: Optional[pd.DataFrame]) -> str:
    """Generate transformation achievements table."""
    if manifest_df is None or manifest_df.empty:
        return "## Transformation Achievements\n\n*Validation pending - run `./run_benchmark.sh --from validate`*\n"
    
    # Count transformation types
    # Success stories: General→GMA, General→S-system, GMA→S-system
    transform_counts = {}
    
    if "original_type" in manifest_df.columns and "recast_type" in manifest_df.columns:
        for _, row in manifest_df.iterrows():
            orig = row.get("original_type", "Unknown")
            recast = row.get("recast_type", "Unknown")
            key = f"{orig} → {recast}"
            transform_counts[key] = transform_counts.get(key, 0) + 1
    
    # Define success story order (most interesting first)
    success_order = [
        "General → S-system",
        "General → GMA", 
        "GMA → S-system",
        "GMA → GMA",
        "S-system → S-system",
    ]
    
    lines = [
        "## Transformation Achievements",
        "",
        "The following table shows successful transformations by type.",
        "**Highlights:** General → S-system/GMA transformations demonstrate ssys lifting",
        "non-polynomial ODEs to canonical forms.",
        "",
        "| Transformation | Count | Significance |",
        "|----------------|-------|--------------|",
    ]
    
    # Add rows in order
    for key in success_order:
        if key in transform_counts:
            count = transform_counts[key]
            if "General → S-system" in key:
                sig = "Full simplification achieved"
            elif "General → GMA" in key:
                sig = "Functional lifting to GMA form"
            elif "GMA → S-system" in key:
                sig = "Sum-to-product reduction"
            elif "GMA → GMA" in key:
                sig = "Already in GMA form (identity)"
            else:
                sig = "Already in S-system form"
            lines.append(f"| {key} | {count} | {sig} |")
    
    # Add any remaining types not in the order list
    for key, count in sorted(transform_counts.items()):
        if key not in success_order:
            lines.append(f"| {key} | {count} | - |")
    
    total = sum(transform_counts.values())
    lines.append(f"| **Total validated** | **{total}** | |")
    
    return "\n".join(lines)


def generate_sample_models_table(manifest_df: Optional[pd.DataFrame]) -> str:
    """Generate sample validated models table."""
    if manifest_df is None or manifest_df.empty:
        return "## Sample Validated Models\n\n*Validation pending*\n"
    
    lines = [
        "## Sample Validated Models",
        "",
        "Examples of successfully transformed and validated models:",
        "",
        "| Model ID | Original Type | Transformed Type | Max Error |",
        "|----------|---------------|------------------|-----------|",
    ]
    
    # Sort by interesting transformations first
    def sort_key(row):
        orig = str(row.get("original_type", ""))
        recast = str(row.get("recast_type", ""))
        if orig == "General" and recast == "S-system":
            return 0
        if orig == "General" and recast == "GMA":
            return 1
        if orig == "GMA" and recast == "S-system":
            return 2
        return 3
    
    sorted_df = manifest_df.copy()
    sorted_df["_sort"] = sorted_df.apply(sort_key, axis=1)
    sorted_df = sorted_df.sort_values("_sort")
    
    # Take top 10
    for _, row in sorted_df.head(10).iterrows():
        model_id = row.get("model_id", "Unknown")
        orig = row.get("original_type", "Unknown")
        recast = row.get("recast_type", "Unknown")
        max_err = row.get("max_error", row.get("max_diff", "N/A"))
        if isinstance(max_err, float):
            max_err = f"{max_err:.2e}"
        lines.append(f"| {model_id} | {orig} | {recast} | {max_err} |")
    
    lines.append("")
    lines.append(f"See `results/validated/manifest.csv` for the complete list of {len(manifest_df)} validated models.")
    
    return "\n".join(lines)


def generate_error_summary(results_df: Optional[pd.DataFrame]) -> str:
    """Generate condensed error summary."""
    if results_df is None or results_df.empty:
        return ""
    
    errors = results_df[results_df["error"].notna()]["error"]
    if len(errors) == 0:
        return ""
    
    # Categorize errors
    categories = {}
    for err in errors:
        err_str = str(err).lower()
        if "timeout" in err_str:
            cat = "Timeout (model too complex)"
        elif "piecewise" in err_str:
            cat = "Unsupported: piecewise functions"
        elif "event" in err_str:
            cat = "Unsupported: discrete events"
        elif "parse" in err_str or "syntax" in err_str:
            cat = "Parse errors"
        elif "symbol" in err_str:
            cat = "Undefined symbols"
        else:
            cat = "Other errors"
        categories[cat] = categories.get(cat, 0) + 1
    
    lines = [
        "## Transformation Failures",
        "",
        "| Failure Category | Count |",
        "|------------------|-------|",
    ]
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")
    
    lines.append(f"| **Total failures** | **{len(errors)}** |")
    
    return "\n".join(lines)


def generate_results_md() -> str:
    """Generate complete RESULTS.md content."""
    manifest_df = load_manifest()
    results_df = load_results()
    
    sections = [
        "# BioModels Batch Transformation Results",
        "",
        f"*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        generate_pipeline_table(),
        "",
        generate_transformation_table(manifest_df),
        "",
        generate_sample_models_table(manifest_df),
        "",
        generate_error_summary(results_df),
        "",
        "## Files",
        "",
        "- `results/batch_recast_results.csv` - Complete results for all candidates",
        "- `results/recasts/` - Successful transformation .ant files",
        "- `results/failures/` - Failure log files",
        "- `results/validation/` - Validation JSON reports",
        "- `results/validated/` - Validated model pairs (SBML + Antimony)",
        "",
        "## Reproducing Results",
        "",
        "```bash",
        "# Activate environment",
        "source activate_dev_env.sh",
        "",
        "# Run full pipeline",
        "cd biomodels_batch",
        "./run_benchmark.sh",
        "",
        "# Or run individual stages",
        "./run_benchmark.sh --only fetch",
        "./run_benchmark.sh --only filter",
        "./run_benchmark.sh --only recast",
        "./run_benchmark.sh --from validate",
        "",
        "# Regenerate this file",
        "python 7_generate_results_md.py --write",
        "```",
    ]
    
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from benchmark data")
    parser.add_argument("--write", action="store_true", help="Write to RESULTS.md (default: print to stdout)")
    args = parser.parse_args()
    
    content = generate_results_md()
    
    if args.write:
        output_path = Path(__file__).parent / "RESULTS.md"
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Updated {output_path}")
    else:
        print(content)


if __name__ == "__main__":
    main()
