#!/usr/bin/env python3
"""
Generate reports from benchmark data.

Consolidated script that:
1. Rebuilds batch_recast_results.csv from actual files
2. Generates RESULTS.md with pipeline summary and tables
3. Optionally generates visualization figures (--figures)

Usage:
    python step6_report.py           # Rebuild CSV + update RESULTS.md
    python step6_report.py --figures # Also generate PNG figures
    python step6_report.py --dry-run # Print RESULTS.md without writing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# CSV Rebuild (from 5_rebuild_results_csv.py)
# ===========================================================================

def rebuild_results_csv(mode: str = "simplified") -> None:
    """Rebuild batch_recast_results.csv from recast and failure files."""
    recasts_dir = Path(config.RECASTS_DIR)
    failures_dir = Path(config.FAILURES_DIR)
    validation_dir = Path(config.VALIDATION_DIR)
    output_file = Path(config.RESULTS_DIR) / "batch_recast_results.csv"

    logger.info("Rebuilding results CSV from files...")

    # Get all candidate model IDs from SBML files
    sbml_candidates_dir = Path(config.SBML_CANDIDATES_DIR)
    if sbml_candidates_dir.exists():
        sbml_files = list(sbml_candidates_dir.glob("*.xml"))
        candidate_ids = {f.stem for f in sbml_files}
        logger.info(f"Found {len(candidate_ids)} SBML candidate files")
    else:
        logger.error("No candidates found")
        return

    # Scan recast files
    recast_files = list(recasts_dir.glob(f"*_{mode}.ant"))
    recast_ids = {f.stem.replace(f"_{mode}", "") for f in recast_files}
    logger.info(f"Found {len(recast_ids)} successful recast files")

    # Scan failure files
    failure_files = list(failures_dir.glob(f"*_{mode}.txt")) + list(failures_dir.glob(f"*_{mode}.log"))
    failure_ids = {f.stem.replace(f"_{mode}", "") for f in failure_files}
    logger.info(f"Found {len(failure_ids)} failure files")

    # Scan validation files (try both patterns for compatibility)
    validation_files = list(validation_dir.glob(f"*_{mode}_numerical.json"))
    if not validation_files:
        # Fallback to old pattern
        validation_files = list(validation_dir.glob(f"*_{mode}_validation.json"))
    validation_data = {}
    for vf in validation_files:
        model_id = vf.stem.replace(f"_{mode}_numerical", "").replace(f"_{mode}_validation", "")
        try:
            with open(vf) as f:
                validation_data[model_id] = json.load(f)
        except Exception:
            pass
    logger.info(f"Found {len(validation_data)} validation files")

    # Build results
    results = []
    for model_id in sorted(candidate_ids):
        if model_id in recast_ids:
            status = "success"
        elif model_id in failure_ids:
            failure_file = failures_dir / f"{model_id}_{mode}.log"
            if not failure_file.exists():
                failure_file = failures_dir / f"{model_id}_{mode}.txt"
            try:
                with open(failure_file) as f:
                    error_text = f.read().strip()
                    status = "timeout" if "TimeoutError" in error_text or "timed out" in error_text else "error"
            except Exception:
                status = "error"
        else:
            status = "pending"

        row = {
            "model_id": model_id,
            "mode": mode,
            "status": status,
            "recast_success": model_id in recast_ids,
            "recast_time": "",
            "validation_attempted": model_id in validation_data,
            "validation_pass": False,
            "error": "",
        }

        if model_id in validation_data:
            row["validation_pass"] = validation_data[model_id].get("overall_pass", False)

        if model_id in failure_ids:
            failure_file = failures_dir / f"{model_id}_{mode}.log"
            if not failure_file.exists():
                failure_file = failures_dir / f"{model_id}_{mode}.txt"
            try:
                with open(failure_file) as f:
                    row["error"] = f.read().strip().split("\n")[0][:100]
            except Exception:
                row["error"] = "Unknown error"

        results.append(row)

    # Write CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        fieldnames = ["model_id", "mode", "status", "recast_success", "recast_time",
                      "validation_attempted", "validation_pass", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")
    validation_pass = sum(1 for r in results if r["validation_pass"])

    logger.info(f"CSV rebuilt: {len(results)} rows, {success_count} success, "
                f"{timeout_count} timeout, {error_count} error, {validation_pass} validated")


# ===========================================================================
# RESULTS.md Generation (from 7_generate_results_md.py)
# ===========================================================================

def count_files(directory: Path, pattern: str) -> int:
    """Count files matching pattern in directory."""
    return len(list(directory.glob(pattern))) if directory.exists() else 0


def load_manifest() -> Optional[pd.DataFrame]:
    """Load validated models manifest."""
    manifest_path = Path(config.RESULTS_DIR) / "validated" / "manifest.csv"
    return pd.read_csv(manifest_path) if manifest_path.exists() else None


def load_results() -> Optional[pd.DataFrame]:
    """Load batch recast results."""
    results_path = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
    return pd.read_csv(results_path) if results_path.exists() else None


def generate_pipeline_table() -> str:
    """Generate pipeline summary table."""
    sbml_count = count_files(Path(config.SBML_DOWNLOADS_DIR), "*.xml")
    candidates_count = count_files(Path(config.SBML_CANDIDATES_DIR), "*.xml")
    transforms_count = count_files(Path(config.RECASTS_DIR), "*.ant")
    validation_count = count_files(Path(config.VALIDATION_DIR), "*_validation.json")
    validated_count = count_files(Path(config.RESULTS_DIR) / "validated", "*.ant")
    
    results_df = load_results()
    timeouts = other_errors = 0
    if results_df is not None and not results_df.empty:
        timeouts = len(results_df[results_df["error"].str.contains("Timeout", na=False)])
        other_errors = len(results_df[results_df["error"].notna()]) - timeouts
    
    filtered_out = max(0, sbml_count - candidates_count)
    
    lines = [
        "## Pipeline Summary",
        "",
        "| Stage | Count | Notes |",
        "|-------|-------|-------|",
        f"| SBML downloads | {sbml_count:,} | Successfully downloaded |",
        f"| Filtered out | {filtered_out:,} | No ODEs, events, delays, etc. |",
        f"| **Transformation candidates** | **{candidates_count:,}** | Passed heuristic filters |",
    ]
    
    if candidates_count > 0:
        lines.append(f"| Successful transformations | {transforms_count:,} | {100*transforms_count/candidates_count:.1f}% success rate |")
    
    if timeouts > 0:
        lines.append(f"| Failed (timeout) | {timeouts:,} | >60s processing time |")
    if other_errors > 0:
        lines.append(f"| Failed (error) | {other_errors:,} | Various errors |")
    if validation_count > 0:
        lines.append(f"| Validation reports | {validation_count:,} | Numerical validation attempted |")
    if validated_count > 0:
        lines.append(f"| **Validated models** | **{validated_count:,}** | Numerical validation passed |")
    
    return "\n".join(lines)


def generate_transformation_table(manifest_df: Optional[pd.DataFrame]) -> str:
    """Generate transformation achievements table."""
    if manifest_df is None or manifest_df.empty:
        return "## Transformation Achievements\n\n*Validation pending - run `./run_benchmark.sh --from validate`*\n"
    
    transform_counts = {}
    if "original_type" in manifest_df.columns and "recast_type" in manifest_df.columns:
        for _, row in manifest_df.iterrows():
            key = f"{row.get('original_type', 'Unknown')} → {row.get('recast_type', 'Unknown')}"
            transform_counts[key] = transform_counts.get(key, 0) + 1
    
    success_order = ["General → S-system", "General → GMA", "GMA → S-system", "GMA → GMA", "S-system → S-system"]
    significance = {
        "General → S-system": "Full simplification achieved",
        "General → GMA": "Functional lifting to GMA form",
        "GMA → S-system": "Sum-to-product reduction",
        "GMA → GMA": "Already in GMA form",
        "S-system → S-system": "Already in S-system form",
    }
    
    lines = [
        "## Transformation Achievements",
        "",
        "| Transformation | Count | Significance |",
        "|----------------|-------|--------------|",
    ]
    
    for key in success_order:
        if key in transform_counts:
            lines.append(f"| {key} | {transform_counts[key]} | {significance.get(key, '-')} |")
    
    for key, count in sorted(transform_counts.items()):
        if key not in success_order:
            lines.append(f"| {key} | {count} | - |")
    
    lines.append(f"| **Total validated** | **{sum(transform_counts.values())}** | |")
    
    return "\n".join(lines)


def generate_sample_models_table(manifest_df: Optional[pd.DataFrame]) -> str:
    """Generate sample validated models table."""
    if manifest_df is None or manifest_df.empty:
        return "## Sample Validated Models\n\n*Validation pending*\n"
    
    lines = [
        "## Sample Validated Models",
        "",
        "| Model ID | Original Type | Transformed Type | Max Error |",
        "|----------|---------------|------------------|-----------|",
    ]
    
    def sort_key(row):
        orig, recast = str(row.get("original_type", "")), str(row.get("recast_type", ""))
        if orig == "General" and recast == "S-system": return 0
        if orig == "General" and recast == "GMA": return 1
        if orig == "GMA" and recast == "S-system": return 2
        return 3
    
    sorted_df = manifest_df.copy()
    sorted_df["_sort"] = sorted_df.apply(sort_key, axis=1)
    sorted_df = sorted_df.sort_values("_sort")
    
    for _, row in sorted_df.head(10).iterrows():
        max_err = row.get("max_error", row.get("max_diff", "N/A"))
        if isinstance(max_err, float):
            max_err = f"{max_err:.2e}"
        lines.append(f"| {row.get('model_id', 'Unknown')} | {row.get('original_type', 'Unknown')} | "
                     f"{row.get('recast_type', 'Unknown')} | {max_err} |")
    
    lines.append(f"\nSee `results/validated/manifest.csv` for complete list ({len(manifest_df)} models).")
    
    return "\n".join(lines)


def generate_results_md() -> str:
    """Generate complete RESULTS.md content."""
    manifest_df = load_manifest()
    
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
        "## Reproducing Results",
        "",
        "```bash",
        "cd biomodels_batch",
        "./run_benchmark.sh              # Default: numerical validation only",
        "./run_benchmark.sh --full       # Full validation (+ JAX + symbolic)",
        "./run_benchmark.sh --from recast  # Re-run from transformation step",
        "python step6_report.py --figures  # Regenerate this file with figures",
        "```",
    ]
    
    return "\n".join(sections)


# ===========================================================================
# Figures Generation (from 4_analyze_results.py)
# ===========================================================================

def generate_figures() -> None:
    """Generate analysis figures (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available - skipping figures")
        return
    
    figures_dir = Path(config.RESULTS_DIR) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = load_results()
    if results_df is None or results_df.empty:
        logger.warning("No results data - skipping figures")
        return
    
    # Figure 1: Pipeline funnel
    logger.info("Generating pipeline_funnel.png...")
    sbml_count = count_files(Path(config.SBML_DOWNLOADS_DIR), "*.xml")
    candidates_count = count_files(Path(config.SBML_CANDIDATES_DIR), "*.xml")
    transforms_count = count_files(Path(config.RECASTS_DIR), "*.ant")
    validated_count = count_files(Path(config.RESULTS_DIR) / "validated", "*.ant")
    
    stages = ["Downloaded\nSBML", "Passed\nFilters", "Transformed", "Validated"]
    counts = [sbml_count, candidates_count, transforms_count, validated_count]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(stages)), counts, color=colors, height=0.6, edgecolor="black")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Models")
    ax.set_title("Pipeline Funnel: BioModels Transformation")
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                f"{count:,}", va="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "pipeline_funnel.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Figure 2: Success rates pie chart
    logger.info("Generating success_rates.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    success = results_df["recast_success"].sum()
    failure = len(results_df) - success
    axes[0].pie([success, failure], labels=["Success", "Failure"], autopct="%1.1f%%",
                startangle=90, colors=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Recast Success Rate")
    
    validated = results_df["validation_attempted"].sum()
    val_pass = results_df["validation_pass"].sum()
    val_fail = validated - val_pass
    not_validated = len(results_df) - validated
    axes[1].bar(["Pass", "Fail", "Not Validated"], [val_pass, val_fail, not_validated],
                color=["#2ecc71", "#e74c3c", "#95a5a6"])
    axes[1].set_title("Validation Results")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "success_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Figure 3: Validation errors histogram
    logger.info("Generating validation_errors.png...")
    validation_dir = Path(config.VALIDATION_DIR)
    if validation_dir.exists():
        max_errors = []
        for json_file in validation_dir.glob("*_validation.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if data.get("numerical", {}).get("passed"):
                    max_err = data.get("numerical", {}).get("max_diff", 0)
                    if max_err and max_err > 0:
                        max_errors.append(max_err)
            except Exception:
                continue
        
        if len(max_errors) >= 5:
            fig, ax = plt.subplots(figsize=(10, 6))
            log_errors = np.log10(max_errors)
            ax.hist(log_errors, bins=30, edgecolor="black", alpha=0.7, color="#2ecc71")
            ax.set_xlabel("log₁₀(Max Numerical Error)")
            ax.set_ylabel("Number of Models")
            ax.set_title("Distribution of Validation Errors")
            median_err = np.median(max_errors)
            ax.axvline(np.log10(median_err), color="red", linestyle="--",
                       label=f"Median: {median_err:.2e}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "validation_errors.png", dpi=150, bbox_inches="tight")
            plt.close()
    
    logger.info(f"Figures saved to {figures_dir}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("--figures", action="store_true", help="Generate PNG figures")
    parser.add_argument("--dry-run", action="store_true", help="Print RESULTS.md without writing")
    parser.add_argument("--mode", choices=["simplified", "canonical"], default="simplified")
    args = parser.parse_args()
    
    # Step 1: Rebuild CSV
    if not args.dry_run:
        rebuild_results_csv(args.mode)
    
    # Step 2: Generate RESULTS.md
    content = generate_results_md()
    
    if args.dry_run:
        print(content)
    else:
        output_path = Path(__file__).parent / "RESULTS.md"
        with open(output_path, "w") as f:
            f.write(content)
        logger.info(f"Updated {output_path}")
    
    # Step 3: Generate figures (optional)
    if args.figures and not args.dry_run:
        generate_figures()
    
    logger.info("Report generation complete!")


if __name__ == "__main__":
    main()
