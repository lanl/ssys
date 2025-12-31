#!/usr/bin/env python3
"""
Analyze batch recast results and generate reports.

Creates comprehensive analysis including:
- Success rate statistics
- Performance metrics
- Complexity correlation analysis
- Visualizations
- Jupyter notebook report

Usage:
    python 4_analyze_results.py

Output:
    results/summary.json - Overall statistics
    results/report.ipynb - Jupyter notebook with analysis
    results/figures/ - Generated plots
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def load_results() -> pd.DataFrame:
    """Load batch recast results CSV."""
    csv_path = Path(config.RESULTS_DIR) / "batch_recast_results.csv"
    if not csv_path.exists():
        logger.error(f"Results not found: {csv_path}")
        logger.error("Run 3_recast_batch.py first")
        return pd.DataFrame()

    return pd.read_csv(csv_path)


def load_candidates() -> pd.DataFrame:
    """Load candidates CSV for complexity info."""
    csv_path = Path(config.CANDIDATES_CSV)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute comprehensive statistics."""
    total = len(df)
    if total == 0:
        return {}

    # Success rates
    recast_success = df["recast_success"].sum()
    validated = df["validation_attempted"].sum()
    val_pass = df["validation_pass"].sum()

    # Performance
    success_times = df[df["recast_success"]]["recast_time"]

    # Error analysis
    errors = df[df["error"].notna()]["error"]

    stats = {
        "total_models": int(total),
        "recast_success_count": int(recast_success),
        "recast_success_rate": float(recast_success / total),
        "validated_count": int(validated),
        "validation_pass_count": int(val_pass),
        "validation_pass_rate": float(val_pass / validated) if validated > 0 else 0,
        "performance": {
            "mean_time": float(success_times.mean()) if len(success_times) > 0 else 0,
            "median_time": float(success_times.median()) if len(success_times) > 0 else 0,
            "min_time": float(success_times.min()) if len(success_times) > 0 else 0,
            "max_time": float(success_times.max()) if len(success_times) > 0 else 0,
            "total_time": float(success_times.sum()) if len(success_times) > 0 else 0,
        },
        "error_count": int(len(errors)),
    }

    return stats


def analyze_by_complexity(results_df: pd.DataFrame, candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze success rates by model complexity."""
    if candidates_df.empty:
        return pd.DataFrame()

    # Merge results with candidate info
    merged = results_df.merge(
        candidates_df[["model_id", "n_species", "n_reactions", "n_parameters"]],
        on="model_id",
        how="left",
    )

    # Bin by complexity
    bins_species = [0, 5, 10, 20, 50, 100]
    bins_reactions = [0, 10, 20, 50, 100, 200]

    merged["species_bin"] = pd.cut(merged["n_species"], bins=bins_species)
    merged["reactions_bin"] = pd.cut(merged["n_reactions"], bins=bins_reactions)

    # Compute success rates per bin
    by_species = (
        merged.groupby("species_bin")
        .agg({"recast_success": ["count", "sum", "mean"]})
        .reset_index()
    )

    by_reactions = (
        merged.groupby("reactions_bin")
        .agg({"recast_success": ["count", "sum", "mean"]})
        .reset_index()
    )

    return merged, by_species, by_reactions


def create_pipeline_funnel(results_df: pd.DataFrame, candidates_df: pd.DataFrame, output_dir: Path):
    """Generate pipeline funnel chart showing data flow through stages."""
    # Get counts at each stage
    sbml_dir = Path(config.SBML_DOWNLOADS_DIR)
    candidates_dir = Path(config.SBML_CANDIDATES_DIR)
    recasts_dir = Path(config.RECASTS_DIR)
    validated_dir = Path(config.RESULTS_DIR) / "validated"
    
    sbml_count = len(list(sbml_dir.glob("*.xml"))) if sbml_dir.exists() else 0
    candidates_count = len(list(candidates_dir.glob("*.xml"))) if candidates_dir.exists() else 0
    transforms_count = len(list(recasts_dir.glob("*.ant"))) if recasts_dir.exists() else 0
    validated_count = len(list(validated_dir.glob("*.ant"))) if validated_dir.exists() else 0
    
    stages = ["Downloaded\nSBML", "Passed\nFilters", "Transformed", "Validated"]
    counts = [sbml_count, candidates_count, transforms_count, validated_count]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart (funnel-like)
    y_pos = range(len(stages))
    bars = ax.barh(y_pos, counts, color=colors, height=0.6, edgecolor="black")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel("Number of Models")
    ax.set_title("Pipeline Funnel: BioModels Transformation")
    
    # Add count labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                f"{count:,}", va="center", fontweight="bold")
    
    # Add percentage annotations
    for i, (count, prev) in enumerate(zip(counts[1:], counts[:-1]), 1):
        if prev > 0:
            pct = 100 * count / prev
            ax.text(counts[i] / 2, i, f"{pct:.0f}%", va="center", ha="center",
                    fontsize=10, color="white", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pipeline_funnel.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_transformation_types(output_dir: Path):
    """Generate bar chart of transformation type success stories."""
    manifest_path = Path(config.RESULTS_DIR) / "validated" / "manifest.csv"
    if not manifest_path.exists():
        logger.warning("No manifest.csv found - skipping transformation_types figure")
        return
    
    manifest_df = pd.read_csv(manifest_path)
    
    if "original_type" not in manifest_df.columns or "recast_type" not in manifest_df.columns:
        logger.warning("Manifest missing type columns - skipping transformation_types figure")
        return
    
    # Count transformation types
    transform_counts = {}
    for _, row in manifest_df.iterrows():
        orig = row.get("original_type", "Unknown")
        recast = row.get("recast_type", "Unknown")
        key = f"{orig} → {recast}"
        transform_counts[key] = transform_counts.get(key, 0) + 1
    
    if not transform_counts:
        return
    
    # Sort by count
    sorted_items = sorted(transform_counts.items(), key=lambda x: -x[1])
    labels, counts = zip(*sorted_items)
    
    # Color code: General→ is green, GMA→ is blue, S-system→ is gray
    colors = []
    for label in labels:
        if label.startswith("General"):
            colors.append("#2ecc71")  # Green - success stories
        elif label.startswith("GMA"):
            colors.append("#3498db")  # Blue
        else:
            colors.append("#95a5a6")  # Gray
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor="black")
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Models")
    ax.set_title("Transformation Achievements\n(Green = General ODE lifted to canonical form)")
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "transformation_types.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_error_breakdown(results_df: pd.DataFrame, output_dir: Path):
    """Generate pie chart of error categories."""
    errors = results_df[results_df["error"].notna()]["error"]
    if len(errors) == 0:
        logger.info("No errors found - skipping error_breakdown figure")
        return
    
    # Categorize errors
    categories = {}
    for err in errors:
        err_str = str(err).lower()
        if "timeout" in err_str:
            cat = "Timeout"
        elif "piecewise" in err_str:
            cat = "Piecewise"
        elif "event" in err_str:
            cat = "Events"
        elif "parse" in err_str or "syntax" in err_str:
            cat = "Parse Error"
        elif "symbol" in err_str:
            cat = "Undefined Symbol"
        else:
            cat = "Other"
        categories[cat] = categories.get(cat, 0) + 1
    
    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors[:len(labels)], startangle=90
    )
    ax.set_title(f"Transformation Failures by Category\n(n={sum(sizes)})")
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_validation_errors(output_dir: Path):
    """Generate histogram of numerical validation errors."""
    validation_dir = Path(config.VALIDATION_DIR)
    if not validation_dir.exists():
        logger.warning("No validation directory - skipping validation_errors figure")
        return
    
    max_errors = []
    for json_file in validation_dir.glob("*_validation.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if data.get("numerical", {}).get("passed"):
                max_err = data.get("numerical", {}).get("max_diff", 0)
                if max_err is not None and max_err > 0:
                    max_errors.append(max_err)
        except Exception:
            continue
    
    if len(max_errors) < 5:
        logger.info("Not enough validation data - skipping validation_errors figure")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Log scale for better visualization
    import numpy as np
    log_errors = np.log10(max_errors)
    
    ax.hist(log_errors, bins=30, edgecolor="black", alpha=0.7, color="#2ecc71")
    ax.set_xlabel("log₁₀(Max Numerical Error)")
    ax.set_ylabel("Number of Models")
    ax.set_title("Distribution of Validation Errors\n(Lower = More Accurate)")
    
    # Add summary stats
    median_err = np.median(max_errors)
    ax.axvline(np.log10(median_err), color="red", linestyle="--",
               label=f"Median: {median_err:.2e}")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "validation_errors.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_model_size_distribution(candidates_df: pd.DataFrame, output_dir: Path):
    """Generate histograms of model complexity metrics."""
    if candidates_df.empty:
        logger.warning("No candidates data - skipping model_size_distribution figure")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("n_species", "Number of Species", "#3498db"),
        ("n_reactions", "Number of Reactions", "#2ecc71"),
        ("n_parameters", "Number of Parameters", "#f39c12"),
    ]
    
    for ax, (col, title, color) in zip(axes, metrics):
        if col in candidates_df.columns:
            data = candidates_df[col].dropna()
            ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color=color)
            ax.set_xlabel(title)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {title}")
            ax.axvline(data.median(), color="red", linestyle="--",
                       label=f"Median: {data.median():.0f}")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_time_vs_complexity(results_df: pd.DataFrame, candidates_df: pd.DataFrame, output_dir: Path):
    """Generate scatter plot of recast time vs model complexity."""
    if candidates_df.empty:
        logger.warning("No candidates data - skipping time_vs_complexity figure")
        return
    
    # Merge results with complexity data
    merged = results_df.merge(
        candidates_df[["model_id", "n_species", "n_reactions"]],
        on="model_id", how="left"
    )
    
    # Filter to successful recasts with valid times
    success = merged[(merged["recast_success"]) & (merged["recast_time"] > 0)]
    
    if len(success) < 10:
        logger.info("Not enough data - skipping time_vs_complexity figure")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time vs species
    ax = axes[0]
    ax.scatter(success["n_species"], success["recast_time"], alpha=0.5, s=20)
    ax.set_xlabel("Number of Species")
    ax.set_ylabel("Recast Time (seconds)")
    ax.set_title("Transformation Time vs Species Count")
    ax.grid(alpha=0.3)
    
    # Time vs reactions
    ax = axes[1]
    ax.scatter(success["n_reactions"], success["recast_time"], alpha=0.5, s=20, color="orange")
    ax.set_xlabel("Number of Reactions")
    ax.set_ylabel("Recast Time (seconds)")
    ax.set_title("Transformation Time vs Reaction Count")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_complexity.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_feature_prevalence(candidates_df: pd.DataFrame, output_dir: Path):
    """Generate bar chart of feature prevalence in models."""
    if candidates_df.empty:
        logger.warning("No candidates data - skipping feature_prevalence figure")
        return
    
    # Count features
    feature_cols = [
        ("has_exp", "exp()"),
        ("has_log", "log/ln()"),
        ("has_sin_cos", "sin/cos"),
        ("has_piecewise", "piecewise"),
    ]
    
    features = []
    counts = []
    for col, label in feature_cols:
        if col in candidates_df.columns:
            features.append(label)
            counts.append(candidates_df[col].sum())
    
    if not features:
        logger.warning("No feature columns found - skipping feature_prevalence figure")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(features, counts, color=colors[:len(features)], edgecolor="black")
    
    ax.set_ylabel("Number of Models")
    ax.set_title(f"Feature Prevalence in Candidate Models\n(n={len(candidates_df)} candidates)")
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                str(count), ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_prevalence.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_visualizations(results_df: pd.DataFrame, candidates_df: pd.DataFrame, output_dir: Path):
    """Generate all analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating pipeline funnel...")
    create_pipeline_funnel(results_df, candidates_df, output_dir)
    
    logger.info("Generating transformation types chart...")
    create_transformation_types(output_dir)
    
    logger.info("Generating error breakdown...")
    create_error_breakdown(results_df, output_dir)
    
    logger.info("Generating validation errors histogram...")
    create_validation_errors(output_dir)
    
    logger.info("Generating model size distribution...")
    create_model_size_distribution(candidates_df, output_dir)
    
    logger.info("Generating time vs complexity...")
    create_time_vs_complexity(results_df, candidates_df, output_dir)
    
    logger.info("Generating feature prevalence...")
    create_feature_prevalence(candidates_df, output_dir)

    # Original figures
    logger.info("Generating success rates overview...")

    # Figure 1: Success rates overview
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart of outcomes
    ax = axes[0]
    success = results_df["recast_success"].sum()
    failure = len(results_df) - success
    ax.pie(
        [success, failure],
        labels=["Success", "Failure"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
    )
    ax.set_title("Recast Success Rate")

    # Bar chart of validation
    ax = axes[1]
    validated = results_df["validation_attempted"].sum()
    val_pass = results_df["validation_pass"].sum()
    val_fail = validated - val_pass
    not_validated = len(results_df) - validated

    categories = ["Pass", "Fail", "Not Validated"]
    counts = [val_pass, val_fail, not_validated]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    ax.bar(categories, counts, color=colors)
    ax.set_title("Validation Results")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "success_rates.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Performance histogram
    if results_df["recast_success"].sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        success_times = results_df[results_df["recast_success"]]["recast_time"]
        ax.hist(success_times, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Recast Time (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Recast Time Distribution")
        ax.axvline(
            success_times.median(),
            color="red",
            linestyle="--",
            label=f"Median: {success_times.median():.2f}s",
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "timing_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Figure 3: Success by complexity
    if not candidates_df.empty:
        merged, _, _ = analyze_by_complexity(results_df, candidates_df)

        if not merged.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Success vs species count
            ax = axes[0]
            species_success = merged.groupby("n_species")["recast_success"].agg(["sum", "count"])
            species_success["rate"] = species_success["sum"] / species_success["count"]

            ax.scatter(
                species_success.index,
                species_success["rate"],
                s=species_success["count"] * 10,
                alpha=0.6,
            )
            ax.set_xlabel("Number of Species")
            ax.set_ylabel("Success Rate")
            ax.set_title("Success Rate vs Model Size (Species)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

            # Success vs reactions count
            ax = axes[1]
            rxn_success = merged.groupby("n_reactions")["recast_success"].agg(["sum", "count"])
            rxn_success["rate"] = rxn_success["sum"] / rxn_success["count"]

            ax.scatter(
                rxn_success.index,
                rxn_success["rate"],
                s=rxn_success["count"] * 10,
                alpha=0.6,
                color="orange",
            )
            ax.set_xlabel("Number of Reactions")
            ax.set_ylabel("Success Rate")
            ax.set_title("Success Rate vs Model Size (Reactions)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "complexity_analysis.png", dpi=150, bbox_inches="tight")
            plt.close()


def generate_notebook(
    results_df: pd.DataFrame, candidates_df: pd.DataFrame, stats: dict, output_path: Path
):
    """Generate Jupyter notebook with analysis."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()

    cells = []

    # Title
    cells.append(
        nbf.v4.new_markdown_cell(
            f"# BioModels Benchmark Results\n\nAnalysis of {stats.get('total_models', 0)} models\n"
        )
    )

    # Summary statistics
    cells.append(nbf.v4.new_markdown_cell("## Summary Statistics"))

    cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import json\n\n"
            "# Load summary\n"
            "with open('summary.json', 'r') as f:\n"
            "    stats = json.load(f)\n\n"
            "print(f\"Total models: {stats['total_models']}\")\n"
            "print(f\"Recast success: {stats['recast_success_count']} "
            f'({stats["recast_success_rate"] * 100:.1f}%)")\n'
            "print(f\"Validation pass: {stats['validation_pass_count']} "
            f'({stats["validation_pass_rate"] * 100:.1f}% of validated)")\n'
        )
    )

    # Visualizations
    cells.append(nbf.v4.new_markdown_cell("## Success Rates"))
    cells.append(
        nbf.v4.new_code_cell(
            "from IPython.display import Image\nImage('figures/success_rates.png')"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Performance"))
    cells.append(nbf.v4.new_code_cell("Image('figures/timing_distribution.png')"))

    cells.append(nbf.v4.new_markdown_cell("## Complexity Analysis"))
    cells.append(nbf.v4.new_code_cell("Image('figures/complexity_analysis.png')"))

    # Detailed results table
    cells.append(nbf.v4.new_markdown_cell("## Detailed Results"))
    cells.append(
        nbf.v4.new_code_cell(
            "# Load results\nresults = pd.read_csv('batch_recast_results.csv')\nresults.head(20)"
        )
    )

    nb["cells"] = cells

    # Write notebook
    with open(output_path, "w") as f:
        nbf.write(nb, f)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Analyze batch recast results")

    parser.parse_args()  # Parse for validation but don't use

    # Set up logging
    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("BioModels Analysis Script")
    logger.info("=" * 60)

    # Load data
    results_df = load_results()
    if results_df.empty:
        return

    candidates_df = load_candidates()

    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(results_df)

    # Save summary JSON
    summary_path = Path(config.SUMMARY_JSON)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\nBenchmark Summary")
    print("=" * 60)
    print(f"Total models: {stats['total_models']}")
    print(
        f"Recast success: {stats['recast_success_count']} "
        f"({stats['recast_success_rate'] * 100:.1f}%)"
    )
    print(f"Validated: {stats['validated_count']}")
    print(
        f"Validation pass: {stats['validation_pass_count']} "
        f"({stats['validation_pass_rate'] * 100:.1f}% of validated)"
    )
    print("\nPerformance:")
    print(f"  Mean time: {stats['performance']['mean_time']:.2f}s")
    print(f"  Median time: {stats['performance']['median_time']:.2f}s")
    print(f"  Total time: {stats['performance']['total_time']:.1f}s")

    # Create visualizations
    logger.info("Generating visualizations...")
    figures_dir = Path(config.RESULTS_DIR) / "figures"
    create_visualizations(results_df, candidates_df, figures_dir)
    logger.info(f"Saved figures to {figures_dir}")

    # Generate notebook
    logger.info("Generating analysis notebook...")
    notebook_path = Path(config.REPORT_NOTEBOOK)
    generate_notebook(results_df, candidates_df, stats, notebook_path)
    logger.info(f"Saved notebook to {notebook_path}")

    logger.info("\nAnalysis complete!")
    logger.info(f"View results: jupyter notebook {notebook_path}")


if __name__ == "__main__":
    main()
