#!/usr/bin/env python3
"""
Filter fetched models to identify recast candidates.

Uses libSBML to analyze SBML files directly (no Antimony conversion).
Applies heuristics to classify models as:
- Already GMA (power-law form, no recasting needed)
- Recast candidates (eligible for transformation)
- Not suitable (events, delays, etc.)

Usage:
    python 2_filter_models.py

Output:
    results/candidates.csv - Classified models with metadata
    data/sbml_candidates/  - Copies of qualifying SBML files
"""

import argparse
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    import libsbml

# Add parent directory to path for ssys import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

logger = logging.getLogger(__name__)


def load_sbml(model_id: str) -> Optional["libsbml.Model"]:
    """Load SBML model using libSBML."""
    try:
        import libsbml
    except ImportError:
        raise ImportError("python-libsbml is required. Install with: pip install python-libsbml")

    sbml_path = Path(config.SBML_DOWNLOADS_DIR) / f"{model_id}.xml"
    if not sbml_path.exists():
        return None

    try:
        doc = libsbml.readSBML(str(sbml_path))

        # Check for fatal errors
        if doc.getNumErrors() > 0:
            for i in range(doc.getNumErrors()):
                err = doc.getError(i)
                if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                    logger.debug(f"SBML error in {model_id}: {err.getMessage()}")

        return doc.getModel()
    except Exception as e:
        logger.error(f"Failed to read {model_id}: {e}")
        return None


def is_power_law_monomial(ast_node) -> bool:
    """
    Check if an AST node represents a power-law monomial.

    A power-law monomial is a product of terms like: k * X1^g1 * X2^g2 * ...
    This means: only multiplication, powers, constants, and names.
    No addition, subtraction, division (ratios), or other functions.
    """
    import libsbml

    if ast_node is None:
        return True  # Empty is trivially a monomial

    node_type = ast_node.getType()

    # Constants and names are always OK
    if node_type in (
        libsbml.AST_INTEGER,
        libsbml.AST_REAL,
        libsbml.AST_REAL_E,
        libsbml.AST_RATIONAL,
        libsbml.AST_NAME,
        libsbml.AST_CONSTANT_E,
        libsbml.AST_CONSTANT_PI,
    ):
        return True

    # Multiplication is OK if all children are monomials
    if node_type == libsbml.AST_TIMES:
        for i in range(ast_node.getNumChildren()):
            if not is_power_law_monomial(ast_node.getChild(i)):
                return False
        return True

    # Power (exponentiation) is OK - base^exponent
    if node_type == libsbml.AST_POWER:
        base = ast_node.getChild(0)
        return is_power_law_monomial(base)

    # Division breaks the monomial form (creates ratios like Michaelis-Menten)
    if node_type == libsbml.AST_DIVIDE:
        return False

    # Addition/subtraction break the monomial form
    if node_type in (libsbml.AST_PLUS, libsbml.AST_MINUS):
        return False

    # Functions (exp, log, sin, etc.) break the monomial form
    if node_type in (
        libsbml.AST_FUNCTION,
        libsbml.AST_FUNCTION_ABS,
        libsbml.AST_FUNCTION_EXP,
        libsbml.AST_FUNCTION_LN,
        libsbml.AST_FUNCTION_LOG,
        libsbml.AST_FUNCTION_SIN,
        libsbml.AST_FUNCTION_COS,
        libsbml.AST_FUNCTION_TAN,
        libsbml.AST_FUNCTION_PIECEWISE,
    ):
        return False

    # Unary minus: check the child
    if node_type == libsbml.AST_MINUS and ast_node.getNumChildren() == 1:
        return is_power_law_monomial(ast_node.getChild(0))

    # For other cases, be conservative and say it's not a monomial
    return False


def is_model_gma(model) -> bool:
    """
    Check if a model is already in GMA (Generalized Mass Action) form.

    A model is GMA if all reaction rate laws are power-law monomials.
    """
    if model is None:
        return False

    # Must have at least one reaction
    if model.getNumReactions() == 0:
        return False

    # Check each reaction's rate law
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()

        if kl is None or kl.getMath() is None:
            return False

        if not is_power_law_monomial(kl.getMath()):
            return False

    return True


def has_sbml_l3_packages(sbml_path: str) -> bool:
    """Check if SBML file uses L3 packages (layout, fbc, etc.)."""
    try:
        with open(sbml_path) as f:
            header = f.read(2000)  # Read just the header

        # Check for common L3 package namespaces
        l3_packages = [
            "layout:",
            "layout_L2",
            "fbc:",
            "comp:",
            "qual:",
            "multi:",
            "render:",
            "groups:",
            "distrib:",
        ]
        return any(pkg in header for pkg in l3_packages)
    except Exception:
        return False


def detect_sbml_features(model, sbml_path: str = None) -> dict:
    """Detect model features using libSBML."""
    import libsbml

    features = {
        "events": False,
        "delays": False,
        "algebraic_rules": False,
        "piecewise": False,
        "piecewise_heavy": False,
        "time_dependent": False,
        "sin_cos": False,  # Supported trig (sin, cos)
        "unsupported_trig": False,  # Unsupported trig (tan, tanh)
        "exp": False,
        "log": False,
        "negative_species": False,
        "is_gma": False,
        "sbml_l3_packages": False,
    }

    if model is None:
        return features

    # Check events
    features["events"] = model.getNumEvents() > 0

    # Check rules
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.getTypeCode() == libsbml.SBML_ALGEBRAIC_RULE:
            features["algebraic_rules"] = True

    def check_formula_for_features(formula_str: str):
        """Check a formula string for special features."""
        if not formula_str:
            return

        formula_lower = formula_str.lower()

        if "delay(" in formula_lower:
            features["delays"] = True

        if "piecewise(" in formula_lower:
            features["piecewise"] = True
            if formula_lower.count("piecewise(") > 2:
                features["piecewise_heavy"] = True

        if "time" in formula_lower or " t " in formula_lower:
            features["time_dependent"] = True

        # Supported trig: sin, cos
        if "sin(" in formula_lower or "cos(" in formula_lower:
            features["sin_cos"] = True
        # Unsupported trig: tan, tanh (ssys only supports sin/cos)
        if "tan(" in formula_lower or "tanh(" in formula_lower:
            features["unsupported_trig"] = True

        if "exp(" in formula_lower:
            features["exp"] = True
        if "log(" in formula_lower or "ln(" in formula_lower:
            features["log"] = True

    # Check all reactions' kinetic laws
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl and kl.getMath():
            formula_str = libsbml.formulaToString(kl.getMath())
            check_formula_for_features(formula_str)

    # Check all rules
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.getMath():
            formula_str = libsbml.formulaToString(rule.getMath())
            check_formula_for_features(formula_str)

    # Check for negative initial species
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if sp.isSetInitialAmount():
            if sp.getInitialAmount() < 0:
                features["negative_species"] = True
        elif sp.isSetInitialConcentration():
            if sp.getInitialConcentration() < 0:
                features["negative_species"] = True

    # Check if model is already GMA
    features["is_gma"] = is_model_gma(model)

    # Check for L3 packages
    if sbml_path:
        features["sbml_l3_packages"] = has_sbml_l3_packages(sbml_path)

    return features


def classify_model(model_id: str, model, sbml_path: str = None) -> dict:
    """Classify a model for recast suitability using libSBML."""
    if model is None:
        return {
            "model_id": model_id,
            "n_species": 0,
            "n_reactions": 0,
            "n_parameters": 0,
            "can_attempt": False,
            "s_system_candidate": False,
            "gma_candidate": False,
            "is_already_gma": False,
            "blockers": "parse_error",
            "warnings": "",
            "has_events": False,
            "has_delays": False,
            "has_piecewise": False,
            "has_sin_cos": False,
            "has_unsupported_trig": False,
            "has_exp": False,
            "has_log": False,
        }

    features = detect_sbml_features(model, sbml_path)

    # Count species (floating only)
    n_species = 0
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if not sp.getBoundaryCondition():
            n_species += 1

    n_reactions = model.getNumReactions()
    n_params = model.getNumParameters()

    # Identify blockers
    blockers = []
    for feature in config.BLOCKING_FEATURES:
        if features.get(feature, False):
            blockers.append(feature)

    if n_species == 0:
        blockers.append("no_dynamics")

    # Identify warnings
    warnings = []
    for feature in config.WARNING_FEATURES:
        if features.get(feature, False):
            warnings.append(feature)

    can_attempt = len(blockers) == 0 and n_species > 0

    s_system_candidate = (
        can_attempt
        and n_species > 0
        and n_species <= 10
        and n_reactions > 0
        and n_reactions <= 20
        and not features.get("piecewise_heavy", False)
        and not features.get("sign_changing_trig", False)
    )

    gma_candidate = (
        can_attempt
        and n_species > 0
        and n_species <= config.MAX_SPECIES
        and n_reactions <= config.MAX_REACTIONS
        and n_params <= config.MAX_PARAMETERS
    )

    return {
        "model_id": model_id,
        "n_species": n_species,
        "n_reactions": n_reactions,
        "n_parameters": n_params,
        "can_attempt": can_attempt,
        "s_system_candidate": s_system_candidate,
        "gma_candidate": gma_candidate,
        "is_already_gma": features.get("is_gma", False),
        "blockers": ",".join(blockers) if blockers else "",
        "warnings": ",".join(warnings) if warnings else "",
        "has_events": features.get("events", False),
        "has_delays": features.get("delays", False),
        "has_piecewise": features.get("piecewise", False),
        "has_sin_cos": features.get("sin_cos", False),
        "has_unsupported_trig": features.get("unsupported_trig", False),
        "has_exp": features.get("exp", False),
        "has_log": features.get("log", False),
    }


def filter_all_models() -> pd.DataFrame:
    """Filter all fetched models using libSBML."""
    sbml_dir = Path(config.SBML_DOWNLOADS_DIR)
    if not sbml_dir.exists():
        logger.error(f"SBML directory not found: {sbml_dir}")
        return pd.DataFrame()

    sbml_files = list(sbml_dir.glob("*.xml"))
    model_ids = [f.stem for f in sbml_files]

    if not model_ids:
        logger.error("No models found to filter")
        return pd.DataFrame()

    logger.info(f"Filtering {len(model_ids)} models...")

    results = []
    for model_id in tqdm(model_ids, desc="Classifying models"):
        sbml_path = str(sbml_dir / f"{model_id}.xml")
        model = load_sbml(model_id)
        classification = classify_model(model_id, model, sbml_path)
        results.append(classification)

    return pd.DataFrame(results)


def copy_candidates(df: pd.DataFrame):
    """Copy qualifying SBML files to candidates directory."""
    candidates = df[df["gma_candidate"]]["model_id"].tolist()

    if not candidates:
        logger.warning("No candidate models to copy")
        return 0

    candidates_dir = Path(config.SBML_CANDIDATES_DIR)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for model_id in candidates:
        src = Path(config.SBML_DOWNLOADS_DIR) / f"{model_id}.xml"
        dst = candidates_dir / f"{model_id}.xml"

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    logger.info(f"Copied {copied} candidate SBML files to {candidates_dir}")
    return copied


def generate_summary(df: pd.DataFrame) -> str:
    """Generate clear, hierarchical summary of filtering results."""
    total = len(df)
    if total == 0:
        return "No models processed."

    # Count blockers for exclusion breakdown
    all_blockers = []
    for blockers_str in df["blockers"]:
        if blockers_str:
            all_blockers.extend(blockers_str.split(","))
    blocker_counts = Counter(all_blockers)

    # Calculate key metrics
    excluded = int(total - df["can_attempt"].sum())
    eligible = int(df["can_attempt"].sum())
    candidates = int(df["gma_candidate"].sum())
    excluded_by_size = eligible - candidates

    # Size categories (among eligible models)
    eligible_df = df[df["can_attempt"]]
    small = len(eligible_df[(eligible_df["n_species"] <= 10) & (eligible_df["n_reactions"] <= 20)])
    medium = len(eligible_df[(eligible_df["n_species"] > 10) & (eligible_df["n_species"] <= 100)])
    large = len(eligible_df[eligible_df["n_species"] > 100])

    # Already GMA
    already_gma_eligible = (
        int(eligible_df["is_already_gma"].sum()) if "is_already_gma" in eligible_df.columns else 0
    )

    # Special features (among all models)
    has_exp = int(df["has_exp"].sum())
    has_log = int(df["has_log"].sum())
    has_sin_cos = int(df["has_sin_cos"].sum())
    has_piecewise = int(df["has_piecewise"].sum())

    # Build summary
    lines = []
    lines.append("")
    lines.append("BioModels Filtering Summary")
    lines.append("===========================")
    lines.append("")
    lines.append(f"INPUT:  {total} models analyzed")
    lines.append("")

    # Funnel visualization
    lines.append("FILTERING FUNNEL:")
    lines.append(f"  {total:4d}  Total models")
    lines.append(f"  -{excluded:4d}  Excluded (blockers)")
    lines.append("  -----")
    lines.append(f"  {eligible:4d}  Eligible (no blockers)")
    lines.append(f"  -{excluded_by_size:4d}  Excluded (too large)")
    lines.append("  -----")
    lines.append(f"  {candidates:4d}  CANDIDATES for transformation")
    lines.append("")

    # Blocker breakdown
    lines.append(f"BLOCKER DETAILS ({excluded} models excluded):")
    lines.append("  Note: Models may have multiple blockers (counts overlap)")

    blocker_labels = {
        "no_dynamics": "No ODEs (0 floating species)",
        "events": "Contains discrete events",
        "parse_error": "SBML parse errors",
        "delays": "Contains delay functions",
        "algebraic_rules": "Algebraic constraints",
        "sbml_l3_packages": "Uses SBML L3 packages",
        "unsupported_trig": "Unsupported trig (tan/tanh)",
    }

    for blocker in ["no_dynamics", "events", "parse_error", "delays",
                    "algebraic_rules", "sbml_l3_packages", "unsupported_trig"]:
        if blocker in blocker_counts:
            label = blocker_labels.get(blocker, blocker)
            count = blocker_counts[blocker]
            lines.append(f"    {count:4d}  {label}")

    # Any unlisted blockers
    for blocker, count in sorted(blocker_counts.items()):
        if blocker not in blocker_labels:
            lines.append(f"    {count:4d}  {blocker}")

    lines.append("")

    # Size distribution
    lines.append(f"SIZE DISTRIBUTION ({eligible} eligible models):")
    lines.append(f"    {small:4d}  Small (≤10 species, ≤20 reactions)")
    lines.append(f"    {medium:4d}  Medium (11-100 species)")
    lines.append(f"    {large:4d}  Large (>100 species) - may be slow")
    lines.append("")
    lines.append(f"    {already_gma_eligible:4d}  Already in GMA form (no transformation needed)")
    lines.append("")

    # Complexity stats
    if len(eligible_df) > 0:
        lines.append("COMPLEXITY STATISTICS (eligible models):")
        lines.append(
            f"    Species:    median={eligible_df['n_species'].median():.0f}, "
            f"range=[{eligible_df['n_species'].min()}, {eligible_df['n_species'].max()}]"
        )
        lines.append(
            f"    Reactions:  median={eligible_df['n_reactions'].median():.0f}, "
            f"range=[{eligible_df['n_reactions'].min()}, {eligible_df['n_reactions'].max()}]"
        )
        lines.append(
            f"    Parameters: median={eligible_df['n_parameters'].median():.0f}, "
            f"range=[{eligible_df['n_parameters'].min()}, {eligible_df['n_parameters'].max()}]"
        )
        lines.append("")

    # Special features
    lines.append("SPECIAL FEATURES (all models):")
    lines.append(f"    {has_exp:4d}  Contains exp()")
    lines.append(f"    {has_log:4d}  Contains log/ln()")
    lines.append(f"    {has_sin_cos:4d}  Contains sin/cos (supported)")
    lines.append(f"    {has_piecewise:4d}  Contains piecewise functions")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Filter models to identify recast candidates")
    parser.add_argument(
        "--output", type=str, default=config.CANDIDATES_CSV, help="Output CSV file path"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy candidate SBML files to candidates directory",
    )

    args = parser.parse_args()

    # Set up logging
    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("BioModels Filter Script (libSBML-based)")
    logger.info("=" * 60)

    # Filter models
    df = filter_all_models()

    if df.empty:
        logger.error("No models to filter. Run 1_fetch_models.py first.")
        return

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Copy candidate SBML files
    if not args.no_copy:
        copy_candidates(df)

    # Print summary
    summary = generate_summary(df)
    print(summary)

    # Also save summary
    summary_path = output_path.parent / "filter_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"Saved summary to {summary_path}")

    logger.info("\nFiltering complete!")


if __name__ == "__main__":
    main()
