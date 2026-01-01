#!/usr/bin/env python3
"""
Collect validated recast models into a separate directory.

This script:
1. Scans validation JSON files for overall_pass=True
2. Copies both original SBML (as Antimony) and recast files
3. Creates a manifest of validated models with classification info

Usage:
    python 6_collect_validated.py
"""

import json
import shutil
from pathlib import Path

import config


def copy_original_sbml(sbml_path: Path, output_path: Path) -> bool:
    """Copy the original SBML file to output directory."""
    try:
        shutil.copy(sbml_path, output_path)
        return True
    except Exception:
        return False


def collect_validated_models():
    """Collect all validated models into results/validated/."""
    validation_dir = Path(config.VALIDATION_DIR)
    recasts_dir = Path(config.RECASTS_DIR)
    sbml_dir = Path(config.SBML_CANDIDATES_DIR)
    output_dir = Path(config.RESULTS_DIR) / "validated"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Collecting Validated Models")
    print("=" * 60)

    # Find all passing validation files (Stage 1 numerical results)
    # New naming: {model_id}_{mode}_numerical.json
    validated = []
    for vfile in sorted(validation_dir.glob("*_numerical.json")):
        with open(vfile) as f:
            data = json.load(f)

        if data.get("overall_pass"):
            # Extract model_id from filename: MODEL_simplified_numerical.json
            model_id = vfile.stem.replace("_simplified_numerical", "")
            model_id = model_id.replace("_canonical_numerical", "")
            validated.append((model_id, data))

    print(f"Found {len(validated)} validated models")
    print()

    # Collect each validated model
    manifest = []
    for model_id, vdata in validated:
        print(f"  {model_id}...", end=" ")

        # Get classification info
        classification = vdata.get("classification", {})
        original_type = classification.get("original", "unknown")
        recast_type = classification.get("recast", "unknown")

        # Copy recast file
        recast_src = recasts_dir / f"{model_id}_simplified.ant"
        if not recast_src.exists():
            recast_src = recasts_dir / f"{model_id}_canonical.ant"

        if recast_src.exists():
            recast_dst = output_dir / f"{model_id}_recast.ant"
            shutil.copy(recast_src, recast_dst)
        else:
            print("SKIP (no recast file)")
            continue

        # Copy original SBML file
        sbml_src = sbml_dir / f"{model_id}.xml"
        if sbml_src.exists():
            original_dst = output_dir / f"{model_id}_original.xml"
            if not copy_original_sbml(sbml_src, original_dst):
                print("WARN (failed to copy SBML)")

        manifest.append({
            "model_id": model_id,
            "original_type": original_type,
            "recast_type": recast_type,
            "max_error": vdata.get("tests", {}).get("numerical", {}).get(
                "max_error", None
            ),
        })
        print("OK")

    # Write manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write manifest as CSV too
    csv_file = output_dir / "manifest.csv"
    with open(csv_file, "w") as f:
        f.write("model_id,original_type,recast_type,max_error\n")
        for m in manifest:
            max_err = m.get("max_error", "")
            if max_err is not None:
                max_err = f"{max_err:.2e}"
            f.write(f"{m['model_id']},{m['original_type']},"
                    f"{m['recast_type']},{max_err}\n")

    print()
    print(f"Collected {len(manifest)} validated models to {output_dir}/")
    print(f"  Manifest: {manifest_file}")
    print(f"  CSV: {csv_file}")

    # Summary by type
    print()
    print("Classification Summary:")
    print("-" * 40)
    orig_types = {}
    recast_types = {}
    for m in manifest:
        orig_types[m["original_type"]] = orig_types.get(
            m["original_type"], 0
        ) + 1
        recast_types[m["recast_type"]] = recast_types.get(
            m["recast_type"], 0
        ) + 1

    print("Original types:")
    for t, c in sorted(orig_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print()
    print("Recast types:")
    for t, c in sorted(recast_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    collect_validated_models()
