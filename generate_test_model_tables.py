#!/usr/bin/env python3
"""
Generate markdown tables for TEST_MODELS.md from recast output.

This script parses validation JSON files and recast .ant files to:
1. Extract original/recast classification
2. Infer which recasting rules were applied
3. Generate markdown tables with provenance

Usage:
    python generate_test_model_tables.py [--update-readme]
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def detect_rules(recast_ant_path: Path, validation_json: dict) -> list[int]:
    """
    Detect which recasting rules were applied by analyzing the recast .ant file.
    
    Rules:
    1 - Exponential lifting (exp)
    2 - Logarithmic lifting (log)  
    3 - Trigonometric lifting (sin/cos)
    4 - Sum handling (denominator or product splitting)
    5 - Product rule (factoring via pool construction)
    6 - ε-splitting (slack variables)
    7 - Clock state (time → T)
    """
    rules = set()
    
    if not recast_ant_path.exists():
        return sorted(rules)
    
    content = recast_ant_path.read_text()
    
    # Rule 1: Exponential lifting - look for exp() in auxiliary definitions
    if re.search(r'//\s*Z_\d+\s*:=\s*exp\(', content):
        rules.add(1)
    
    # Rule 2: Logarithmic lifting - look for log() in auxiliary definitions  
    if re.search(r'//\s*Z_\d+\s*:=\s*log\(', content):
        rules.add(2)
    
    # Rule 3: Trigonometric lifting - look for sin/cos in auxiliary definitions
    # Note: sin/cos auxiliaries have +2 offset, so look for patterns like "sin(...) + 2"
    if re.search(r'//\s*Z_\d+\s*:=\s*(sin|cos)\(', content):
        rules.add(3)
    
    # Rule 4: Sum handling - look for Y_* := ... + ... (sum in definition)
    # Also detect when denominators are lifted (rational functions)
    if re.search(r'//\s*Y_\d+\s*:=\s*[^+]+\+', content):
        rules.add(4)
    # Also check for sum handling in Z_* definitions (e.g., Z_1 := K + X)
    if re.search(r'//\s*Z_\d+\s*:=\s*[A-Za-z_]\w*\s*\+', content):
        rules.add(4)
    
    # Rule 5: Product rule - look for factor_map with products (X = Z_1*Z_2)
    # Pattern: "// X = Z_1*Z_2" or more factors
    if re.search(r'//\s*\w+\s*=\s*Z_\d+\s*\*\s*Z_\d+', content):
        rules.add(5)
    
    # Rule 6: ε-splitting - look for epsilon in equations
    if re.search(r'\bepsilon\b', content):
        rules.add(6)
    
    # Rule 7: Clock state - look for T := time or T' = 1
    if re.search(r'//\s*T\s*:=\s*time', content) or re.search(r"\bT'\s*=\s*1\s*;", content):
        rules.add(7)
    
    # Also check for assignment rules with T (time-varying coefficients)
    if re.search(r':=\s*[^;]*\bT\b', content):
        rules.add(7)
    
    return sorted(rules)


def format_rules(rules: list[int]) -> str:
    """Format rule list for table display."""
    if not rules:
        return "—"
    return ", ".join(str(r) for r in rules)


def has_time_varying_coefficients(content: str) -> bool:
    """
    Check if model has time-varying coefficients (assignment rules with T).
    
    Models with assignment rules like `beta_t := f(T)` or `beta_t := f(t)`
    are not strict GMA/S-system because coefficients depend on time.
    """
    # Look for assignment rules section (case-insensitive)
    if "assignment rules" not in content.lower():
        return False
    
    # Extract assignment rules section
    # Check if any rule references clock state T or t (lowercase for
    # models using t as clock variable)
    # Pattern: `:= ... T ...` or `:= ... t ...` (word boundary)
    if re.search(r':=\s*[^;]*\bT\b', content):
        return True
    # Also check for lowercase t in assignment rules (some models use t)
    if re.search(r':=\s*[^;]*\bt\b', content):
        return True
    return False


def get_recast_classification_from_file(recast_ant: Path) -> str:
    """
    Get the actual classification from recast .ant file content.
    
    This is more reliable than the JSON classification because it checks
    the actual format markers in the generated file.
    
    Categories:
    - S-system: 1-2 monomial terms per equation
    - GMA: Multiple monomials with constant coefficients
    - GMA (time-varying coefficients): Power-law with T-dependent coefficients
    - General: Contains non-monomial terms (shouldn't happen for successful recast)
    """
    if not recast_ant.exists():
        return "Unknown"
    
    content = recast_ant.read_text()
    
    # Check for time-varying coefficients FIRST (overrides other classifications)
    is_time_varying = has_time_varying_coefficients(content)
    
    # Check for GMA marker
    if "// GMA (Generalized Mass Action) format" in content:
        if is_time_varying:
            return "GMA (time-varying coefficients)"
        return "GMA"
    
    # Check for S-system dynamics marker
    if "// S-SYSTEM DYNAMICS" in content:
        if is_time_varying:
            return "GMA (time-varying coefficients)"
        # Check for Canonical S-system (strict two-term)
        if "Canonical S-system" in content:
            return "Canonical S-system"
        return "S-system"
    
    # Fallback: General if no markers found
    return "General"


def parse_output_directory(out_dir: Path) -> list[dict]:
    """Parse all validation JSON files in an output directory."""
    results = []
    
    for json_file in sorted(out_dir.glob("*_validation.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract model name from filename
            model_name = json_file.stem.replace("_validation", "")
            
            # Get original classification from JSON (this is reliable)
            original = data.get("classification", {}).get("original", "Unknown")
            
            # Find corresponding recast .ant file
            recast_ant = out_dir / f"{model_name}_recast.ant"
            
            # Get recast classification from file content (more reliable)
            recast = get_recast_classification_from_file(recast_ant)
            
            # Detect rules
            rules = detect_rules(recast_ant, data)
            
            results.append({
                "name": model_name,
                "original": original,
                "recast": recast,
                "rules": rules,
                "passed": data.get("overall_pass", False)
            })
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
    
    return results


def generate_table(results: list[dict], dir_name: str, include_source: bool = False) -> str:
    """Generate markdown table from results."""
    lines = []
    
    if include_source:
        lines.append("| File | Description | Original | Recast | Rules | Source |")
        lines.append("|------|-------------|----------|--------|-------|--------|")
    else:
        lines.append("| File | Description | Original | Recast | Rules |")
        lines.append("|------|-------------|----------|--------|-------|")
    
    for r in results:
        # Format model name (remove prefix like m01_, S1987_, etc.)
        display_name = r["name"]
        
        # Description placeholder - could be enhanced with model metadata
        description = "—"
        
        rules_str = format_rules(r["rules"])
        
        if include_source:
            lines.append(f"| {display_name} | {description} | {r['original']} | {r['recast']} | {rules_str} | |")
        else:
            lines.append(f"| {display_name} | {description} | {r['original']} | {r['recast']} | {rules_str} |")
    
    return "\n".join(lines)


def generate_summary_table(all_results: dict[str, list[dict]]) -> str:
    """Generate summary statistics table with 4 categories."""
    lines = []
    lines.append(
        "| Directory | → S-system | → GMA | → GMA (time-varying) | Total |"
    )
    lines.append(
        "|-----------|------------|-------|----------------------|-------|"
    )
    
    totals = {"S-system": 0, "GMA": 0, "GMA_TV": 0, "total": 0}
    
    for dir_name, results in sorted(all_results.items()):
        counts = {"S-system": 0, "GMA": 0, "GMA_TV": 0}
        for r in results:
            recast = r["recast"]
            if "time-varying" in recast:
                counts["GMA_TV"] += 1
            elif "S-system" in recast or recast == "Canonical S-system":
                counts["S-system"] += 1
            elif recast == "GMA":
                counts["GMA"] += 1
            else:
                # General or Unknown - group with GMA for now
                counts["GMA"] += 1
        
        total = len(results)
        lines.append(
            f"| `{dir_name}/` | {counts['S-system']} | "
            f"{counts['GMA']} | {counts['GMA_TV']} | {total} |"
        )
        
        totals["S-system"] += counts["S-system"]
        totals["GMA"] += counts["GMA"]
        totals["GMA_TV"] += counts["GMA_TV"]
        totals["total"] += total
    
    lines.append(
        f"| **Total** | **{totals['S-system']}** | "
        f"**{totals['GMA']}** | **{totals['GMA_TV']}** | "
        f"**{totals['total']}** |"
    )
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    # Find all output directories
    base_dir = Path(".")
    out_dirs = sorted(base_dir.glob("out_test_models*"))
    
    if not out_dirs:
        print("No output directories found (out_test_models*)", file=sys.stderr)
        sys.exit(1)
    
    all_results = {}
    
    print("=" * 70)
    print("TEST MODEL TABLE GENERATION")
    print("=" * 70)
    print(f"Generated: {datetime.now().isoformat()}")
    print()
    
    for out_dir in out_dirs:
        # Extract test_models directory name from out_* directory
        dir_name = out_dir.name.replace("out_", "")
        
        print(f"Processing {out_dir}...")
        results = parse_output_directory(out_dir)
        
        if results:
            all_results[dir_name] = results
            print(f"  Found {len(results)} models")
    
    print()
    
    # Generate summary table
    print("## Summary by Directory")
    print()
    print(generate_summary_table(all_results))
    print()
    
    # Generate individual tables
    for dir_name, results in sorted(all_results.items()):
        print(f"## {dir_name}/")
        print()
        print(generate_table(results, dir_name))
        print()
    
    # Provenance
    print("---")
    print()
    print("**Provenance**: Generated by `generate_test_model_tables.py` from validation JSON files.")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
