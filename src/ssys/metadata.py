"""Metadata parsing and emission helpers for generated Antimony."""

import re

from ssys.types import SolverRequirement


def format_antimony_number(value: object) -> str:
    """Format a numeric Antimony literal without losing float round-trip precision."""
    val = float(value)
    if val == 0.0:
        return "0"
    text = repr(val)
    if text.endswith(".0") and val.is_integer():
        return text[:-2]
    return text


def _format_antimony_number(value: object) -> str:
    """Backward-compatible alias for Antimony numeric formatting."""
    return format_antimony_number(value)


def extract_sim_metadata(
    text: str,
) -> tuple[float | None, float | None, int | None, float | None, float | None]:
    """
    Extract @SIM metadata from Antimony comments.

    Format: // @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6 EPS_SLACK=1e-10
    """
    t_start = None
    t_end = None
    n_steps = None
    eps_init = None
    eps_slack = None

    sim_marker_pattern = re.compile(r"@SIM\b")
    key_value_pattern = re.compile(r"(\w+)\s*=\s*([0-9.eE+-]+)")

    for line in text.splitlines():
        if "//" not in line:
            continue
        comment_part = line.split("//", 1)[1]
        if not sim_marker_pattern.search(comment_part):
            continue
        for match in key_value_pattern.finditer(comment_part):
            key = match.group(1).upper()
            value = match.group(2)
            try:
                if key == "T_START":
                    t_start = float(value)
                elif key == "T_END":
                    t_end = float(value)
                elif key == "N_STEPS":
                    n_steps = int(float(value))
                elif key == "EPS_INIT":
                    eps_init = float(value)
                elif key == "EPS_SLACK":
                    eps_slack = float(value)
            except ValueError:
                pass

    return t_start, t_end, n_steps, eps_init, eps_slack


def _extract_sim_metadata(
    text: str,
) -> tuple[float | None, float | None, int | None, float | None, float | None]:
    """Backward-compatible alias for shared SIM metadata parsing."""
    return extract_sim_metadata(text)


def normalize_solver_requirement(value: str | SolverRequirement | None) -> SolverRequirement | None:
    """Normalize a solver requirement value from metadata, enum, or raw string."""
    if value is None:
        return None
    if isinstance(value, SolverRequirement):
        return value
    normalized = str(value).strip().lower()
    for requirement in SolverRequirement:
        if normalized == requirement.value:
            return requirement
    return None


def extract_solver_requirement_metadata(text: str) -> SolverRequirement | None:
    """
    Extract solver requirement metadata from generated Antimony comments.

    Preferred format:
        // @SSYS SOLVER_REQUIREMENT=ode_with_assignment_rules
    """
    key_value_pattern = re.compile(r"SOLVER_REQUIREMENT\s*=\s*([A-Za-z_]+)", re.IGNORECASE)
    label_pattern = re.compile(r"Solver requirement\s*:\s*([A-Za-z_]+)", re.IGNORECASE)

    for line in text.splitlines():
        if "//" not in line:
            continue
        comment = line.split("//", 1)[1]
        match = key_value_pattern.search(comment) or label_pattern.search(comment)
        if match:
            return normalize_solver_requirement(match.group(1))
    return None


def _extract_solver_requirement_metadata(text: str) -> SolverRequirement | None:
    """Backward-compatible alias for shared solver metadata parsing."""
    return extract_solver_requirement_metadata(text)


def format_solver_metadata_lines(requirement: SolverRequirement) -> list[str]:
    return [
        f"// @SSYS SOLVER_REQUIREMENT={requirement.value}",
        f"// Solver requirement: {requirement.value}",
    ]


def _format_solver_metadata_lines(requirement: SolverRequirement) -> list[str]:
    """Backward-compatible alias for shared solver metadata formatting."""
    return format_solver_metadata_lines(requirement)


def format_sim_metadata_lines(result) -> list[str]:
    """Format @SIM metadata as Antimony comment lines."""
    lines = []

    sim_parts = []
    if result.sim_t_start is not None:
        sim_parts.append(f"T_START={format_antimony_number(result.sim_t_start)}")
    if result.sim_t_end is not None:
        sim_parts.append(f"T_END={format_antimony_number(result.sim_t_end)}")
    if result.sim_n_steps is not None:
        sim_parts.append(f"N_STEPS={result.sim_n_steps}")
    if result.eps_init is not None:
        sim_parts.append(f"EPS_INIT={format_antimony_number(result.eps_init)}")
    if result.eps_slack is not None:
        sim_parts.append(f"EPS_SLACK={format_antimony_number(result.eps_slack)}")

    if sim_parts:
        lines.append(f"// @SIM {' '.join(sim_parts)}")
        if result.eps_init is not None:
            lines.append("// Note: Zero-valued initial conditions are replaced with EPS_INIT during recasting.")

    return lines


def _format_sim_metadata_lines(result) -> list[str]:
    """Backward-compatible alias for shared SIM metadata formatting."""
    return format_sim_metadata_lines(result)


__all__ = [
    "_extract_sim_metadata",
    "_extract_solver_requirement_metadata",
    "_format_antimony_number",
    "_format_sim_metadata_lines",
    "_format_solver_metadata_lines",
    "extract_sim_metadata",
    "extract_solver_requirement_metadata",
    "format_antimony_number",
    "format_sim_metadata_lines",
    "format_solver_metadata_lines",
    "normalize_solver_requirement",
]
