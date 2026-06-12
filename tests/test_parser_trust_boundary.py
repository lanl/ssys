"""Parser trust-boundary audit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

from ssys._recaster.parsing import _sympify_sbml_formula
from ssys.cli import main
from ssys.types import SBMLParseError


def test_parser_trust_boundary_docs_state_trusted_local_only() -> None:
    audit = Path("PARSER_TRUST_BOUNDARY.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    public_api = Path("PUBLIC_API.md").read_text(encoding="utf-8")

    assert "trusted-local" in audit
    assert "scientific model files" in audit
    assert "No `--parser hardened`" in audit
    assert "no hardened parser mode" in audit
    assert "not as safe untrusted uploads" in readme
    assert "[PARSER_TRUST_BOUNDARY.md]" in readme
    assert "`--parser {sbml,legacy}`" in public_api


def test_cli_exposes_only_documented_trusted_parser_modes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["ssys-recast", "--help"])

    with pytest.raises(SystemExit) as exc:
        main()
    output = capsys.readouterr().out

    assert exc.value.code == 0
    assert "trusted scientific inputs" in output
    assert "{legacy,sbml}" in output
    assert "hardened" not in output


def test_sbml_formula_gate_rejects_dunder_call_before_sympify() -> None:
    x = sp.Symbol("X", positive=True)

    with pytest.raises(SBMLParseError) as exc_info:
        _sympify_sbml_formula(
            "__import__('os').system('echo unsafe') + X",
            {"X": x},
            source="inline",
            kind="kinetic_law",
            reaction_id="r1",
        )

    err = exc_info.value
    assert err.kind == "kinetic_law"
    assert "unsupported function" in err.message
    assert "__import__" in err.message
