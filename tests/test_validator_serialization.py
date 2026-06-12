"""Behavioral tests for generated artifact roundtrip validation."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from ssys._validator.serialization import validate_generated_output_roundtrip
from ssys.validator import ValidationResult

_DEFAULT_MODEL = object()


class _FakeAntimony:
    def __init__(
        self,
        *,
        load_code: int = 1,
        module_name: str = "recast",
        sbml: str = "<sbml/>",
        last_error: str = "",
        crash_on_load: bool = False,
    ):
        self.load_code = load_code
        self.module_name = module_name
        self.sbml = sbml
        self.last_error = last_error
        self.crash_on_load = crash_on_load
        self.loaded_text = ""
        self.cleared = False
        self.requested_module = ""

    def clearPreviousLoads(self):
        self.cleared = True

    def loadAntimonyString(self, text):
        self.loaded_text = text
        if self.crash_on_load:
            raise RuntimeError("antimony crashed")
        return self.load_code

    def getLastError(self):
        return self.last_error

    def getMainModuleName(self):
        return self.module_name

    def getSBMLString(self, module_name):
        self.requested_module = module_name
        return self.sbml


class _FakeSbmlError:
    def __init__(self, severity: str, message: str):
        self.severity = severity
        self.message = message

    def getSeverityAsString(self):
        return self.severity

    def getMessage(self):
        return self.message


class _FakeSbmlDocument:
    def __init__(self, *, errors=(), model=_DEFAULT_MODEL, diagnostics: str = "sbml diagnostics"):
        self.errors = list(errors)
        self.model = model
        self.diagnostics = diagnostics

    def getErrorLog(self):
        return SimpleNamespace(toString=lambda: self.diagnostics)

    def getNumErrors(self):
        return len(self.errors)

    def getError(self, index):
        return self.errors[index]

    def getModel(self):
        return self.model


def _install_roundtrip_modules(monkeypatch: pytest.MonkeyPatch, antimony, document):
    monkeypatch.setitem(__import__("sys").modules, "antimony", antimony)
    monkeypatch.setitem(
        __import__("sys").modules,
        "libsbml",
        SimpleNamespace(readSBMLFromString=lambda sbml: document),
    )


def test_roundtrip_reports_missing_parser_dependency(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def missing_parser(name, *args, **kwargs):
        if name in {"antimony", "libsbml"}:
            raise ImportError(f"missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_parser)

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.UNSUPPORTED
    assert result.reason == "unsupported"
    assert result.metadata["antimony_parse_success"] is False
    assert "missing antimony" in result.details


def test_roundtrip_reports_unreadable_recast_file(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_roundtrip_modules(monkeypatch, _FakeAntimony(), _FakeSbmlDocument())

    result = validate_generated_output_roundtrip(str(tmp_path / "missing.ant"))

    assert result.result == ValidationResult.FAIL
    assert result.reason == "failed"
    assert result.metadata["antimony_parse_success"] is False
    assert "Could not read recast file" in result.details


def test_roundtrip_reports_antimony_parse_failure(monkeypatch: pytest.MonkeyPatch):
    antimony = _FakeAntimony(load_code=-1, last_error="line 2: unexpected token")
    _install_roundtrip_modules(monkeypatch, antimony, _FakeSbmlDocument())

    result = validate_generated_output_roundtrip("unused.ant", recast_text="bad antimony")

    assert result.result == ValidationResult.FAIL
    assert result.metadata["antimony_load_code"] == -1
    assert result.metadata["antimony_parse_success"] is False
    assert result.metadata["parser_diagnostics"] == "line 2: unexpected token"
    assert "Antimony parse failed" in result.details


def test_roundtrip_reports_missing_main_module(monkeypatch: pytest.MonkeyPatch):
    antimony = _FakeAntimony(module_name="")
    _install_roundtrip_modules(monkeypatch, antimony, _FakeSbmlDocument())

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.FAIL
    assert result.metadata["antimony_parse_success"] is True
    assert result.metadata["parser_diagnostics"] == "No main Antimony module found"
    assert "no main module" in result.details


def test_roundtrip_reports_sbml_conversion_failure(monkeypatch: pytest.MonkeyPatch):
    antimony = _FakeAntimony(sbml="", last_error="conversion failed")
    _install_roundtrip_modules(monkeypatch, antimony, _FakeSbmlDocument())

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.FAIL
    assert result.metadata["module_name"] == "recast"
    assert result.metadata["sbml_conversion_success"] is False
    assert result.metadata["parser_diagnostics"] == "conversion failed"
    assert "Antimony to SBML conversion failed" in result.details


def test_roundtrip_reports_serious_sbml_errors(monkeypatch: pytest.MonkeyPatch):
    document = _FakeSbmlDocument(
        errors=[
            _FakeSbmlError("Warning", "annotation ignored"),
            _FakeSbmlError("Error", "invalid species reference"),
        ],
        diagnostics="invalid species reference",
    )
    _install_roundtrip_modules(monkeypatch, _FakeAntimony(), document)

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.FAIL
    assert result.metadata["sbml_conversion_success"] is True
    assert result.metadata["libsbml_error_count"] == 2
    assert result.metadata["parser_diagnostics"] == "invalid species reference"
    assert "SBML parse failed" in result.details


def test_roundtrip_reports_sbml_without_model(monkeypatch: pytest.MonkeyPatch):
    document = _FakeSbmlDocument(model=None, diagnostics="")
    _install_roundtrip_modules(monkeypatch, _FakeAntimony(), document)

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.FAIL
    assert result.metadata["parser_diagnostics"] == "SBML document has no model"
    assert "no model" in result.details


def test_roundtrip_passes_and_records_parser_metadata(monkeypatch: pytest.MonkeyPatch):
    antimony = _FakeAntimony()
    _install_roundtrip_modules(monkeypatch, antimony, _FakeSbmlDocument())

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.PASS
    assert result.reason is None
    assert antimony.cleared is True
    assert antimony.loaded_text == "model m()\nend"
    assert antimony.requested_module == "recast"
    assert result.metadata["antimony_parse_success"] is True
    assert result.metadata["sbml_conversion_success"] is True
    assert result.metadata["sbml_parse_success"] is True
    assert result.metadata["libsbml_error_count"] == 0


def test_roundtrip_crash_is_inconclusive_with_diagnostic(monkeypatch: pytest.MonkeyPatch):
    _install_roundtrip_modules(
        monkeypatch,
        _FakeAntimony(crash_on_load=True),
        _FakeSbmlDocument(),
    )

    result = validate_generated_output_roundtrip("unused.ant", recast_text="model m()\nend")

    assert result.result == ValidationResult.INCONCLUSIVE
    assert result.reason == "inconclusive"
    assert result.metadata["parser_diagnostics"] == "antimony crashed"
    assert "Roundtrip check crashed" in result.details
