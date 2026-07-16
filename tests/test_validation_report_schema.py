"""JSON Schema contract tests for validation reports."""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from ssys.cli import recast_file
from ssys.validator import (
    VALIDATION_REPORT_SCHEMA_VERSION,
    load_validation_report_schema,
    validate_recast_pair,
)


def _schema_validator() -> Draft202012Validator:
    schema = load_validation_report_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _assert_schema_valid(data: dict) -> None:
    validator = _schema_validator()
    errors = sorted(validator.iter_errors(data), key=lambda error: error.json_path)
    assert not errors, "\n".join(f"{error.json_path}: {error.message}" for error in errors)


def test_packaged_validation_report_schema_matches_report_version():
    schema = load_validation_report_schema()

    assert schema["$id"] == "urn:ssys:schema:validation-report:1.0"
    assert schema["properties"]["schema_version"]["const"] == VALIDATION_REPORT_SCHEMA_VERSION
    Draft202012Validator.check_schema(schema)


def test_emitted_structural_validation_report_matches_schema(tmp_path):
    original = tmp_path / "original.ant"
    original.write_text("""
        model original()
            species X;
            X' = -k*X;
            k = 0.5;
            X = 1.0;
        end
    """)
    recast = tmp_path / "recast.ant"
    recast.write_text("""
        model recast()
            // @SSYS SOLVER_REQUIREMENT=ode_only
            species X;
            X' = -k*X;
            k = 0.5;
            X = 1.0;
        end
    """)
    output_json = tmp_path / "validation.json"

    report = validate_recast_pair(
        str(original),
        str(recast),
        output_json=str(output_json),
        parser="sbml",
        profile="structural",
    )
    data = json.loads(output_json.read_text())

    assert data["schema_version"] == VALIDATION_REPORT_SCHEMA_VERSION
    assert data == report.to_dict()
    assert data["validation_profile"]["name"] == "structural"
    _assert_schema_valid(data)


def test_emitted_parser_failure_report_matches_schema(tmp_path):
    original = tmp_path / "original.ant"
    original.write_text("""
        model original()
            species X;
            X' = -k*X;
            k = 0.5;
            X = 1.0;
        end
    """)
    recast = tmp_path / "bad_recast.ant"
    recast.write_text("model bad_recast()\n    DNA := Z_1;\nend\n")
    output_json = tmp_path / "validation.json"

    report = validate_recast_pair(
        str(original),
        str(recast),
        output_json=str(output_json),
        parser="sbml",
        profile="strict",
    )
    data = json.loads(output_json.read_text())

    assert data["schema_version"] == VALIDATION_REPORT_SCHEMA_VERSION
    assert data == report.to_dict()
    assert data["overall_result"] == "failed"
    assert data["tests"]["mapping"]["reason"] == "parser_failed"
    _assert_schema_valid(data)


def test_cli_validation_crash_report_matches_schema(tmp_path, monkeypatch):
    original = tmp_path / "original.ant"
    original.write_text("""
        species X
        X' = -k*X
        k = 0.5
        X = 1.0
    """)
    outdir = tmp_path / "out"
    outdir.mkdir()

    def crash_validation(*args, **kwargs):
        raise RuntimeError("forced validation crash")

    monkeypatch.setattr("ssys.validator.validate_recast_pair", crash_validation)

    _, _, _, validation_path = recast_file(
        str(original),
        str(outdir),
        validate=True,
        validation_profile="strict",
    )
    assert validation_path is not None
    data = json.loads((outdir / "original_validation.json").read_text())

    assert data["schema_version"] == VALIDATION_REPORT_SCHEMA_VERSION
    assert data["overall_result"] == "failed"
    assert data["tests"]["parser"]["name"] == "validator_crash"
    assert data["tests"]["symbolic"]["reason"] == "validation_crashed"
    _assert_schema_valid(data)
