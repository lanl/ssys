"""Generated artifact validation and serialization helpers."""

from typing import Any

from ssys._validator.report import EquivalenceTest, ValidationResult


def validate_generated_output_roundtrip(
    recast_file: str, recast_text: str | None = None
) -> "EquivalenceTest":
    """Validate that generated Antimony parses and converts through SBML."""
    metadata: dict[str, Any] = {
        "antimony_parse_success": False,
        "sbml_conversion_success": False,
        "sbml_parse_success": False,
        "parser_diagnostics": "",
    }

    try:
        import antimony
        import libsbml
    except ImportError as e:
        metadata["parser_diagnostics"] = str(e)
        return EquivalenceTest(
            name="generated_output_roundtrip",
            result=ValidationResult.UNSUPPORTED,
            details=f"Antimony/SBML parser dependency unavailable: {e}",
            metadata=metadata,
        )

    try:
        text = recast_text if recast_text is not None else open(recast_file).read()
    except OSError as e:
        metadata["parser_diagnostics"] = str(e)
        return EquivalenceTest(
            name="generated_output_roundtrip",
            result=ValidationResult.FAIL,
            details=f"Could not read recast file: {e}",
            metadata=metadata,
        )

    try:
        antimony.clearPreviousLoads()
        load_code = antimony.loadAntimonyString(text)
        metadata["antimony_load_code"] = load_code
        antimony_diagnostics = antimony.getLastError() or ""
        metadata["parser_diagnostics"] = antimony_diagnostics

        if load_code < 0:
            return EquivalenceTest(
                name="generated_output_roundtrip",
                result=ValidationResult.FAIL,
                details=f"Antimony parse failed: {antimony_diagnostics}",
                metadata=metadata,
            )
        metadata["antimony_parse_success"] = True

        module_name = antimony.getMainModuleName()
        metadata["module_name"] = module_name
        if not module_name:
            metadata["parser_diagnostics"] = "No main Antimony module found"
            return EquivalenceTest(
                name="generated_output_roundtrip",
                result=ValidationResult.FAIL,
                details="Antimony parse succeeded but no main module was found",
                metadata=metadata,
            )

        sbml = antimony.getSBMLString(module_name)
        conversion_diagnostics = antimony.getLastError() or ""
        if not sbml:
            metadata["parser_diagnostics"] = conversion_diagnostics
            return EquivalenceTest(
                name="generated_output_roundtrip",
                result=ValidationResult.FAIL,
                details=f"Antimony to SBML conversion failed: {conversion_diagnostics}",
                metadata=metadata,
            )
        metadata["sbml_conversion_success"] = True

        document = libsbml.readSBMLFromString(sbml)
        error_log = document.getErrorLog()
        sbml_diagnostics = error_log.toString() if error_log is not None else ""
        metadata["sbml_diagnostics"] = sbml_diagnostics
        metadata["libsbml_error_count"] = document.getNumErrors()

        serious_errors = []
        for i in range(document.getNumErrors()):
            err = document.getError(i)
            severity = err.getSeverityAsString().lower()
            if severity in {"error", "fatal"}:
                serious_errors.append(err.getMessage())

        if serious_errors:
            metadata["parser_diagnostics"] = sbml_diagnostics
            return EquivalenceTest(
                name="generated_output_roundtrip",
                result=ValidationResult.FAIL,
                details=f"SBML parse failed: {sbml_diagnostics}",
                metadata=metadata,
            )

        if document.getModel() is None:
            metadata["parser_diagnostics"] = sbml_diagnostics or "SBML document has no model"
            return EquivalenceTest(
                name="generated_output_roundtrip",
                result=ValidationResult.FAIL,
                details="SBML parse succeeded but no model was found",
                metadata=metadata,
            )

        metadata["sbml_parse_success"] = True
        return EquivalenceTest(
            name="generated_output_roundtrip",
            result=ValidationResult.PASS,
            details="Generated Antimony parsed and converted through SBML",
            metadata=metadata,
        )
    except Exception as e:
        metadata["parser_diagnostics"] = str(e)
        return EquivalenceTest(
            name="generated_output_roundtrip",
            result=ValidationResult.INCONCLUSIVE,
            details=f"Roundtrip check crashed: {e}",
            metadata=metadata,
        )
