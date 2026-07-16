"""Negative corpus tests for unsupported and malformed model inputs."""

from dataclasses import dataclass

import pytest

from ssys.recaster import (
    SBMLParseError,
    parse_antimony_via_sbml,
    parse_sbml_from_string,
    recast_to_ssystem,
    ssystem_to_antimony,
)


def _minimal_sbml(*, species: str | None = None, extra: str = "") -> str:
    species_block = species or """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m" substanceUnits="mole" timeUnits="second" extentUnits="mole">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
{species_block}
    </listOfSpecies>
{extra}
  </model>
</sbml>"""


def _invalid_identifier_sbml() -> str:
    return _minimal_sbml(
        species="""
      <species id="1bad" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
    )


@dataclass(frozen=True)
class NegativeSbmlCase:
    name: str
    sbml: str
    expected_kind: str
    expected_message: str


NEGATIVE_SBML_CASES = [
    NegativeSbmlCase(
        name="event",
        sbml=_minimal_sbml(
            extra="""
    <listOfEvents>
      <event id="pulse" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci> time </ci><cn> 1 </cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="S">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 0 </cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>"""
        ),
        expected_kind="unsupported_feature",
        expected_message="events",
    ),
    NegativeSbmlCase(
        name="delay",
        sbml=_minimal_sbml(
            extra="""
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><delay/><ci> S </ci><cn> 1 </cn></apply>
        </math>
      </rateRule>
    </listOfRules>"""
        ),
        expected_kind="unsupported_feature",
        expected_message="delays",
    ),
    NegativeSbmlCase(
        name="constraint",
        sbml=_minimal_sbml(
            extra="""
    <listOfConstraints>
      <constraint>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><gt/><ci> S </ci><cn> 0 </cn></apply>
        </math>
      </constraint>
    </listOfConstraints>"""
        ),
        expected_kind="unsupported_feature",
        expected_message="constraints",
    ),
    NegativeSbmlCase(
        name="unknown_function",
        sbml=_minimal_sbml(
            extra="""
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><ci> unsupported </ci><ci> S </ci></apply>
        </math>
      </rateRule>
    </listOfRules>"""
        ),
        expected_kind="rate_rule",
        expected_message="unsupported function(s): unsupported",
    ),
    NegativeSbmlCase(
        name="invalid_identifier",
        sbml=_invalid_identifier_sbml(),
        expected_kind="invalid_identifier",
        expected_message="invalid SBML identifier",
    ),
    NegativeSbmlCase(
        name="malformed_formula",
        sbml=_minimal_sbml(
            extra="""
    <listOfRules>
      <rateRule variable="S"/>
    </listOfRules>"""
        ),
        expected_kind="rate_rule",
        expected_message="missing math formula",
    ),
    NegativeSbmlCase(
        name="unhandled_algebraic_form",
        sbml=_minimal_sbml(
            extra="""
    <listOfRules>
      <algebraicRule>
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> missing </ci></math>
      </algebraicRule>
    </listOfRules>"""
        ),
        expected_kind="algebraic_rule",
        expected_message="unknown identifier(s): missing",
    ),
    NegativeSbmlCase(
        name="ambiguous_duplicate_rate_rule",
        sbml=_minimal_sbml(
            extra="""
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 1 </cn></math>
      </rateRule>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 2 </cn></math>
      </rateRule>
    </listOfRules>"""
        ),
        expected_kind="ambiguous_model",
        expected_message="multiple rate rules for variable S",
    ),
    # GH #237: a speciesReference stoichiometry driven by a rule is time-varying
    # and not power-law-recastable; reject rather than freeze it at its load value.
    NegativeSbmlCase(
        name="variable_stoichiometry_rule",
        sbml="""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m" substanceUnits="mole" timeUnits="second" extentUnits="mole">
    <listOfCompartments><compartment id="cell" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5" constant="true"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="sA">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><cn>1</cn><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol></apply></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants><speciesReference id="sA" species="A" stoichiometry="1" constant="false"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>""",
        expected_kind="unsupported_feature",
        expected_message="variable stoichiometry",
    ),
    # GH #237: an L2 <stoichiometryMath> that reads a species does not constant-fold.
    NegativeSbmlCase(
        name="variable_stoichiometry_math",
        sbml="""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m">
    <listOfCompartments><compartment id="cell" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="2" boundaryCondition="false"/>
      <species id="B" compartment="cell" initialConcentration="0" boundaryCondition="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3"/></listOfParameters>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A">
            <stoichiometryMath><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>B</ci></math></stoichiometryMath>
          </speciesReference>
        </listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>""",
        expected_kind="unsupported_feature",
        expected_message="variable stoichiometry",
    ),
    # GH #231: a compartment driven by a rate rule has a time-varying volume; the
    # concentration ODE of a species it owns is missing the -[S]·(dV/dt)/V dilution
    # term, so it is rejected rather than silently mis-integrated.
    NegativeSbmlCase(
        name="rate_rule_compartment_volume",
        sbml="""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m" substanceUnits="mole" timeUnits="second" extentUnits="mole">
    <listOfCompartments><compartment id="cell" size="1" spatialDimensions="3" constant="false"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="cell"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math></rateRule>
    </listOfRules>
  </model>
</sbml>""",
        expected_kind="unsupported_feature",
        expected_message="time-varying compartment volume",
    ),
    # GH #231: an assignment-rule compartment whose RHS reads time is time-varying.
    NegativeSbmlCase(
        name="assignment_rule_compartment_volume",
        sbml="""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m" substanceUnits="mole" timeUnits="second" extentUnits="mole">
    <listOfCompartments><compartment id="cell" size="1" spatialDimensions="3" constant="false"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5" constant="true"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="cell"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><plus/><cn>1</cn><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol></apply>
      </math></assignmentRule>
    </listOfRules>
  </model>
</sbml>""",
        expected_kind="unsupported_feature",
        expected_message="time-varying compartment volume",
    ),
]


@pytest.mark.parametrize("case", NEGATIVE_SBML_CASES, ids=[case.name for case in NEGATIVE_SBML_CASES])
def test_negative_sbml_corpus_rejects_before_recast_artifact(case: NegativeSbmlCase):
    """Unsupported SBML fixtures fail with structured diagnostics before output generation."""
    with pytest.raises(SBMLParseError) as exc_info:
        sym = parse_sbml_from_string(case.sbml)
        result = recast_to_ssystem(sym)
        ssystem_to_antimony(result)

    err = exc_info.value
    assert err.kind == case.expected_kind
    assert case.expected_message in err.message


def test_malformed_antimony_rejects_before_recast_artifact():
    """Malformed Antimony cannot enter the recast/output pipeline."""
    text = """
    model malformed()
      S' = k*
      S = 1
      k = 0.5
    end
    """

    with pytest.raises(ValueError, match="Antimony parsing error"):
        sym = parse_antimony_via_sbml(text)
        result = recast_to_ssystem(sym)
        ssystem_to_antimony(result)
