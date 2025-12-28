---
title: 'ssys: Exact algebraic recasting of ODE models into canonical S-system form'
tags:
  - systems biology
  - dynamical systems
  - S-systems
  - power-law formalism
  - symbolic computation
  - Antimony
  - SBML
authors:
  - name: William S. Hlavacek
    orcid: 0000-0003-4383-8711
    affiliation: 1
affiliations:
  - name: Theoretical Biology & Biophysics Group, Theoretical Division, Los Alamos National Laboratory, Los Alamos, NM 87545, USA
    index: 1
date: 2025-12-28
bibliography: paper.bib
---

# Summary

`ssys` is a Python toolkit for exact algebraic transformation of ordinary differential equation (ODE) models into canonical S-system or Generalized Mass Action (GMA) form. Given a model in Antimony or SBML format, `ssys` produces a mathematically equivalent representation where each equation has the form:
$$\frac{dX_i}{dt} = \alpha_i \prod_j X_j^{g_{ij}} - \beta_i \prod_j X_j^{h_{ij}}$$
The transformation introduces auxiliary variables as needed to decompose arbitrary nonlinearities—including rational functions, exponentials, logarithms, and trigonometric functions—into products of power-law terms. The recast is exact: the original and transformed systems have identical dynamics.

The software provides a command-line interface for batch processing, a three-test validation suite (symbolic, numerical, trajectory), and generates Jupyter notebooks with LaTeX renderings and trajectory comparisons. It parses models via the reference Antimony library and libRoadRunner, ensuring compatibility with standard SBML semantics.

# Statement of need

S-systems are a canonical ODE form developed within Biochemical Systems Theory [@Savageau1969; @SavageauVoit1987]. Any ODE system composed of elementary functions can, in principle, be exactly recast into S-system or GMA form [@SavageauVoit1987]. The canonical S-system representation offers several advantages: steady-state equations become linear in log-space, enabling algebraic analysis of identifiability and sensitivity; the uniform power-law structure facilitates parameter estimation via linear regression techniques [@Daniels2015]; and exact recast models can serve as ground truth for validating structure-learning algorithms such as SINDy [@Brunton2016], which infer governing equations from data using sparse regression.

Despite the theoretical utility of S-systems, no general-purpose, open-source tool previously existed to perform exact recasting of arbitrary ODE models. `ssys` fills this gap by providing:

- **Exact transformation.** The recast is algebraically equivalent to the original, verified by symbolic differentiation and numerical comparison.
- **Broad applicability.** The tool handles rational functions, composite transcendental functions, and time-dependent coefficients through systematic lifting procedures.
- **Standard formats.** Input/output in Antimony format ensures interoperability with SBML-based workflows and model repositories such as BioModels [@Malik2020].
- **Validation infrastructure.** A three-test suite (symbolic Jacobian verification, pointwise numerical sampling, trajectory comparison) provides certificates of correctness.

The package enables researchers to convert published models into canonical form for analysis, create ground-truth benchmarks for structure-learning algorithms, and leverage the log-linear properties of S-systems for identifiability studies.

# State of the field

Antimony is a widely used human-readable modeling language for systems biology [@Smith2009Antimony], with transparent interoperability to SBML [@Hucka2003]. The S-system formalism is classical [@Savageau1969; @Voit2013], but general-purpose recasting tools are scarce. We are not aware of an open-source package that ingests arbitrary Antimony/SBML ODEs and returns an exact S-system with explicit state mappings and correctness certificates. `ssys` aims to be that missing bridge.

# Functionality

- **SBML-first parsing.** Models are parsed via the Antimony library and libRoadRunner, then converted to SymPy symbolic expressions for manipulation.
- **Function lifting.** Composite functions (exp, log, sin, cos, etc.) and rational denominators are lifted to auxiliary variables with chain-rule-derived ODEs, preserving exactness.
- **Sum splitting.** Sums of monomials are factored into products via pool-auxiliary construction, yielding the canonical two-term S-system form.
- **Validation.** Three independent tests verify correctness: symbolic Jacobian chain-rule verification, numerical sampling at random positive points, and trajectory comparison via libRoadRunner simulation.
- **Output modes.** Simplified mode preserves zeros; canonical mode ensures strict two-term form with epsilon slack variables.

# Quality control

The package includes 213 unit tests covering parsing, recasting, validation, and CLI functionality. Test suites include 29 core models, 42 literature examples from Savageau, Voit, and others, and 18 BioModels-derived cases. Continuous integration runs all tests on each commit.

# Availability

`ssys` is available on GitLab at https://lisdi-git.lanl.gov/hlavacek/ssys under the BSD-3-Clause license. Installation requires Python ≥3.10, SymPy, NumPy, libRoadRunner, and the Antimony library.

```bash
pip install -e ".[dev]"
ssys-recast --manifest models.manifest --outdir output --validate
```

# Acknowledgements

This work was supported by the U.S. Department of Energy through the Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy (Contract No. 89233218CNA000001).

# References
