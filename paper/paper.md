---
title: 'ssys: Exact algebraic recasting of ODE models into S-system or GMA form'
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
date: 2025-12-31
bibliography: paper.bib
header-includes:
  - \hyphenpenalty=10000
  - \exhyphenpenalty=10000
---

# Summary

`ssys` is a Python toolkit for exact algebraic transformation of ordinary differential equation (ODE) models into canonical S-system or Generalized Mass Action (GMA) form. Given a model in Antimony format [@Smith2009Antimony; @Heydarabadipour2026Antimony], or an SBML file [@Keating2020SBML] parsed with `ssys.parse_sbml`, `ssys` produces a mathematically equivalent representation where each equation has one of the following forms:

**S-system form** (difference of two monomials):
\begin{equation}
\dot X_i = \alpha_i \prod_j X_j^{g_{ij}} - \beta_i \prod_j X_j^{h_{ij}}, \quad \alpha_i,\beta_i \geq 0 .
\label{eq:ssystem}
\end{equation}

Equation \ref{eq:ssystem} is a relaxed S-system form that permits zero-valued right-hand-side terms; canonical S-system form requires both constant coefficients in every equation to be strictly positive.

**GMA form** (sum of monomials):
\begin{equation}
\frac{dX_i}{dt} = \sum_k \gamma_{ik} \prod_j X_j^{f_{ijk}} .
\label{eq:gma}
\end{equation}

The transformation introduces auxiliary variables as needed to decompose a broad class of nonlinearities---including rational functions, exponentials, logarithms, sine, and cosine---into products of power-law terms. The recast is exact: the original and transformed systems have identical dynamics on the invariant constraint manifold defined by auxiliary variable definitions, given consistent initial conditions.

The software provides a command-line interface (CLI) for batch processing Antimony models, a three-test validation suite (symbolic, numerical, trajectory), and generates Jupyter notebooks [@Kluyver2016Jupyter] with LaTeX renderings and trajectory comparisons. It converts Antimony models to SBML with the Antimony library [@Heydarabadipour2026Antimony; @Smith2009Antimony], parses the resulting SBML with libSBML [@Bornstein2008libSBML], and exposes direct SBML-file parsing through `ssys.parse_sbml`.

# Statement of need

S-systems and GMA systems are canonical ODE forms developed within Biochemical Systems Theory (BST) [@Savageau1976; @SavageauVoit1987]. Savageau & Voit (1987) proved that systems built from sums, products, and compositions of elementary functions can be recast into S-system form; their theorem provides a minimum estimate of the class of recastable systems. These canonical representations offer several advantages: S-system steady-state equations become linear in log-space, enabling algebraic analysis of identifiability [@Villaverde2019Observability] and sensitivity [@Savageau1971ParameterSensitivity]; the uniform power-law structure facilitates parameter estimation via linear regression techniques [@Daniels2015b]; and exact recast models can serve as ground truth for validating structure-learning algorithms that infer governing equations from data [@Daniels2015].

Despite the theoretical utility of S-systems and GMA systems, no general-purpose, open-source tool previously existed to perform exact recasting of arbitrary ODE models. `ssys` fills this gap by providing:

- **Exact transformation.** The recast is algebraically equivalent to the original on the constraint manifold, verified by symbolic differentiation and numerical comparison.
- **Broad applicability.** The tool handles rational functions, composite transcendental functions, and time-dependent coefficients through systematic lifting procedures. Because power-law terms require positive bases, S-system variables must remain strictly positive; `ssys` does not automatically translate variables that can be zero or negative, so users must preprocess those models into the positive orthant before recasting.
- **Standard formats.** Input/output in Antimony format ensures interoperability with SBML-based workflows and model repositories such as BioModels [@Malik2020].
- **Validation infrastructure.** A three-test suite (symbolic Jacobian verification, pointwise numerical sampling, trajectory comparison) provides certificates of correctness.

The package enables researchers to convert published models into canonical form for analysis, create ground-truth benchmarks for structure-learning algorithms, and leverage the log-linear properties of S-systems for algebraic steady-state analyses.

# State of the field

Antimony is a human-readable modeling language for systems biology, with transparent interoperability with SBML. The S-system and GMA formalisms are classical [@Savageau1976; @SavageauVoit1987; @VoitKemp2025], but existing software support has focused on analysis, construction, or approximation rather than exact general-purpose recasting. Classical BST computational-analysis workflows focus on systems already expressed in S-system or GMA form [@Voit2000ComputationalAnalysis]. PLMaddon [@Vera2007PLMaddon] can generate power-law approximations via Taylor expansion, but such local approximations differ fundamentally from exact recasting with auxiliary variables. More recently, BSTModelKit.jl [@Vadhin2026BSTModelKit] provides Julia tools for constructing and analyzing BST models, though it emphasizes model construction and analysis from declarative specifications rather than recasting arbitrary ODE/SBML/Antimony models. To the best of our knowledge, no existing open-source tool performs exact algebraic recasting from arbitrary SBML/Antimony ODEs to S-system or GMA form with explicit constraint-manifold handling and validation certificates.

# Software design

`ssys` is organized as a parser, symbolic transformation engine, formatter, and validation layer. Antimony models are converted to SBML with the Antimony library and then parsed with libSBML; SBML files can be parsed directly with `ssys.parse_sbml`. Both paths produce SymPy [@Meurer2017SymPy] symbolic expressions, which serve as the internal representation for exact algebraic rewriting.

The recaster applies local exact transformations: composite functions (exp, log, sin, cos, etc.) and rational denominators are lifted to auxiliary variables with chain-rule-derived ODEs, and sums of monomials are factored into products via pool-auxiliary construction. The simplified output mode keeps structural zeros visible, while canonical mode rewrites equations to the strict two-term S-system convention. Correctness is checked by three independent validators: symbolic Jacobian chain-rule verification, pointwise numerical sampling, and trajectory comparison via libRoadRunner simulation [@Somogyi2015libRoadRunner; @Welsh2023libRoadRunner].

# Quality control

The package includes 712 pytest-collected test cases covering parsing, recasting, validation, and CLI functionality. Of these, 11 are integration-marked test cases, including 8 slow manifest tests that entail transformation and validation of 117 handwritten Antimony models across two output modes. Most of these models were drawn from published examples used to introduce, solve, or analyze S-system and related power-law representations: the complete Savageau & Voit example set [@SavageauVoit1987], later S-system examples and applications [@Voit1988Recasting; @Irvine1988SSystemSolution; @Savageau1988IntroSSystems; @Voit1988NewMethodologies; @RustVoit1990; @Voit1990Endemic; @Voit1992Symmetries; @Voit1993ChaoticInput; @Savageau1993Roots], metabolic optimization and model-inference examples using power-law canonical forms [@MarinSanguino2007; @Pozo2011; @Daniels2015b], and nonlinear dynamics or control examples that test polynomial and quasipolynomial transformations [@HernandezBermejo1998; @PapachristodoulouPrajna2005; @Anghel2013Lyapunov; @Anguelov2018Gompertz; @Zhang2022PLL].

# BioModels database benchmark

To assess applicability to real-world models, we applied `ssys` to ODE models from the BioModels database [@Malik2020]. The benchmark suite (included in the repository under `biomodels_batch/`) operates in three phases: (1) fetch SBML models from BioModels, filtering for ODE-based models; (2) identify transformation candidates by excluding models with unsupported features or more than 100 dynamic species, 200 reactions, or 500 parameters; and (3) batch transform the candidates. After filtering 1,644 SBML models downloaded from BioModels, 978 candidates remained. Of these, the transformation algorithm completed for 848 (86.7%), and numerical validation confirmed mathematical equivalence for 738 transformed models. Among the validated models, 365 achieved meaningful structural canonization: 281 models in general form were converted to GMA, 77 models in GMA form were converted to S-system form, and 7 models in general form were converted to S-system form. The remaining validated models were already in GMA or S-system form, or retained general form after exact rewriting. Unsuccessful cases were mostly due to unsupported model features, parsing or loading errors, complexity limits, or timeouts. Benchmark scripts and the tracked summary are in `biomodels_batch/`; rerunning the benchmark writes detailed outputs under `biomodels_batch/results/`.

# Research impact statement

`ssys` makes the Savageau and Voit recasting construction available as a practical tool for SBML and Antimony models. The curated handwritten examples provide regression tests with known recasting targets, while the BioModels benchmark demonstrates that exact recasting can be applied at repository scale to hundreds of published ODE models. The package therefore supports both direct BST analysis of existing models and construction of validated ground-truth benchmarks for methods that infer dynamical equations from data.

# Availability

`ssys` is available on GitHub at https://github.com/lanl/ssys. Installation requires Python >= 3.10 (and < 3.13), SymPy, NumPy [@Harris2020NumPy], libRoadRunner, and the Antimony library. The repository includes comprehensive documentation: theoretical background and recasting rules (`RECASTING.md`), test model descriptions (`TEST_MODELS.md`), and usage instructions (`README.md`).

Installation: `uv pip install -e ".[dev]"`. For a working example, run `python recast_models.py test_models1` to recast and validate 29 test-suite models.

# AI usage disclosure

Generative AI assistance was used during software review, debugging, benchmark analysis, bibliography checking, and manuscript editing. The author made the scientific, architectural, and editorial decisions, reviewed AI-assisted changes, ran verification tests and paper builds, and is responsible for the correctness and licensing of the submitted materials.

# Acknowledgments

This work was inspired by a phone call with Michael A. Savageau in early 2025. I will always be grateful to Mike for his kind and patient mentorship of me as a graduate student at the University of Michigan, his exemplary and inspiring scholarship, and his enduring support.

This work was supported by the National Institutes of Health (NIH) National Institute of General Medical Sciences (NIGMS) under grant R01GM111510 and by the U.S. Department of Energy through the Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy (Contract No. 89233218CNA000001).

# References
