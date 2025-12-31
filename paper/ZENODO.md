# Zenodo Archival DOI - JOSS Requirement

This document tracks the Zenodo archival release requirement for JOSS submission.

## JOSS Requirement

JOSS requires an archived release of the software with a DOI. From the JOSS submission checklist:

> "The software must be open source and deposited in a long-term archive (e.g., Zenodo, figshare, or a DOI-issuing repository)."

## Steps to Create Zenodo DOI

### Before JOSS Submission

1. **Enable Zenodo-GitHub integration:**
   - Go to https://zenodo.org and log in with GitHub
   - Navigate to Settings → GitHub
   - Enable the `lanl/ssys` repository

2. **Create a GitHub Release:**
   - Tag the release (e.g., `v1.0.0`)
   - Write release notes summarizing features
   - Publish the release

3. **Zenodo automatically archives:**
   - Zenodo will detect the release and create a DOI
   - A badge will be available for the README

4. **Update paper metadata:**
   - Add the DOI to CITATION.cff
   - Add Zenodo badge to README.md

### Zenodo Metadata to Prepare

When creating the Zenodo record, use:

- **Title:** ssys: Exact algebraic recasting of ODE models into S-system or GMA form
- **Authors:** William S. Hlavacek (ORCID: 0000-0003-4383-8711)
- **Affiliation:** Los Alamos National Laboratory
- **License:** (match LICENSE file)
- **Keywords:** S-systems, GMA, ODE recasting, systems biology, Antimony, SBML
- **Related identifiers:** Link to JOSS paper DOI (after acceptance)

## Checklist

- [ ] GitHub repository is public at https://github.com/lanl/ssys
- [ ] Zenodo-GitHub integration enabled
- [ ] GitHub release created with semantic version tag
- [ ] Zenodo DOI generated
- [ ] DOI added to CITATION.cff
- [ ] DOI badge added to README.md
- [ ] DOI included in JOSS submission form

## References

- JOSS submission requirements: https://joss.readthedocs.io/en/latest/submitting.html
- Zenodo-GitHub integration: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- Zenodo: https://zenodo.org
