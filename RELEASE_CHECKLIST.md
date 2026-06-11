# ssys Release Checklist

Run this checklist before tagging a public release.

## Metadata Consistency

- Confirm `pyproject.toml`, `src/ssys/__init__.py`, `CITATION.cff`, `README.md`, and `CHANGELOG.md` agree on the release version.
- Confirm `CITATION.cff`, `README.md`, and `CHANGELOG.md` agree on the release date.
- Validate citation metadata:

  ```bash
  uv run --with cffconvert cffconvert --validate
  ```

## Required Gates

- Run the `Release Candidate` GitHub Actions workflow for the commit being tagged.
- Confirm the workflow builds both sdist and wheel artifacts.
- Confirm wheel smoke jobs pass on Ubuntu, macOS, and Windows for Python 3.10, 3.11, and 3.12.
- Confirm full integration validation runs with `SSYS_REQUIRE_DAE_VALIDATION=1`.
- Archive the dependency-version artifacts from the release-candidate workflow.
