# ssys Release Checklist

Run this checklist before tagging a public release.

## Metadata Consistency

- Confirm the automated metadata gate passes:

  ```bash
  python tools/check_release_metadata.py
  ```

- Validate citation metadata:

  ```bash
  uv run --with cffconvert cffconvert --validate
  ```

- Validate built package metadata:

  ```bash
  uv run --with build python -m build
  uv run --with twine twine check dist/*
  ```

## Local Required Gates

- Run the broad local engineering checks:

  ```bash
  uv run ruff check .
  uv run mypy
  uv run pytest -q
  python3 tools/check_release_metadata.py
  ```

- Build and smoke-test both local artifact formats:

  ```bash
  python tools/local_artifact_smoke.py --all-supported-pythons \
    --evidence-dir release-evidence/artifact-smoke
  ```

- Run the critical-module coverage gate:

  ```bash
  uv run python tools/check_critical_coverage.py --run-pytest \
    --coverage-json release-evidence/critical-coverage.json
  ```

- Run the critical-module maintainability gate:

  ```bash
  uv run python tools/check_maintainability.py
  ```

- Run the representative performance budget gate:

  ```bash
  uv run python tools/check_performance_budgets.py \
    --evidence-dir release-evidence/performance
  ```

- Run the validation-report schema contract tests:

  ```bash
  uv run pytest tests/test_validation_report_schema.py -q
  ```

- Record dependency and supply-chain evidence:

  ```bash
  python tools/check_dependency_risk.py \
    --evidence-dir release-evidence/dependency-risk
  ```

- Run full integration validation locally with the DAE extra and
  `SSYS_REQUIRE_DAE_VALIDATION=1`. Use the strict validation profile for
  release evidence; partial profiles are diagnostic only.
- Run representative backend cross-checks locally:
  `SSYS_REQUIRE_DAE_VALIDATION=1 pytest tests/test_solver_cross_checks.py -v`.
- Run and archive the local BioModels benchmark evidence:

  ```bash
  uv run python tools/run_biomodels_benchmark.py \
    --artifact release-evidence/artifact-smoke/dist/ssys-0.5.5-py3-none-any.whl \
    --evidence-dir release-evidence/biomodels \
    --from-stage filter \
    --force \
    --min-candidates 900 \
    --min-recasts 800 \
    --min-validation-reports 200 \
    --min-validated 200
  ```

- Use the wheel path recorded by the artifact-smoke `summary.json`; update the
  versioned filename in the command above if the package version changes.

- Create the hashed local release-evidence manifest after the gates above have
  written their evidence directories:

  ```bash
  uv run python tools/archive_release_evidence.py \
    --evidence-dir release-evidence \
    --require artifact-smoke \
    --require dependency-risk \
    --require biomodels \
    --require performance
  ```

- Archive dependency versions, command logs, validation reports, benchmark
  summaries, performance reports, and artifact hashes under the local
  `release-evidence/` directory.
