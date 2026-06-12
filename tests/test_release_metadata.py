"""Tests for release metadata consistency checks."""

from pathlib import Path

from tools.check_release_metadata import check_release_metadata


def test_release_metadata_is_consistent():
    repo_root = Path(__file__).resolve().parents[1]

    assert check_release_metadata(repo_root) == []
