"""Smoke tests for release metadata and packaging readiness."""

import re
import subprocess
from pathlib import Path

import lumina_lob

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_package():
    """The package version is a valid semver string."""
    version = lumina_lob.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"Invalid semver: {version}"


def test_version_matches_pyproject():
    """pyproject.toml version matches the package version."""
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = lumina_lob.__version__
    assert f'version = "{version}"' in pyproject_text


def test_build_workflow_has_publish_job():
    """The tag-triggered build workflow includes a PyPI publish job."""
    workflow = ROOT / ".github" / "workflows" / "build.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "publish:" in text
    assert "pypi" in text
    assert "pypa/gh-action-pypi-publish" in text


def test_git_tag_exists():
    """A release tag matching the package version exists in git history."""
    version = lumina_lob.__version__
    result = subprocess.run(
        ["git", "tag", "-l", f"v{version}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"v{version}" in result.stdout
