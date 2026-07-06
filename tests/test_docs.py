"""Smoke tests for the MkDocs documentation site configuration."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = ROOT / "mkdocs.yml"
DOCS_DIR = ROOT / "docs"


def test_mkdocs_config_exists_and_is_valid_yaml():
    """mkdocs.yml exists and parses as valid YAML."""
    assert MKDOCS_YML.is_file(), f"{MKDOCS_YML} not found"
    config = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert config.get("site_name") == "Lumina LOB"
    assert "nav" in config


def test_documentation_pages_exist():
    """Every page referenced in mkdocs nav exists on disk."""
    config = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    nav = config["nav"]

    def collect_paths(items):
        for item in items:
            if isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, list):
                        yield from collect_paths(value)
                    elif isinstance(value, str) and not value.startswith("http"):
                        yield value
            elif isinstance(item, str):
                yield item

    for path in collect_paths(nav):
        page = DOCS_DIR / path
        assert page.is_file(), f"Missing documentation page: {page}"


def test_docs_extras_declared():
    """pyproject.toml declares a docs extra with mkdocs dependencies."""
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.optional-dependencies]" in pyproject_text
    assert "docs = [" in pyproject_text
    assert "mkdocs" in pyproject_text
    assert "mkdocs-material" in pyproject_text
    assert "mkdocstrings" in pyproject_text


def test_docs_workflow_exists():
    """The GitHub Pages deployment workflow exists."""
    workflow = ROOT / ".github" / "workflows" / "docs.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "mkdocs build" in text
    assert "actions/deploy-pages" in text
