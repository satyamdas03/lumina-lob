"""Smoke tests for the social launch drafts."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "blog" / "social-launch.md"
ASSETS_DIR = ROOT / "blog" / "assets"


def _front_matter_and_body():
    """Split the file into YAML front matter and markdown body."""
    text = POST.read_text(encoding="utf-8")
    assert text.startswith("---")
    parts = text.split("---", 2)
    assert len(parts) >= 3
    return parts[1], parts[2]


def test_social_launch_file_exists():
    """The social launch draft file exists."""
    assert POST.is_file(), f"Missing social launch draft: {POST}"


def test_front_matter_is_valid_yaml():
    """Front matter is valid YAML with required fields."""
    front, _ = _front_matter_and_body()
    meta = yaml.safe_load(front)
    assert isinstance(meta, dict)
    assert "title" in meta
    assert "description" in meta
    assert "author" in meta
    assert "date" in meta


def test_required_links_present():
    """Both LinkedIn and X drafts reference the repo, docs, and blog post."""
    _, body = _front_matter_and_body()
    required = [
        "github.com/satyamdas03/lumina-lob",
        "satyamdas03.github.io/lumina-lob",
        "build-a-lob-simulator.md",
    ]
    for link in required:
        assert link in body, f"Missing required link: {link}"


def test_social_card_asset_exists():
    """The social media teaser card image exists."""
    card = ASSETS_DIR / "social_card.png"
    assert card.is_file(), f"Missing social card asset: {card}"
