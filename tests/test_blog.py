"""Smoke tests for the technical blog post draft."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "blog" / "build-a-lob-simulator.md"
ASSETS_DIR = ROOT / "blog" / "assets"


def _front_matter_and_body():
    """Split the post into YAML front matter and markdown body."""
    text = POST.read_text(encoding="utf-8")
    assert text.startswith("---"), "Post must start with YAML front matter"
    parts = text.split("---", 2)
    assert len(parts) >= 3, "Front matter not closed"
    return parts[1], parts[2]


def test_blog_post_exists():
    """The technical blog post markdown file exists."""
    assert POST.is_file(), f"Missing blog post: {POST}"


def test_front_matter_is_valid_yaml():
    """Front matter is valid YAML with required fields."""
    front, _ = _front_matter_and_body()
    meta = yaml.safe_load(front)
    assert isinstance(meta, dict)
    assert "title" in meta
    assert "description" in meta
    assert "author" in meta
    assert "date" in meta
    assert "tags" in meta


def test_markdown_body_references_image_assets():
    """All image assets referenced in the post exist on disk."""
    _, body = _front_matter_and_body()
    for line in body.splitlines():
        if line.strip().startswith("!["):
            # Extract the URL inside the parentheses.
            start = line.find("](")
            end = line.find(")", start)
            assert start != -1 and end != -1, f"Malformed image reference: {line}"
            url = line[start + 2 : end]
            if url.startswith("http"):
                continue
            asset = ASSETS_DIR / url.replace("assets/", "")
            assert asset.is_file(), f"Referenced image asset missing: {asset}"


def test_required_assets_exist():
    """The expected visual assets are present."""
    for name in ("depth_ladder.png", "history.png"):
        asset = ASSETS_DIR / name
        assert asset.is_file(), f"Missing blog asset: {asset}"
