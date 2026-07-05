"""Smoke tests for report notebooks."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"


def _load_notebook(name: str) -> dict:
    path = NOTEBOOKS / name
    if not path.exists():
        pytest.fail(f"Notebook not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_report_notebook_exists_and_has_cells():
    """Verify the benchmark report notebook is valid JSON and contains expected cells."""
    nb = _load_notebook("05_benchmark_report.ipynb")
    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 3
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert "Python vs C++ Engine Benchmark Report" in source
    assert "run_benchmark" in source
    assert "LUMINA_BENCHMARK_QUICK" in source
