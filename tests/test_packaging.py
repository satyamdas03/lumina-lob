"""Smoke tests for package metadata and top-level imports."""

import importlib

import lumina_lob

PUBLIC_SUBMODULES = [
    "lumina_lob.core",
    "lumina_lob.agents",
    "lumina_lob.market_model",
    "lumina_lob.data",
    "lumina_lob.rl",
    "lumina_lob.simulation",
    "lumina_lob.viz",
]


def test_package_version_is_defined():
    """The installed package exposes a version string."""
    assert hasattr(lumina_lob, "__version__")
    assert isinstance(lumina_lob.__version__, str)
    assert len(lumina_lob.__version__.split(".")) >= 2


def test_public_submodules_importable():
    """Every public submodule can be imported without error."""
    for name in PUBLIC_SUBMODULES:
        mod = importlib.import_module(name)
        assert mod is not None
        assert name in mod.__name__
