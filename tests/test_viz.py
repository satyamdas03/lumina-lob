"""Tests for visualization helpers."""

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for tests

from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side
from lumina_lob.viz import plot_depth_ladder


def test_plot_depth_ladder_returns_figure():
    """Verify the depth-ladder plot accepts a Python OrderBook and returns a Figure."""
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 10, OrderType.LIMIT))
    engine.process(Order(2, Side.BID, 99, 5, OrderType.LIMIT))
    engine.process(Order(3, Side.ASK, 101, 7, OrderType.LIMIT))
    engine.process(Order(4, Side.ASK, 102, 3, OrderType.LIMIT))

    fig, ax = plot_depth_ladder(book, top_n=2)
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) > 0


def test_plot_depth_ladder_requires_positive_top_n():
    """Top-n must be positive."""
    book = OrderBook()
    with pytest.raises(ValueError, match="top_n must be positive"):
        plot_depth_ladder(book, top_n=0)


def test_plot_depth_ladder_requires_levels():
    """An empty book has no price levels to plot."""
    book = OrderBook()
    with pytest.raises(ValueError, match="book has no price levels"):
        plot_depth_ladder(book)


def test_plot_depth_ladder_accepts_cpp_book():
    """The plot should also accept the C++ OrderBook exposed via pybind11."""
    _core = pytest.importorskip("lumina_lob._core")
    book = _core.OrderBook()
    engine = _core.MatchingEngine(book)
    engine.process(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.ASK, 102, 8, _core.OrderType.LIMIT)

    fig, ax = plot_depth_ladder(book, top_n=1)
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) > 0
