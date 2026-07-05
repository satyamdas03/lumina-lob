"""Tests for visualization helpers."""

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for tests

from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side
from lumina_lob.viz import plot_depth_ladder, plot_simulation_history


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


def test_plot_simulation_history_returns_figure():
    """Verify the history plot accepts a list of step records."""
    history = [
        {"step": 1, "mid_price": 100.0, "spread": 2, "trade_count": 0, "trade_volume": 0},
        {"step": 2, "mid_price": 101.0, "spread": 1, "trade_count": 1, "trade_volume": 10},
        {"step": 3, "mid_price": 100.5, "spread": 3, "trade_count": 2, "trade_volume": 25},
    ]
    fig, axes = plot_simulation_history(history)
    assert fig is not None
    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Mid price"
    assert axes[1].get_ylabel() == "Spread"
    assert axes[2].get_ylabel() == "Volume"


def test_plot_simulation_history_empty_raises():
    """An empty history cannot be plotted."""
    with pytest.raises(ValueError, match="history is empty"):
        plot_simulation_history([])


def test_plot_simulation_history_missing_columns_raises():
    """Required columns must be present."""
    history = [{"step": 1, "mid_price": 100.0}]
    with pytest.raises(ValueError, match="missing required columns"):
        plot_simulation_history(history)


def test_plot_simulation_history_empty_dataframe_raises():
    """An empty pandas DataFrame raises the same empty-history error."""
    import pandas as pd

    df = pd.DataFrame(columns=["step", "mid_price", "spread", "trade_count", "trade_volume"])
    with pytest.raises(ValueError, match="history is empty"):
        plot_simulation_history(df)


def test_plot_simulation_history_accepts_dataframe():
    """The history plot also accepts a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"step": 1, "mid_price": 100.0, "spread": 2, "trade_count": 0, "trade_volume": 0},
            {"step": 2, "mid_price": 101.0, "spread": 1, "trade_count": 1, "trade_volume": 5},
        ]
    )
    fig, axes = plot_simulation_history(df)
    assert fig is not None
    assert len(axes) == 3

