"""Tests for visualization helpers."""

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for tests

from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side
from lumina_lob.viz import (
    SimulationAnimator,
    plot_depth_ladder,
    plot_simulation_history,
    run_animation,
    save_animation,
)


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


def _make_simple_simulation():
    from lumina_lob.agents import NoiseTrader
    from lumina_lob.simulation import Simulation

    return Simulation(
        agents=[NoiseTrader(arrival_rate=10.0, size_min=1, size_max=5, tick_size=1.0, seed=1)],
        seed=2,
    )


def test_simulation_animator_steps_and_redraws():
    """The animator advances the simulation and updates both panels."""
    sim = _make_simple_simulation()
    animator = SimulationAnimator(sim, top_n=3, history_window=5)
    assert animator.fig is not None
    assert animator.ax_depth is not None
    assert animator.ax_price is not None

    for _ in range(3):
        animator.update()

    assert len(sim.history) == 3
    assert animator._price_line is not None
    x_data, y_data = animator._price_line.get_data()
    assert len(x_data) == 3
    assert len(y_data) == 3


def test_run_animation_returns_funcanimation():
    """run_animation returns a matplotlib FuncAnimation object."""
    from matplotlib.animation import FuncAnimation

    sim = _make_simple_simulation()
    anim = run_animation(sim, n_steps=2, top_n=3)
    assert isinstance(anim, FuncAnimation)
    # Advance a couple of frames manually to confirm the callback is wired.
    # FuncAnimation draws the first frame at creation, so history already has 1 entry.
    anim._step(0)  # type: ignore[attr-defined]
    anim._step(1)  # type: ignore[attr-defined]
    assert len(sim.history) == 3


def test_draw_depth_ladder_empty_book():
    """Internal depth-ladder renderer shows an empty title when the book is empty."""
    import matplotlib.pyplot as plt

    from lumina_lob.viz.realtime import _draw_depth_ladder

    fig, ax = plt.subplots()
    book = OrderBook()
    _draw_depth_ladder(ax, book, top_n=3)
    assert "empty" in ax.get_title().lower()


def test_update_price_axis_with_empty_history():
    """Internal price-axis update is a no-op when no history is available."""
    sim = _make_simple_simulation()
    animator = SimulationAnimator(sim, top_n=3, history_window=5)
    # Calling before any steps should not crash even though history is empty.
    animator._update_price_axis()
    x_data, y_data = animator._price_line.get_data()
    assert len(x_data) == 0
    assert len(y_data) == 0


def test_update_price_axis_with_all_none_mids():
    """Internal price-axis update is a no-op when every record lacks a mid price."""
    sim = _make_simple_simulation()
    sim.history = [
        {"step": 1, "mid_price": None, "trade_count": 0},
        {"step": 2, "mid_price": None, "trade_count": 0},
    ]
    animator = SimulationAnimator(sim, top_n=3, history_window=5)
    animator._update_price_axis()
    x_data, y_data = animator._price_line.get_data()
    assert len(x_data) == 0
    assert len(y_data) == 0


def test_save_animation_gif_creates_file(tmp_path):
    """save_animation can write a short GIF using the Pillow writer."""
    import matplotlib.animation as _animation

    if not _animation.writers.is_available("pillow"):
        pytest.skip("Pillow writer not available")

    sim = _make_simple_simulation()
    anim = run_animation(sim, n_steps=3, top_n=3)
    out = tmp_path / "replay.gif"
    save_animation(anim, out, fps=2)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_animation_unsupported_extension(tmp_path):
    """save_animation rejects unknown file extensions."""
    sim = _make_simple_simulation()
    anim = run_animation(sim, n_steps=2, top_n=3)
    out = tmp_path / "replay.png"
    with pytest.raises(ValueError, match="Unsupported animation format"):
        save_animation(anim, out)


def test_save_animation_missing_writer_raises(monkeypatch, tmp_path):
    """save_animation reports when the required backend writer is missing."""
    import matplotlib.animation as _animation

    sim = _make_simple_simulation()
    anim = run_animation(sim, n_steps=2, top_n=3)
    out = tmp_path / "replay.gif"
    monkeypatch.setattr(_animation.writers, "is_available", lambda _name: False)
    with pytest.raises(ValueError, match="writer is not available"):
        save_animation(anim, out)

