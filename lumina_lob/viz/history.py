"""Time-series visualisation for simulation history."""

from __future__ import annotations

from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for visualization. "
        "Install it with `pip install matplotlib` or use the viz extras."
    ) from exc


def _to_dataframe(history: Any) -> Any:
    """Normalise a list of dicts or a pandas DataFrame to a DataFrame."""
    import pandas as pd

    if isinstance(history, list):
        if not history:
            raise ValueError("history is empty")
        return pd.DataFrame(history)
    # Assume pandas DataFrame-like
    if len(history) == 0:
        raise ValueError("history is empty")
    return history


def plot_simulation_history(history: Any) -> tuple[Figure, Any]:
    """Plot mid price, spread, and trades from a simulation history.

    Parameters
    ----------
    history:
        A list of step records (as returned by ``Simulation.run``) or a
        ``pandas.DataFrame`` produced by ``Simulation.to_dataframe()``.

    Returns
    -------
    ``(fig, axes)`` from matplotlib. ``axes`` is a 3-element array with:
    - mid price + trade markers
    - bid-ask spread
    - trade volume per step
    """
    df = _to_dataframe(history)

    required = {"step", "mid_price", "spread", "trade_count", "trade_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    assert len(axes) == 3  # appease type checkers / maintainers

    # Mid price with trade markers.
    ax_mid = axes[0]
    ax_mid.plot(df["step"], df["mid_price"], color="#2980b9", label="Mid price")
    trade_steps = df.loc[df["trade_count"] > 0, "step"]
    trade_prices = df.loc[df["trade_count"] > 0, "mid_price"]
    if not trade_steps.empty:
        ax_mid.scatter(trade_steps, trade_prices, color="#e67e22", marker="x", label="Trade")
    ax_mid.set_ylabel("Mid price")
    ax_mid.set_title("Simulation time series")
    ax_mid.legend(loc="best")
    ax_mid.grid(axis="y", linestyle="--", alpha=0.5)

    # Spread.
    ax_spread = axes[1]
    ax_spread.plot(df["step"], df["spread"], color="#8e44ad", label="Spread")
    ax_spread.set_ylabel("Spread")
    ax_spread.legend(loc="best")
    ax_spread.grid(axis="y", linestyle="--", alpha=0.5)

    # Trade volume.
    ax_vol = axes[2]
    ax_vol.bar(df["step"], df["trade_volume"], color="#c0392b", label="Trade volume")
    ax_vol.set_xlabel("Step")
    ax_vol.set_ylabel("Volume")
    ax_vol.legend(loc="best")
    ax_vol.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig, axes
