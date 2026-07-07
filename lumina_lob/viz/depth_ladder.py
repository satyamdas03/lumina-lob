"""Depth-ladder visualization for an order book."""

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

from lumina_lob.core import Side

try:
    from lumina_lob import _core  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _core = None


def _detect_side_enums(book: Any) -> tuple[Any, Any]:
    """Return the correct Side enum objects for the given book instance."""
    if _core is not None and isinstance(book, _core.OrderBook):
        return _core.Side.BID, _core.Side.ASK
    return Side.BID, Side.ASK


def _to_level_items(raw: Any, reverse: bool) -> list[tuple[float, int]]:
    """Normalise `depth()` output to a list of (price, qty) pairs."""
    if isinstance(raw, dict):
        items = sorted(raw.items(), key=lambda item: item[0], reverse=reverse)
    else:
        # C++ extension returns a list-like of (price, qty) pairs already ordered.
        items = list(raw)
    return items


def plot_depth_ladder(book: Any, top_n: int = 10) -> tuple[Figure, Any]:
    """Render a horizontal depth-ladder plot for *book*.

    Parameters
    ----------
    book:
        An ``OrderBook`` instance (Python ``lumina_lob.core.OrderBook`` or the
        C++ ``lumina_lob._core.OrderBook``).
    top_n:
        Number of price levels to show on each side.

    Returns
    -------
    ``(fig, ax)`` from matplotlib.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    bid_side, ask_side = _detect_side_enums(book)
    bid_items = _to_level_items(book.depth(bid_side, top_n), reverse=True)
    ask_items = _to_level_items(book.depth(ask_side, top_n), reverse=False)

    all_prices = sorted({price for price, _ in bid_items + ask_items})
    if not all_prices:
        raise ValueError("book has no price levels to plot")

    y_positions = {price: idx for idx, price in enumerate(all_prices)}

    fig, ax = plt.subplots(figsize=(8, max(4, len(all_prices) * 0.4)))

    for price, qty in bid_items:
        ax.barh(
            y_positions[price],
            -qty,
            color="#2ecc71",
            edgecolor="black",
            height=0.6,
            label="Bid" if price == bid_items[0][0] else "",
        )

    for price, qty in ask_items:
        ax.barh(
            y_positions[price],
            qty,
            color="#e74c3c",
            edgecolor="black",
            height=0.6,
            label="Ask" if price == ask_items[0][0] else "",
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(all_prices)))
    ax.set_yticklabels(str(p) for p in all_prices)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.set_title("Limit Order Book Depth Ladder")
    ax.legend(loc="lower right")
    fig.tight_layout()

    return fig, ax
