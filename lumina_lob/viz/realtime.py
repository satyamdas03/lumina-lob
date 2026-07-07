"""Real-time simulation visualisation using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import matplotlib.animation as _animation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for visualization. "
        "Install it with `pip install matplotlib` or use the viz extras."
    ) from exc

from lumina_lob.simulation import Simulation

from .depth_ladder import _detect_side_enums, _to_level_items


def _draw_depth_ladder(ax: Any, book: Any, top_n: int) -> None:
    """Redraw the depth ladder on *ax* from *book*."""
    ax.clear()
    bid_side, ask_side = _detect_side_enums(book)
    bid_items = _to_level_items(book.depth(bid_side, top_n), reverse=True)
    ask_items = _to_level_items(book.depth(ask_side, top_n), reverse=False)

    all_prices = sorted({price for price, _ in bid_items + ask_items})
    if not all_prices:
        ax.set_title("Depth ladder (empty)")
        return

    y_positions = {price: idx for idx, price in enumerate(all_prices)}

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
    ax.set_title("Live depth ladder")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)


class SimulationAnimator:
    """Animate a running ``Simulation`` with depth ladder + mid-price trace."""

    def __init__(
        self,
        simulation: Simulation,
        top_n: int = 10,
        history_window: int = 50,
        interval_ms: int = 200,
    ) -> None:
        self.simulation = simulation
        self.top_n = top_n
        self.history_window = history_window
        self.interval_ms = interval_ms

        self.fig = plt.figure(figsize=(12, 5))
        self.ax_depth = self.fig.add_subplot(1, 2, 1)
        self.ax_price = self.fig.add_subplot(1, 2, 2)

        self._price_line, = self.ax_price.plot([], [], color="#2980b9", label="Mid price")
        self._trade_scatter = self.ax_price.scatter([], [], color="#e67e22", marker="x", label="Trade")
        self.ax_price.set_xlabel("Step")
        self.ax_price.set_ylabel("Mid price")
        self.ax_price.set_title("Mid-price trace")
        self.ax_price.legend(loc="best")
        self.ax_price.grid(axis="y", linestyle="--", alpha=0.5)

    def _update_price_axis(self) -> None:
        """Rescale the price trace to the available history."""
        history = self.simulation.history[-self.history_window :]
        if not history:
            return
        valid = [(r["step"], r["mid_price"]) for r in history if r["mid_price"] is not None]
        if not valid:
            return
        steps, mids = zip(*valid, strict=False)
        self._price_line.set_data(steps, mids)
        self.ax_price.set_xlim(min(steps), max(max(steps), min(steps) + 1))
        y_min, y_max = min(mids), max(mids)
        pad = max((y_max - y_min) * 0.1, 0.01)
        self.ax_price.set_ylim(y_min - pad, y_max + pad)

        trade_steps = [
            cast(int, r["step"]) for r in history
            if cast(int, r["trade_count"]) > 0 and r["mid_price"] is not None
        ]
        trade_mids = [
            cast(float, r["mid_price"]) for r in history
            if cast(int, r["trade_count"]) > 0 and r["mid_price"] is not None
        ]
        if trade_mids:
            self._trade_scatter.set_offsets(np.column_stack([trade_steps, trade_mids]))
        else:
            self._trade_scatter.set_offsets(np.zeros((0, 2)))

    def update(self, _frame: int | None = None) -> tuple[Any, ...]:
        """Advance the simulation by one step and redraw both panels."""
        self.simulation.step()
        _draw_depth_ladder(self.ax_depth, self.simulation.book, self.top_n)
        self._update_price_axis()
        self.fig.tight_layout()
        return (self._price_line, self._trade_scatter)

    def run(self, n_steps: int = 100) -> FuncAnimation:
        """Return a ``FuncAnimation`` that runs *n_steps* frames."""
        return FuncAnimation(
            self.fig,
            self.update,
            frames=n_steps,
            interval=self.interval_ms,
            blit=False,
            repeat=False,
        )


def run_animation(
    simulation: Simulation,
    n_steps: int = 100,
    top_n: int = 10,
    history_window: int = 50,
    interval_ms: int = 200,
) -> FuncAnimation:
    """Create and return a running animation for *simulation*.

    Parameters
    ----------
    simulation:
        The ``Simulation`` to animate.
    n_steps:
        Number of animation frames.
    top_n:
        Number of price levels to show on each side of the depth ladder.
    history_window:
        How many recent steps to show in the mid-price trace.
    interval_ms:
        Delay between frames in milliseconds.

    Returns
    -------
    A ``matplotlib.animation.FuncAnimation`` instance. Call ``plt.show()`` or
    save it to display/record.
    """
    animator = SimulationAnimator(
        simulation,
        top_n=top_n,
        history_window=history_window,
        interval_ms=interval_ms,
    )
    return animator.run(n_steps)


def save_animation(
    animation: FuncAnimation,
    path: str | Path,
    fps: int = 5,
) -> None:
    """Save a simulation animation to *path* as GIF or MP4.

    Parameters
    ----------
    animation:
        A ``matplotlib.animation.FuncAnimation`` instance (e.g. from
        ``run_animation``).
    path:
        Destination file path. Supported extensions: ``.gif`` (requires
        Pillow) and ``.mp4`` (requires ffmpeg).
    fps:
        Frames per second for the output file.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the required writer is not
        available.
    """
    out = Path(path)
    writer_map = {".gif": "pillow", ".mp4": "ffmpeg"}
    ext = out.suffix.lower()
    if ext not in writer_map:
        raise ValueError(f"Unsupported animation format: {ext!r}. Use .gif or .mp4.")

    writer = writer_map[ext]
    if not _animation.writers.is_available(writer):
        raise ValueError(
            f"{writer!r} writer is not available. "
            f"Install it to save {ext} animations (e.g. `pip install pillow` for .gif)."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(out), writer=writer, fps=fps)
