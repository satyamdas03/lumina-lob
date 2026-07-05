"""Replay historical tick events through the matching engine."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from lumina_lob.core.book import OrderBook
from lumina_lob.core.matching import MatchingEngine
from lumina_lob.core.order import Order, OrderType, Side


class ReplayEngine:
    """Replay a merged stream of quote updates and trades through the engine.

    For every quote event, the previous synthetic best-bid/best-ask limit orders
    are cancelled and replaced by fresh ones at the new quoted prices.  For
    every trade event, a market order in the trade direction is submitted with
    the traded quantity.  After each event the current best spread is recorded,
    allowing a comparison between the real quote spread distribution and the
    spread distribution produced by the engine.
    """

    def __init__(self, book: Optional[OrderBook] = None, engine: Optional[MatchingEngine] = None, default_quote_size: int = 100):
        self.book = book if book is not None else OrderBook()
        self.engine = engine if engine is not None else MatchingEngine(self.book)
        self.default_quote_size = int(default_quote_size)
        self._order_id = 0
        self._quote_order_ids: list[int] = []
        self._timestamps: list[pd.Timestamp] = []
        self._spreads: list[float] = []
        self._best_bids: list[Optional[float]] = []
        self._best_asks: list[Optional[float]] = []
        self._trade_count = 0
        self._trade_volume = 0.0

    def replay(
        self,
        events: pd.DataFrame,
        event_col: str = "event_type",
        bid_col: str = "bid_px",
        ask_col: str = "ask_px",
        side_col: str = "side",
        size_col: str = "size",
    ) -> pd.DataFrame:
        """Process ``events`` in row order and return a metrics DataFrame.

        Parameters
        ----------
        events:
            DataFrame with at least ``timestamp`` and ``event_type`` columns.
            ``event_type`` values must be ``"quote"`` or ``"trade"``.  Quote rows
            must contain ``bid_col`` and ``ask_col``.  Trade rows must contain
            ``side_col`` and ``size_col``.
        event_col:
            Name of the column that distinguishes quote and trade events.
        bid_col, ask_col:
            Column names for quoted bid/ask prices.
        side_col, size_col:
            Column names for trade side and trade size.

        Returns
        -------
        pandas.DataFrame
            One row per event with columns ``timestamp``, ``spread``,
            ``best_bid``, and ``best_ask``.
        """
        if events.empty:
            raise ValueError("events DataFrame is empty")
        if "timestamp" not in events.columns or event_col not in events.columns:
            raise ValueError(f"events DataFrame must contain 'timestamp' and '{event_col}' columns")

        for _, row in events.iterrows():
            kind = str(row[event_col]).lower()
            self._clear_quotes()
            if kind == "quote":
                self._post_quote(row, bid_col, ask_col)
            elif kind == "trade":
                self._execute_trade(row, side_col, size_col)
            else:
                raise ValueError(f"unsupported event_type: {row[event_col]}")
            self._record(row["timestamp"])

        return pd.DataFrame(
            {
                "timestamp": self._timestamps,
                "spread": self._spreads,
                "best_bid": self._best_bids,
                "best_ask": self._best_asks,
            }
        )

    def _clear_quotes(self) -> None:
        for order_id in self._quote_order_ids:
            self.book.cancel(order_id)
        self._quote_order_ids = []

    def _post_quote(self, row: pd.Series, bid_col: str, ask_col: str) -> None:
        bid = row.get(bid_col)
        ask = row.get(ask_col)
        if pd.notna(bid):
            order_id = self._next_order_id()
            order = Order(
                order_id=order_id,
                side=Side.BID,
                price=float(bid),
                qty=self.default_quote_size,
                order_type=OrderType.LIMIT,
            )
            self.engine.process(order)
            self._quote_order_ids.append(order_id)
        if pd.notna(ask):
            order_id = self._next_order_id()
            order = Order(
                order_id=order_id,
                side=Side.ASK,
                price=float(ask),
                qty=self.default_quote_size,
                order_type=OrderType.LIMIT,
            )
            self.engine.process(order)
            self._quote_order_ids.append(order_id)

    def _execute_trade(self, row: pd.Series, side_col: str, size_col: str) -> None:
        side_value = row.get(side_col)
        size_value = row.get(size_col)
        if pd.isna(size_value):
            return
        size = int(size_value)
        if size <= 0:
            return
        sign = _sign_from_value(side_value)
        if sign == 0:
            return
        side = Side.BID if sign > 0 else Side.ASK
        order = Order(
            order_id=self._next_order_id(),
            side=side,
            price=None,
            qty=size,
            order_type=OrderType.MARKET,
        )
        self.engine.process(order)
        self._trade_count += 1
        self._trade_volume += size

    def _record(self, timestamp) -> None:
        self._timestamps.append(pd.Timestamp(timestamp))
        bid = self.book.best_bid
        ask = self.book.best_ask
        self._best_bids.append(bid)
        self._best_asks.append(ask)
        if bid is not None and ask is not None:
            self._spreads.append(float(ask - bid))
        else:
            self._spreads.append(np.nan)

    def _next_order_id(self) -> int:
        self._order_id += 1
        return self._order_id


def validate_spread_distribution(real_spreads: pd.Series, simulated_spreads: pd.Series, bins: Optional[int] = None) -> float:
    """Return a histogram-overlap similarity score between two spread distributions.

    The score is in ``[0, 1]`` where ``1`` means identical normalized histograms and
    ``0`` means no overlap.
    """
    real = np.asarray(real_spreads.dropna(), dtype=float)
    sim = np.asarray(simulated_spreads.dropna(), dtype=float)
    if real.size == 0 or sim.size == 0:
        return 0.0

    if bins is None:
        bins = max(5, int(min(real.size, sim.size) ** 0.5))
    lo = min(real.min(), sim.min())
    hi = max(real.max(), sim.max())
    if hi <= lo:
        return 1.0 if np.allclose(real, sim) else 0.0

    edges = np.linspace(lo, hi, bins + 1)
    real_hist, _ = np.histogram(real, bins=edges)
    sim_hist, _ = np.histogram(sim, bins=edges)
    real_density = real_hist / real_hist.sum() if real_hist.sum() else np.zeros_like(real_hist)
    sim_density = sim_hist / sim_hist.sum() if sim_hist.sum() else np.zeros_like(sim_hist)
    overlap = np.minimum(real_density, sim_density).sum()
    return float(overlap)


def _sign_from_value(value) -> int:
    """Map a side label to +1 (buy/bid), -1 (sell/ask), or 0 (unknown)."""
    if value is None:
        return 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0
    lowered = str(value).lower().strip()
    if lowered in {"buy", "bid", "long", "b", "1", "+1"}:
        return 1
    if lowered in {"sell", "ask", "short", "s", "-1"}:
        return -1
    return 0
