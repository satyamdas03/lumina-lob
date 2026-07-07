"""Order book with bid/ask price levels."""
from __future__ import annotations

from typing import Any

from lumina_lob.core.event_log import EventLog
from lumina_lob.core.order import Order, Side
from lumina_lob.core.price_level import PriceLevel


class OrderBook:
    """Price-time priority order book."""

    def __init__(self, event_log: EventLog | None = None) -> None:
        self.bids: dict[float, PriceLevel] = {}
        self.asks: dict[float, PriceLevel] = {}
        self.orders: dict[int, Order] = {}
        self.trades: list[tuple[int, int, int]] = []  # (buy_id, sell_id, qty)
        self.event_log = event_log if event_log is not None else EventLog()

    # ---------- helpers ----------
    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def mid_price(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _side_levels(self, side: Side) -> dict[float, PriceLevel]:
        return self.bids if side == Side.BID else self.asks

    # ---------- public ----------
    def add(self, order: Order) -> None:
        """Insert order into book. Market orders are not stored."""
        if order.order_id in self.orders:
            raise ValueError(f"duplicate order_id {order.order_id}")
        self.orders[order.order_id] = order
        price = order.price
        if price is None:
            raise ValueError("resting order must have a price")
        if order.side == Side.BID:
            level = self.bids.setdefault(price, PriceLevel(price))
        else:
            level = self.asks.setdefault(price, PriceLevel(price))
        level.append(order)
        self.event_log.log_add(
            order_id=order.order_id,
            side=order.side.name,
            price=order.price,
            qty=order.qty,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
        )

    def cancel(self, order_id: int) -> bool:
        """Cancel resting order. Return True if cancelled."""
        order = self.orders.pop(order_id, None)
        if order is None:
            return False
        price = order.price
        if price is None:
            return False
        levels = self._side_levels(order.side)
        level = levels.get(price)
        if level is None:
            return False
        level.remove(order)
        if level.is_empty():
            del levels[price]
        self.event_log.log_cancel(
            order_id=order_id,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
        )
        return True

    def modify(self, order_id: int, new_qty: int) -> bool:
        """Reduce resting order to new total qty. Remove if fully filled."""
        order = self.orders.get(order_id)
        if order is None:
            return False
        if new_qty <= 0:
            raise ValueError("new qty must be positive")
        price = order.price
        if price is None:
            return False
        levels = self._side_levels(order.side)
        level = levels.get(price)
        if level is None:
            return False
        level.reduce(order, new_qty)
        if order.is_filled:
            level.remove(order)
            self.orders.pop(order_id, None)
        if level.is_empty():
            del levels[price]
        self.event_log.log_modify(
            order_id=order_id,
            new_qty=new_qty,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
        )
        return True

    def full_depth(self, side: Side) -> dict[float, int]:
        """Return all price levels and total qty for a side."""
        levels = self._side_levels(side)
        return {p: levels[p].total_qty for p in sorted(levels.keys(), reverse=(side == Side.BID))}

    def depth(self, side: Side, n: int = 5) -> dict[float, int]:
        """Return top N price levels and total qty."""
        levels = self._side_levels(side)
        prices = sorted(levels.keys(), reverse=(side == Side.BID))
        return {p: levels[p].total_qty for p in prices[:n]}

    def snapshot(self) -> dict[str, dict[float, int]]:
        return {"bids": self.depth(Side.BID), "asks": self.depth(Side.ASK)}

    def full_snapshot(self) -> dict[str, dict[float, int]]:
        return {"bids": self.full_depth(Side.BID), "asks": self.full_depth(Side.ASK)}

    def to_pandas(self) -> Any:
        """Return book depth as pandas DataFrame with columns [side, price, qty, order_count]."""
        import pandas as pd

        rows: list[dict[str, object]] = []
        for side in (Side.BID, Side.ASK):
            for price, level in self._side_levels(side).items():
                rows.append({
                    "side": side.name,
                    "price": price,
                    "qty": level.total_qty,
                    "order_count": len(level),
                })
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.orders)
