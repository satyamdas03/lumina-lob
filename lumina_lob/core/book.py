"""Order book with bid/ask price levels."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

from lumina_lob.core.order import Order, Side
from lumina_lob.core.price_level import PriceLevel


class OrderBook:
    """Price-time priority order book."""

    def __init__(self) -> None:
        self.bids: Dict[int, PriceLevel] = {}
        self.asks: Dict[int, PriceLevel] = {}
        self.orders: Dict[int, Order] = {}
        self.trades: list[Tuple[int, int, int]] = []  # (buy_id, sell_id, qty)

    # ---------- helpers ----------
    @property
    def best_bid(self) -> Optional[int]:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[int]:
        return min(self.asks) if self.asks else None

    @property
    def spread(self) -> Optional[int]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _side_levels(self, side: Side) -> Dict[int, PriceLevel]:
        return self.bids if side == Side.BID else self.asks

    # ---------- public ----------
    def add(self, order: Order) -> None:
        """Insert order into book. Market orders are not stored."""
        if order.order_id in self.orders:
            raise ValueError(f"duplicate order_id {order.order_id}")
        self.orders[order.order_id] = order
        if order.side == Side.BID:
            level = self.bids.setdefault(order.price, PriceLevel(order.price))
        else:
            level = self.asks.setdefault(order.price, PriceLevel(order.price))
        level.append(order)

    def cancel(self, order_id: int) -> bool:
        """Cancel resting order. Return True if cancelled."""
        order = self.orders.pop(order_id, None)
        if order is None:
            return False
        levels = self._side_levels(order.side)
        level = levels.get(order.price)
        if level is None:
            return False
        level.remove(order)
        if level.is_empty():
            del levels[order.price]
        return True

    def modify(self, order_id: int, new_qty: int) -> bool:
        """Reduce resting order to new total qty. Remove if fully filled."""
        order = self.orders.get(order_id)
        if order is None:
            return False
        if new_qty <= 0:
            raise ValueError("new qty must be positive")
        levels = self._side_levels(order.side)
        level = levels.get(order.price)
        if level is None:
            return False
        level.reduce(order, new_qty)
        if order.is_filled:
            level.remove(order)
            self.orders.pop(order_id, None)
        if level.is_empty():
            del levels[order.price]
        return True

    def depth(self, side: Side, n: int = 5) -> Dict[int, int]:
        """Return top N price levels and total qty."""
        levels = self._side_levels(side)
        prices = sorted(levels.keys(), reverse=(side == Side.BID))
        return {p: levels[p].total_qty for p in prices[:n]}

    def snapshot(self) -> Dict[str, Dict[int, int]]:
        return {"bids": self.depth(Side.BID), "asks": self.depth(Side.ASK)}

    def __len__(self) -> int:
        return len(self.orders)
