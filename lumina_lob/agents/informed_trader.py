"""Informed trader: directional signal with temporary/permanent impact tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np

from lumina_lob.agents.base import Agent
from lumina_lob.core import Order, OrderBook, OrderType, Side


@dataclass
class InformedTrader(Agent):
    """Trader that trades in the direction of a private signal.

    The informed trader submits aggressive market orders or large crossing
    limit orders to move the fair value. It tracks its own traded volume so
    downstream impact models can estimate temporary and permanent price impact.

    Parameters
    ----------
    signal:
        Direction of private information: ``bullish`` or ``bearish``.
    trade_size:
        Base quantity per order. Must be positive.
    participation_rate:
        Probability of submitting an order on any given ``act()`` call.
        Must be in [0, 1].
    order_type:
        ``market`` for aggressive fills, ``limit`` for large crossing orders.
    price_offset:
        For limit orders, how many ticks past the best price to cross. Default 1.
    tick_size:
        Minimum price increment. Default 1.0.
    seed:
        Optional RNG seed.
    """

    signal: Literal["bullish", "bearish"] = "bullish"
    trade_size: int = 100
    participation_rate: float = 0.5
    order_type: Literal["market", "limit"] = "market"
    price_offset: int = 1
    tick_size: float = 1.0
    seed: Optional[int] = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _next_order_id: int = field(init=False, repr=False)
    _total_traded: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.signal not in ("bullish", "bearish"):
            raise ValueError("signal must be 'bullish' or 'bearish'")
        if self.trade_size <= 0:
            raise ValueError("trade_size must be positive")
        if not 0.0 <= self.participation_rate <= 1.0:
            raise ValueError("participation_rate must be in [0, 1]")
        if self.price_offset < 0:
            raise ValueError("price_offset must be non-negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self._rng = np.random.default_rng(self.seed)
        self._next_order_id = 1
        self._total_traded = 0

    @property
    def side(self) -> Side:
        """Trading side implied by the signal."""
        return Side.BID if self.signal == "bullish" else Side.ASK

    @property
    def total_traded(self) -> int:
        """Cumulative quantity traded by this agent."""
        return self._total_traded

    def act(self, reference_price: float, book: OrderBook) -> List[Order]:
        """Generate an order if the trader participates this step."""
        if self._rng.random() >= self.participation_rate:
            return []

        side = self.side
        qty = self.trade_size

        if self.order_type == "market":
            order = Order(
                order_id=self._next_order_id,
                side=side,
                price=None,
                qty=qty,
                order_type=OrderType.MARKET,
            )
        else:
            price = self._crossing_price(reference_price, side, book)
            order = Order(
                order_id=self._next_order_id,
                side=side,
                price=price,
                qty=qty,
                order_type=OrderType.LIMIT,
            )

        self._next_order_id += 1
        self._total_traded += qty
        return [order]

    def _crossing_price(self, reference_price: float, side: Side, book: OrderBook) -> int:
        """Price that guarantees immediate execution in the signal direction."""
        if side == Side.BID:
            best_ask = book.best_ask
            if best_ask is not None:
                tick = round((best_ask + self.price_offset * self.tick_size) / self.tick_size)
            else:
                tick = round(reference_price / self.tick_size) + self.price_offset
        else:
            best_bid = book.best_bid
            if best_bid is not None:
                tick = round((best_bid - self.price_offset * self.tick_size) / self.tick_size)
            else:
                tick = round(reference_price / self.tick_size) - self.price_offset
        return max(1, int(round(tick * self.tick_size)))
