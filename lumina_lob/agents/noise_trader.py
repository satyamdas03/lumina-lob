"""Noise trader: random Poisson arrivals with randomized size and side."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np

from lumina_lob.agents.base import Agent
from lumina_lob.core import Order, OrderBook, Side


@dataclass
class NoiseTrader(Agent):
    """Liquidity-demanding agent that submits random limit orders.

    Parameters
    ----------
    arrival_rate:
        Expected number of orders emitted per `act()` call (Poisson).
    size_dist:
        Distribution for order quantity: ``uniform`` or ``lognormal``.
    size_min:
        Minimum order quantity for uniform distribution. Default 1.
    size_max:
        Maximum order quantity for uniform distribution. Default 10.
    size_mu:
        Mean of log quantity for log-normal distribution. Default 1.0.
    size_sigma:
        Standard deviation of log quantity for log-normal distribution. Default 0.5.
    side_bias:
        Probability of generating a bid (0.5 = neutral). Default 0.5.
    price_offset_max:
        Maximum number of ticks away from the rounded reference price. Default 5.
    tick_size:
        Price tick size. Default 1.0.
    seed:
        Optional RNG seed for reproducibility.
    """

    arrival_rate: float = 1.0
    size_dist: Literal["uniform", "lognormal"] = "uniform"
    size_min: int = 1
    size_max: int = 10
    size_mu: float = 1.0
    size_sigma: float = 0.5
    side_bias: float = 0.5
    price_offset_max: int = 5
    tick_size: float = 1.0
    seed: Optional[int] = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _next_order_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.arrival_rate < 0:
            raise ValueError("arrival_rate must be non-negative")
        if self.size_min <= 0:
            raise ValueError("size_min must be positive")
        if self.size_max < self.size_min:
            raise ValueError("size_max must be >= size_min")
        if self.size_sigma < 0:
            raise ValueError("size_sigma must be non-negative")
        if not 0.0 <= self.side_bias <= 1.0:
            raise ValueError("side_bias must be in [0, 1]")
        if self.price_offset_max < 0:
            raise ValueError("price_offset_max must be non-negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self._rng = np.random.default_rng(self.seed)
        self._next_order_id = 1

    def act(self, reference_price: float, book: OrderBook) -> List[Order]:
        """Generate a batch of random limit orders."""
        n_orders = self._rng.poisson(lam=self.arrival_rate)
        orders: List[Order] = []
        for _ in range(n_orders):
            side = Side.BID if self._rng.random() < self.side_bias else Side.ASK
            qty = self._draw_qty()
            price = self._draw_price(reference_price, side)
            orders.append(
                Order(
                    order_id=self._next_order_id,
                    side=side,
                    price=price,
                    qty=qty,
                )
            )
            self._next_order_id += 1
        return orders

    def _draw_qty(self) -> int:
        if self.size_dist == "uniform":
            return int(self._rng.integers(self.size_min, self.size_max + 1))
        # lognormal
        return max(self.size_min, int(self._rng.lognormal(self.size_mu, self.size_sigma)))

    def _draw_price(self, reference_price: float, side: Side) -> int:
        ticks_away = int(self._rng.integers(0, self.price_offset_max + 1))
        mid_tick = round(reference_price / self.tick_size)
        if side == Side.BID:
            tick = mid_tick - ticks_away
        else:
            tick = mid_tick + ticks_away
        price = tick * self.tick_size
        return max(1, int(round(price)))
