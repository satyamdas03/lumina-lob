"""Market maker: symmetric quotes around reference price with inventory limits."""
from __future__ import annotations

from dataclasses import dataclass, field

from lumina_lob.agents.base import Agent
from lumina_lob.core import Order, OrderBook, Side


@dataclass
class MarketMaker(Agent):
    """Simple market maker that quotes symmetrically around the reference price.

    The agent maintains a target half-spread and a maximum inventory (long/short).
    If inventory reaches the limit on one side, it stops quoting that side until
    the position comes back within bounds.

    Parameters
    ----------
    spread_half_width:
        Half-spread in price ticks. Must be positive.
    quote_size:
        Quantity to quote on each side. Must be positive.
    max_inventory:
        Maximum absolute inventory allowed. Must be non-negative.
    tick_size:
        Minimum price increment. Default 1.0.
    """

    spread_half_width: float = 2.0
    quote_size: int = 10
    max_inventory: int = 100
    tick_size: float = 1.0

    _inventory: int = field(init=False, repr=False)
    _next_order_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.spread_half_width <= 0:
            raise ValueError("spread_half_width must be positive")
        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.max_inventory < 0:
            raise ValueError("max_inventory must be non-negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self._inventory = 0
        self._next_order_id = 1

    @property
    def inventory(self) -> int:
        """Current signed inventory (positive = long, negative = short)."""
        return self._inventory

    def act(self, reference_price: float, book: OrderBook) -> list[Order]:
        """Submit bid/ask quotes around reference price, respecting inventory limits."""
        bid_price, ask_price = self._quote_prices(reference_price)
        orders: list[Order] = []

        if self._can_quote_bid():
            orders.append(
                Order(
                    order_id=self._next_order_id,
                    side=Side.BID,
                    price=bid_price,
                    qty=self.quote_size,
                )
            )
            self._next_order_id += 1

        if self._can_quote_ask():
            orders.append(
                Order(
                    order_id=self._next_order_id,
                    side=Side.ASK,
                    price=ask_price,
                    qty=self.quote_size,
                )
            )
            self._next_order_id += 1

        return orders

    def _quote_prices(self, reference_price: float) -> tuple[float, float]:
        """Compute bid and ask quote prices rounded to ticks."""
        half_spread = self.spread_half_width * self.tick_size
        bid_tick = round((reference_price - half_spread) / self.tick_size)
        ask_tick = round((reference_price + half_spread) / self.tick_size)
        # Ensure positive ask tick and non-overlapping spread
        bid_tick = max(1, bid_tick)
        ask_tick = max(bid_tick + 1, ask_tick)
        return float(bid_tick * self.tick_size), float(ask_tick * self.tick_size)

    def _can_quote_bid(self) -> bool:
        """Can quote a bid if not at the maximum long position."""
        return self._inventory < self.max_inventory

    def _can_quote_ask(self) -> bool:
        """Can quote an ask if not at the maximum short position."""
        return self._inventory > -self.max_inventory

    def on_fill(self, side: Side, qty: int) -> None:
        """Update inventory when one of the market maker's orders is filled.

        Call this from the simulation loop after processing the agent's orders.
        """
        if side == Side.BID:
            # Bought qty -> inventory increases
            self._inventory += qty
        elif side == Side.ASK:
            # Sold qty -> inventory decreases
            self._inventory -= qty

    def reset_inventory(self, value: int = 0) -> None:
        """Reset inventory to a target value."""
        self._inventory = value
