"""Skewed market maker: inventory-sensitive quoting."""
from __future__ import annotations

from dataclasses import dataclass, field

from lumina_lob.agents.base import Agent
from lumina_lob.core import Order, OrderBook, Side


@dataclass
class SkewedMarketMaker(Agent):
    """Market maker that skews quotes based on signed inventory.

    As inventory grows long, the market maker lowers bids and offers to attract
    sells. As inventory grows short, it raises bids and offers to attract buys.
    The skew is linear in the signed inventory ratio.

    Parameters
    ----------
    base_half_spread:
        Base half-spread in ticks before skew. Must be positive.
    quote_size:
        Quantity to quote on each side. Must be positive.
    max_inventory:
        Maximum absolute inventory allowed. Must be non-negative.
    skew_factor:
        How aggressively to skew quotes per unit of inventory ratio.
        inventory_ratio = inventory / max_inventory. Default 2.0.
    tick_size:
        Minimum price increment. Default 1.0.
    """

    base_half_spread: float = 2.0
    quote_size: int = 10
    max_inventory: int = 100
    skew_factor: float = 2.0
    tick_size: float = 1.0

    _inventory: int = field(init=False, repr=False)
    _next_order_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.base_half_spread <= 0:
            raise ValueError("base_half_spread must be positive")
        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.max_inventory < 0:
            raise ValueError("max_inventory must be non-negative")
        if self.skew_factor < 0:
            raise ValueError("skew_factor must be non-negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self._inventory = 0
        self._next_order_id = 1

    @property
    def inventory(self) -> int:
        """Current signed inventory (positive = long, negative = short)."""
        return self._inventory

    def act(self, reference_price: float, book: OrderBook) -> list[Order]:
        """Submit inventory-skewed bid/ask quotes."""
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
        """Compute bid and ask quote prices with inventory skew."""
        if self.max_inventory == 0:
            ratio = 0.0
        else:
            ratio = self._inventory / self.max_inventory

        # Inventory skew:
        #   long (ratio > 0) -> raise bid and lower ask to encourage selling
        #   short (ratio < 0) -> lower bid and raise ask to encourage buying
        skew_ticks = self.skew_factor * ratio

        bid_half = self.base_half_spread + skew_ticks
        ask_half = self.base_half_spread - skew_ticks

        # Keep a minimum one-tick spread on each side of mid
        min_half = 0.5
        bid_half = max(min_half, bid_half)
        ask_half = max(min_half, ask_half)

        bid_tick = round((reference_price - bid_half * self.tick_size) / self.tick_size)
        ask_tick = round((reference_price + ask_half * self.tick_size) / self.tick_size)

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
        """Update inventory when one of the market maker's orders is filled."""
        if side == Side.BID:
            self._inventory += qty
        elif side == Side.ASK:
            self._inventory -= qty

    def reset_inventory(self, value: int = 0) -> None:
        """Reset inventory to a target value."""
        self._inventory = value
