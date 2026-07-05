"""Order data model."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Side(Enum):
    BID = auto()
    ASK = auto()


class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()
    IOC = auto()   # Immediate or Cancel: fill what you can, rest cancelled
    FOK = auto()   # Fill or Kill: all or nothing


@dataclass
class Order:
    """Single order in the book."""

    order_id: int
    side: Side
    price: Optional[int]
    qty: int
    order_type: OrderType = OrderType.LIMIT
    filled_qty: int = field(default=0, init=False)
    next: Optional["Order"] = field(default=None, repr=False)
    prev: Optional["Order"] = field(default=None, repr=False)

    def __post_init__(self):
        if self.price is not None and self.price <= 0:
            raise ValueError("price must be positive")
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("market order cannot have price")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("limit order must have price")

    @property
    def remaining_qty(self) -> int:
        return self.qty - self.filled_qty

    @property
    def is_filled(self) -> bool:
        return self.remaining_qty == 0

    def fill(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("fill amount must be positive")
        if amount > self.remaining_qty:
            raise ValueError("fill amount exceeds remaining qty")
        self.filled_qty += amount

    def reduce_qty(self, new_qty: int) -> int:
        """Reduce total desired qty. Return amount removed."""
        if new_qty < self.filled_qty:
            raise ValueError("new qty cannot be below filled qty")
        if new_qty > self.qty:
            raise ValueError("cannot increase qty via reduce")
        removed = self.qty - new_qty
        self.qty = new_qty
        return removed
