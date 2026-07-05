"""Price level: queue of orders at same price using doubly-linked list."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .order import Order


@dataclass
class PriceLevel:
    """Orders at one price level. FIFO queue via linked list."""

    price: int
    head: Optional[Order] = field(default=None, init=False)
    tail: Optional[Order] = field(default=None, init=False)
    total_qty: int = field(default=0, init=False)
    order_count: int = field(default=0, init=False)

    def append(self, order: Order) -> None:
        """Add order to tail."""
        if self.tail is None:
            self.head = self.tail = order
            order.prev = order.next = None
        else:
            order.prev = self.tail
            order.next = None
            self.tail.next = order
            self.tail = order
        self.total_qty += order.remaining_qty
        self.order_count += 1

    def remove(self, order: Order) -> bool:
        """Remove order from queue. Return True if found."""
        if order.prev is None and order.next is None and self.head is not order:
            return False
        if order.prev:
            order.prev.next = order.next
        else:
            self.head = order.next
        if order.next:
            order.next.prev = order.prev
        else:
            self.tail = order.prev
        self.total_qty -= order.remaining_qty
        self.order_count -= 1
        order.prev = order.next = None
        return True

    def fill(self, amount: int) -> int:
        """Fill from front of queue. Return filled amount."""
        remaining = amount
        while remaining > 0 and self.head:
            front = self.head
            can_fill = min(front.remaining_qty, remaining)
            front.fill(can_fill)
            self.total_qty -= can_fill
            remaining -= can_fill
            if front.is_filled:
                self.remove(front)
        return amount - remaining

    def reduce(self, order: Order, new_qty: int) -> int:
        """Reduce order qty at this level. Return removed amount."""
        removed = order.reduce_qty(new_qty)
        self.total_qty -= removed
        return removed

    def is_empty(self) -> bool:
        return self.head is None

    def __iter__(self):
        node = self.head
        while node:
            yield node
            node = node.next

    def __len__(self) -> int:
        return self.order_count
