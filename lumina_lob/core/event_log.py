"""Nanosecond-precision event journal for order book lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import perf_counter_ns
from typing import Dict, List, Optional


class EventType(Enum):
    ADD = auto()
    CANCEL = auto()
    MODIFY = auto()
    FILL = auto()


@dataclass
class Event:
    """Single order book event."""

    event_id: int
    timestamp_ns: int
    event_type: EventType
    order_id: int
    side: Optional[str] = None
    price: Optional[int] = None
    qty: Optional[int] = None
    filled_qty: Optional[int] = None
    counterparty_id: Optional[int] = None
    trade_qty: Optional[int] = None
    best_bid: Optional[int] = None
    best_ask: Optional[int] = None


class EventLog:
    """Append-only journal of all book events."""

    def __init__(self) -> None:
        self.events: List[Event] = []
        self._counter: int = 0

    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def _now_ns(self) -> int:
        return perf_counter_ns()

    def log_add(self, order_id: int, side: str, price: Optional[int], qty: int, best_bid: Optional[int], best_ask: Optional[int]) -> Event:
        ev = Event(
            event_id=self._next_id(),
            timestamp_ns=self._now_ns(),
            event_type=EventType.ADD,
            order_id=order_id,
            side=side,
            price=price,
            qty=qty,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        self.events.append(ev)
        return ev

    def log_cancel(self, order_id: int, best_bid: Optional[int], best_ask: Optional[int]) -> Event:
        ev = Event(
            event_id=self._next_id(),
            timestamp_ns=self._now_ns(),
            event_type=EventType.CANCEL,
            order_id=order_id,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        self.events.append(ev)
        return ev

    def log_modify(self, order_id: int, new_qty: int, best_bid: Optional[int], best_ask: Optional[int]) -> Event:
        ev = Event(
            event_id=self._next_id(),
            timestamp_ns=self._now_ns(),
            event_type=EventType.MODIFY,
            order_id=order_id,
            qty=new_qty,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        self.events.append(ev)
        return ev

    def log_fill(self, order_id: int, counterparty_id: int, trade_qty: int, price: int, side: str, filled_qty: int, best_bid: Optional[int], best_ask: Optional[int]) -> Event:
        ev = Event(
            event_id=self._next_id(),
            timestamp_ns=self._now_ns(),
            event_type=EventType.FILL,
            order_id=order_id,
            side=side,
            price=price,
            counterparty_id=counterparty_id,
            trade_qty=trade_qty,
            filled_qty=filled_qty,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        self.events.append(ev)
        return ev

    def to_dicts(self) -> List[Dict]:
        return [
            {
                "event_id": e.event_id,
                "timestamp_ns": e.timestamp_ns,
                "event_type": e.event_type.name,
                "order_id": e.order_id,
                "side": e.side,
                "price": e.price,
                "qty": e.qty,
                "filled_qty": e.filled_qty,
                "counterparty_id": e.counterparty_id,
                "trade_qty": e.trade_qty,
                "best_bid": e.best_bid,
                "best_ask": e.best_ask,
            }
            for e in self.events
        ]

    def __len__(self) -> int:
        return len(self.events)
