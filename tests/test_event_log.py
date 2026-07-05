"""Unit tests for event log."""
from __future__ import annotations

from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side
from lumina_lob.core.event_log import EventLog, EventType


def test_event_log_add_and_fill():
    log = EventLog()
    book = OrderBook(event_log=log)
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.ASK, 100, 3))
    # order 2 fully fills against order 1, so only ADD(order1) + FILL logged
    assert len(log.events) == 2
    assert log.events[0].event_type == EventType.ADD
    assert log.events[0].order_id == 1
    assert log.events[1].event_type == EventType.FILL
    assert log.events[1].trade_qty == 3
    assert log.events[1].counterparty_id == 1


def test_event_log_cancel():
    log = EventLog()
    book = OrderBook(event_log=log)
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    book.cancel(1)
    assert len(log.events) == 2
    assert log.events[1].event_type == EventType.CANCEL
    assert log.events[1].order_id == 1


def test_event_log_modify():
    log = EventLog()
    book = OrderBook(event_log=log)
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 10))
    book.modify(1, 6)
    assert log.events[1].event_type == EventType.MODIFY
    assert log.events[1].qty == 6


def test_event_log_to_dicts():
    log = EventLog()
    book = OrderBook(event_log=log)
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    dicts = log.to_dicts()
    assert len(dicts) == 1
    assert dicts[0]["event_type"] == "ADD"
