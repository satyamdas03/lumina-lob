"""Unit tests for order book."""
from __future__ import annotations

import pytest

from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side


def test_add_limit_orders():
    book = OrderBook()
    book.add(Order(1, Side.BID, 100, 10))
    book.add(Order(2, Side.ASK, 101, 5))
    assert book.best_bid == 100
    assert book.best_ask == 101
    assert book.spread == 1
    assert book.mid_price == 100.5


def test_price_time_priority_match():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 100, 5))
    engine.process(Order(2, Side.ASK, 100, 3))
    engine.process(Order(3, Side.BID, 100, 7))
    # aggressor fully filled, not stored; resting order 2 has 1 left
    assert len(book.orders) == 1
    assert 3 not in book.orders
    assert book.orders[2].remaining_qty == 1
    assert len(book.trades) == 2


def test_partial_fill_and_rest():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 3))
    engine.process(Order(2, Side.ASK, 100, 10))
    assert book.orders[2].remaining_qty == 7
    assert book.best_ask == 100
    assert book.best_bid is None
    assert len(book.trades) == 1
    assert book.trades[0] == (2, 1, 3)


def test_cancel_order():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 100, 5))
    assert book.cancel(1) is True
    assert len(book.orders) == 1
    assert book.depth(Side.BID)[100] == 5


def test_cancel_missing_order():
    book = OrderBook()
    assert book.cancel(99) is False


def test_market_order_consumes_best():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 99, 5))
    market = Order(3, Side.ASK, None, 7, OrderType.MARKET)
    engine.process(market)
    assert market.is_filled
    assert book.best_bid == 99
    assert book.depth(Side.BID)[99] == 3


def test_limit_order_no_match_rest():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.ASK, 102, 5))
    assert book.best_bid == 100
    assert book.best_ask == 102
    assert len(book.trades) == 0


def test_invalid_market_order_with_price():
    with pytest.raises(ValueError):
        Order(1, Side.BID, 100, 1, OrderType.MARKET)


def test_invalid_limit_order_without_price():
    with pytest.raises(ValueError):
        Order(1, Side.BID, None, 1, OrderType.LIMIT)
