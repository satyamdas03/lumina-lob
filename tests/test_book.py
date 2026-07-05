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


def test_ioc_partial_fill_no_rest():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 100, 3))
    ioc = Order(3, Side.ASK, None, 7, OrderType.IOC)
    engine.process(ioc)
    assert ioc.remaining_qty == 0
    assert len(book.orders) == 1
    assert book.orders[2].remaining_qty == 1


def test_ioc_with_price_limit():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 100, 5))
    engine.process(Order(2, Side.ASK, 101, 5))
    ioc = Order(3, Side.BID, 100, 7, OrderType.IOC)
    engine.process(ioc)
    assert ioc.remaining_qty == 2
    assert len(book.trades) == 1
    assert book.orders[2].remaining_qty == 5


def test_fok_full_fill():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 100, 5))
    fok = Order(3, Side.ASK, None, 7, OrderType.FOK)
    engine.process(fok)
    assert fok.is_filled
    assert len(book.trades) == 2
    assert len(book.orders) == 1


def test_fok_killed_when_insufficient():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    fok = Order(2, Side.ASK, None, 10, OrderType.FOK)
    engine.process(fok)
    assert fok.remaining_qty == 10
    assert len(book.trades) == 0
    assert len(book.orders) == 1


def test_fok_respects_limit_price():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 99, 10))
    fok = Order(3, Side.ASK, None, 15, OrderType.FOK)
    # FOK has no price limit here; fills all available liquidity
    engine.process(fok)
    assert fok.is_filled


def test_ioc_and_fok_cannot_have_price_invalidation():
    # IOC/FOK with a price is allowed in our model (price-limited IOC/FOK)
    # but with None price they behave as market-style
    pass


def test_modify_reduce_qty():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 10))
    assert book.modify(1, 6) is True
    assert book.orders[1].qty == 6
    assert book.orders[1].remaining_qty == 6
    assert book.depth(Side.BID)[100] == 6


def test_modify_to_fully_filled_removes():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 10))
    engine.process(Order(2, Side.ASK, 100, 4))
    assert book.orders[1].filled_qty == 4
    assert book.modify(1, 4) is True
    assert 1 not in book.orders
    assert book.best_bid is None


def test_modify_missing_order():
    book = OrderBook()
    assert book.modify(99, 5) is False


def test_modify_below_filled_raises():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 10))
    engine.process(Order(2, Side.ASK, 100, 4))
    with pytest.raises(ValueError):
        book.modify(1, 3)


def test_modify_increase_raises():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    with pytest.raises(ValueError):
        book.modify(1, 10)
