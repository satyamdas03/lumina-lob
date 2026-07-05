"""Tests for the basic market maker agent."""
from __future__ import annotations

import pytest

from lumina_lob.agents import MarketMaker
from lumina_lob.core import OrderBook, Side


def test_validation_spread_half_width():
    with pytest.raises(ValueError, match="spread_half_width must be positive"):
        MarketMaker(spread_half_width=0.0)


def test_validation_quote_size():
    with pytest.raises(ValueError, match="quote_size must be positive"):
        MarketMaker(quote_size=0)


def test_validation_max_inventory():
    with pytest.raises(ValueError, match="max_inventory must be non-negative"):
        MarketMaker(max_inventory=-1)


def test_validation_tick_size():
    with pytest.raises(ValueError, match="tick_size must be positive"):
        MarketMaker(tick_size=0.0)


def test_quotes_symmetric_around_reference():
    book = OrderBook()
    mm = MarketMaker(spread_half_width=2.0, quote_size=10, tick_size=1.0)
    orders = mm.act(reference_price=100.0, book=book)
    assert len(orders) == 2
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    assert bid.price == 98
    assert ask.price == 102
    assert bid.qty == 10
    assert ask.qty == 10


def test_quotes_positive_prices():
    book = OrderBook()
    mm = MarketMaker(spread_half_width=5.0, quote_size=1, tick_size=1.0)
    orders = mm.act(reference_price=2.0, book=book)
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    assert bid.price >= 1
    assert ask.price > bid.price


def test_bid_suppressed_when_inventory_at_max():
    book = OrderBook()
    mm = MarketMaker(spread_half_width=2.0, quote_size=10, max_inventory=100)
    mm.reset_inventory(100)
    orders = mm.act(reference_price=100.0, book=book)
    sides = [o.side for o in orders]
    assert Side.BID not in sides
    assert Side.ASK in sides


def test_ask_suppressed_when_inventory_at_min():
    book = OrderBook()
    mm = MarketMaker(spread_half_width=2.0, quote_size=10, max_inventory=100)
    mm.reset_inventory(-100)
    orders = mm.act(reference_price=100.0, book=book)
    sides = [o.side for o in orders]
    assert Side.ASK not in sides
    assert Side.BID in sides


def test_max_inventory_zero_allows_no_quotes():
    book = OrderBook()
    mm = MarketMaker(spread_half_width=2.0, quote_size=10, max_inventory=0)
    orders = mm.act(reference_price=100.0, book=book)
    assert orders == []


def test_inventory_increases_on_bid_fill():
    mm = MarketMaker(quote_size=5)
    assert mm.inventory == 0
    mm.on_fill(Side.BID, 5)
    assert mm.inventory == 5


def test_inventory_decreases_on_ask_fill():
    mm = MarketMaker(quote_size=5)
    mm.on_fill(Side.ASK, 5)
    assert mm.inventory == -5


def test_order_ids_increment():
    book = OrderBook()
    mm = MarketMaker()
    o1 = mm.act(reference_price=100.0, book=book)
    o2 = mm.act(reference_price=100.0, book=book)
    ids = [o.order_id for batch in (o1, o2) for o in batch]
    assert ids == list(range(1, len(ids) + 1))


def test_tick_size_rounding():
    book = OrderBook()
    mm = MarketMaker(spread_half_width=1.0, quote_size=1, tick_size=0.5)
    orders = mm.act(reference_price=100.0, book=book)
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    assert bid.price == 99.5
    assert ask.price == 100.5
