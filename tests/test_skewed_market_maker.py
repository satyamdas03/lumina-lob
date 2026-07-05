"""Tests for the skewed market maker agent."""
from __future__ import annotations

import pytest

from lumina_lob.agents import SkewedMarketMaker
from lumina_lob.core import OrderBook, Side


def test_validation_base_half_spread():
    with pytest.raises(ValueError, match="base_half_spread must be positive"):
        SkewedMarketMaker(base_half_spread=0.0)


def test_validation_quote_size():
    with pytest.raises(ValueError, match="quote_size must be positive"):
        SkewedMarketMaker(quote_size=0)


def test_validation_max_inventory():
    with pytest.raises(ValueError, match="max_inventory must be non-negative"):
        SkewedMarketMaker(max_inventory=-1)


def test_validation_skew_factor():
    with pytest.raises(ValueError, match="skew_factor must be non-negative"):
        SkewedMarketMaker(skew_factor=-0.1)


def test_validation_tick_size():
    with pytest.raises(ValueError, match="tick_size must be positive"):
        SkewedMarketMaker(tick_size=0.0)


def test_quotes_symmetric_when_inventory_flat():
    book = OrderBook()
    mm = SkewedMarketMaker(base_half_spread=2.0, quote_size=10, skew_factor=0.0)
    orders = mm.act(reference_price=100.0, book=book)
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    assert bid.price == 98
    assert ask.price == 102


def test_long_inventory_skews_quotes_down():
    book = OrderBook()
    mm = SkewedMarketMaker(
        base_half_spread=2.0,
        quote_size=10,
        max_inventory=100,
        skew_factor=2.0,
    )
    mm.reset_inventory(50)
    orders = mm.act(reference_price=100.0, book=book)
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    # With ratio 0.5 and skew_factor 2, bid half-spread grows by 1 tick,
    # ask half-spread shrinks by 1 tick -> bid lower, ask lower.
    assert bid.price == 97
    assert ask.price == 101


def test_short_inventory_skews_quotes_up():
    book = OrderBook()
    mm = SkewedMarketMaker(
        base_half_spread=2.0,
        quote_size=10,
        max_inventory=100,
        skew_factor=2.0,
    )
    mm.reset_inventory(-50)
    orders = mm.act(reference_price=100.0, book=book)
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    # With ratio -0.5 and skew_factor 2, bid half-spread shrinks by 1 tick,
    # ask half-spread grows by 1 tick -> bid higher, ask higher.
    assert bid.price == 99
    assert ask.price == 103


def test_bid_suppressed_when_inventory_at_max():
    book = OrderBook()
    mm = SkewedMarketMaker(base_half_spread=2.0, quote_size=10, max_inventory=100)
    mm.reset_inventory(100)
    orders = mm.act(reference_price=100.0, book=book)
    sides = [o.side for o in orders]
    assert Side.BID not in sides
    assert Side.ASK in sides


def test_ask_suppressed_when_inventory_at_min():
    book = OrderBook()
    mm = SkewedMarketMaker(base_half_spread=2.0, quote_size=10, max_inventory=100)
    mm.reset_inventory(-100)
    orders = mm.act(reference_price=100.0, book=book)
    sides = [o.side for o in orders]
    assert Side.ASK not in sides
    assert Side.BID in sides


def test_max_inventory_zero_allows_no_quotes():
    book = OrderBook()
    mm = SkewedMarketMaker(base_half_spread=2.0, quote_size=10, max_inventory=0)
    orders = mm.act(reference_price=100.0, book=book)
    assert orders == []


def test_inventory_increases_on_bid_fill():
    mm = SkewedMarketMaker(quote_size=5)
    mm.on_fill(Side.BID, 5)
    assert mm.inventory == 5


def test_inventory_decreases_on_ask_fill():
    mm = SkewedMarketMaker(quote_size=5)
    mm.on_fill(Side.ASK, 5)
    assert mm.inventory == -5


def test_skew_is_bounded_by_min_half_spread():
    book = OrderBook()
    mm = SkewedMarketMaker(
        base_half_spread=1.0,
        quote_size=1,
        max_inventory=100,
        skew_factor=100.0,
    )
    mm.reset_inventory(99)
    orders = mm.act(reference_price=100.0, book=book)
    bid = next(o for o in orders if o.side == Side.BID)
    ask = next(o for o in orders if o.side == Side.ASK)
    # Even with huge skew, minimum half-spread keeps bid < ask
    assert bid.price < ask.price
    assert ask.price - bid.price >= 1
