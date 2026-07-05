"""Tests for the informed trader agent."""
from __future__ import annotations

import pytest

from lumina_lob.agents import InformedTrader
from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side


def test_validation_signal():
    with pytest.raises(ValueError, match="signal must be 'bullish' or 'bearish'"):
        InformedTrader(signal="neutral")  # type: ignore[arg-type]


def test_validation_trade_size():
    with pytest.raises(ValueError, match="trade_size must be positive"):
        InformedTrader(trade_size=0)


def test_validation_participation_rate():
    with pytest.raises(ValueError, match="participation_rate must be in"):
        InformedTrader(participation_rate=1.5)


def test_validation_price_offset():
    with pytest.raises(ValueError, match="price_offset must be non-negative"):
        InformedTrader(price_offset=-1)


def test_validation_tick_size():
    with pytest.raises(ValueError, match="tick_size must be positive"):
        InformedTrader(tick_size=0.0)


def test_zero_participation_returns_no_orders():
    book = OrderBook()
    agent = InformedTrader(participation_rate=0.0, seed=42)
    orders = agent.act(reference_price=100.0, book=book)
    assert orders == []
    assert agent.total_traded == 0


def test_bullish_only_buys():
    book = OrderBook()
    agent = InformedTrader(
        signal="bullish",
        participation_rate=1.0,
        trade_size=50,
        order_type="market",
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert len(orders) == 1
    assert orders[0].side == Side.BID
    assert orders[0].qty == 50


def test_bearish_only_sells():
    book = OrderBook()
    agent = InformedTrader(
        signal="bearish",
        participation_rate=1.0,
        trade_size=50,
        order_type="market",
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert len(orders) == 1
    assert orders[0].side == Side.ASK
    assert orders[0].qty == 50


def test_market_order_type():
    book = OrderBook()
    agent = InformedTrader(
        participation_rate=1.0,
        order_type="market",
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert orders[0].order_type == OrderType.MARKET
    assert orders[0].price is None


def test_limit_order_crosses_ask():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 101, 10))
    agent = InformedTrader(
        signal="bullish",
        participation_rate=1.0,
        trade_size=5,
        order_type="limit",
        price_offset=2,
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert orders[0].order_type == OrderType.LIMIT
    assert orders[0].side == Side.BID
    assert orders[0].price >= 103


def test_limit_order_crosses_bid():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 99, 10))
    agent = InformedTrader(
        signal="bearish",
        participation_rate=1.0,
        trade_size=5,
        order_type="limit",
        price_offset=2,
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert orders[0].order_type == OrderType.LIMIT
    assert orders[0].side == Side.ASK
    assert orders[0].price <= 97


def test_total_traded_accumulates():
    book = OrderBook()
    agent = InformedTrader(participation_rate=1.0, trade_size=20, seed=1)
    agent.act(reference_price=100.0, book=book)
    agent.act(reference_price=100.0, book=book)
    assert agent.total_traded == 40


def test_order_ids_increment():
    book = OrderBook()
    agent = InformedTrader(participation_rate=1.0, seed=1)
    o1 = agent.act(reference_price=100.0, book=book)
    o2 = agent.act(reference_price=100.0, book=book)
    assert o1[0].order_id == 1
    assert o2[0].order_id == 2


def test_limit_price_when_book_empty():
    book = OrderBook()
    agent = InformedTrader(
        signal="bullish",
        participation_rate=1.0,
        order_type="limit",
        price_offset=3,
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert orders[0].price == 103


def test_bearish_limit_price_when_book_empty():
    book = OrderBook()
    agent = InformedTrader(
        signal="bearish",
        participation_rate=1.0,
        order_type="limit",
        price_offset=3,
        seed=1,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert orders[0].price == 97


def test_reproducible_with_seed():
    a1 = InformedTrader(participation_rate=0.5, seed=42)
    a2 = InformedTrader(participation_rate=0.5, seed=42)
    counts1 = sum(len(a1.act(100.0, OrderBook())) for _ in range(100))
    counts2 = sum(len(a2.act(100.0, OrderBook())) for _ in range(100))
    assert counts1 == counts2
