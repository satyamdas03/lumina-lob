"""Tests for the noise trader agent."""
from __future__ import annotations

import pytest

from lumina_lob.agents import NoiseTrader
from lumina_lob.core import OrderBook, Side


def test_zero_arrival_rate():
    book = OrderBook()
    agent = NoiseTrader(arrival_rate=0.0, seed=42)
    orders = agent.act(reference_price=100.0, book=book)
    assert orders == []


def test_validation_arrival_rate():
    with pytest.raises(ValueError, match="arrival_rate must be non-negative"):
        NoiseTrader(arrival_rate=-1.0)


def test_validation_size_min():
    with pytest.raises(ValueError, match="size_min must be positive"):
        NoiseTrader(size_min=0)


def test_validation_size_max():
    with pytest.raises(ValueError, match="size_max must be >= size_min"):
        NoiseTrader(size_min=10, size_max=1)


def test_validation_size_sigma():
    with pytest.raises(ValueError, match="size_sigma must be non-negative"):
        NoiseTrader(size_sigma=-0.1)


def test_validation_side_bias():
    with pytest.raises(ValueError, match="side_bias must be in"):
        NoiseTrader(side_bias=1.5)


def test_validation_price_offset_max():
    with pytest.raises(ValueError, match="price_offset_max must be non-negative"):
        NoiseTrader(price_offset_max=-1)


def test_validation_tick_size():
    with pytest.raises(ValueError, match="tick_size must be positive"):
        NoiseTrader(tick_size=0.0)


def test_positive_intensity_generates_orders():
    book = OrderBook()
    agent = NoiseTrader(arrival_rate=10.0, seed=123)
    orders = agent.act(reference_price=100.0, book=book)
    assert len(orders) > 0


def test_order_ids_increment():
    book = OrderBook()
    agent = NoiseTrader(arrival_rate=5.0, seed=1)
    orders = agent.act(reference_price=100.0, book=book)
    ids = [o.order_id for o in orders]
    assert ids == list(range(1, len(orders) + 1))


def test_uniform_sizes_in_bounds():
    book = OrderBook()
    agent = NoiseTrader(
        arrival_rate=50.0,
        size_dist="uniform",
        size_min=3,
        size_max=7,
        seed=7,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert all(3 <= o.qty <= 7 for o in orders)


def test_lognormal_sizes_positive():
    book = OrderBook()
    agent = NoiseTrader(
        arrival_rate=50.0,
        size_dist="lognormal",
        size_min=1,
        seed=7,
    )
    orders = agent.act(reference_price=100.0, book=book)
    assert all(o.qty >= 1 for o in orders)


def test_side_bias_toward_bid():
    book = OrderBook()
    agent = NoiseTrader(arrival_rate=200.0, side_bias=0.9, seed=99)
    orders = agent.act(reference_price=100.0, book=book)
    bid_ratio = sum(1 for o in orders if o.side == Side.BID) / len(orders)
    assert bid_ratio > 0.75


def test_side_bias_toward_ask():
    book = OrderBook()
    agent = NoiseTrader(arrival_rate=200.0, side_bias=0.1, seed=99)
    orders = agent.act(reference_price=100.0, book=book)
    ask_ratio = sum(1 for o in orders if o.side == Side.ASK) / len(orders)
    assert ask_ratio > 0.75


def test_prices_centered_around_reference():
    book = OrderBook()
    agent = NoiseTrader(
        arrival_rate=100.0,
        price_offset_max=3,
        tick_size=1.0,
        seed=42,
    )
    orders = agent.act(reference_price=100.0, book=book)
    for o in orders:
        assert 97 <= o.price <= 103


def test_prices_positive_with_low_reference():
    book = OrderBook()
    agent = NoiseTrader(
        arrival_rate=100.0,
        price_offset_max=5,
        seed=42,
    )
    orders = agent.act(reference_price=2.0, book=book)
    assert all(o.price >= 1 for o in orders)


def test_reproducible_with_seed():
    a1 = NoiseTrader(arrival_rate=5.0, seed=42)
    a2 = NoiseTrader(arrival_rate=5.0, seed=42)
    o1 = a1.act(reference_price=100.0, book=OrderBook())
    o2 = a2.act(reference_price=100.0, book=OrderBook())
    assert [(o.order_id, o.side, o.price, o.qty) for o in o1] == [
        (o.order_id, o.side, o.price, o.qty) for o in o2
    ]
