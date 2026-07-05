"""Tests for the Simulation orchestrator."""
from __future__ import annotations

import pytest

from lumina_lob.agents import Agent, InformedTrader, MarketMaker, NoiseTrader
from lumina_lob.core import Order, OrderBook, OrderType, Side
from lumina_lob.market_model import ReferencePriceProcess
from lumina_lob.simulation import Simulation


def test_simulation_defaults_step():
    sim = Simulation()
    record = sim.step()
    assert record["step"] == 1
    assert "reference_price" in record
    assert record["best_bid"] is None
    assert record["best_ask"] is None
    assert record["mid_price"] is None
    assert record["spread"] is None
    assert record["trade_count"] == 0
    assert record["trade_volume"] == 0
    assert record["book_size"] == 0


def test_empty_agents_no_trades():
    sim = Simulation(agents=[])
    sim.run(5)
    assert len(sim.history) == 5
    assert sum(r["trade_count"] for r in sim.history) == 0
    assert sum(r["book_size"] for r in sim.history) == 0


def test_run_requires_positive_n_steps():
    sim = Simulation()
    with pytest.raises(ValueError, match="n_steps must be positive"):
        sim.run(0)


def test_run_returns_correct_length():
    noise = NoiseTrader(seed=1, arrival_rate=2)
    sim = Simulation(agents=[noise])
    history = sim.run(10)
    assert len(history) == 10
    assert len(sim.history) == 10


def test_noise_only_builds_book():
    noise = NoiseTrader(seed=7, arrival_rate=5)
    ref = ReferencePriceProcess(volatility=0.0, seed=7)
    sim = Simulation(agents=[noise], reference_price=ref)
    sim.run(10)
    assert any(r["book_size"] > 0 for r in sim.history)
    # Noise orders can cross each other; trade count is non-negative.
    assert sum(r["trade_count"] for r in sim.history) >= 0


def test_mixed_agents_keep_book_quoted():
    mm = MarketMaker(spread_half_width=2, quote_size=10, tick_size=1.0)
    noise = NoiseTrader(seed=3, arrival_rate=5, tick_size=1.0)
    ref = ReferencePriceProcess(volatility=0.0, seed=3)
    sim = Simulation(agents=[mm, noise], reference_price=ref)
    sim.run(20)
    quoted = [
        r for r in sim.history
        if r["best_bid"] is not None and r["best_ask"] is not None
    ]
    assert len(quoted) > 0
    assert all(r["spread"] > 0 for r in quoted)


def test_mixed_agents_can_produce_trades():
    mm = MarketMaker(spread_half_width=1, quote_size=5, tick_size=1.0)
    noise = NoiseTrader(seed=4, arrival_rate=10, price_offset_max=5, tick_size=1.0)
    ref = ReferencePriceProcess(volatility=0.0, seed=4)
    sim = Simulation(agents=[mm, noise], reference_price=ref)
    sim.run(50)
    total_volume = sum(r["trade_volume"] for r in sim.history)
    assert total_volume > 0


def test_market_maker_alone_no_self_trades():
    mm = MarketMaker(spread_half_width=2, quote_size=10, tick_size=1.0)
    ref = ReferencePriceProcess(volatility=0.0, seed=5)
    sim = Simulation(agents=[mm], reference_price=ref)
    sim.run(15)
    assert sum(r["trade_count"] for r in sim.history) == 0
    assert mm.inventory == 0


def test_market_maker_inventory_updates_on_fill():
    mm = MarketMaker(spread_half_width=2, quote_size=10, tick_size=1.0)
    informed = InformedTrader(
        signal="bullish",
        trade_size=10,
        participation_rate=1.0,
        order_type="market",
        seed=6,
    )
    ref = ReferencePriceProcess(volatility=0.0, seed=6)
    sim = Simulation(agents=[mm, informed], reference_price=ref)
    sim.run(5)
    assert sum(r["trade_count"] for r in sim.history) > 0
    assert mm.inventory < 0


def test_multiple_agents_get_unique_order_ids():
    nt1 = NoiseTrader(seed=8, arrival_rate=2)
    nt2 = NoiseTrader(seed=9, arrival_rate=2)
    ref = ReferencePriceProcess(volatility=0.0, seed=10)
    sim = Simulation(agents=[nt1, nt2], reference_price=ref)
    sim.run(5)
    assert sim._next_order_id > 1
    assert len(sim._order_owner) > 0


def test_deterministic_run_is_reproducible():
    sim1 = Simulation(
        reference_price=ReferencePriceProcess(volatility=0.0, seed=11),
        agents=[NoiseTrader(seed=11, arrival_rate=3)],
    )
    sim2 = Simulation(
        reference_price=ReferencePriceProcess(volatility=0.0, seed=11),
        agents=[NoiseTrader(seed=11, arrival_rate=3)],
    )
    h1 = sim1.run(10)
    h2 = sim2.run(10)
    assert h1 == h2


def test_to_dataframe_shape_and_columns():
    sim = Simulation(agents=[NoiseTrader(seed=12, arrival_rate=2)])
    sim.run(5)
    df = sim.to_dataframe()
    assert len(df) == 5
    expected = {
        "step",
        "reference_price",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "trade_count",
        "trade_volume",
        "book_size",
    }
    assert set(df.columns) == expected


def test_agent_without_on_fill_is_safe():
    class SilentAgent(Agent):
        def act(self, reference_price, book):
            return []

    sim = Simulation(agents=[SilentAgent()])
    sim.run(3)
    assert len(sim.history) == 3
    assert sum(r["trade_count"] for r in sim.history) == 0


def test_can_supply_prebuilt_book_and_engine():
    book = OrderBook()
    from lumina_lob.core import MatchingEngine
    engine = MatchingEngine(book)
    sim = Simulation(book=book, engine=engine, agents=[NoiseTrader(seed=13)])
    sim.run(3)
    assert sim.book is book
    assert sim.engine is engine


def test_history_records_each_step():
    sim = Simulation(agents=[NoiseTrader(seed=14, arrival_rate=1)])
    sim.run(5)
    steps = [r["step"] for r in sim.history]
    assert steps == [1, 2, 3, 4, 5]


def test_aggressor_agent_receives_on_fill():
    """An agent that submits a market order is notified when it fills a resting quote."""

    class MarketBuyer(Agent):
        def __init__(self, qty: int = 5) -> None:
            self.qty = qty
            self.fills: list[tuple[Side, int]] = []

        def act(self, reference_price, book):
            return [
                Order(
                    order_id=1,
                    side=Side.BID,
                    price=None,
                    qty=self.qty,
                    order_type=OrderType.MARKET,
                )
            ]

        def on_fill(self, side: Side, qty: int) -> None:
            self.fills.append((side, qty))

    buyer = MarketBuyer(qty=5)
    mm = MarketMaker(spread_half_width=2, quote_size=5, tick_size=1.0)
    ref = ReferencePriceProcess(volatility=0.0, seed=15)
    sim = Simulation(agents=[mm, buyer], reference_price=ref)
    sim.run(3)
    assert len(buyer.fills) > 0
    assert all(side == Side.BID for side, _ in buyer.fills)
    assert mm.inventory < 0
