"""Tests for the Gymnasium market-maker environment."""
from __future__ import annotations

import numpy as np
import pytest

from lumina_lob.core.order import Side
from lumina_lob.rl.env import MarketMakerEnv


def zero_action():
    return np.zeros(4, dtype=np.float32)


def test_env_has_expected_spaces():
    env = MarketMakerEnv()
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32


def test_reset_returns_observation_and_info():
    env = MarketMakerEnv(seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert "reference_price" in info
    assert env.simulation is not None


def test_step_returns_valid_tuple():
    env = MarketMakerEnv(max_steps=5, warmup_steps=2, seed=1)
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(zero_action())
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert not terminated


def test_truncated_at_max_steps():
    env = MarketMakerEnv(max_steps=3, warmup_steps=0, seed=2)
    env.reset(seed=2)
    terminated = False
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(zero_action())
    assert not terminated
    assert truncated


def test_observation_time_fraction_increases():
    env = MarketMakerEnv(max_steps=10, warmup_steps=1, seed=3)
    env.reset(seed=3)
    _, _, _, _, _ = env.step(zero_action())
    t1 = env._current_step
    obs1 = env._get_observation()
    _, _, _, _, _ = env.step(zero_action())
    obs2 = env._get_observation()
    assert obs2[-1] > obs1[-1]


def test_determinism_with_same_seed():
    env1 = MarketMakerEnv(max_steps=5, warmup_steps=2, seed=7)
    env2 = MarketMakerEnv(max_steps=5, warmup_steps=2, seed=7)
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    assert np.allclose(obs1, obs2)
    for _ in range(5):
        obs1, *_ = env1.step(zero_action())
        obs2, *_ = env2.step(zero_action())
    assert np.allclose(obs1, obs2)


def test_step_before_reset_raises():
    env = MarketMakerEnv()
    with pytest.raises(RuntimeError, match="reset before calling"):
        env.step(zero_action())


def test_observation_features_documented():
    env = MarketMakerEnv()
    assert len(env.observation_features) == 10
    assert env.observation_features[-1] == "time_fraction"


def test_observation_zero_when_simulation_none():
    env = MarketMakerEnv()
    env.simulation = None
    obs = env._get_observation()
    assert obs.shape == (10,)
    assert np.allclose(obs, 0.0)


def test_update_inventory_buy_and_sell():
    env = MarketMakerEnv()
    env.reset(seed=1)

    # Agent buys 100 shares at price 100 -> inventory +100, cash -10_000
    env._update_inventory(Side.BID, 100, 100.0)
    assert env._inventory == 100.0
    assert env._cash == -10_000.0
    assert env._avg_fill_price == 100.0

    # Agent buys another 50 shares at 110 -> VWAP = 103.333...
    env._update_inventory(Side.BID, 50, 110.0)
    assert env._inventory == 150.0
    assert env._avg_fill_price == (100 * 100 + 50 * 110) / 150

    # Agent sells 150 shares at 120 -> flat, cash increases by proceeds
    expected_cash = -10_000.0 - 5500.0 + 150 * 120.0
    env._update_inventory(Side.ASK, 150, 120.0)
    assert env._inventory == 0.0
    assert env._cash == expected_cash
    assert env._avg_fill_price == 0.0


def test_invalid_action_shape_raises():
    env = MarketMakerEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="action shape"):
        env.step(0.0)


def test_action_space_extremes_map_to_quote_params():
    env = MarketMakerEnv(
        tick_size=0.1,
        max_quote_offset_ticks=10,
        min_quote_size=5,
        max_quote_size=50,
    )
    env.reset(seed=1)

    # Minimum action -> smallest offsets and sizes
    env._pending_action = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    orders = env._action_to_quotes(env._pending_action, env._reference_price)
    assert len(orders) == 2
    bid, ask = orders
    assert bid.side == Side.BID
    assert ask.side == Side.ASK
    # Offset zero means quote at mid (or one tick apart to avoid crossing)
    assert bid.qty == env.min_quote_size
    assert ask.qty == env.min_quote_size

    # Maximum action -> largest offsets and sizes
    env._pending_action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    orders = env._action_to_quotes(env._pending_action, env._reference_price)
    bid, ask = orders
    assert bid.qty == env.max_quote_size
    assert ask.qty == env.max_quote_size
    assert ask.price > bid.price
    # Ask should be at least mid + max offset
    mid = env.simulation.book.mid_price or env._reference_price
    assert ask.price >= mid


def test_agent_proxy_act_no_action_returns_empty():
    env = MarketMakerEnv()
    env.reset(seed=1)
    assert env._proxy.act(env._reference_price, env.simulation.book) == []


def test_step_posts_agent_quotes_to_book():
    env = MarketMakerEnv(
        max_steps=5,
        warmup_steps=1,
        tick_size=0.1,
        max_quote_offset_ticks=2,
        seed=4,
    )
    env.reset(seed=4)
    action = np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
    env.step(action)

    agent_orders = [
        order for order in env.simulation.book.orders.values()
        if getattr(order, "_agent_quote", False)
    ]
    assert len(agent_orders) == 2
    sides = {order.side for order in agent_orders}
    assert sides == {Side.BID, Side.ASK}


def test_agent_quotes_cancelled_between_steps():
    env = MarketMakerEnv(
        max_steps=5,
        warmup_steps=1,
        tick_size=0.1,
        seed=5,
    )
    env.reset(seed=5)
    action1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    env.step(action1)
    old_ids = {
        order.order_id for order in env.simulation.book.orders.values()
        if getattr(order, "_agent_quote", False)
    }
    assert old_ids

    action2 = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    env.step(action2)
    new_ids = {
        order.order_id for order in env.simulation.book.orders.values()
        if getattr(order, "_agent_quote", False)
    }
    assert not old_ids & new_ids


def test_agent_proxy_on_fill_updates_inventory():
    env = MarketMakerEnv()
    env.reset(seed=1)
    # Force a reference price for predictable cash accounting
    env._reference_price = 100.0
    env._proxy.on_fill(Side.BID, 10)
    assert env._inventory == 10.0
    assert env._cash == -1000.0
    env._proxy.on_fill(Side.ASK, 10)
    assert env._inventory == 0.0
    assert env._cash == 0.0


def test_fill_price_fallback_to_reference():
    env = MarketMakerEnv()
    env.reset(seed=1)
    # Wipe the simulation so mid is unavailable
    env.simulation = None
    env._reference_price = 50.0
    env._update_inventory(Side.BID, 5)
    assert env._inventory == 5.0
    assert env._cash == -250.0


def test_action_to_quotes_no_simulation_returns_empty():
    env = MarketMakerEnv()
    env.simulation = None
    assert env._action_to_quotes(zero_action(), 100.0) == []


def test_cancel_agent_quotes_no_simulation_is_safe():
    env = MarketMakerEnv()
    env.simulation = None
    env._cancel_agent_quotes()  # should not raise
