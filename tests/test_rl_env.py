"""Tests for the Gymnasium market-maker environment."""
from __future__ import annotations

import numpy as np
import pytest

from lumina_lob.core.order import Side
from lumina_lob.rl.env import MarketMakerEnv


def test_env_has_expected_spaces():
    env = MarketMakerEnv()
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 1


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
    obs, reward, terminated, truncated, info = env.step(0)
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
        _, _, terminated, truncated, _ = env.step(0)
    assert not terminated
    assert truncated


def test_observation_time_fraction_increases():
    env = MarketMakerEnv(max_steps=10, warmup_steps=1, seed=3)
    env.reset(seed=3)
    _, _, _, _, _ = env.step(0)
    t1 = env._current_step
    obs1 = env._get_observation()
    _, _, _, _, _ = env.step(0)
    obs2 = env._get_observation()
    assert obs2[-1] > obs1[-1]


def test_determinism_with_same_seed():
    env1 = MarketMakerEnv(max_steps=5, warmup_steps=2, seed=7)
    env2 = MarketMakerEnv(max_steps=5, warmup_steps=2, seed=7)
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    assert np.allclose(obs1, obs2)
    for _ in range(5):
        obs1, *_ = env1.step(0)
        obs2, *_ = env2.step(0)
    assert np.allclose(obs1, obs2)


def test_step_before_reset_raises():
    env = MarketMakerEnv()
    with pytest.raises(RuntimeError, match="reset before calling"):
        env.step(0)


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
