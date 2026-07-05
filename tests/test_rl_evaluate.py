"""Tests for heuristic policy evaluation helpers."""
from __future__ import annotations

import numpy as np

from lumina_lob.rl.env import MarketMakerEnv
from lumina_lob.rl.evaluate import (
    SimpleMarketMakerPolicy,
    evaluate_heuristic_policy,
    summarize_results,
)


def make_env(seed: int = 0) -> MarketMakerEnv:
    return MarketMakerEnv(seed=seed, max_steps=5, warmup_steps=1)


def test_policy_returns_valid_action():
    env = make_env()
    env.reset(seed=1)
    policy = SimpleMarketMakerPolicy()
    action = policy(env)
    assert action.shape == env.action_space.shape
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_policy_skews_with_inventory():
    env = make_env()
    env.reset(seed=1)
    policy = SimpleMarketMakerPolicy(base_offset_ticks=2, max_inventory_skew=1.0)

    # No inventory -> symmetric offsets
    action_flat = policy(env)
    assert action_flat[0] == action_flat[1]

    # Long inventory should tighten ask (lower offset) and widen bid
    env._inventory = 500.0
    action_long = policy(env)
    assert action_long[1] < action_long[0]

    # Short inventory should tighten bid and widen ask
    env._inventory = -500.0
    action_short = policy(env)
    assert action_short[0] < action_short[1]


def test_evaluate_heuristic_policy_runs_episodes():
    results = evaluate_heuristic_policy(
        lambda: make_env(seed=2),
        SimpleMarketMakerPolicy(),
        n_episodes=3,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r.total_reward, float)
        assert isinstance(r.total_pnl, float)
        assert r.n_steps > 0


def test_summarize_results():
    env = make_env()
    env.reset(seed=1)
    results = evaluate_heuristic_policy(
        lambda: make_env(seed=3),
        SimpleMarketMakerPolicy(),
        n_episodes=2,
    )
    summary = summarize_results(results)
    assert "mean_reward" in summary
    assert "mean_pnl" in summary
    assert "max_abs_inventory" in summary
    assert summary["max_abs_inventory"] >= 0.0


def test_policy_with_no_simulation_returns_zeros():
    env = make_env()
    env.reset(seed=1)
    env.simulation = None
    policy = SimpleMarketMakerPolicy()
    action = policy(env)
    assert np.allclose(action, 0.0)


def test_summarize_empty_results():
    assert summarize_results([]) == {}
