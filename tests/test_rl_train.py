"""Smoke tests for RL training helpers."""
from __future__ import annotations

import pytest

from lumina_lob.rl.env import MarketMakerEnv
from lumina_lob.rl.train import (
    evaluate_agent,
    make_env,
    save_model,
    train_ppo,
    train_sac,
)


def test_make_env_factory():
    factory = make_env(seed=7)
    env = factory()
    assert isinstance(env.unwrapped, MarketMakerEnv)
    env.reset(seed=7)


def test_train_ppo_smoke():
    env = MarketMakerEnv(seed=1, max_steps=20, warmup_steps=1)
    env.reset(seed=1)
    model = train_ppo(env, total_timesteps=64, verbose=0)
    assert model is not None
    env.close()


def test_train_sac_smoke():
    env = MarketMakerEnv(seed=2, max_steps=20, warmup_steps=1)
    env.reset(seed=2)
    model = train_sac(env, total_timesteps=32, verbose=0)
    assert model is not None
    env.close()


def test_evaluate_agent_smoke():
    env = make_env(seed=3)()
    env.reset(seed=3)
    model = train_ppo(env, total_timesteps=64, verbose=0)
    mean_reward, std_reward = evaluate_agent(model, env, n_eval_episodes=2)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    env.close()


def test_save_model(tmp_path):
    env = MarketMakerEnv(seed=4, max_steps=10, warmup_steps=1)
    env.reset(seed=4)
    model = train_ppo(env, total_timesteps=64, verbose=0)
    path = tmp_path / "ppo_mm"
    save_model(model, path)
    assert path.exists() or (path.with_suffix(".zip")).exists()
    env.close()


def test_train_helpers_exported():
    import lumina_lob.rl as rl

    assert callable(rl.make_env)
    assert callable(rl.train_ppo)
    assert callable(rl.train_sac)
    assert callable(rl.evaluate_agent)
    assert callable(rl.save_model)
