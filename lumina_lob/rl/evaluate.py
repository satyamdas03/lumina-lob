"""Evaluate heuristic and trained RL market-making policies."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from lumina_lob.rl.env import MarketMakerEnv


@dataclass
class EpisodeResult:
    """Summary statistics for one episode."""

    total_reward: float
    total_pnl: float
    final_inventory: float
    max_inventory: float
    min_inventory: float
    n_steps: int


class SimpleMarketMakerPolicy:
    """Deterministic heuristic that skews quotes based on inventory.

    Long inventory widens the bid offset and tightens the ask offset to
    encourage selling; short inventory does the opposite.  Quote sizes are
    always at the configured minimum.
    """

    def __init__(
        self,
        base_offset_ticks: int = 2,
        max_inventory_skew: float = 1.0,
    ) -> None:
        self.base_offset_ticks = max(0, int(base_offset_ticks))
        self.max_inventory_skew = float(max_inventory_skew)

    def __call__(self, env: MarketMakerEnv) -> np.ndarray:
        """Return an action vector for the current env state."""
        base_env = getattr(env, "unwrapped", env)
        if not isinstance(base_env, MarketMakerEnv):
            base_env = env

        if base_env.simulation is None:
            shape = env.action_space.shape
            if shape is None:
                shape = (4,)
            return np.zeros(shape, dtype=np.float32)

        max_position = 1_000.0
        inventory_ratio = base_env._inventory / max_position

        bid_skew = self.base_offset_ticks + max(0.0, inventory_ratio * self.max_inventory_skew)
        ask_skew = self.base_offset_ticks + max(0.0, -inventory_ratio * self.max_inventory_skew)

        # Map offsets to [-1, 1] given the env's maximum quote offset.
        max_offset = max(1, base_env.max_quote_offset_ticks)
        bid_action = bid_skew / max_offset * 2.0 - 1.0
        ask_action = ask_skew / max_offset * 2.0 - 1.0

        bid_action = max(-1.0, min(1.0, bid_action))
        ask_action = max(-1.0, min(1.0, ask_action))

        return np.array(
            [bid_action, ask_action, -1.0, -1.0],
            dtype=np.float32,
        )


def _unwrap_env(env: gym.Env) -> MarketMakerEnv:
    """Return the inner ``MarketMakerEnv`` from a possibly wrapped env."""
    base = getattr(env, "unwrapped", env)
    if isinstance(base, MarketMakerEnv):
        return base
    # Fallback: some wrappers expose the wrapped env as `.env`.
    inner = getattr(env, "env", env)
    base = getattr(inner, "unwrapped", inner)
    if isinstance(base, MarketMakerEnv):
        return base
    raise TypeError(f"Expected MarketMakerEnv, got {type(env)!r}")  # pragma: no cover


def evaluate_heuristic_policy(
    env_factory: Callable[[], gym.Env],
    policy: Callable[[gym.Env], np.ndarray],
    n_episodes: int = 5,
) -> list[EpisodeResult]:
    """Run a heuristic policy for several episodes and return summaries."""
    results: list[EpisodeResult] = []
    for _ in range(n_episodes):
        env = env_factory()
        base_env = _unwrap_env(env)
        obs, _ = env.reset()
        total_reward = 0.0
        max_inv = 0.0
        min_inv = 0.0
        truncated = False
        while not truncated:
            action = policy(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            max_inv = max(max_inv, base_env._inventory)
            min_inv = min(min_inv, base_env._inventory)

        total_pnl = base_env._cash + base_env._inventory * base_env._reference_price
        results.append(
            EpisodeResult(
                total_reward=total_reward,
                total_pnl=total_pnl,
                final_inventory=base_env._inventory,
                max_inventory=max_inv,
                min_inventory=min_inv,
                n_steps=base_env._current_step,
            )
        )
    return results


def summarize_results(results: list[EpisodeResult]) -> dict[str, float]:
    """Return mean and std of key metrics across episodes."""
    if not results:
        return {}
    return {
        "mean_reward": float(np.mean([r.total_reward for r in results])),
        "mean_pnl": float(np.mean([r.total_pnl for r in results])),
        "mean_final_inventory": float(np.mean([r.final_inventory for r in results])),
        "max_abs_inventory": float(
            np.max([max(abs(r.max_inventory), abs(r.min_inventory)) for r in results])
        ),
    }
