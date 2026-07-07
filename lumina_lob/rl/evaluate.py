"""Evaluate heuristic and trained RL market-making policies."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
        if env.simulation is None:
            shape = env.action_space.shape
            if shape is None:
                shape = (4,)
            return np.zeros(shape, dtype=np.float32)

        max_position = 1_000.0
        inventory_ratio = env._inventory / max_position

        bid_skew = self.base_offset_ticks + max(0.0, inventory_ratio * self.max_inventory_skew)
        ask_skew = self.base_offset_ticks + max(0.0, -inventory_ratio * self.max_inventory_skew)

        # Map offsets to [-1, 1] given the env's maximum quote offset.
        max_offset = max(1, env.max_quote_offset_ticks)
        bid_action = bid_skew / max_offset * 2.0 - 1.0
        ask_action = ask_skew / max_offset * 2.0 - 1.0

        bid_action = max(-1.0, min(1.0, bid_action))
        ask_action = max(-1.0, min(1.0, ask_action))

        return np.array(
            [bid_action, ask_action, -1.0, -1.0],
            dtype=np.float32,
        )


def evaluate_heuristic_policy(
    env_factory: Callable[[], MarketMakerEnv],
    policy: Callable[[MarketMakerEnv], np.ndarray],
    n_episodes: int = 5,
) -> list[EpisodeResult]:
    """Run a heuristic policy for several episodes and return summaries."""
    results: list[EpisodeResult] = []
    for _ in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset()
        total_reward = 0.0
        max_inv = 0.0
        min_inv = 0.0
        truncated = False
        while not truncated:
            action = policy(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            max_inv = max(max_inv, env._inventory)
            min_inv = min(min_inv, env._inventory)

        total_pnl = env._cash + env._inventory * env._reference_price
        results.append(
            EpisodeResult(
                total_reward=total_reward,
                total_pnl=total_pnl,
                final_inventory=env._inventory,
                max_inventory=max_inv,
                min_inventory=min_inv,
                n_steps=env._current_step,
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
