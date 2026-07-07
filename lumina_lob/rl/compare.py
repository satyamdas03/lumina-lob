"""Compare PPO and SAC market-maker agents on ``MarketMakerEnv``."""
from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from lumina_lob.rl.train import evaluate_agent, train_ppo, train_sac


def compare_ppo_sac(
    env_factory: Callable[[], gym.Env[np.ndarray, np.ndarray]],
    total_timesteps: int = 10_000,
    n_eval_episodes: int = 5,
) -> dict[str, dict[str, float]]:
    """Train PPO and SAC agents and return their evaluation statistics.

    Parameters
    ----------
    env_factory:
        Callable that returns a fresh Gymnasium environment.
    total_timesteps:
        Number of timesteps to train each agent for.
    n_eval_episodes:
        Number of evaluation episodes per agent.

    Returns
    -------
    A dictionary with ``ppo`` and ``sac`` keys, each mapping to
    ``{"mean": float, "std": float}`` reward statistics.
    """
    ppo_model = train_ppo(env_factory(), total_timesteps=total_timesteps)
    sac_model = train_sac(env_factory(), total_timesteps=total_timesteps)

    ppo_mean, ppo_std = evaluate_agent(ppo_model, env_factory(), n_eval_episodes)
    sac_mean, sac_std = evaluate_agent(sac_model, env_factory(), n_eval_episodes)

    return {
        "ppo": {"mean": ppo_mean, "std": ppo_std},
        "sac": {"mean": sac_mean, "std": sac_std},
    }
