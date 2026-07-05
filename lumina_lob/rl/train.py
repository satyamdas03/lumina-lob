"""Training and evaluation helpers for RL market makers."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from lumina_lob.rl.env import MarketMakerEnv


def make_env(seed: int = 0) -> Callable[[], gym.Env]:
    """Return a no-argument factory that creates a fresh ``MarketMakerEnv``.

    The environment is wrapped with SB3's ``Monitor`` so training/evaluation
    logs episode statistics without warnings.
    """

    def _init() -> gym.Env:
        return Monitor(MarketMakerEnv(seed=seed))

    return _init


def train_ppo(
    env: gym.Env,
    total_timesteps: int = 10_000,
    verbose: int = 0,
    **kwargs,
) -> PPO:
    """Train a PPO agent on ``MarketMakerEnv``.

    Parameters
    ----------
    env:
        A Gymnasium environment (typically ``MarketMakerEnv``).
    total_timesteps:
        Number of timesteps to train for.
    verbose:
        SB3 verbosity level.
    kwargs:
        Extra keyword arguments forwarded to ``PPO``.

    Returns
    -------
    A trained ``PPO`` model.
    """
    device = kwargs.pop("device", "cpu")
    model = PPO("MlpPolicy", env, verbose=verbose, device=device, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def train_sac(
    env: gym.Env,
    total_timesteps: int = 10_000,
    verbose: int = 0,
    **kwargs,
) -> SAC:
    """Train an SAC agent on ``MarketMakerEnv``.

    Parameters
    ----------
    env:
        A Gymnasium environment (typically ``MarketMakerEnv``).
    total_timesteps:
        Number of timesteps to train for.
    verbose:
        SB3 verbosity level.
    kwargs:
        Extra keyword arguments forwarded to ``SAC``.

    Returns
    -------
    A trained ``SAC`` model.
    """
    device = kwargs.pop("device", "cpu")
    model = SAC("MlpPolicy", env, verbose=verbose, device=device, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_agent(
    model,
    env: gym.Env,
    n_eval_episodes: int = 5,
) -> Tuple[float, float]:
    """Evaluate a trained model and return mean and std episode reward."""
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    return float(mean_reward), float(std_reward)


def save_model(model, path: Path) -> None:
    """Save a Stable-Baselines3 model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
