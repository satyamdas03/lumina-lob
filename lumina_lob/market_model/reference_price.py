"""Reference price process for simulated markets.

Models the latent fair price as a discrete-time jump-diffusion:

    log(S_{t+1}) = log(S_t) + (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z + J

where Z ~ N(0, 1) and J is a compound Poisson jump with log-normal jump size.
The price is floored at a small positive epsilon to avoid zero/negative quotes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ReferencePriceProcess:
    """Discrete-time jump-diffusion reference price simulator.

    Parameters
    ----------
    initial_price:
        Starting reference price. Must be positive.
    drift:
        Instantaneous drift (per unit time). Default 0.0.
    volatility:
        Instantaneous volatility (per unit time). Must be non-negative. Default 0.2.
    dt:
        Time step size. Must be positive. Default 1.0.
    jump_intensity:
        Expected number of jumps per unit time. Must be non-negative. Default 0.0.
    jump_mean:
        Mean of the log jump size. Default 0.0.
    jump_std:
        Standard deviation of the log jump size. Must be non-negative. Default 0.05.
    seed:
        Optional RNG seed for reproducibility.
    min_price:
        Floor for the price path. Default 0.0001.
    """

    initial_price: float = 100.0
    drift: float = 0.0
    volatility: float = 0.2
    dt: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.05
    seed: Optional[int] = None
    min_price: float = 0.0001

    _rng: np.random.Generator = field(init=False, repr=False)
    _path: List[float] = field(init=False, repr=False)
    _steps: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.jump_intensity < 0:
            raise ValueError("jump_intensity must be non-negative")
        if self.jump_std < 0:
            raise ValueError("jump_std must be non-negative")
        if self.min_price <= 0:
            raise ValueError("min_price must be positive")

        self._rng = np.random.default_rng(self.seed)
        self._path = [float(self.initial_price)]
        self._steps = 0

    @property
    def price(self) -> float:
        """Current reference price."""
        return self._path[-1]

    @property
    def path(self) -> List[float]:
        """Full price history including the initial price."""
        return list(self._path)

    def _draw_jump(self, n: int) -> np.ndarray:
        """Draw compound Poisson jumps for n steps."""
        if self.jump_intensity == 0.0:
            return np.zeros(n, dtype=float)
        arrivals = self._rng.poisson(lam=self.jump_intensity * self.dt, size=n)
        if self.jump_std == 0.0:
            return self.jump_mean * arrivals
        log_jumps = self._rng.normal(
            loc=self.jump_mean * arrivals,
            scale=self.jump_std * np.sqrt(arrivals),
            size=n,
        )
        return log_jumps

    def step(self) -> float:
        """Advance one time step and return the new price."""
        prev = self._path[-1]
        diffusion = (self.drift - 0.5 * self.volatility**2) * self.dt
        diffusion += self.volatility * np.sqrt(self.dt) * self._rng.standard_normal()
        jump = self._draw_jump(1)[0]
        new_price = max(prev * np.exp(diffusion + jump), self.min_price)
        self._path.append(float(new_price))
        self._steps += 1
        return new_price

    def simulate(self, n_steps: int) -> List[float]:
        """Simulate n_steps ahead and return the full path."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        prev = self._path[-1]
        diffusion_drift = (self.drift - 0.5 * self.volatility**2) * self.dt
        shocks = self._rng.standard_normal(n_steps)
        jumps = self._draw_jump(n_steps)
        log_returns = diffusion_drift + self.volatility * np.sqrt(self.dt) * shocks + jumps
        cum_log_returns = np.cumsum(log_returns)
        prices = prev * np.exp(cum_log_returns)
        prices = np.maximum(prices, self.min_price)
        self._path.extend(float(p) for p in prices)
        self._steps += n_steps
        return self.path

    def reset(self, price: Optional[float] = None) -> None:
        """Reset the process to the initial price or a new price."""
        start = self.initial_price if price is None else price
        if start <= 0:
            raise ValueError("reset price must be positive")
        self._path = [float(start)]
        self._steps = 0
