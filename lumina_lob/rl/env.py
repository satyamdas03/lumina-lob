"""Gymnasium environment for training an RL market maker."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from lumina_lob.agents.informed_trader import InformedTrader
from lumina_lob.agents.noise_trader import NoiseTrader
from lumina_lob.core.book import OrderBook
from lumina_lob.core.matching import MatchingEngine
from lumina_lob.core.order import Side
from lumina_lob.market_model.reference_price import ReferencePriceProcess
from lumina_lob.simulation import Simulation


class MarketMakerEnv(gym.Env):
    """A Gymnasium environment where an agent acts as a single market maker.

    The observation combines the current limit-order-book state, the agent's
    inventory, and its mark-to-market P&L.  The action space is intentionally a
    placeholder in this checkpoint (CP3.1); the agent's quotes will be wired to
    actions in CP3.2.

    Parameters
    ----------
    max_steps:
        Maximum number of steps per episode.
    warmup_steps:
        Number of steps to run before the agent starts observing, so the book
        has initial liquidity.
    tick_size:
        Price tick size used for rounding and normalisation.
    seed:
        Optional global RNG seed.
    """

    def __init__(
        self,
        max_steps: int = 200,
        warmup_steps: int = 10,
        tick_size: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.tick_size = float(tick_size)
        self._seed = seed

        self.observation_features: List[str] = [
            "best_bid_norm",
            "best_ask_norm",
            "mid_price_norm",
            "spread_norm",
            "bid_depth_norm",
            "ask_depth_norm",
            "inventory_norm",
            "cash_norm",
            "unrealised_pnl_norm",
            "time_fraction",
        ]
        n_features = len(self.observation_features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )
        # Placeholder action space; CP3.2 will replace this with quote controls.
        self.action_space = gym.spaces.Discrete(1)

        self.simulation: Optional[Simulation] = None
        self._inventory = 0.0
        self._cash = 0.0
        self._avg_fill_price = 0.0
        self._current_step = 0
        self._reference_price = 100.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed

        book = OrderBook()
        engine = MatchingEngine(book)
        reference_price = ReferencePriceProcess(seed=effective_seed)
        self._reference_price = float(reference_price.price)

        noise = NoiseTrader(
            arrival_rate=0.5,
            tick_size=self.tick_size,
            seed=effective_seed,
        )
        informed = InformedTrader(
            signal="bullish",
            trade_size=50,
            order_type="market",
            participation_rate=0.1,
            tick_size=self.tick_size,
            seed=effective_seed,
        )

        self.simulation = Simulation(
            book=book,
            engine=engine,
            reference_price=reference_price,
            agents=[noise, informed],
            seed=effective_seed,
        )

        self._inventory = 0.0
        self._cash = 0.0
        self._avg_fill_price = 0.0
        self._current_step = 0

        for _ in range(self.warmup_steps):
            self.simulation.step()
            self._current_step += 1

        observation = self._get_observation()
        info: Dict = {"reference_price": self._reference_price}
        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance the market by one step and return the new observation."""
        if self.simulation is None:
            raise RuntimeError("Environment must be reset before calling step()")

        # Action is currently a placeholder; CP3.2 will translate it into quotes.
        _ = action

        metrics = self.simulation.step()
        self._current_step += 1
        self._reference_price = float(metrics["reference_price"])

        observation = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = self._current_step >= self.max_steps
        info = {
            "reference_price": self._reference_price,
            "inventory": self._inventory,
            "cash": self._cash,
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build the normalised observation vector."""
        if self.simulation is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        book = self.simulation.book
        best_bid = book.best_bid if book.best_bid is not None else self._reference_price
        best_ask = book.best_ask if book.best_ask is not None else self._reference_price
        mid = book.mid_price if book.mid_price is not None else self._reference_price
        spread = best_ask - best_bid

        bid_depth = sum(level.total_qty for level in book.bids.values())
        ask_depth = sum(level.total_qty for level in book.asks.values())

        unrealised_pnl = self._inventory * (mid - self._avg_fill_price) if self._avg_fill_price else 0.0

        max_position = 1_000.0
        norm = max(self._reference_price, 1e-9)
        obs = np.array(
            [
                best_bid / norm,
                best_ask / norm,
                mid / norm,
                spread / norm,
                bid_depth / max_position,
                ask_depth / max_position,
                self._inventory / max_position,
                self._cash / norm,
                unrealised_pnl / norm,
                self._current_step / self.max_steps,
            ],
            dtype=np.float32,
        )
        return obs

    def _update_inventory(self, side, qty: int, price: float) -> None:
        """Record a fill against the agent's inventory."""
        qty = float(qty)
        if side == Side.BID:  # BID: agent bought
            cost = qty * price
            total_value = self._inventory * self._avg_fill_price + cost
            self._inventory += qty
            self._cash -= cost
            if self._inventory > 0:
                self._avg_fill_price = total_value / self._inventory
        else:  # ASK: agent sold
            proceeds = qty * price
            self._inventory -= qty
            self._cash += proceeds
            if self._inventory == 0:
                self._avg_fill_price = 0.0
