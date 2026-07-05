"""Gymnasium environment for training an RL market maker."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from lumina_lob.agents.informed_trader import InformedTrader
from lumina_lob.agents.noise_trader import NoiseTrader
from lumina_lob.core.book import OrderBook
from lumina_lob.core.matching import MatchingEngine
from lumina_lob.core.order import Order, Side
from lumina_lob.market_model.reference_price import ReferencePriceProcess
from lumina_lob.simulation import Simulation


class MarketMakerEnv(gym.Env):
    """A Gymnasium environment where an agent acts as a single market maker.

    The observation combines the current limit-order-book state, the agent's
    inventory, and its mark-to-market P&L.  The action is a continuous vector
    that controls bid/ask quote offsets and sizes.

    Parameters
    ----------
    max_steps:
        Maximum number of steps per episode.
    warmup_steps:
        Number of steps to run before the agent starts observing, so the book
        has initial liquidity.
    tick_size:
        Price tick size used for rounding and normalisation.
    max_quote_offset_ticks:
        Maximum number of ticks away from the mid price the agent may quote.
    min_quote_size:
        Minimum quantity per quote.
    max_quote_size:
        Maximum quantity per quote.
    inventory_penalty:
        Coefficient for the quadratic inventory penalty applied each step.
    seed:
        Optional global RNG seed.
    """

    def __init__(
        self,
        max_steps: int = 200,
        warmup_steps: int = 10,
        tick_size: float = 0.01,
        max_quote_offset_ticks: int = 5,
        min_quote_size: int = 10,
        max_quote_size: int = 100,
        inventory_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.tick_size = float(tick_size)
        self.max_quote_offset_ticks = max(0, int(max_quote_offset_ticks))
        self.min_quote_size = max(1, int(min_quote_size))
        self.max_quote_size = max(self.min_quote_size, int(max_quote_size))
        self.inventory_penalty = float(inventory_penalty)
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
        # Continuous action: [bid_offset, ask_offset, bid_size, ask_size] in [-1, 1].
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.simulation: Optional[Simulation] = None
        self._inventory = 0.0
        self._cash = 0.0
        self._avg_fill_price = 0.0
        self._current_step = 0
        self._reference_price = 100.0
        self._previous_total_pnl = 0.0
        self._pending_action: Optional[np.ndarray] = None
        self._proxy = self._AgentProxy(self)

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
            agents=[noise, informed, self._proxy],
            seed=effective_seed,
        )

        self._inventory = 0.0
        self._cash = 0.0
        self._avg_fill_price = 0.0
        self._current_step = 0
        self._pending_action = None

        for _ in range(self.warmup_steps):
            self.simulation.step()
            self._current_step += 1

        self._previous_total_pnl = self._total_pnl()

        observation = self._get_observation()
        info: Dict = {"reference_price": self._reference_price}
        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance the market by one step and return the new observation."""
        if self.simulation is None:
            raise RuntimeError("Environment must be reset before calling step()")

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != self.action_space.shape:
            raise ValueError(
                f"action shape {action_arr.shape} does not match {self.action_space.shape}"
            )
        self._pending_action = action_arr

        metrics = self.simulation.step()
        self._current_step += 1
        self._reference_price = float(metrics["reference_price"])

        observation = self._get_observation()
        reward = self._compute_reward()
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

    def _update_inventory(
        self, side, qty: int, price: Optional[float] = None
    ) -> None:
        """Record a fill against the agent's inventory."""
        qty = float(qty)
        fill_price = self._fill_price(price)
        if side == Side.BID:  # BID: agent bought
            cost = qty * fill_price
            total_value = self._inventory * self._avg_fill_price + cost
            self._inventory += qty
            self._cash -= cost
            if self._inventory > 0:
                self._avg_fill_price = total_value / self._inventory
        else:  # ASK: agent sold
            proceeds = qty * fill_price
            self._inventory -= qty
            self._cash += proceeds
            if self._inventory == 0:
                self._avg_fill_price = 0.0

    def _fill_price(self, price: Optional[float]) -> float:
        """Resolve a fill price, falling back to mid or reference price."""
        if price is not None:
            return float(price)
        if self.simulation is not None:
            mid = self.simulation.book.mid_price
            if mid is not None:
                return float(mid)
        return float(self._reference_price)

    def _total_pnl(self) -> float:
        """Mark-to-market value of cash plus inventory at the current mid price."""
        mid = self._reference_price
        if self.simulation is not None:
            mid = self.simulation.book.mid_price if self.simulation.book.mid_price is not None else mid
        return float(self._cash + self._inventory * mid)

    def _compute_reward(self) -> float:
        """Return the change in mark-to-market P&L minus inventory penalty."""
        total_pnl = self._total_pnl()
        pnl_delta = total_pnl - self._previous_total_pnl
        self._previous_total_pnl = total_pnl
        penalty = self.inventory_penalty * (self._inventory ** 2)
        return float(pnl_delta - penalty)

    def _action_to_quotes(
        self, action: np.ndarray, reference_price: float
    ) -> List[Order]:
        """Translate a continuous action into bid/ask limit orders."""
        if self.simulation is None:
            return []

        book = self.simulation.book
        mid = book.mid_price if book.mid_price is not None else reference_price

        bid_offset = int(round(((float(action[0]) + 1.0) / 2.0) * self.max_quote_offset_ticks))
        ask_offset = int(round(((float(action[1]) + 1.0) / 2.0) * self.max_quote_offset_ticks))

        bid_size = int(round(((float(action[2]) + 1.0) / 2.0) * (self.max_quote_size - self.min_quote_size) + self.min_quote_size))
        ask_size = int(round(((float(action[3]) + 1.0) / 2.0) * (self.max_quote_size - self.min_quote_size) + self.min_quote_size))

        bid_tick = max(1, round((mid - bid_offset * self.tick_size) / self.tick_size))
        ask_tick = max(bid_tick + 1, round((mid + ask_offset * self.tick_size) / self.tick_size))
        bid_price = bid_tick * self.tick_size
        ask_price = ask_tick * self.tick_size

        orders: List[Order] = []
        if bid_size > 0:
            order = Order(
                order_id=0,
                side=Side.BID,
                price=bid_price,
                qty=bid_size,
            )
            order._agent_quote = True  # type: ignore[attr-defined]
            orders.append(order)
        if ask_size > 0:
            order = Order(
                order_id=0,
                side=Side.ASK,
                price=ask_price,
                qty=ask_size,
            )
            order._agent_quote = True  # type: ignore[attr-defined]
            orders.append(order)
        return orders

    def _cancel_agent_quotes(self) -> None:
        """Cancel any resting orders placed by the RL agent."""
        if self.simulation is None:
            return
        book = self.simulation.book
        for order_id in list(book.orders.keys()):
            if getattr(book.orders[order_id], "_agent_quote", False):
                book.cancel(order_id)

    class _AgentProxy:
        """Thin agent wrapper that turns the env action into limit orders."""

        def __init__(self, env: "MarketMakerEnv") -> None:
            self.env = env

        def act(self, reference_price: float, book: OrderBook) -> List[Order]:
            """Cancel old quotes and submit new ones from the pending action."""
            self.env._cancel_agent_quotes()
            action = self.env._pending_action
            if action is None:
                return []
            return self.env._action_to_quotes(action, reference_price)

        def on_fill(self, side: Side, qty: int) -> None:
            """Forward fill notifications to the environment inventory tracker."""
            self.env._update_inventory(side, qty)
