"""Simulation orchestrator: runs agents and matching engine through time."""
from __future__ import annotations

from typing import Dict, List, Optional

from lumina_lob.agents.base import Agent
from lumina_lob.core import MatchingEngine, OrderBook, Side
from lumina_lob.market_model.reference_price import ReferencePriceProcess


class Simulation:
    """Orchestrate a multi-agent limit order book simulation.

    A ``Simulation`` owns one ``OrderBook``, one ``MatchingEngine``, one
    ``ReferencePriceProcess``, and a list of ``Agent`` instances. At each
    discrete time step the reference price is advanced, every agent is asked
    for orders, and those orders are submitted to the matching engine.  Fill
    notifications are routed back to agents that implement ``on_fill`` (e.g.
    market makers updating inventory).

    Parameters
    ----------
    book:
        ``OrderBook`` to use. A fresh book is created if omitted.
    engine:
        ``MatchingEngine`` to use. Created from ``book`` if omitted.
    reference_price:
        ``ReferencePriceProcess`` to use. Created with default parameters if
        omitted.  If ``seed`` is also provided it is passed to the default
        process.
    agents:
        List of ``Agent`` instances participating in the simulation.
    seed:
        Optional RNG seed for the default ``ReferencePriceProcess``.
    """

    def __init__(
        self,
        book: Optional[OrderBook] = None,
        engine: Optional[MatchingEngine] = None,
        reference_price: Optional[ReferencePriceProcess] = None,
        agents: Optional[List[Agent]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.book = book if book is not None else OrderBook()
        self.engine = engine if engine is not None else MatchingEngine(self.book)
        if reference_price is not None:
            self.reference_price = reference_price
        else:
            self.reference_price = ReferencePriceProcess(seed=seed)
        self.agents: List[Agent] = list(agents) if agents is not None else []
        self.history: List[Dict[str, object]] = []
        self._next_order_id = 1
        self._order_owner: Dict[int, Agent] = {}
        self._order_side: Dict[int, Side] = {}

    def step(self) -> Dict[str, object]:
        """Advance the simulation by one time step.

        Returns
        -------
        A dictionary of metrics for the step:
        ``step``, ``reference_price``, ``best_bid``, ``best_ask``,
        ``mid_price``, ``spread``, ``trade_count``, ``trade_volume``,
        ``book_size``.
        """
        reference_price = self.reference_price.step()
        pre_trade_count = len(self.book.trades)

        for agent in self.agents:
            orders = agent.act(reference_price, self.book)
            for order in orders:
                self._submit(order, agent)

        self._notify_fills(pre_trade_count)

        new_trades = self.book.trades[pre_trade_count:]
        trade_count = len(new_trades)
        trade_volume = sum(qty for _, _, qty in new_trades)
        signed_volume = 0
        for aggressor_id, _, qty in new_trades:
            side = self._order_side.get(aggressor_id)
            if side == Side.BID:
                signed_volume += qty
            elif side == Side.ASK:
                signed_volume -= qty

        record = {
            "step": len(self.history) + 1,
            "reference_price": reference_price,
            "best_bid": self.book.best_bid,
            "best_ask": self.book.best_ask,
            "mid_price": self.book.mid_price,
            "spread": self.book.spread,
            "trade_count": trade_count,
            "trade_volume": trade_volume,
            "signed_volume": signed_volume,
            "book_size": len(self.book),
        }
        self.history.append(record)
        return record

    def run(self, n_steps: int) -> List[Dict[str, object]]:
        """Run ``n_steps`` and return the full history list."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        for _ in range(n_steps):
            self.step()
        return list(self.history)

    def to_dataframe(self):
        """Return the simulation history as a pandas ``DataFrame``."""
        import pandas as pd

        return pd.DataFrame(self.history)

    def _submit(self, order, agent: Agent) -> None:
        """Assign a globally unique order id and route the order to the engine."""
        new_id = self._next_order_id
        self._next_order_id += 1
        order.order_id = new_id
        self._order_owner[new_id] = agent
        self._order_side[new_id] = order.side
        self.engine.process(order)

    def _notify_fills(self, pre_trade_count: int) -> None:
        """Call ``on_fill`` on agents whose orders traded this step.

        Self-trades (where the aggressor and resting order belong to the same
        agent) are skipped so that an agent does not artificially move its own
        inventory.
        """
        for aggressor_id, resting_id, qty in self.book.trades[pre_trade_count:]:
            owner_aggr = self._order_owner.get(aggressor_id)
            owner_rest = self._order_owner.get(resting_id)
            if owner_aggr is owner_rest:
                continue

            side_aggr = self._order_side.get(aggressor_id)
            side_rest = self._order_side.get(resting_id)

            if owner_aggr is not None:
                on_fill = getattr(owner_aggr, "on_fill", None)
                if callable(on_fill):
                    on_fill(side_aggr, qty)

            if owner_rest is not None:
                on_fill = getattr(owner_rest, "on_fill", None)
                if callable(on_fill):
                    on_fill(side_rest, qty)
