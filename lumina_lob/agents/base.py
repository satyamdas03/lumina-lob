"""Abstract base class for market agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from lumina_lob.core import Order, OrderBook


class Agent(ABC):
    """Base class for all market agents.

    An agent observes the current reference price and order book state, then
    emits zero or more orders to be processed by the matching engine.
    """

    @abstractmethod
    def act(self, reference_price: float, book: OrderBook) -> List[Order]:
        """Return a list of orders to submit at this simulation step."""
        ...
