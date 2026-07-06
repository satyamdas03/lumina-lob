# Building a Custom Agent

Agents in Lumina LOB follow a simple protocol. This tutorial shows how to write a custom agent and plug it into a simulation.

## The agent protocol

An agent must subclass `lumina_lob.Agent` and implement `act(reference_price, book)`.

```python
from lumina_lob import Agent, Order, OrderBook, Side

class BuyAndHoldAgent(Agent):
    def __init__(self):
        self.has_traded = False

    def act(self, reference_price: float, book: OrderBook):
        if self.has_traded:
            return []
        self.has_traded = True
        mid = book.mid_price
        if mid is None:
            return []
        return [Order(
            order_id=0,  # the simulation assigns a fresh id automatically
            side=Side.BID,
            price=mid,
            qty=1,
        )]
```

`act` receives:

- `reference_price` — current fundamental reference price.
- `book` — the current `OrderBook`.

The method must return a list of `Order` objects to submit. The `Simulation` assigns a globally unique order id, so any placeholder id works.

## Register the agent

Pass the agent into the `Simulation` constructor like any built-in agent:

```python
from lumina_lob import Simulation

sim = Simulation(
    agents=[BuyAndHoldAgent()],
    seed=123,
)
sim.run(n_steps=50)
```

## Reacting to fills

Agents can optionally implement `on_fill(side, qty)` to update internal state when one of their orders trades.

```python
from lumina_lob import Side

class InventoryTrackingAgent(Agent):
    def __init__(self):
        self.inventory = 0

    def act(self, reference_price: float, book: OrderBook):
        # implement quoting logic...
        return []

    def on_fill(self, side: Side, qty: int):
        if side == Side.BID:
            self.inventory += qty
        else:
            self.inventory -= qty
```

`side` is the `Side` of the filled order and `qty` is the traded quantity.

## Best practices

- Keep `act()` deterministic given `reference_price` and `book` so simulations are reproducible with a fixed seed.
- Do not mutate the book directly — return `Order` objects and let the matching engine process them.
- Implement `on_fill` for any strategy that tracks inventory or P&L.
