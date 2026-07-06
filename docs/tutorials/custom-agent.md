# Building a Custom Agent

Agents in Lumina LOB follow a simple protocol. This tutorial shows how to write a custom agent and plug it into a simulation.

## The agent protocol

An agent must subclass `lumina_lob.Agent` and implement `step(state)`.

```python
from lumina_lob import Agent, Order, Side

class BuyAndHoldAgent(Agent):
    def __init__(self):
        self.has_traded = False

    def step(self, state):
        if self.has_traded:
            return []
        self.has_traded = True
        mid = state.mid_price
        return [Order(
            order_id=state.next_order_id,
            side=Side.BID,
            price=mid,
            qty=1,
        )]
```

The `state` object exposes fields such as:

- `book` — the current `OrderBook`.
- `mid_price` — current mid price, or `None` if one side is empty.
- `spread` — current bid–ask spread.
- `reference_price` — current fundamental reference price.
- `step` — integer step counter.
- `next_order_id` — a fresh order id from the simulation.

## Register the agent

Pass the agent into the `Simulation` constructor like any built-in agent:

```python
from lumina_lob import Simulation

sim = Simulation(
    agents=[BuyAndHoldAgent()],
    steps=50,
    seed=123,
)
sim.run()
```

## Reacting to fills

Agents can optionally implement `on_fill(fill)` to update internal state when one of their orders trades.

```python
from lumina_lob import Fill

class InventoryTrackingAgent(Agent):
    def __init__(self):
        self.inventory = 0

    def step(self, state):
        # implement quoting logic...
        return []

    def on_fill(self, fill: Fill):
        if fill.side == Side.BID:
            self.inventory += fill.qty
        else:
            self.inventory -= fill.qty
```

`Fill` contains the traded side, price, quantity, and the local order id that produced the fill.

## Best practices

- Keep `step()` deterministic given `state` so simulations are reproducible with a fixed seed.
- Do not mutate the book directly — return `Order` objects and let the matching engine process them.
- Use `state.next_order_id` to avoid id collisions.
- Implement `on_fill` for any strategy that tracks inventory or P&L.
