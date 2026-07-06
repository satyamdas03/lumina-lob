# Quickstart

This tutorial walks through the core objects in Lumina LOB: orders, the order book, the matching engine, and a simple simulation.

## Create an order book

```python
from lumina_lob import Order, OrderBook, MatchingEngine, Side

book = OrderBook()
engine = MatchingEngine(book)
```

`OrderBook` stores all resting orders. `MatchingEngine` processes incoming orders and records trades.

## Submit orders

```python
engine.process(Order(1, Side.BID, 100, 10))
engine.process(Order(2, Side.ASK, 100, 4))
engine.process(Order(3, Side.BID, 101, 6))

print(book.snapshot())
print(book.trades)
```

The first order is a **bid** for 10 shares at price 100. The second order is an **ask** for 4 shares at price 100 — it crosses the existing bid and produces a trade. The third order is a bid at 101 that does not cross and rests on the book.

## Supported order types

Lumina supports:

- `LIMIT` — rest on the book at a given price.
- `MARKET` — cross immediately against the opposite side.
- `IOC` (Immediate-or-Cancel) — execute immediately, cancel the remainder.
- `FOK` (Fill-or-Kill) — execute the entire quantity or cancel entirely.
- `CANCEL` — remove a resting order.
- `MODIFY` — reduce the quantity of a resting order.

```python
from lumina_lob import OrderType

# IOC order: execute whatever is available and cancel the rest
engine.process(Order(4, Side.ASK, 101, 100, order_type=OrderType.IOC))
```

## Run a simulation

The `Simulation` class wires agents to the matching engine and records a history DataFrame.

```python
from lumina_lob import Simulation, NoiseTrader

sim = Simulation(
    agents=[NoiseTrader(intensity=5.0, max_qty=10)],
    steps=100,
    seed=42,
)
sim.run()
print(sim.history.head())
```

See the [architecture tutorial](architecture.md) for how agents, impact models, and the engine fit together.
