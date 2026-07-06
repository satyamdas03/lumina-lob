# Architecture

Lumina LOB is organized into a small set of focused packages. This page explains what each package does and how they connect.

## Package overview

```
lumina_lob/
├── core/          # Order book + matching engine
├── agents/        # Trading agents (noise, informed, market makers)
├── market_model/  # Reference price + market impact models
├── data/          # Polygon.io / Databento downloaders + calibration
├── simulation.py  # Simulation orchestrator
├── rl/            # Gymnasium environment + RL training helpers
└── viz/           # Matplotlib visualizations + animation export
```

## Core engine

The core engine lives in `lumina_lob.core`.

- `Order` represents a single order with side, price, quantity, and type.
- `PriceLevel` holds all orders at the same price, ordered by arrival time.
- `OrderBook` maintains the bid/ask price levels and provides public snapshots.
- `MatchingEngine` routes incoming orders against the resting book using price-time priority.
- `EventLog` records every submission, cancellation, modification, fill, and trade with a nanosecond timestamp.

The matching engine is the only object that should modify a book directly in normal use. All public operations go through `engine.process(order)`.

## Agents

Agents implement a single-step protocol. Each step they receive the current market state and return zero or more orders to submit.

- `NoiseTrader` — random buy/sell pressure calibrated from real data.
- `InformedTrader` — trades in a fixed direction to simulate information arrival.
- `MarketMaker` — basic two-sided quoting.
- `SkewedMarketMaker` — adjusts quotes based on inventory.

## Market model

The `market_model` package separates the *fundamental* price from the *order-book* prices.

- `ReferencePrice` — random-walk fundamental price with optional drift and volatility.
- `PropagatorImpact` / `AlmgrenChrissImpact` — temporary and permanent market impact models.

## Simulation

`Simulation` is the orchestrator. It owns the book, the matching engine, the agents, and the reference-price/impact models. Each step:

1. Advances the reference price.
2. Lets every agent react to the current state.
3. Processes the resulting orders through the matching engine.
4. Updates market impact.
5. Records a row in `history`.

## Data + calibration

`lumina_lob.data` contains downloaders for Polygon.io and Databento, plus calibration helpers that fit arrival-rate distributions, spread distributions, and impact parameters from real tick data.

## RL

`lumina_lob.rl` exposes a `gymnasium` environment where the action is quote offsets/sizes and the reward is the step change in P&L minus an inventory penalty. Training helpers support PPO and SAC via `stable-baselines3`.

## Visualization

`lumina_lob.viz` provides:

- Depth-ladder plots of the current book.
- Time-series plots of mid price, spread, and trades.
- Real-time `SimulationAnimator` for live simulation playback.
- GIF/MP4 export via `save_animation`.
