# Lumina LOB

A production-grade, educational **Limit Order Book (LOB) simulator** with algorithmic market-making and reinforcement learning support.

> **Mission:** Build the most complete open-source LOB simulator that demonstrates tick-level market microstructure — matching engines, market impact, adverse selection, inventory risk, and RL-based market making.

## Why this matters

Quant firms like Jane Street, Citadel Securities, Optiver, and IMC make markets at the tick level. Most open-source simulators are either too academic or too toy-like. **Lumina LOB** fills the gap with realistic agents, performance benchmarks, and RL training.

## Current status

- ✅ Price-time priority matching engine (Python)
- ✅ Doubly-linked order queues per price level
- ✅ Market orders + limit orders + cancellations
- ✅ Unit tests for matching, partial fills, price-time priority
- 🚧 Agents + market impact model
- 🚧 RL market-maker environment
- 🚧 C++ performance benchmark
- 🚧 Data calibration + visualization

## Quickstart

```bash
git clone https://github.com/satyamdas03/lumina-lob.git
cd lumina-lob
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Example

```python
from lumina_lob import Order, OrderBook, MatchingEngine, Side

book = OrderBook()
engine = MatchingEngine(book)

engine.process(Order(1, Side.BID, 100, 10))
engine.process(Order(2, Side.ASK, 100, 4))
engine.process(Order(3, Side.BID, 101, 6))

print(book.snapshot())
print(book.trades)
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| v0.1 | Core matching engine | ✅ Done |
| v0.2 | Agents + market impact model | 🚧 In progress |
| v0.3 | RL market-maker environment | 🔲 Planned |
| v0.4 | C++ performance benchmark | 🔲 Planned |
| v0.5 | Data calibration + visualization | 🔲 Planned |

## Project spec

See [`PROJECT_SPEC.txt`](PROJECT_SPEC.txt) for the full vision, market gap, build phases, and interview talking points.

## License

MIT © Satyam Das
