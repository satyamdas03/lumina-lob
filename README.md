# Lumina LOB

A production-grade, educational **Limit Order Book (LOB) simulator** with algorithmic market-making and reinforcement learning support.

> **Mission:** Build the most complete open-source LOB simulator that demonstrates tick-level market microstructure — matching engines, market impact, adverse selection, inventory risk, and RL-based market making.

## Why this matters

Quant firms like Jane Street, Citadel Securities, Optiver, and IMC make markets at the tick level. Most open-source simulators are either too academic or too toy-like. **Lumina LOB** fills the gap with realistic agents, performance benchmarks, and RL training.

## What you can do with it

- Run a price-time priority matching engine from scratch
- Simulate noise traders, informed traders, and market makers
- Train an RL market-maker with PPO/SAC
- Benchmark the hot path in C++ vs Python
- Visualize order-book depth and agent P&L in real time
- Calibrate arrival-rate and impact models against real tick data

## Tech stack

- Python 3.11+ · NumPy · pandas · polars
- Custom linked-list order queues + heapq
- Cython / pybind11 + C++17 performance layer
- Stable-Baselines3 for RL
- Matplotlib / Plotly / WebGL for visualization
- Polygon.io / Databento for market data

## Quickstart (coming soon)

```bash
git clone https://github.com/satyamdas03/lumina-lob.git
cd lumina-lob
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| v0.1 | Core matching engine (orders, cancels, fills) | 🚧 In progress |
| v0.2 | Agents + market impact model | 🔲 Planned |
| v0.3 | RL market-maker environment | 🔲 Planned |
| v0.4 | C++ performance benchmark | 🔲 Planned |
| v0.5 | Data calibration + visualization | 🔲 Planned |

## Project spec

See [`PROJECT_SPEC.txt`](PROJECT_SPEC.txt) for the full vision, market gap, build phases, and interview talking points.

## License

MIT © Satyam Das
