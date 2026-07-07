# Lumina LOB

[![CI](https://github.com/satyamdas03/lumina-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/satyamdas03/lumina-lob/actions/workflows/ci.yml)

A production-grade, educational **Limit Order Book (LOB) simulator** with algorithmic market-making, reinforcement learning support, and real-time visualization.

> **Mission:** Build the most complete open-source LOB simulator that demonstrates tick-level market microstructure — matching engines, market impact, adverse selection, inventory risk, RL-based market making, and replay against real tick data.

## Why this matters

Quant firms like Jane Street, Citadel Securities, Optiver, and IMC make markets at the tick level. Most open-source simulators are either too academic or too toy-like. **Lumina LOB** fills the gap with realistic agents, performance benchmarks, calibration to real data, and RL training.

## Documentation

Full docs site: **https://satyamdas03.github.io/lumina-lob**

## Blog post

Draft: [`blog/build-a-lob-simulator.md`](blog/build-a-lob-simulator.md) — *"Build a Limit Order Book Simulator from Scratch"*.  
Social launch drafts: [`blog/social-launch.md`](blog/social-launch.md) — LinkedIn post + X/Twitter thread ready for human review/posting.

## Install

```bash
pip install lumina-lob
```

The package is published to PyPI. If `pip install lumina-lob` is not yet live, the wheels/sdist are built and the release tag is ready; PyPI trusted publishing must be configured by the repository owner for the first upload.

### First-time PyPI trusted publishing setup (repository owner only)

1. Log in to [pypi.org](https://pypi.org) as `satyamdas03`.
2. Go to **Account settings → Publishing**.
3. Add a new **pending publisher**:
   - **PyPI project name:** `lumina-lob`
   - **Owner:** `satyamdas03`
   - **Repository name:** `lumina-lob`
   - **Workflow name:** `.github/workflows/build.yml`
   - **Environment name:** `pypi`
4. Save. The next push of a `v*.*.*` tag will automatically publish.
5. The `v0.1.1` tag is already pushed; after configuring the publisher, re-run the failed **Build wheels** workflow run or push a new `v*.*.*` tag.

The GitHub Actions run for `v0.1.1` confirmed that all 9 OS/Python matrix builds pass and produce both wheels and an sdist; only the final publish step fails until this configuration is done.

Optional visualization support:

```bash
pip install lumina-lob[viz]
```

Development install from source:

```bash
git clone https://github.com/satyamdas03/lumina-lob.git
cd lumina-lob
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,viz,docs]"
pytest
```

The C++ accelerated core builds automatically when a compiler and `pybind11` are available; otherwise the package falls back to the pure-Python engine.

## Quickstart

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

For an interactive walkthrough, open the first available notebook: [`notebooks/02_agents_and_impact.ipynb`](notebooks/02_agents_and_impact.ipynb).

## Features

| Feature | Status |
|---|---|
| Price-time priority matching engine (Python) | ✅ Done |
| Optional C++17 hot path via pybind11 | ✅ Done |
| Limit, market, IOC, FOK, cancel, modify orders | ✅ Done |
| Event log with nanosecond timestamps | ✅ Done |
| Noise trader, informed trader, market makers | ✅ Done |
| Propagator / Almgren-Chriss market impact | ✅ Done |
| Simulation orchestrator + pandas history export | ✅ Done |
| Polygon.io + Databento data downloaders | ✅ Done |
| Calibration to real tick data | ✅ Done |
| Gymnasium RL market-maker environment (PPO/SAC) | ✅ Done |
| Matplotlib depth ladder + history + real-time animator | ✅ Done |
| GIF/MP4 replay export | ✅ Done |
| PyPI packaging | ✅ Done |
| GitHub Actions CI | ✅ Done |
| MkDocs documentation site | ✅ Done |
| Technical blog post | ✅ Done (draft) |
| Social launch (LinkedIn/X) | ✅ Done (drafts) |
| GitHub release + PyPI publish | ✅ Ready (tag v0.1.0 pushed; PyPI trusted publishing required) |

## Roadmap

The project is built in six phases:

| Phase | Goal | Status |
|---|---|---|
| Phase 0 | Core engine hardening | ✅ Done |
| Phase 1 | Market model + agents | ✅ Done |
| Phase 2 | Data + calibration | ✅ Done |
| Phase 3 | RL market maker | ✅ Done |
| Phase 4 | C++ performance layer | ✅ Done |
| Phase 5 | Visualization | ✅ Done |
| Phase 6 | Packaging + publication | ✅ Done |

## Project spec

See [`PROJECT_SPEC.txt`](PROJECT_SPEC.txt) for the full vision, market gap, build phases, and interview talking points.

## License

MIT © Satyam Das
