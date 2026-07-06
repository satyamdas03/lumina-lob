# Lumina LOB — Complete Build Plan

## Final version vision

A public, well-documented open-source limit order book simulator that:

1. Runs a price-time priority matching engine in pure Python and a C++17 hot path
2. Simulates realistic agents: noise traders, informed traders, market makers
3. Trains RL market-makers via Stable-Baselines3
4. Calibrates to real tick data from Polygon/Databento
5. Visualizes book depth, price evolution, and P&L in real time
6. Ships with reproducible notebooks, unit tests, and a published technical blog post
7. Becomes one of the top 5 Google results for "limit order book simulator python"

Target user: quant-interview candidates, ML researchers, students, and hiring managers at Jane Street / Citadel / Optiver / IMC.

## Success metrics (must be measured)

| Metric | Target | How measured |
|---|---|---|
| Python engine throughput | >100k events/sec | `benchmarks/engine_benchmark.py` |
| C++ engine throughput | >10M events/sec | `benchmarks/cpp_benchmark.py` |
| Unit test coverage | >90% | `pytest --cov` |
| RL market maker P&L | positive net vs baseline | `notebooks/rl_market_maker.ipynb` |
| Real spread reproduction | within 20% of real data | `notebooks/calibration.ipynb` |
| GitHub stars | >200 within 6 months | GitHub API |
| Blog post traffic | >5k views | Substack/LinkedIn analytics |

## Architecture

```
lumina_lob/
├── core/
│   ├── order.py          # Order, Side, OrderType
│   ├── price_level.py    # doubly-linked FIFO queue
│   ├── book.py           # OrderBook
│   ├── matching.py       # MatchingEngine
│   └── event_log.py      # nanosecond event journal
├── agents/
│   ├── base.py           # Agent ABC
│   ├── noise_trader.py   # random arrival Poisson model
│   ├── informed_trader.py # directional signal with impact
│   └── market_maker.py   # inventory-skewed quoting
├── market_model/
│   ├── reference_price.py # Brownian + jump process
│   ├── impact.py         # propagator / Almgren-Chriss
│   └── arrival.py        # calibrated order arrival rates
├── rl/
│   ├── env.py            # Gymnasium market-making env
│   ├── rewards.py        # P&L + inventory penalty
│   └── train.py          # PPO/SAC training script
├── data/
│   ├── polygon.py        # Polygon.io ingest
│   ├── databento.py      # Databento ingest
│   └── calibration.py    # fit arrival + impact params
├── cpp/
│   ├── include/          # C++ headers
│   ├── src/              # C++ implementation
│   ├── bindings.cpp      # pybind11
│   └── CMakeLists.txt
├── viz/
│   ├── depth_ladder.py   # matplotlib book plot
│   └── realtime.py       # WebGL / Plotly streaming
├── benchmarks/
│   └── engine_benchmark.py
├── tests/
│   └── test_*.py
└── notebooks/
    ├── 01_matching_engine_demo.ipynb
    ├── 02_agents_and_impact.ipynb
    ├── 03_rl_market_maker.ipynb
    └── 04_calibration.ipynb
```

## Checkpoint list

Each checkpoint is small enough for one focused session. After each, we stop and report.

### Phase 0: Core engine hardening

- [x] **CP0.1** Move existing modules into `lumina_lob/core/` and fix imports
- [x] **CP0.2** Add order modification (reduce quantity)
- [x] **CP0.3** Add IOC and FOK order types
- [x] **CP0.4** Add event log: every add/cancel/match recorded with timestamp
- [x] **CP0.5** Add full depth snapshot and `to_pandas()` helpers
- [x] **CP0.6** Reach 90% test coverage on core engine

### Phase 1: Market model + agents

- [x] **CP1.1** Implement reference-price process (Brownian + jump)
- [x] **CP1.2** Implement noise trader with Poisson arrivals and random sizes
- [x] **CP1.3** Implement informed trader with temporary/permanent impact
- [x] **CP1.4** Implement basic market-maker (symmetric quotes, inventory limit)
- [x] **CP1.5** Implement skewed market-maker (inventory-sensitive quoting)
- [x] **CP1.6** Add propagator-style market impact model
- [x] **CP1.7** Build `Simulation` orchestrator that runs agents + engine together
- [x] **CP1.8** Notebook: agents + impact demo

### Phase 2: Data + calibration

- [x] **CP2.1** Polygon.io EOD + tick data downloader
- [x] **CP2.2** Databento downloader with $125 free credits
- [x] **CP2.3** Calibrate arrival-rate distributions from real data
- [x] **CP2.4** Calibrate impact parameters from real data
- [x] **CP2.5** Replay real tick data through engine and validate spread distribution
- [x] **CP2.6** Notebook: calibration demo

### Phase 3: RL market maker

- [x] **CP3.1** Define Gymnasium observation space (book state + inventory + P&L)
- [x] **CP3.2** Define action space (bid/ask quote offsets + sizes)
- [x] **CP3.3** Implement reward function (P&L - inventory penalty - spread cost)
- [x] **CP3.4** Train PPO baseline
- [x] **CP3.5** Train SAC comparison
- [x] **CP3.6** Evaluate RL vs heuristic market-makers
- [x] **CP3.7** Notebook: RL market maker training + evaluation

### Phase 4: C++ performance layer

- [x] **CP4.1** Port OrderBook + MatchingEngine to C++17
- [x] **CP4.2** Add pybind11 bindings
- [x] **CP4.3** Add build script (`setup.py` / `CMakeLists.txt`)
- [x] **CP4.4** Add throughput benchmark (Python vs C++)
- [x] **CP4.5** Notebook: benchmark report

### Phase 5: Visualization

- [x] **CP5.1** Matplotlib depth-ladder plot
- [x] **CP5.2** Time-series plot of mid price, spread, and trades
- [x] **CP5.3** Real-time streaming visualizer for simulation
- [x] **CP5.4** GIF/MP4 export of simulation replay

### Phase 6: Packaging + publication

- [x] **CP6.1** Package for PyPI (`pip install lumina-lob`)
- [x] **CP6.2** GitHub Actions CI (test matrix Python 3.11–3.13)
- [ ] **CP6.3** Write full documentation site (MkDocs)
- [ ] **CP6.4** Write technical blog post "Build a Limit Order Book Simulator from Scratch"
- [ ] **CP6.5** LinkedIn/X launch
- [ ] **CP6.6** Pin repo on GitHub profile

## Execution mode

We work checkpoint-by-checkpoint. After each checkpoint:
1. Run tests and ensure green
2. Commit and push
3. Report: what was done, key files changed, test result, next checkpoint
4. Wait for user to say "continue"

No batching multiple checkpoints without explicit user approval.

## Notes

- Keep all code typed, tested, and documented.
- Every new public function gets a docstring.
- Every new module gets at least one unit test.
- Prefer simple, correct code over clever code.
- Blog post will be the primary traffic driver; code quality must back it up.
