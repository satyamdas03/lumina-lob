"""Tests for reference price process."""
from __future__ import annotations

import numpy as np
import pytest

from lumina_lob.market_model import ReferencePriceProcess


def test_initial_price_positive():
    with pytest.raises(ValueError, match="initial_price must be positive"):
        ReferencePriceProcess(initial_price=0.0)


def test_volatility_non_negative():
    with pytest.raises(ValueError, match="volatility must be non-negative"):
        ReferencePriceProcess(volatility=-0.1)


def test_dt_positive():
    with pytest.raises(ValueError, match="dt must be positive"):
        ReferencePriceProcess(dt=0.0)


def test_jump_intensity_non_negative():
    with pytest.raises(ValueError, match="jump_intensity must be non-negative"):
        ReferencePriceProcess(jump_intensity=-1.0)


def test_jump_std_non_negative():
    with pytest.raises(ValueError, match="jump_std must be non-negative"):
        ReferencePriceProcess(jump_std=-0.05)


def test_min_price_positive():
    with pytest.raises(ValueError, match="min_price must be positive"):
        ReferencePriceProcess(min_price=0.0)


def test_n_steps_positive():
    proc = ReferencePriceProcess()
    with pytest.raises(ValueError, match="n_steps must be positive"):
        proc.simulate(0)


def test_reset_price_positive():
    proc = ReferencePriceProcess()
    with pytest.raises(ValueError, match="reset price must be positive"):
        proc.reset(price=-1.0)


def test_deterministic_drift():
    """With zero volatility and zero jumps, price grows at drift."""
    proc = ReferencePriceProcess(
        initial_price=100.0,
        drift=0.10,
        volatility=0.0,
        dt=1.0,
        seed=42,
    )
    path = proc.simulate(10)
    expected = [100.0 * np.exp(0.10 * i) for i in range(11)]
    assert np.allclose(path, expected)


def test_no_negative_price():
    proc = ReferencePriceProcess(
        initial_price=100.0,
        drift=-10.0,
        volatility=5.0,
        dt=1.0,
        seed=12345,
    )
    path = proc.simulate(1000)
    assert all(p > 0 for p in path)
    assert any(p == pytest.approx(0.0001, abs=1e-9) for p in path) or min(path) > 0.0001


def test_simulate_path_length():
    proc = ReferencePriceProcess(initial_price=50.0, seed=7)
    path = proc.simulate(100)
    assert len(path) == 101
    assert proc.path == path


def test_step_advances_one():
    proc = ReferencePriceProcess(initial_price=100.0, seed=7)
    first = proc.step()
    assert len(proc.path) == 2
    assert proc.price == first


def test_jump_occurs_with_high_intensity():
    """With very high jump intensity over many steps we expect at least one jump."""
    proc = ReferencePriceProcess(
        initial_price=100.0,
        volatility=0.0,
        jump_intensity=10.0,
        jump_mean=0.0,
        jump_std=0.5,
        dt=1.0,
        seed=999,
    )
    path = proc.simulate(100)
    # Without jumps the path would be flat at 100. With jumps it should differ.
    assert any(abs(p - 100.0) > 1e-6 for p in path[1:])


def test_reproducible_with_seed():
    p1 = ReferencePriceProcess(seed=42)
    p2 = ReferencePriceProcess(seed=42)
    assert p1.simulate(50) == p2.simulate(50)


def test_reset():
    proc = ReferencePriceProcess(initial_price=100.0, seed=1)
    proc.simulate(10)
    proc.reset()
    assert proc.price == 100.0
    assert len(proc.path) == 1
    proc.reset(price=50.0)
    assert proc.price == 50.0


def test_draw_jump_zero_intensity():
    proc = ReferencePriceProcess(jump_intensity=0.0)
    jumps = proc._draw_jump(100)
    assert np.all(jumps == 0.0)


def test_draw_jump_zero_std_deterministic():
    proc = ReferencePriceProcess(
        jump_intensity=1.0,
        jump_mean=0.05,
        jump_std=0.0,
        dt=1.0,
        seed=42,
    )
    jumps = proc._draw_jump(1000)
    # Every jump should be exactly 0.05 times the Poisson count, no randomness beyond arrival count
    assert all(j == 0.05 * count or j == 0.0 for j, count in zip(jumps, np.round(jumps / 0.05)))


def test_compound_jump_sum():
    """Multiple jumps in one step should sum to integer multiple of jump_mean (when std=0)."""
    proc = ReferencePriceProcess(
        jump_intensity=5.0,
        jump_mean=0.02,
        jump_std=0.0,
        dt=1.0,
        seed=11,
    )
    jumps = proc._draw_jump(100)
    counts = np.round(jumps / 0.02).astype(int)
    assert np.allclose(jumps, counts * 0.02)
