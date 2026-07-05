"""Tests for tick-data replay and spread-distribution validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumina_lob.data.replay import ReplayEngine, validate_spread_distribution


def _make_quote_events() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01 09:30:00", periods=5, freq="1s")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_type": ["quote"] * 5,
            "bid_px": [100.00, 100.01, 100.02, 100.03, 100.04],
            "ask_px": [100.05, 100.06, 100.07, 100.05, 100.06],
            "side": [None] * 5,
            "size": [None] * 5,
        }
    )


def test_empty_events_raises():
    engine = ReplayEngine()
    with pytest.raises(ValueError, match="events DataFrame is empty"):
        engine.replay(pd.DataFrame(columns=["timestamp", "event_type"]))


def test_missing_columns_raises():
    engine = ReplayEngine()
    events = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "event_type": ["quote"] * 3,
        }
    )
    with pytest.raises(ValueError, match="events DataFrame must contain"):
        engine.replay(events)


def test_unknown_event_type_raises():
    engine = ReplayEngine()
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1s"),
            "event_type": ["quote", "auction"],
            "bid_px": [100.0, 100.0],
            "ask_px": [100.1, 100.1],
            "side": [None, None],
            "size": [None, None],
        }
    )
    with pytest.raises(ValueError, match="unsupported event_type"):
        engine.replay(events)


def test_quote_replay_records_spread_distribution():
    quotes = _make_quote_events()
    engine = ReplayEngine()
    simulated = engine.replay(quotes)

    real_spreads = quotes["ask_px"] - quotes["bid_px"]
    assert len(simulated) == len(quotes)
    assert simulated["spread"].isna().sum() == 0
    assert np.allclose(simulated["spread"].values, real_spreads.values, rtol=1e-9)
    score = validate_spread_distribution(real_spreads, simulated["spread"])
    assert score == pytest.approx(1.0, abs=1e-9)


def test_trade_consumes_liquidity():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:30:00", periods=3, freq="1s"),
            "event_type": ["quote", "trade", "quote"],
            "bid_px": [100.00, np.nan, 100.01],
            "ask_px": [100.05, np.nan, 100.04],
            "side": [None, "buy", None],
            "size": [None, 50, None],
        }
    )
    engine = ReplayEngine()
    simulated = engine.replay(events)

    assert engine._trade_count == 1
    assert engine._trade_volume == 50
    # After the first quote, spread is 0.05.  After the buy market order the ask
    # side is partially consumed; because the quote size is 100, best ask remains.
    assert simulated["spread"].iloc[0] == pytest.approx(0.05, rel=1e-9)
    assert simulated["spread"].iloc[2] == pytest.approx(0.03, rel=1e-9)


def test_trade_with_unknown_side_is_skipped():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1s"),
            "event_type": ["quote", "trade"],
            "bid_px": [100.00, np.nan],
            "ask_px": [100.05, np.nan],
            "side": [None, "unknown"],
            "size": [None, 10],
        }
    )
    engine = ReplayEngine()
    engine.replay(events)
    assert engine._trade_count == 0
    assert engine._trade_volume == 0


def test_trade_with_non_positive_size_is_skipped():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "event_type": ["quote", "trade", "trade"],
            "bid_px": [100.00, np.nan, np.nan],
            "ask_px": [100.05, np.nan, np.nan],
            "side": [None, "buy", "sell"],
            "size": [None, 0, -5],
        }
    )
    engine = ReplayEngine()
    engine.replay(events)
    assert engine._trade_count == 0
    assert engine._trade_volume == 0


def test_trade_with_nan_size_is_skipped():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1s"),
            "event_type": ["quote", "trade"],
            "bid_px": [100.00, np.nan],
            "ask_px": [100.05, np.nan],
            "side": [None, "buy"],
            "size": [None, np.nan],
        }
    )
    engine = ReplayEngine()
    engine.replay(events)
    assert engine._trade_count == 0


def test_trade_with_none_or_zero_side_is_skipped():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "event_type": ["quote", "trade", "trade"],
            "bid_px": [100.00, np.nan, np.nan],
            "ask_px": [100.05, np.nan, np.nan],
            "side": [None, None, 0],
            "size": [None, 10, 10],
        }
    )
    engine = ReplayEngine()
    engine.replay(events)
    assert engine._trade_count == 0


def test_validate_spread_distribution_perfect_match():
    s = pd.Series([0.01, 0.02, 0.02, 0.03, 0.03, 0.03])
    assert validate_spread_distribution(s, s) == pytest.approx(1.0, abs=1e-9)


def test_validate_spread_distribution_partial_overlap():
    real = pd.Series([0.01] * 10 + [0.05] * 10)
    sim = pd.Series([0.01] * 10 + [0.09] * 10)
    score = validate_spread_distribution(real, sim, bins=5)
    assert 0.0 < score < 1.0


def test_validate_spread_distribution_empty_returns_zero():
    real = pd.Series([0.01, 0.02, 0.03])
    assert validate_spread_distribution(real, pd.Series([], dtype=float)) == 0.0
    assert validate_spread_distribution(pd.Series([], dtype=float), real) == 0.0


def test_validate_spread_distribution_constant_equal():
    s = pd.Series([0.02, 0.02, 0.02])
    assert validate_spread_distribution(s, s) == pytest.approx(1.0, abs=1e-9)


def test_default_quote_size_override():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1s"),
            "event_type": ["quote", "trade"],
            "bid_px": [100.00, np.nan],
            "ask_px": [100.05, np.nan],
            "side": [None, "buy"],
            "size": [None, 200],
        }
    )
    engine = ReplayEngine(default_quote_size=150)
    engine.replay(events)
    # The 150-lot ask is fully consumed by the 200-lot buy, leaving a missing ask.
    assert engine.book.best_ask is None


def test_numeric_and_string_side_values():
    """Side inference should handle both numeric signs and string labels."""
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1s"),
            "event_type": ["quote", "trade", "trade", "trade"],
            "bid_px": [100.00, np.nan, np.nan, np.nan],
            "ask_px": [100.05, np.nan, np.nan, np.nan],
            "side": [None, 1, -1, "sell"],
            "size": [None, 10, 10, 10],
        }
    )
    engine = ReplayEngine()
    engine.replay(events)
    assert engine._trade_count == 3
    assert engine._trade_volume == 30


def test_sign_from_value_maps_none():
    from lumina_lob.data.replay import _sign_from_value

    assert _sign_from_value(None) == 0
