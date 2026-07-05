"""Tests for calibration utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumina_lob.data.calibration import CalibratedParams, calibrate


def _make_trades(start: str, n: int = 10, freq: str = "1s") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "size": np.random.default_rng(42).integers(1, 11, size=n),
            "price": 100.0 + np.arange(n) * 0.01,
        }
    )


def _make_quotes(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_px_00": 100.0 - np.arange(len(timestamps)) * 0.01,
            "ask_px_00": 100.0 + 0.02 - np.arange(len(timestamps)) * 0.01,
        }
    )


def test_calibrate_with_lognormal_sizes():
    trades = _make_trades("2024-01-01 09:30:00", n=60, freq="1s")
    quotes = _make_quotes(trades["timestamp"])
    params = calibrate(trades, quotes)

    assert isinstance(params, CalibratedParams)
    assert params.size_dist_method == "lognormal"
    assert params.arrival_rate == pytest.approx(1.0, rel=1e-9)
    assert params.size_lognorm_mu is not None
    assert params.size_lognorm_sigma is not None
    assert params.size_lognorm_sigma >= 0
    assert params.mean_spread == pytest.approx(0.02, rel=1e-9)
    assert params.tick_size == pytest.approx(0.01, rel=1e-9)
    assert params.time_unit == "S"


def test_calibrate_empirical_size_distribution():
    trades = _make_trades("2024-01-01 09:30:00", n=100, freq="1s")
    params = calibrate(trades, size_method="empirical")

    assert params.size_dist_method == "empirical"
    assert params.size_hist is not None
    assert abs(params.size_hist.sum() - 1.0) < 1e-9
    assert params.size_lognorm_mu is None
    assert params.size_lognorm_sigma is None


def test_calibrate_without_quotes():
    trades = _make_trades("2024-01-01 09:30:00", n=30, freq="1s")
    params = calibrate(trades)

    assert params.arrival_rate == pytest.approx(1.0, rel=1e-9)
    assert params.mean_spread is None
    assert params.tick_size == 0.01


def test_arrival_rate_per_minute():
    trades = _make_trades("2024-01-01 09:30:00", n=60, freq="1s")
    params = calibrate(trades, time_unit="min")

    assert params.arrival_rate == pytest.approx(60.0, rel=1e-9)
    assert params.time_unit == "min"


def test_single_trade_zero_rate():
    trades = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01 09:30:00")], "size": [10]})
    params = calibrate(trades)

    assert params.arrival_rate == 0.0


def test_empty_trades_raises():
    trades = pd.DataFrame(columns=["timestamp", "size"])
    with pytest.raises(ValueError, match="trades DataFrame is empty"):
        calibrate(trades)


def test_missing_columns_raises():
    trades = pd.DataFrame({"ts": [pd.Timestamp("2024-01-01")], "qty": [10]})
    with pytest.raises(ValueError, match="trades DataFrame must contain"):
        calibrate(trades)


def test_unsupported_time_unit_raises():
    trades = _make_trades("2024-01-01 09:30:00", n=10, freq="1s")
    with pytest.raises(ValueError, match="unsupported time_unit"):
        calibrate(trades, time_unit="fortnight")


def test_unsupported_size_method_raises():
    trades = _make_trades("2024-01-01 09:30:00", n=10, freq="1s")
    with pytest.raises(ValueError, match="unsupported size_method"):
        calibrate(trades, size_method="gamma")


def test_quote_column_validation():
    trades = _make_trades("2024-01-01 09:30:00", n=10, freq="1s")
    quotes = pd.DataFrame(
        {
            "timestamp": trades["timestamp"],
            "bid": 99.0,
            "ask": 101.0,
        }
    )
    with pytest.raises(ValueError, match="quotes DataFrame must contain"):
        calibrate(trades, quotes, bid_col="bid_px_00", ask_col="ask_px_00")


def test_tick_size_inference():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "bid_px_00": [100.00, 100.01, 100.02],
            "ask_px_00": [100.02, 100.03, 100.04],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": quotes["timestamp"],
            "size": [10, 10, 10],
        }
    )
    params = calibrate(trades, quotes)
    assert params.tick_size == pytest.approx(0.01, rel=1e-9)
    assert params.mean_spread == pytest.approx(0.02, rel=1e-9)


def test_size_filtering_ignores_non_positive():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s"),
            "size": [0, -1, 2, 3, 4],
        }
    )
    params = calibrate(trades, size_method="lognormal")
    # only three positive sizes; lognormal still computable
    assert params.size_lognorm_mu is not None
    assert params.size_lognorm_sigma is not None


def test_all_non_positive_sizes_raises():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "size": [0, -1, 0],
        }
    )
    with pytest.raises(ValueError, match="no positive sizes to fit"):
        calibrate(trades)


def test_duplicate_timestamps_zero_rate():
    trades = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 09:30:00")] * 5,
            "size": [10] * 5,
        }
    )
    params = calibrate(trades)
    assert params.arrival_rate == 0.0


def test_all_negative_spreads_zero_mean_spread():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "bid_px_00": [100.05, 100.06, 100.07],
            "ask_px_00": [100.00, 100.00, 100.00],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": quotes["timestamp"],
            "size": [10, 10, 10],
        }
    )
    params = calibrate(trades, quotes)
    assert params.mean_spread == 0.0


def test_uniform_prices_default_tick_size():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
            "bid_px_00": [100.0, 100.0, 100.0],
            "ask_px_00": [100.0, 100.0, 100.0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": quotes["timestamp"],
            "size": [10, 10, 10],
        }
    )
    params = calibrate(trades, quotes)
    assert params.tick_size == 0.01


def test_tiny_price_diffs_default_tick_size():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1s"),
            "bid_px_00": [100.0, 100.0],
            "ask_px_00": [100.0, 100.0 + 1e-12],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": quotes["timestamp"],
            "size": [10, 10],
        }
    )
    params = calibrate(trades, quotes)
    assert params.tick_size == 0.01


def test_time_unit_seconds_validation():
    from lumina_lob.data.calibration import _time_unit_seconds

    with pytest.raises(ValueError, match="unsupported time_unit"):
        _time_unit_seconds("week")
