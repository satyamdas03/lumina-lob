"""Calibration utilities for fitting agent parameters to real market data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CalibratedParams:
    """Parameters estimated from real market data for simulation agents.

    Attributes
    ----------
    arrival_rate:
        Expected number of events per ``time_unit`` (e.g., per second).
    size_dist_method:
        Either ``"lognormal"`` or ``"empirical"``.
    size_lognorm_mu:
        Mean of log-sizes when ``size_dist_method == "lognormal"``.
    size_lognorm_sigma:
        Standard deviation of log-sizes when ``size_dist_method == "lognormal"``.
    size_hist:
        Normalized empirical size histogram when ``size_dist_method == "empirical"``.
    mean_spread:
        Average bid-ask spread in price units, if quote data was supplied.
    tick_size:
        Minimum price increment inferred from the spread / price grid.
    time_unit:
        Unit used for ``arrival_rate`` (``"S"`` = seconds, ``"min"`` = minutes, etc.).
    """

    arrival_rate: float
    size_dist_method: str
    size_lognorm_mu: Optional[float] = None
    size_lognorm_sigma: Optional[float] = None
    size_hist: Optional[pd.Series] = None
    mean_spread: Optional[float] = None
    tick_size: float = 0.01
    time_unit: str = "S"


def calibrate(
    trades: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    size_method: str = "lognormal",
    time_unit: str = "S",
    bid_col: str = "bid_px_00",
    ask_col: str = "ask_px_00",
) -> CalibratedParams:
    """Estimate agent parameters from a trades DataFrame and optional quotes.

    Parameters
    ----------
    trades:
        DataFrame with at least ``timestamp`` and ``size`` columns.
    quotes:
        Optional DataFrame with at least ``timestamp``, ``bid_col`` and ``ask_col``.
    size_method:
        ``"lognormal"`` fits ``log(size)``; ``"empirical"`` returns a normalized
        histogram.
    time_unit:
        Unit for ``arrival_rate``.  Pandas frequency string, e.g. ``"S"``,
        ``"min"``, ``"H"``.
    bid_col, ask_col:
        Column names for best bid and ask in ``quotes``.

    Returns
    -------
    CalibratedParams
    """
    if "timestamp" not in trades.columns or "size" not in trades.columns:
        raise ValueError("trades DataFrame must contain 'timestamp' and 'size' columns")
    if trades.empty:
        raise ValueError("trades DataFrame is empty")

    arrival_rate = _estimate_arrival_rate(trades["timestamp"], time_unit)
    size_result = _fit_size_distribution(trades["size"], size_method)
    spread = None
    tick_size = 0.01
    if quotes is not None:
        _validate_quote_columns(quotes, bid_col, ask_col)
        tick_size = _estimate_tick_size(quotes, bid_col, ask_col)
        spread = _estimate_mean_spread(quotes, bid_col, ask_col)

    return CalibratedParams(
        arrival_rate=arrival_rate,
        size_dist_method=size_method,
        mean_spread=spread,
        tick_size=tick_size,
        time_unit=time_unit,
        **size_result,
    )


def _estimate_arrival_rate(timestamps: pd.Series, time_unit: str) -> float:
    """Poisson rate = 1 / mean inter-arrival time in the requested unit."""
    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    if len(ts) <= 1:
        return 0.0

    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty or deltas.mean() <= 0:
        return 0.0

    unit_seconds = _time_unit_seconds(time_unit)
    return float(unit_seconds / deltas.mean())


def _time_unit_seconds(unit: str) -> float:
    """Convert a pandas frequency alias to seconds."""
    mapping = {
        "S": 1.0,
        "s": 1.0,
        "min": 60.0,
        "T": 60.0,
        "H": 3600.0,
        "D": 86400.0,
    }
    if unit in mapping:
        return mapping[unit]
    raise ValueError(f"unsupported time_unit: {unit}")


def _fit_size_distribution(sizes: pd.Series, method: str) -> dict:
    """Return size distribution parameters for the requested method."""
    sizes = sizes[sizes > 0]
    if sizes.empty:
        raise ValueError("no positive sizes to fit")

    if method == "lognormal":
        log_sizes = np.log(sizes)
        return {
            "size_lognorm_mu": float(log_sizes.mean()),
            "size_lognorm_sigma": float(log_sizes.std(ddof=0)),
        }
    if method == "empirical":
        hist = sizes.value_counts(normalize=True).sort_index()
        return {"size_hist": hist}
    raise ValueError(f"unsupported size_method: {method}")


def _validate_quote_columns(quotes: pd.DataFrame, bid_col: str, ask_col: str) -> None:
    if bid_col not in quotes.columns or ask_col not in quotes.columns:
        raise ValueError(f"quotes DataFrame must contain '{bid_col}' and '{ask_col}' columns")


def _estimate_mean_spread(quotes: pd.DataFrame, bid_col: str, ask_col: str) -> float:
    spreads = quotes[ask_col] - quotes[bid_col]
    spreads = spreads[spreads > 0]
    if spreads.empty:
        return 0.0
    return float(spreads.mean())


def _estimate_tick_size(quotes: pd.DataFrame, bid_col: str, ask_col: str) -> float:
    """Infer minimum price increment from unique positive price differences."""
    prices = pd.concat([quotes[bid_col], quotes[ask_col]], ignore_index=True).dropna()
    unique_sorted = np.unique(prices)
    if len(unique_sorted) < 2:
        return 0.01
    diffs = np.diff(unique_sorted)
    positive_diffs = diffs[diffs > 1e-9]
    if positive_diffs.size == 0:
        return 0.01
    return float(positive_diffs.min())
