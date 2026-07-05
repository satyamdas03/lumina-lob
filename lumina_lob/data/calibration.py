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
    permanent_impact:
        Permanent price-impact coefficient per unit signed volume, if price data
        was supplied.
    temporary_impact:
        Temporary price-impact coefficient per unit signed volume, if price data
        was supplied.
    impact_decay:
        Decay factor for the propagator-style temporary impact residual.
    """

    arrival_rate: float
    size_dist_method: str
    size_lognorm_mu: Optional[float] = None
    size_lognorm_sigma: Optional[float] = None
    size_hist: Optional[pd.Series] = None
    mean_spread: Optional[float] = None
    tick_size: float = 0.01
    time_unit: str = "S"
    permanent_impact: Optional[float] = None
    temporary_impact: Optional[float] = None
    impact_decay: Optional[float] = None


def calibrate(
    trades: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    size_method: str = "lognormal",
    time_unit: str = "S",
    bid_col: str = "bid_px_00",
    ask_col: str = "ask_px_00",
    price_col: str = "price",
    side_col: str = "side",
) -> CalibratedParams:
    """Estimate agent parameters from a trades DataFrame and optional quotes.

    Parameters
    ----------
    trades:
        DataFrame with at least ``timestamp`` and ``size`` columns. If ``price_col``
        and ``side_col`` are present, propagator-style impact coefficients are also
        estimated.
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
    price_col:
        Column name for trade prices in ``trades``.
    side_col:
        Column name for trade side in ``trades``.

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
    impact_result = {}
    if quotes is not None:
        _validate_quote_columns(quotes, bid_col, ask_col)
        tick_size = _estimate_tick_size(quotes, bid_col, ask_col)
        spread = _estimate_mean_spread(quotes, bid_col, ask_col)
    if price_col in trades.columns and side_col in trades.columns:
        impact_result = _fit_propagator_impact(trades[price_col], _signed_volume(trades, side_col))

    return CalibratedParams(
        arrival_rate=arrival_rate,
        size_dist_method=size_method,
        mean_spread=spread,
        tick_size=tick_size,
        time_unit=time_unit,
        **size_result,
        **impact_result,
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


def _signed_volume(trades: pd.DataFrame, side_col: str) -> np.ndarray:
    """Convert trade sizes to signed volumes using the side column."""
    sizes = np.asarray(trades["size"], dtype=float)
    signs = _sign_from_side(trades[side_col])
    return sizes * signs


def _sign_from_side(series: pd.Series) -> np.ndarray:
    """Map side labels to +1 (buy), -1 (sell) or 0 (unknown)."""
    values = series.to_numpy()
    # Fast path for numeric signs.
    if pd.api.types.is_numeric_dtype(series):
        numeric = np.asarray(values, dtype=float)
        return np.where(numeric > 0, 1, np.where(numeric < 0, -1, 0))

    lowered = np.asarray([str(v).lower().strip() for v in values])
    buy_mask = np.isin(lowered, ["buy", "bid", "long", "b", "1", "+1"])
    sell_mask = np.isin(lowered, ["sell", "ask", "short", "s", "-1"])
    return np.where(buy_mask, 1, np.where(sell_mask, -1, 0))


def _fit_propagator_impact(prices: pd.Series, signed_volumes: np.ndarray) -> dict:
    """Estimate propagator-style impact coefficients from price and signed volume.

    The model is:

        dprice_t = permanent * q_t + temporary * exposure_{t-1} + noise_t
        exposure_t = q_t + decay * exposure_{t-1}

    where ``exposure_t`` is the accumulated temporary-impact footprint of signed
    volume.  For each candidate ``decay`` in a fixed grid we build the exposure
    series, then solve a 2-variable least-squares problem for ``permanent`` and
    ``temporary``.  The candidate with the lowest residual sum of squares is
    returned.  This avoids the window-bias issues of cumulative estimators and
    handles both purely permanent and mixed temporary/permanent data.
    """
    prices = np.asarray(prices, dtype=float)
    q = np.asarray(signed_volumes, dtype=float)
    if prices.size != q.size or prices.size < 2:
        return {
            "permanent_impact": None,
            "temporary_impact": None,
            "impact_decay": None,
        }

    dprice = np.diff(prices)
    q_lag = q[1:]  # volume associated with each observed price change

    if not np.any(q_lag != 0):
        return {
            "permanent_impact": None,
            "temporary_impact": None,
            "impact_decay": None,
        }

    decay_candidates = np.linspace(0.05, 0.95, 19)
    best = {"rss": np.inf}
    for decay in decay_candidates:
        exposure = _build_exposure(q, decay)[1:]
        X = np.column_stack([q_lag, exposure])
        # Remove any rows with NaN/Inf before fitting.
        mask = np.isfinite(X).all(axis=1) & np.isfinite(dprice)
        Xf = X[mask]
        yf = dprice[mask]
        if Xf.shape[0] < 2 or np.linalg.matrix_rank(Xf) < 2:
            continue
        coeffs, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        pred = Xf @ coeffs
        rss = float(np.sum((yf - pred) ** 2))
        if rss < best["rss"]:
            best = {
                "rss": rss,
                "decay": float(decay),
                "permanent": float(coeffs[0]),
                "temporary": float(coeffs[1]),
            }

    if best["rss"] is np.inf:
        return {
            "permanent_impact": None,
            "temporary_impact": None,
            "impact_decay": None,
        }

    return {
        "permanent_impact": best["permanent"],
        "temporary_impact": best["temporary"],
        "impact_decay": best["decay"],
    }


def _build_exposure(q: np.ndarray, decay: float) -> np.ndarray:
    """Build the temporary-impact exposure series ``exposure_t = q_t + decay * exposure_{t-1}``."""
    exposure = np.zeros_like(q, dtype=float)
    for t in range(1, q.size):
        exposure[t] = q[t] + decay * exposure[t - 1]
    return exposure
