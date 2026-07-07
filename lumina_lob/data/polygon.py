"""Polygon.io downloader for end-of-day bars and tick/trade data."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests


class PolygonClient:
    """Lightweight client for Polygon.io market data.

    Parameters
    ----------
    api_key:
        Polygon API key. If omitted, read from the ``POLYGON_API_KEY``
        environment variable.
    base_url:
        Polygon API base URL.
    cache_dir:
        Directory to cache raw JSON responses.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.polygon.io",
        cache_dir: str = ".cache/polygon",
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key is required (pass api_key or set POLYGON_API_KEY)")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_daily_bars(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for ``ticker`` between ``start_date`` and ``end_date``.

        Dates must be in ``YYYY-MM-DD`` format.  Results are cached by ticker
        and date range.

        Returns
        -------
        DataFrame with columns:
        ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``,
        ``vwap``, ``transactions``.
        """
        endpoint = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"apiKey": self.api_key}
        cache_key = f"{ticker.upper()}_{start_date}_{end_date}_bars.json"
        data = self._fetch_json(endpoint, params, cache_key)
        return self._bars_to_dataframe(data.get("results", []))

    def get_trades(self, ticker: str, date: str) -> pd.DataFrame:
        """Fetch trades (ticks) for ``ticker`` on ``date``.

        Returns
        -------
        DataFrame with columns:
        ``timestamp``, ``price``, ``size``, ``exchange``, ``conditions``.
        """
        endpoint = f"{self.base_url}/v3/trades/{ticker}"
        params = {"timestamp": date, "apiKey": self.api_key}
        cache_key = f"{ticker.upper()}_{date}_trades.json"
        data = self._fetch_json(endpoint, params, cache_key)
        return self._trades_to_dataframe(data.get("results", []))

    def _fetch_json(
        self, url: str, params: dict[str, Any], cache_key: str
    ) -> dict[str, Any]:
        """Return JSON from cache or fetch from Polygon and cache it."""
        cache_path = self.cache_dir / cache_key.replace("/", "_")
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as fh:
                return cast(dict[str, Any], json.load(fh))

        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Polygon API error {response.status_code}: {response.text}"
            )
        data: dict[str, Any] = cast(dict[str, Any], response.json())
        cache_path.write_text(response.text, encoding="utf-8")
        return data

    @staticmethod
    def _bars_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "transactions",
                ]
            )
        rows = []
        for r in results:
            rows.append(
                {
                    "timestamp": pd.to_datetime(r["t"], unit="ms", utc=True),
                    "open": float(r["o"]),
                    "high": float(r["h"]),
                    "low": float(r["l"]),
                    "close": float(r["c"]),
                    "volume": int(r["v"]),
                    "vwap": float(r.get("vw", 0.0)),
                    "transactions": int(r.get("n", 0)),
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _trades_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame(
                columns=["timestamp", "price", "size", "exchange", "conditions"]
            )
        rows = []
        for r in results:
            rows.append(
                {
                    "timestamp": pd.to_datetime(r["sip_timestamp"], unit="ns", utc=True),
                    "price": float(r["price"]),
                    "size": int(r["size"]),
                    "exchange": int(r.get("exchange", 0)),
                    "conditions": r.get("conditions", []),
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values("timestamp").reset_index(drop=True)
