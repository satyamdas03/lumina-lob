"""Databento downloader for historical trades and quotes."""
from __future__ import annotations

import os
from pathlib import Path

import databento as db
import pandas as pd


class DatabentoClient:
    """Lightweight client for Databento historical market data.

    Parameters
    ----------
    api_key:
        Databento API key. If omitted, read from the ``DATABENTO_API_KEY``
        environment variable.
    cache_dir:
        Directory to cache raw DBN files.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = ".cache/databento",
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("Databento API key is required (pass api_key or set DATABENTO_API_KEY)")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = db.Historical(key=self.api_key)

    def get_trades(
        self,
        symbol: str,
        start: str,
        end: str,
        dataset: str = "XNAS.ITCH",
    ) -> pd.DataFrame:
        """Fetch trade records for ``symbol`` between ``start`` and ``end``.

        Returns
        -------
        DataFrame with Databento ``trades`` schema columns.
        """
        return self._get_range(dataset, symbol, "trades", start, end)

    def get_quotes(
        self,
        symbol: str,
        start: str,
        end: str,
        dataset: str = "XNAS.ITCH",
        schema: str = "bbo",
    ) -> pd.DataFrame:
        """Fetch quote records for ``symbol`` between ``start`` and ``end``.

        Default schema is ``bbo`` (best bid/offer). Use ``mbo`` for market-by-order.

        Returns
        -------
        DataFrame with the requested quote schema columns.
        """
        if schema not in {"bbo", "mbo", "mbp-1", "mbp-10", "tbbo"}:
            raise ValueError(f"unsupported quote schema: {schema}")
        return self._get_range(dataset, symbol, schema, start, end)

    def _get_range(
        self,
        dataset: str,
        symbol: str,
        schema: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        cache_key = f"{dataset}_{symbol}_{schema}_{start}_{end}.dbn"
        cache_path = self.cache_dir / cache_key.replace("/", "_").replace(":", "-")

        if cache_path.exists():
            store = db.DBNStore.from_file(str(cache_path))
            return store.to_df()

        store = self._client.timeseries.get_range(
            dataset=dataset,
            symbols=symbol,
            schema=schema,
            start=start,
            end=end,
            stype_in="raw_symbol",
            path=str(cache_path),
        )
        return store.to_df()
