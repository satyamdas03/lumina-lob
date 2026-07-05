"""Tests for the Polygon.io data downloader."""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from lumina_lob.data.polygon import PolygonClient


SAMPLE_BAR_RESPONSE = {
    "ticker": "AAPL",
    "results": [
        {
            "v": 1000000,
            "vw": 150.25,
            "o": 149.5,
            "c": 151.0,
            "h": 151.5,
            "l": 149.0,
            "t": 1609459200000,
            "n": 50000,
        },
        {
            "v": 2000000,
            "o": 151.0,
            "c": 152.0,
            "h": 152.5,
            "l": 150.5,
            "t": 1609545600000,
            "n": 60000,
        },
    ],
}

SAMPLE_TRADE_RESPONSE = {
    "results": [
        {
            "sip_timestamp": 1609459200000000000,
            "price": 150.25,
            "size": 100,
            "exchange": 4,
            "conditions": [37],
        },
        {
            "sip_timestamp": 1609459200001000000,
            "price": 150.3,
            "size": 50,
            "exchange": 4,
            "conditions": [],
        },
    ],
}


def _make_mock_response(json_data: dict, status_code: int = 200) -> Mock:
    mock = Mock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.text = json.dumps(json_data)
    return mock


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Polygon API key is required"):
        PolygonClient()


def test_get_daily_bars_parses_response(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response(SAMPLE_BAR_RESPONSE)
        df = client.get_daily_bars("AAPL", "2021-01-01", "2021-01-02")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]
    assert df["open"].tolist() == [149.5, 151.0]
    assert df["close"].tolist() == [151.0, 152.0]
    assert df["volume"].tolist() == [1000000, 2000000]
    assert df["transactions"].tolist() == [50000, 60000]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_get_daily_bars_empty_results(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response({"results": []})
        df = client.get_daily_bars("AAPL", "2021-01-01", "2021-01-02")

    assert df.empty
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]


def test_get_trades_parses_response(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response(SAMPLE_TRADE_RESPONSE)
        df = client.get_trades("AAPL", "2021-01-01")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "price", "size", "exchange", "conditions"]
    assert df["price"].tolist() == [150.25, 150.3]
    assert df["size"].tolist() == [100, 50]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_cache_avoids_second_network_call(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response(SAMPLE_BAR_RESPONSE)
        df1 = client.get_daily_bars("AAPL", "2021-01-01", "2021-01-02")
        df2 = client.get_daily_bars("AAPL", "2021-01-01", "2021-01-02")

    assert mock_get.call_count == 1
    pd.testing.assert_frame_equal(df1, df2)


def test_api_error_raises_runtime_error(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response({"error": "bad request"}, status_code=400)
        with pytest.raises(RuntimeError, match="Polygon API error 400"):
            client.get_daily_bars("AAPL", "2021-01-01", "2021-01-02")


def test_get_trades_empty_results(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response({"results": []})
        df = client.get_trades("AAPL", "2021-01-01")

    assert df.empty
    assert list(df.columns) == ["timestamp", "price", "size", "exchange", "conditions"]


def test_vwap_default_when_missing(tmp_path):
    client = PolygonClient(api_key="test_key", cache_dir=str(tmp_path))
    response = {
        "results": [
            {
                "v": 1000,
                "o": 10.0,
                "c": 11.0,
                "h": 12.0,
                "l": 9.0,
                "t": 1609459200000,
                "n": 100,
            }
        ]
    }
    with patch("lumina_lob.data.polygon.requests.get") as mock_get:
        mock_get.return_value = _make_mock_response(response)
        df = client.get_daily_bars("TICK", "2021-01-01", "2021-01-01")

    assert df["vwap"].tolist() == [0.0]
