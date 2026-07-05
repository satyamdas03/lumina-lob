"""Tests for the Databento historical data downloader."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from lumina_lob.data.databento import DatabentoClient


def _mock_store(df: pd.DataFrame):
    store = Mock()
    store.to_df.return_value = df
    return store


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Databento API key is required"):
        DatabentoClient()


def test_get_trades_requests_range(tmp_path):
    sample_df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2024-01-01T09:30", "2024-01-01T09:31"]),
            "price": [150.25, 150.30],
            "size": [100, 200],
        }
    )
    mock_client = Mock()
    mock_client.timeseries.get_range.return_value = _mock_store(sample_df)

    with patch("lumina_lob.data.databento.db.Historical", return_value=mock_client):
        client = DatabentoClient(api_key="test_key", cache_dir=str(tmp_path))
        df = client.get_trades("AAPL", "2024-01-01T09:30", "2024-01-01T16:00")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    mock_client.timeseries.get_range.assert_called_once()
    call_kwargs = mock_client.timeseries.get_range.call_args.kwargs
    assert call_kwargs["dataset"] == "XNAS.ITCH"
    assert call_kwargs["symbols"] == "AAPL"
    assert call_kwargs["schema"] == "trades"
    assert call_kwargs["start"] == "2024-01-01T09:30"
    assert call_kwargs["end"] == "2024-01-01T16:00"
    assert call_kwargs["stype_in"] == "raw_symbol"
    assert "path" in call_kwargs


def test_get_quotes_invalid_schema(tmp_path):
    with patch("lumina_lob.data.databento.db.Historical", return_value=Mock()):
        client = DatabentoClient(api_key="test_key", cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="unsupported quote schema"):
        client.get_quotes("AAPL", "2024-01-01T09:30", "2024-01-01T16:00", schema="invalid")


def test_get_quotes_requests_bbo(tmp_path):
    sample_df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2024-01-01T09:30"]),
            "bid_px_00": [150.20],
            "ask_px_00": [150.35],
        }
    )
    mock_client = Mock()
    mock_client.timeseries.get_range.return_value = _mock_store(sample_df)

    with patch("lumina_lob.data.databento.db.Historical", return_value=mock_client):
        client = DatabentoClient(api_key="test_key", cache_dir=str(tmp_path))
        df = client.get_quotes("AAPL", "2024-01-01T09:30", "2024-01-01T16:00")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    call_kwargs = mock_client.timeseries.get_range.call_args.kwargs
    assert call_kwargs["schema"] == "bbo"


def test_cache_uses_from_file_on_second_call(tmp_path):
    live_df = pd.DataFrame({"price": [1.0, 2.0]})
    cached_df = pd.DataFrame({"price": [3.0, 4.0]})
    mock_client = Mock()
    mock_client.timeseries.get_range.return_value = _mock_store(live_df)

    cache_path = tmp_path / "XNAS.ITCH_AAPL_trades_2024-01-01_2024-01-02.dbn"
    from_file_store = _mock_store(cached_df)

    with patch("lumina_lob.data.databento.db.Historical", return_value=mock_client), \
         patch("lumina_lob.data.databento.db.DBNStore.from_file", return_value=from_file_store) as mock_from_file:
        client = DatabentoClient(api_key="test_key", cache_dir=str(tmp_path))
        df1 = client.get_trades("AAPL", "2024-01-01", "2024-01-02")
        # simulate a cached file appearing
        cache_path.write_text("dummy cached dbn bytes")
        df2 = client.get_trades("AAPL", "2024-01-01", "2024-01-02")

    assert mock_client.timeseries.get_range.call_count == 1
    mock_from_file.assert_called_once_with(str(cache_path))
    assert df1["price"].tolist() == [1.0, 2.0]
    assert df2["price"].tolist() == [3.0, 4.0]
