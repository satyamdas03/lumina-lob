"""Unit tests for depth snapshot and pandas export."""
from __future__ import annotations

from lumina_lob.core import MatchingEngine, Order, OrderBook, Side


def test_full_depth():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 99, 7))
    engine.process(Order(3, Side.BID, 98, 2))
    engine.process(Order(4, Side.ASK, 101, 4))
    depth = book.full_depth(Side.BID)
    assert depth == {100: 5, 99: 7, 98: 2}


def test_depth_top_n():
    book = OrderBook()
    engine = MatchingEngine(book)
    for i, p in enumerate([100, 99, 98, 97, 96, 95]):
        engine.process(Order(i + 1, Side.BID, p, 1))
    assert book.depth(Side.BID, n=3) == {100: 1, 99: 1, 98: 1}


def test_full_snapshot():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.ASK, 101, 4))
    snap = book.full_snapshot()
    assert snap["bids"] == {100: 5}
    assert snap["asks"] == {101: 4}


def test_to_pandas():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 100, 3))
    engine.process(Order(3, Side.ASK, 101, 4))
    df = book.to_pandas()
    assert len(df) == 2
    bid_row = df[df["side"] == "BID"].iloc[0]
    assert bid_row["price"] == 100
    assert bid_row["qty"] == 8
    assert bid_row["order_count"] == 2
    ask_row = df[df["side"] == "ASK"].iloc[0]
    assert ask_row["price"] == 101
    assert ask_row["qty"] == 4
