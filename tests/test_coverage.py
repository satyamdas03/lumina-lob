"""Coverage tests for core engine edge cases."""
from __future__ import annotations

import pytest

from lumina_lob.core import MatchingEngine, Order, OrderBook, Side, OrderType


# ---------- Order validation ----------

def test_order_price_must_be_positive():
    with pytest.raises(ValueError, match="price must be positive"):
        Order(1, Side.BID, 0, 1)


def test_order_qty_must_be_positive():
    with pytest.raises(ValueError, match="qty must be positive"):
        Order(1, Side.BID, 100, 0)


def test_market_order_cannot_have_price():
    with pytest.raises(ValueError, match="market order cannot have price"):
        Order(1, Side.BID, 100, 1, order_type=OrderType.MARKET)


def test_limit_order_must_have_price():
    with pytest.raises(ValueError, match="limit order must have price"):
        Order(1, Side.BID, None, 1, order_type=OrderType.LIMIT)


def test_fill_negative_amount():
    o = Order(1, Side.BID, 100, 5)
    with pytest.raises(ValueError, match="fill amount must be positive"):
        o.fill(0)


def test_fill_too_much():
    o = Order(1, Side.BID, 100, 5)
    with pytest.raises(ValueError, match="fill amount exceeds remaining qty"):
        o.fill(6)


# ---------- PriceLevel ----------

def test_remove_returns_false_when_order_not_in_level():
    from lumina_lob.core.price_level import PriceLevel

    level = PriceLevel(100)
    orphan = Order(1, Side.BID, 100, 1)
    assert level.remove(orphan) is False


def test_remove_middle_order():
    from lumina_lob.core.price_level import PriceLevel

    level = PriceLevel(100)
    o1 = Order(1, Side.BID, 100, 1)
    o2 = Order(2, Side.BID, 100, 2)
    o3 = Order(3, Side.BID, 100, 3)
    level.append(o1)
    level.append(o2)
    level.append(o3)
    level.remove(o2)
    assert list(level) == [o1, o3]
    assert level.total_qty == 4
    assert len(level) == 2


def test_price_level_fill():
    from lumina_lob.core.price_level import PriceLevel

    level = PriceLevel(100)
    o1 = Order(1, Side.BID, 100, 5)
    o2 = Order(2, Side.BID, 100, 3)
    level.append(o1)
    level.append(o2)
    filled = level.fill(6)
    assert filled == 6
    assert o1.is_filled
    assert o1.filled_qty == 5
    assert o2.filled_qty == 1
    assert level.total_qty == 2
    assert len(level) == 1


def test_price_level_fill_entire_queue():
    from lumina_lob.core.price_level import PriceLevel

    level = PriceLevel(100)
    o1 = Order(1, Side.BID, 100, 2)
    o2 = Order(2, Side.BID, 100, 3)
    level.append(o1)
    level.append(o2)
    filled = level.fill(10)
    assert filled == 5
    assert level.is_empty()
    assert level.total_qty == 0


# ---------- Book properties ----------

def test_spread_none_without_both_sides():
    book = OrderBook()
    assert book.spread is None
    book.add(Order(1, Side.BID, 100, 5))
    assert book.spread is None


def test_mid_price_none_without_both_sides():
    book = OrderBook()
    assert book.mid_price is None
    book.add(Order(1, Side.ASK, 101, 5))
    assert book.mid_price is None


def test_duplicate_order_id():
    book = OrderBook()
    book.add(Order(1, Side.BID, 100, 5))
    with pytest.raises(ValueError, match="duplicate order_id"):
        book.add(Order(1, Side.ASK, 101, 5))


def test_cancel_missing_level():
    book = OrderBook()
    order = Order(1, Side.BID, 100, 5)
    book.add(order)
    # Corrupt the side levels so price level disappears
    del book.bids[100]
    assert book.cancel(1) is False


def test_modify_non_positive_qty():
    book = OrderBook()
    book.add(Order(1, Side.BID, 100, 5))
    with pytest.raises(ValueError, match="new qty must be positive"):
        book.modify(1, 0)


def test_modify_missing_level():
    book = OrderBook()
    book.add(Order(1, Side.BID, 100, 5))
    del book.bids[100]
    assert book.modify(1, 3) is False


def test_snapshot():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 99, 7))
    engine.process(Order(3, Side.ASK, 101, 4))
    snap = book.snapshot()
    assert snap["bids"] == {100: 5, 99: 7}
    assert snap["asks"] == {101: 4}


def test_book_len():
    book = OrderBook()
    assert len(book) == 0
    book.add(Order(1, Side.BID, 100, 5))
    assert len(book) == 1


# ---------- Matching engine edge cases ----------

def test_market_order_stops_when_filled():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 101, 2))
    engine.process(Order(2, Side.ASK, 102, 50))
    engine.process(Order(3, Side.BID, None, 2, order_type=OrderType.MARKET))
    assert book.trades == [(3, 1, 2)]
    assert 102 in book.asks


def test_ioc_ask_with_price_limit():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.BID, 98, 5))
    # IOC ASK with price limit 99: hits 100 (>= 99), then stops at 98 (< 99)
    engine.process(Order(3, Side.ASK, 99, 10, order_type=OrderType.IOC))
    assert book.trades == [(3, 1, 5)]
    assert 98 in book.bids
    assert book.bids[98].total_qty == 5


def test_ioc_stops_when_filled():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 101, 2))
    engine.process(Order(2, Side.ASK, 102, 50))
    engine.process(Order(3, Side.BID, 101, 2, order_type=OrderType.IOC))
    assert book.trades == [(3, 1, 2)]
    assert 102 in book.asks


def test_fok_with_price_limit_bid():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 101, 3))
    engine.process(Order(2, Side.ASK, 102, 3))
    # FOK BID with price limit 101: only 101 available, exactly enough
    engine.process(Order(3, Side.BID, 101, 3, order_type=OrderType.FOK))
    assert book.trades == [(3, 1, 3)]
    assert 102 in book.asks


def test_fok_with_price_limit_ask():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 99, 3))
    engine.process(Order(2, Side.BID, 98, 3))
    # FOK ASK with price limit 99: hits 99, skips 98, exactly enough
    engine.process(Order(3, Side.ASK, 99, 3, order_type=OrderType.FOK))
    assert book.trades == [(3, 1, 3)]
    assert 98 in book.bids


def test_fok_ask_cancels_when_price_limit_too_high():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 98, 5))
    # FOK ASK with price limit 99: 98 < 99, so no fill
    engine.process(Order(2, Side.ASK, 99, 3, order_type=OrderType.FOK))
    assert book.trades == []
    assert 1 in book.orders
    assert 2 not in book.orders


def test_fok_bid_cancels_when_price_limit_too_low():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 102, 5))
    # FOK BID with price limit 101: 102 > 101, so no fill
    engine.process(Order(2, Side.BID, 101, 3, order_type=OrderType.FOK))
    assert book.trades == []
    assert 1 in book.orders
    assert 2 not in book.orders


def test_limit_buy_does_not_cross_when_price_too_low():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 101, 5))
    engine.process(Order(2, Side.BID, 100, 3))
    assert 2 in book.orders
    assert book.trades == []


def test_limit_sell_does_not_cross_when_price_too_high():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.BID, 100, 5))
    engine.process(Order(2, Side.ASK, 101, 3))
    assert 2 in book.orders
    assert book.trades == []


def test_fill_at_price_stops_when_order_filled():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 101, 5))
    engine.process(Order(2, Side.ASK, 101, 5))
    engine.process(Order(3, Side.BID, 101, 5, order_type=OrderType.LIMIT))
    assert book.trades == [(3, 1, 5)]
    assert 2 in book.orders


def test_limit_buy_stops_when_next_ask_above_limit():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 100, 3))
    engine.process(Order(2, Side.ASK, 102, 5))
    engine.process(Order(3, Side.BID, 101, 10, order_type=OrderType.LIMIT))
    # Fills 3 at 100, then next ask 102 > 101 so stops; remaining 7 rests
    assert book.trades == [(3, 1, 3)]
    assert 2 in book.orders
    assert 3 in book.orders
    assert book.bids[101].total_qty == 7


def test_limit_buy_deletes_empty_ask_level():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process(Order(1, Side.ASK, 100, 3))
    engine.process(Order(2, Side.BID, 101, 3, order_type=OrderType.LIMIT))
    assert book.trades == [(2, 1, 3)]
    assert 100 not in book.asks
    assert 1 not in book.orders


# ---------- EventLog ----------

def test_event_log_len():
    from lumina_lob.core.event_log import EventLog

    log = EventLog()
    assert len(log) == 0
    log.log_add(1, "BID", 100, 5, 100, 101)
    assert len(log) == 1
