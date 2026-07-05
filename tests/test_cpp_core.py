"""Smoke tests for the C++17 core exposed via pybind11."""

import pytest

pybind11 = pytest.importorskip("pybind11")

try:
    from lumina_lob import _core
except ImportError as exc:  # pragma: no cover
    pytest.fail(f"C++ extension lumina_lob._core is not built: {exc}")


@pytest.fixture
def fresh_book():
    return _core.OrderBook()


@pytest.fixture
def engine(fresh_book):
    return _core.MatchingEngine(fresh_book)


def test_order_validation():
    o = _core.Order(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    assert o.remaining_qty == 10
    assert not o.is_filled

    with pytest.raises(ValueError):
        _core.Order(2, _core.Side.BID, -5, 10, _core.OrderType.LIMIT)

    with pytest.raises(ValueError):
        _core.Order(3, _core.Side.BID, None, 10, _core.OrderType.LIMIT)

    with pytest.raises(ValueError):
        _core.Order(4, _core.Side.BID, 100, 10, _core.OrderType.MARKET)

    m = _core.Order(5, _core.Side.ASK, None, 5, _core.OrderType.MARKET)
    assert m.price is None


def test_order_book_add_cancel(fresh_book):
    fresh_book.add(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    assert fresh_book.best_bid() == 100
    assert fresh_book.best_ask() is None

    fresh_book.add(2, _core.Side.ASK, 102, 5, _core.OrderType.LIMIT)
    assert fresh_book.best_ask() == 102
    assert fresh_book.spread() == 2
    assert fresh_book.mid_price() == 101.0

    assert fresh_book.cancel(1)
    assert len(fresh_book) == 1
    assert fresh_book.best_bid() is None

    assert not fresh_book.cancel(99)


def test_limit_order_match(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.ASK, 100, 4, _core.OrderType.LIMIT)

    assert len(fresh_book) == 1
    assert len(fresh_book.trades()) == 1
    aggressor, resting, qty = fresh_book.trades()[0]
    assert aggressor == 2
    assert resting == 1
    assert qty == 4
    assert fresh_book.best_bid() == 100
    assert fresh_book.full_depth(_core.Side.BID)[0][1] == 6


def test_market_order(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.ASK, 101, 3, _core.OrderType.LIMIT)
    engine.process(3, _core.Side.ASK, None, 7, _core.OrderType.MARKET)

    assert len(fresh_book) == 2
    aggressor, resting, qty = fresh_book.trades()[-1]
    assert qty == 7
    assert aggressor == 3
    assert resting == 1


def test_ioc(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.BID, 99, 5, _core.OrderType.LIMIT)

    engine.process(3, _core.Side.ASK, 100, 15, _core.OrderType.IOC)

    assert len(fresh_book.trades()) == 1
    assert fresh_book.trades()[-1][2] == 10
    assert len(fresh_book) == 1
    assert fresh_book.best_bid() == 99


def test_fok(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 5, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.ASK, None, 10, _core.OrderType.FOK)
    assert len(fresh_book.trades()) == 0
    assert len(fresh_book) == 1

    engine.process(3, _core.Side.ASK, None, 5, _core.OrderType.FOK)
    assert len(fresh_book.trades()) == 1
    assert len(fresh_book) == 0


def test_modify(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    assert fresh_book.depth(_core.Side.BID)[0][1] == 10

    fresh_book.modify(1, 4)
    assert fresh_book.depth(_core.Side.BID)[0][1] == 4

    with pytest.raises(ValueError):
        fresh_book.modify(1, 0)


def test_depth_and_snapshot(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 1, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.BID, 99, 2, _core.OrderType.LIMIT)
    engine.process(3, _core.Side.ASK, 101, 3, _core.OrderType.LIMIT)

    bids = fresh_book.full_depth(_core.Side.BID)
    assert bids[0][0] == 100
    assert bids[1][0] == 99
    assert bids[0][1] == 1

    asks = fresh_book.full_depth(_core.Side.ASK)
    assert asks[0][0] == 101
    assert asks[0][1] == 3

    bid_snap, ask_snap = fresh_book.snapshot()
    assert len(bid_snap) == 2
    assert len(ask_snap) == 1


def test_event_log(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 10, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.ASK, 100, 4, _core.OrderType.LIMIT)

    log = fresh_book.event_log()
    assert log.size() > 0
    assert len(log.events) == log.size()

    dicts = log.to_dicts()
    assert len(dicts) == log.size()
    assert any(d["event_type"] == "FILL" for d in dicts)


def test_price_level_orders(engine, fresh_book):
    engine.process(1, _core.Side.BID, 100, 5, _core.OrderType.LIMIT)
    engine.process(2, _core.Side.BID, 100, 7, _core.OrderType.LIMIT)

    level = fresh_book.bids()[100]
    assert level.order_count() == 2
    assert level.total_qty() == 12
    orders = level.orders()
    assert len(orders) == 2
    assert orders[0].order_id == 1
    assert orders[1].order_id == 2
