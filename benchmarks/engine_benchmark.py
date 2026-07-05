"""Benchmark the pure-Python engine against the C++17 extension."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Iterable

from lumina_lob.core import MatchingEngine, Order, OrderBook, OrderType, Side

OrderSpec = tuple[int, Side, int, int, OrderType]


def _make_orders(n: int, seed: int) -> list[OrderSpec]:
    """Generate a deterministic sequence of alternating bid/ask limit orders."""
    rng = random.Random(seed)
    orders: list[OrderSpec] = []
    for i in range(n):
        side = Side.BID if i % 2 == 0 else Side.ASK
        # Alternate around a central price so roughly half the orders cross.
        price = 10_000 + rng.randint(-2, 2)
        qty = rng.randint(1, 100)
        orders.append((i + 1, side, price, qty, OrderType.LIMIT))
    return orders


def _run_python(orders: Iterable[OrderSpec]) -> float:
    """Run orders through the pure-Python engine and return events/sec."""
    book = OrderBook()
    engine = MatchingEngine(book)
    orders = list(orders)
    start = time.perf_counter()
    for oid, side, price, qty, otype in orders:
        engine.process(Order(oid, side, price, qty, otype))
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return float("inf")
    return len(orders) / elapsed


def _run_cpp(orders: Iterable[OrderSpec]) -> float | None:
    """Run orders through the C++ extension and return events/sec, or None if unavailable."""
    try:
        from lumina_lob import _core  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        return None

    book = _core.OrderBook()
    engine = _core.MatchingEngine(book)
    orders = list(orders)

    def _to_cpp_side(side: Side) -> int:
        return _core.Side.BID if side == Side.BID else _core.Side.ASK

    def _to_cpp_type(otype: OrderType) -> int:  # noqa: ARG001
        # The benchmark currently only uses LIMIT orders.
        return _core.OrderType.LIMIT

    start = time.perf_counter()
    for oid, side, price, qty, otype in orders:
        engine.process(oid, _to_cpp_side(side), price, qty, _to_cpp_type(otype))
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return float("inf")
    return len(orders) / elapsed


def _fmt_rate(rate: float | None) -> str:
    if rate is None:
        return "not built"
    return f"{rate:,.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Lumina LOB engines.")
    parser.add_argument("--orders", type=int, default=200_000, help="Number of orders to submit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic orders")
    args = parser.parse_args(argv)

    if args.orders <= 0:
        print("ERROR: --orders must be positive", file=sys.stderr)
        return 1

    orders = _make_orders(args.orders, args.seed)

    py_rate = _run_python(orders)
    cpp_rate = _run_cpp(orders)

    print(f"Orders submitted: {args.orders}")
    print(f"Python engine:    {_fmt_rate(py_rate)} events/sec")
    print(f"C++ engine:       {_fmt_rate(cpp_rate)} events/sec")
    if cpp_rate is not None and py_rate > 0:
        print(f"Speedup:          {cpp_rate / py_rate:.1f}x")
    else:
        print("Speedup:          N/A")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
