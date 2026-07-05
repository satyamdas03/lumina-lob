"""Matching engine: price-time priority fills."""
from __future__ import annotations

from typing import Optional

from lumina_lob.core.book import OrderBook
from lumina_lob.core.order import Order, OrderType, Side


class MatchingEngine:
    """Match incoming orders against resting liquidity."""

    def __init__(self, book: OrderBook) -> None:
        self.book = book

    def process(self, order: Order) -> None:
        """Route order: match market/limit aggressive, rest passive remainder."""
        if order.order_type == OrderType.MARKET:
            self._match_market(order)
            return
        if order.order_type == OrderType.IOC:
            self._match_ioc(order)
            return
        if order.order_type == OrderType.FOK:
            self._match_fok(order)
            return
        # Limit order
        if order.side == Side.BID:
            if self.book.best_ask is not None and order.price >= self.book.best_ask:
                self._match_buy(order)
        else:
            if self.book.best_bid is not None and order.price <= self.book.best_bid:
                self._match_sell(order)
        if not order.is_filled:
            self.book.add(order)

    def _match_ioc(self, order: Order) -> None:
        """Immediate or Cancel: match aggressively, cancel remainder, leave nothing in book."""
        opposite = Side.ASK if order.side == Side.BID else Side.BID
        levels = self.book._side_levels(opposite)
        prices = sorted(levels.keys(), reverse=(opposite == Side.BID))
        for p in prices:
            if order.is_filled:
                break
            # IOC can have a price limit; if set, enforce it
            if order.price is not None:
                if order.side == Side.BID and p > order.price:
                    break
                if order.side == Side.ASK and p < order.price:
                    break
            level = levels[p]
            self._fill_at_price(order, level)
            if level.is_empty():
                del levels[p]

    def _match_fok(self, order: Order) -> None:
        """Fill or Kill: all or nothing."""
        opposite = Side.ASK if order.side == Side.BID else Side.BID
        levels = self.book._side_levels(opposite)
        prices = sorted(levels.keys(), reverse=(opposite == Side.BID))
        available = 0
        for p in prices:
            if order.price is not None:
                if order.side == Side.BID and p > order.price:
                    break
                if order.side == Side.ASK and p < order.price:
                    break
            available += levels[p].total_qty
            if available >= order.qty:
                break
        if available >= order.qty:
            self._match_ioc(order)

    def _match_market(self, order: Order) -> None:
        opposite = Side.ASK if order.side == Side.BID else Side.BID
        levels = self.book._side_levels(opposite)
        prices = sorted(levels.keys(), reverse=(opposite == Side.BID))
        for p in prices:
            if order.is_filled:
                break
            level = levels[p]
            self._fill_at_price(order, level)
            if level.is_empty():
                del levels[p]

    def _match_buy(self, order: Order) -> None:
        while not order.is_filled:
            best_ask = self.book.best_ask
            if best_ask is None or best_ask > order.price:
                break
            level = self.book.asks[best_ask]
            self._fill_at_price(order, level)
            if level.is_empty():
                del self.book.asks[best_ask]

    def _match_sell(self, order: Order) -> None:
        while not order.is_filled:
            best_bid = self.book.best_bid
            if best_bid is None or best_bid < order.price:
                break
            level = self.book.bids[best_bid]
            self._fill_at_price(order, level)
            if level.is_empty():
                del self.book.bids[best_bid]

    def _fill_at_price(self, order: Order, level) -> None:
        for resting in list(level):
            if order.is_filled:
                break
            amount = min(order.remaining_qty, resting.remaining_qty)
            order.fill(amount)
            resting.fill(amount)
            level.total_qty -= amount
            self.book.trades.append((order.order_id, resting.order_id, amount))
            if resting.is_filled:
                level.remove(resting)
                self.book.orders.pop(resting.order_id, None)
