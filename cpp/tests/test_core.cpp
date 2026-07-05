#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

#include "lumina_lob/lumina_lob.hpp"

using namespace lumina_lob;

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond) do { \
    ++tests_run; \
    if (!(cond)) { \
        std::cerr << "FAIL: " << #cond << " at line " << __LINE__ << std::endl; \
        ++tests_failed; \
    } \
} while (0)

void test_order_validation() {
    Order o1(1, Side::BID, 100, 10, OrderType::LIMIT);
    CHECK(o1.remaining_qty() == 10);

    try {
        Order bad(2, Side::BID, -5, 10, OrderType::LIMIT);
        CHECK(false);
    } catch (const std::invalid_argument&) {
        CHECK(true);
    }

    try {
        Order bad(3, Side::BID, std::nullopt, 10, OrderType::LIMIT);
        CHECK(false);
    } catch (const std::invalid_argument&) {
        CHECK(true);
    }

    try {
        Order bad(4, Side::BID, 100, 10, OrderType::MARKET);
        CHECK(false);
    } catch (const std::invalid_argument&) {
        CHECK(true);
    }

    Order m(5, Side::ASK, std::nullopt, 5, OrderType::MARKET);
    CHECK(m.price == std::nullopt);
}

void test_price_level() {
    PriceLevel level(100);
    auto o1 = std::make_unique<Order>(1, Side::BID, 100, 5, OrderType::LIMIT);
    level.append(std::move(o1));
    CHECK(level.total_qty() == 5);
    CHECK(level.order_count() == 1);

    auto o2 = std::make_unique<Order>(2, Side::BID, 100, 3, OrderType::LIMIT);
    auto& ref2 = level.append(std::move(o2));
    CHECK(level.total_qty() == 8);

    int64_t filled = level.fill(4);
    CHECK(filled == 4);
    CHECK(level.total_qty() == 4);

    level.reduce(ref2, 2);
    CHECK(level.total_qty() == 3);

    bool removed = level.remove(ref2);
    CHECK(removed);
    CHECK(level.total_qty() == 1);
    CHECK(level.order_count() == 1);
}

void test_order_book_add_cancel() {
    OrderBook book;
    auto o1 = std::make_unique<Order>(1, Side::BID, 100, 10, OrderType::LIMIT);
    book.add(std::move(o1));
    CHECK(book.best_bid().value() == 100);
    CHECK(!book.best_ask().has_value());

    auto o2 = std::make_unique<Order>(2, Side::ASK, 102, 5, OrderType::LIMIT);
    book.add(std::move(o2));
    CHECK(book.best_ask().value() == 102);
    CHECK(book.spread().value() == 2);
    CHECK(book.mid_price().value() == 101.0);

    bool ok = book.cancel(1);
    CHECK(ok);
    CHECK(book.size() == 1);
    CHECK(!book.best_bid().has_value());

    CHECK(!book.cancel(99));
}

void test_order_book_modify() {
    OrderBook book;
    auto o = std::make_unique<Order>(1, Side::BID, 100, 10, OrderType::LIMIT);
    book.add(std::move(o));
    CHECK(book.depth(Side::BID).front().second == 10);

    book.modify(1, 4);
    CHECK(book.depth(Side::BID).front().second == 4);

    try {
        book.modify(1, 0);
        CHECK(false); // should throw for non-positive qty
    } catch (const std::invalid_argument&) {
        CHECK(true);
    }

    // Remove the remaining order via cancel.
    CHECK(book.cancel(1));
    CHECK(book.size() == 0);
}

void test_limit_order_match() {
    OrderBook book;
    MatchingEngine engine(book);

    engine.process(std::make_unique<Order>(1, Side::BID, 100, 10, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(2, Side::ASK, 100, 4, OrderType::LIMIT));

    CHECK(book.size() == 1);
    CHECK(book.trades().size() == 1);
    auto [aggr, rest, qty] = book.trades().front();
    CHECK(aggr == 2);
    CHECK(rest == 1);
    CHECK(qty == 4);
    CHECK(book.best_bid().value() == 100);
    CHECK(book.depth(Side::BID).front().second == 6);
    CHECK(book.event_log()->size() > 0);
}

void test_market_order() {
    OrderBook book;
    MatchingEngine engine(book);

    engine.process(std::make_unique<Order>(1, Side::BID, 100, 10, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(2, Side::ASK, 101, 3, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(3, Side::ASK, std::nullopt, 7, OrderType::MARKET));

    CHECK(book.size() == 2); // bid 10 + ask 3 remain
    CHECK(book.trades().size() == 1);
    auto [aggr, rest, qty] = book.trades().back();
    CHECK(qty == 7);
    CHECK(aggr == 3);
    CHECK(rest == 1);
}

void test_ioc() {
    OrderBook book;
    MatchingEngine engine(book);

    engine.process(std::make_unique<Order>(1, Side::BID, 100, 10, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(2, Side::BID, 99, 5, OrderType::LIMIT));

    // IOC ask with price limit 100: matches only the bid at 100, leaves the rest
    engine.process(std::make_unique<Order>(3, Side::ASK, 100, 15, OrderType::IOC));

    CHECK(book.trades().size() == 1);
    CHECK(std::get<2>(book.trades().back()) == 10);
    CHECK(book.size() == 1); // remaining bid at 99
    CHECK(book.best_bid().value() == 99);
}

void test_fok() {
    OrderBook book;
    MatchingEngine engine(book);

    engine.process(std::make_unique<Order>(1, Side::BID, 100, 5, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(2, Side::ASK, std::nullopt, 10, OrderType::FOK));
    CHECK(book.trades().empty()); // not enough liquidity
    CHECK(book.size() == 1);

    engine.process(std::make_unique<Order>(3, Side::ASK, std::nullopt, 5, OrderType::FOK));
    CHECK(book.trades().size() == 1);
    CHECK(book.size() == 0);
}

void test_full_depth_snapshot() {
    OrderBook book;
    MatchingEngine engine(book);

    engine.process(std::make_unique<Order>(1, Side::BID, 100, 1, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(2, Side::BID, 99, 2, OrderType::LIMIT));
    engine.process(std::make_unique<Order>(3, Side::ASK, 101, 3, OrderType::LIMIT));

    auto bids = book.full_depth(Side::BID);
    CHECK(bids.size() == 2);
    CHECK(bids[0].first == 100);
    CHECK(bids[1].first == 99);
    CHECK(bids[0].second == 1);

    auto asks = book.full_depth(Side::ASK);
    CHECK(asks.size() == 1);
    CHECK(asks[0].first == 101);

    auto [bid_snap, ask_snap] = book.snapshot();
    CHECK(bid_snap.size() == 2);
    CHECK(ask_snap.size() == 1);
}

int main() {
    test_order_validation();
    test_price_level();
    test_order_book_add_cancel();
    test_order_book_modify();
    test_limit_order_match();
    test_market_order();
    test_ioc();
    test_fok();
    test_full_depth_snapshot();

    std::cout << "run " << tests_run << " checks, " << tests_failed << " failed" << std::endl;
    return tests_failed == 0 ? 0 : 1;
}
