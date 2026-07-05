#include "lumina_lob/matching.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "lumina_lob/side.hpp"

namespace lumina_lob {

void MatchingEngine::process(std::unique_ptr<Order> order) {
    switch (order->order_type) {
        case OrderType::MARKET:
            match_market(*order);
            return;
        case OrderType::IOC:
            match_ioc(*order);
            return;
        case OrderType::FOK:
            match_fok(*order);
            return;
        default:
            break;
    }

    // Limit order
    if (order->side == Side::BID) {
        auto best_ask = book_.best_ask();
        if (best_ask.has_value() && order->price.value() >= best_ask.value()) {
            match_buy(*order);
        }
    } else {
        auto best_bid = book_.best_bid();
        if (best_bid.has_value() && order->price.value() <= best_bid.value()) {
            match_sell(*order);
        }
    }

    if (!order->is_filled()) {
        book_.add(std::move(order));
    }
}

void MatchingEngine::match_market(Order& order) {
    Side opposite = order.side == Side::BID ? Side::ASK : Side::BID;
    if (opposite == Side::BID) {
        for (auto it = book_.bids().rbegin(); it != book_.bids().rend() && !order.is_filled(); ) {
            auto& level = it->second;
            fill_at_price(order, level);
            if (level.is_empty()) {
                it = decltype(it)(book_.bids().erase(std::next(it).base()));
            } else {
                ++it;
            }
        }
    } else {
        for (auto it = book_.asks().begin(); it != book_.asks().end() && !order.is_filled(); ) {
            auto& level = it->second;
            fill_at_price(order, level);
            if (level.is_empty()) {
                it = book_.asks().erase(it);
            } else {
                ++it;
            }
        }
    }
}

void MatchingEngine::match_ioc(Order& order) {
    Side opposite = order.side == Side::BID ? Side::ASK : Side::BID;
    if (opposite == Side::BID) {
        for (auto it = book_.bids().rbegin(); it != book_.bids().rend() && !order.is_filled(); ) {
            int64_t p = it->first;
            if (order.price.has_value() && p < order.price.value()) {
                break;
            }
            auto& level = it->second;
            fill_at_price(order, level);
            if (level.is_empty()) {
                it = decltype(it)(book_.bids().erase(std::next(it).base()));
            } else {
                ++it;
            }
        }
    } else {
        for (auto it = book_.asks().begin(); it != book_.asks().end() && !order.is_filled(); ) {
            int64_t p = it->first;
            if (order.price.has_value() && p > order.price.value()) {
                break;
            }
            auto& level = it->second;
            fill_at_price(order, level);
            if (level.is_empty()) {
                it = book_.asks().erase(it);
            } else {
                ++it;
            }
        }
    }
}

void MatchingEngine::match_fok(Order& order) {
    Side opposite = order.side == Side::BID ? Side::ASK : Side::BID;
    int64_t available = 0;
    if (opposite == Side::BID) {
        for (auto it = book_.bids().rbegin(); it != book_.bids().rend(); ++it) {
            int64_t p = it->first;
            if (order.price.has_value() && p < order.price.value()) {
                break;
            }
            available += it->second.total_qty();
            if (available >= order.remaining_qty()) {
                break;
            }
        }
    } else {
        for (auto it = book_.asks().begin(); it != book_.asks().end(); ++it) {
            int64_t p = it->first;
            if (order.price.has_value() && p > order.price.value()) {
                break;
            }
            available += it->second.total_qty();
            if (available >= order.remaining_qty()) {
                break;
            }
        }
    }

    if (available >= order.remaining_qty()) {
        match_ioc(order);
    }
}

void MatchingEngine::match_buy(Order& order) {
    while (!order.is_filled()) {
        auto best_ask = book_.best_ask();
        if (!best_ask.has_value() || best_ask.value() > order.price.value()) {
            break;
        }
        auto it = book_.asks().find(best_ask.value());
        if (it == book_.asks().end()) {
            break;
        }
        auto& level = it->second;
        fill_at_price(order, level);
        if (level.is_empty()) {
            book_.asks().erase(it);
        }
    }
}

void MatchingEngine::match_sell(Order& order) {
    while (!order.is_filled()) {
        auto best_bid = book_.best_bid();
        if (!best_bid.has_value() || best_bid.value() < order.price.value()) {
            break;
        }
        auto it = book_.bids().find(best_bid.value());
        if (it == book_.bids().end()) {
            break;
        }
        auto& level = it->second;
        fill_at_price(order, level);
        if (level.is_empty()) {
            book_.bids().erase(it);
        }
    }
}

void MatchingEngine::fill_at_price(Order& order, PriceLevel& level) {
    std::vector<Order*> snapshot;
    snapshot.reserve(static_cast<size_t>(level.order_count()));
    for (const auto& uptr : level.orders()) {
        snapshot.push_back(uptr.get());
    }

    for (Order* resting : snapshot) {
        if (order.is_filled()) {
            break;
        }
        int64_t amount = std::min(order.remaining_qty(), resting->remaining_qty());
        order.fill(amount);
        resting->fill(amount);
        level.record_fill(amount);
        book_.record_trade(order.order_id, resting->order_id, amount);
        book_.event_log()->log_fill(
            order.order_id,
            resting->order_id,
            amount,
            level.price,
            to_string(order.side),
            order.filled_qty,
            book_.best_bid(),
            book_.best_ask());

        if (resting->is_filled()) {
            int64_t id = resting->order_id;
            level.remove(*resting);
            book_.remove_order(id);
        }
    }
}

} // namespace lumina_lob
