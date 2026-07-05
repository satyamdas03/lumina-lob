#include "lumina_lob/book.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace lumina_lob {

OrderBook::OrderBook(EventLog* event_log) {
    if (event_log != nullptr) {
        event_log_ptr_ = event_log;
    } else {
        event_log_ptr_ = &event_log_;
    }
}

std::optional<int64_t> OrderBook::best_bid() const {
    if (bids_.empty()) {
        return std::nullopt;
    }
    return bids_.rbegin()->first;
}

std::optional<int64_t> OrderBook::best_ask() const {
    if (asks_.empty()) {
        return std::nullopt;
    }
    return asks_.begin()->first;
}

std::optional<int64_t> OrderBook::spread() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb.has_value() || !ba.has_value()) {
        return std::nullopt;
    }
    return ba.value() - bb.value();
}

std::optional<double> OrderBook::mid_price() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb.has_value() || !ba.has_value()) {
        return std::nullopt;
    }
    return (static_cast<double>(bb.value()) + static_cast<double>(ba.value())) / 2.0;
}

Order& OrderBook::add(std::unique_ptr<Order> order) {
    if (orders_.find(order->order_id) != orders_.end()) {
        throw std::invalid_argument("duplicate order_id " + std::to_string(order->order_id));
    }
    auto price = order->price.value();
    auto& levels = side_levels(order->side);
    auto it = levels.try_emplace(price, price).first;
    auto& level = it->second;
    Order& stored = level.append(std::move(order));
    orders_[stored.order_id] = &stored;

    event_log_ptr_->log_add(
        stored.order_id,
        to_string(stored.side),
        stored.price,
        stored.qty,
        best_bid(),
        best_ask());

    return stored;
}

bool OrderBook::cancel(int64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    Order* order = it->second;
    orders_.erase(it);

    auto& levels = side_levels(order->side);
    auto level_it = levels.find(order->price.value());
    if (level_it != levels.end()) {
        level_it->second.remove(*order);
        if (level_it->second.is_empty()) {
            levels.erase(level_it);
        }
    }

    event_log_ptr_->log_cancel(order_id, best_bid(), best_ask());
    return true;
}

bool OrderBook::modify(int64_t order_id, int64_t new_qty) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    if (new_qty <= 0) {
        throw std::invalid_argument("new qty must be positive");
    }
    Order* order = it->second;

    auto& levels = side_levels(order->side);
    auto level_it = levels.find(order->price.value());
    if (level_it == levels.end()) {
        return false;
    }
    auto& level = level_it->second;
    level.reduce(*order, new_qty);
    if (order->is_filled()) {
        level.remove(*order);
        orders_.erase(order_id);
    }
    if (level.is_empty()) {
        levels.erase(level_it);
    }

    event_log_ptr_->log_modify(order_id, new_qty, best_bid(), best_ask());
    return true;
}

std::vector<std::pair<int64_t, int64_t>> OrderBook::full_depth(Side side) const {
    std::vector<std::pair<int64_t, int64_t>> result;
    const auto& levels = side_levels(side);
    if (side == Side::BID) {
        for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
            result.emplace_back(it->first, it->second.total_qty());
        }
    } else {
        for (const auto& [price, level] : levels) {
            result.emplace_back(price, level.total_qty());
        }
    }
    return result;
}

std::vector<std::pair<int64_t, int64_t>> OrderBook::depth(Side side, int64_t n) const {
    auto full = full_depth(side);
    if (static_cast<int64_t>(full.size()) > n) {
        full.resize(static_cast<size_t>(n));
    }
    return full;
}

std::pair<std::vector<std::pair<int64_t, int64_t>>,
          std::vector<std::pair<int64_t, int64_t>>> OrderBook::snapshot() const {
    return {depth(Side::BID), depth(Side::ASK)};
}

std::pair<std::vector<std::pair<int64_t, int64_t>>,
          std::vector<std::pair<int64_t, int64_t>>> OrderBook::full_snapshot() const {
    return {full_depth(Side::BID), full_depth(Side::ASK)};
}

void OrderBook::record_trade(int64_t aggressor_id, int64_t resting_id, int64_t qty) {
    trades_.emplace_back(aggressor_id, resting_id, qty);
}

void OrderBook::remove_order(int64_t order_id) {
    orders_.erase(order_id);
}

std::map<int64_t, PriceLevel>& OrderBook::side_levels(Side side) {
    return side == Side::BID ? bids_ : asks_;
}

const std::map<int64_t, PriceLevel>& OrderBook::side_levels(Side side) const {
    return side == Side::BID ? bids_ : asks_;
}

} // namespace lumina_lob
