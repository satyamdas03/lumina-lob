#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "event_log.hpp"
#include "order.hpp"
#include "price_level.hpp"
#include "side.hpp"

namespace lumina_lob {

class OrderBook {
public:
    explicit OrderBook(EventLog* event_log = nullptr);

    std::optional<int64_t> best_bid() const;
    std::optional<int64_t> best_ask() const;
    std::optional<int64_t> spread() const;
    std::optional<double> mid_price() const;

    // Take ownership of an order and insert it into the book. Returns a
    // reference to the stored order (stable until cancelled/filled).
    Order& add(std::unique_ptr<Order> order);

    bool cancel(int64_t order_id);
    bool modify(int64_t order_id, int64_t new_qty);

    std::vector<std::pair<int64_t, int64_t>> full_depth(Side side) const;
    std::vector<std::pair<int64_t, int64_t>> depth(Side side, int64_t n = 5) const;

    std::pair<std::vector<std::pair<int64_t, int64_t>>,
              std::vector<std::pair<int64_t, int64_t>>> snapshot() const;

    std::pair<std::vector<std::pair<int64_t, int64_t>>,
              std::vector<std::pair<int64_t, int64_t>>> full_snapshot() const;

    std::map<int64_t, PriceLevel>& bids() noexcept { return bids_; }
    std::map<int64_t, PriceLevel>& asks() noexcept { return asks_; }
    const std::map<int64_t, PriceLevel>& bids() const noexcept { return bids_; }
    const std::map<int64_t, PriceLevel>& asks() const noexcept { return asks_; }
    const std::unordered_map<int64_t, Order*>& orders() const noexcept { return orders_; }
    const std::vector<std::tuple<int64_t, int64_t, int64_t>>& trades() const noexcept { return trades_; }
    EventLog* event_log() noexcept { return event_log_ptr_; }

    size_t size() const noexcept { return orders_.size(); }

    // Internal helpers used by the matching engine.
    void record_trade(int64_t aggressor_id, int64_t resting_id, int64_t qty);
    void remove_order(int64_t order_id);


private:
    std::map<int64_t, PriceLevel> bids_;
    std::map<int64_t, PriceLevel> asks_;
    std::unordered_map<int64_t, Order*> orders_;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> trades_;
    EventLog event_log_;
    EventLog* event_log_ptr_ = nullptr;

    std::map<int64_t, PriceLevel>& side_levels(Side side);
    const std::map<int64_t, PriceLevel>& side_levels(Side side) const;
};

} // namespace lumina_lob
