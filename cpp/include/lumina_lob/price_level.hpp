#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <stdexcept>

#include "order.hpp"

namespace lumina_lob {

class PriceLevel {
public:
    int64_t price;

    explicit PriceLevel(int64_t price) : price(price) {}

    // Append an order to the tail of the queue. Returns a reference to the
    // stored order (stable for the lifetime of the level).
    Order& append(std::unique_ptr<Order> order);

    // Remove a specific order from this level. Returns true if found.
    bool remove(const Order& order);

    // Fill `amount` from the front of the queue. Returns filled quantity.
    int64_t fill(int64_t amount);

    // Reduce an order's total desired quantity. Returns amount removed.
    int64_t reduce(Order& order, int64_t new_qty);

    // Decrement the level's displayed total quantity by a filled amount.
    void record_fill(int64_t amount) { total_qty_ -= amount; }

    bool is_empty() const noexcept;
    int64_t total_qty() const noexcept { return total_qty_; }
    int64_t order_count() const noexcept { return order_count_; }

    const std::list<std::shared_ptr<Order>>& orders() const noexcept { return orders_; }
    std::list<std::shared_ptr<Order>>& orders() noexcept { return orders_; }

private:
    std::list<std::shared_ptr<Order>> orders_;
    int64_t total_qty_ = 0;
    int64_t order_count_ = 0;
};

} // namespace lumina_lob
