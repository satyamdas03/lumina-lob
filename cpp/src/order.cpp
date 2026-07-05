#include "lumina_lob/order.hpp"

#include <stdexcept>

namespace lumina_lob {

Order::Order(int64_t order_id,
             Side side,
             std::optional<int64_t> price,
             int64_t qty,
             OrderType order_type)
    : order_id(order_id), side(side), price(price), qty(qty), order_type(order_type) {
    if (price.has_value() && price.value() <= 0) {
        throw std::invalid_argument("price must be positive");
    }
    if (qty <= 0) {
        throw std::invalid_argument("qty must be positive");
    }
    if (order_type == OrderType::MARKET && price.has_value()) {
        throw std::invalid_argument("market order cannot have price");
    }
    if (order_type == OrderType::LIMIT && !price.has_value()) {
        throw std::invalid_argument("limit order must have price");
    }
}

int64_t Order::remaining_qty() const noexcept {
    return qty - filled_qty;
}

bool Order::is_filled() const noexcept {
    return remaining_qty() == 0;
}

void Order::fill(int64_t amount) {
    if (amount <= 0) {
        throw std::invalid_argument("fill amount must be positive");
    }
    if (amount > remaining_qty()) {
        throw std::invalid_argument("fill amount exceeds remaining qty");
    }
    filled_qty += amount;
}

int64_t Order::reduce_qty(int64_t new_qty) {
    if (new_qty < filled_qty) {
        throw std::invalid_argument("new qty cannot be below filled qty");
    }
    if (new_qty > qty) {
        throw std::invalid_argument("cannot increase qty via reduce");
    }
    int64_t removed = qty - new_qty;
    qty = new_qty;
    return removed;
}

} // namespace lumina_lob
