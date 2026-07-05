#include "lumina_lob/price_level.hpp"

#include <algorithm>

namespace lumina_lob {

Order& PriceLevel::append(std::unique_ptr<Order> order) {
    total_qty_ += order->remaining_qty();
    order_count_ += 1;
    orders_.push_back(std::move(order));
    return *orders_.back();
}

bool PriceLevel::remove(const Order& order) {
    auto it = std::find_if(orders_.begin(), orders_.end(),
                           [&order](const std::unique_ptr<Order>& ptr) { return ptr.get() == &order; });
    if (it == orders_.end()) {
        return false;
    }
    total_qty_ -= (*it)->remaining_qty();
    order_count_ -= 1;
    orders_.erase(it);
    return true;
}

int64_t PriceLevel::fill(int64_t amount) {
    if (amount <= 0) {
        return 0;
    }
    int64_t remaining = amount;
    while (remaining > 0 && !orders_.empty()) {
        Order* front = orders_.front().get();
        int64_t can_fill = std::min(front->remaining_qty(), remaining);
        front->fill(can_fill);
        total_qty_ -= can_fill;
        remaining -= can_fill;
        if (front->is_filled()) {
            orders_.pop_front();
            order_count_ -= 1;
        }
    }
    return amount - remaining;
}

int64_t PriceLevel::reduce(Order& order, int64_t new_qty) {
    int64_t removed = order.reduce_qty(new_qty);
    total_qty_ -= removed;
    return removed;
}

bool PriceLevel::is_empty() const noexcept {
    return orders_.empty();
}

} // namespace lumina_lob
