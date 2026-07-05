#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "side.hpp"
#include "order_type.hpp"

namespace lumina_lob {

struct Order {
    int64_t order_id;
    Side side;
    std::optional<int64_t> price;
    int64_t qty;
    OrderType order_type;
    int64_t filled_qty = 0;

    Order(int64_t order_id,
          Side side,
          std::optional<int64_t> price,
          int64_t qty,
          OrderType order_type = OrderType::LIMIT);

    int64_t remaining_qty() const noexcept;
    bool is_filled() const noexcept;
    void fill(int64_t amount);
    int64_t reduce_qty(int64_t new_qty);
};

} // namespace lumina_lob
