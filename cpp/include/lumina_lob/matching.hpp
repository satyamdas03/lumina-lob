#pragma once

#include <memory>

#include "book.hpp"
#include "order.hpp"
#include "side.hpp"

namespace lumina_lob {

class MatchingEngine {
public:
    explicit MatchingEngine(OrderBook& book) : book_(book) {}

    void process(std::unique_ptr<Order> order);

private:
    OrderBook& book_;

    void match_market(Order& order);
    void match_ioc(Order& order);
    void match_fok(Order& order);
    void match_buy(Order& order);
    void match_sell(Order& order);
    void fill_at_price(Order& order, PriceLevel& level);
};

} // namespace lumina_lob
