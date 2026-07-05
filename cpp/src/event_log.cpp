#include "lumina_lob/event_log.hpp"

#include <optional>
#include <string>
#include <unordered_map>

namespace lumina_lob {

int64_t EventLog::next_id() {
    return ++counter_;
}

int64_t EventLog::now_ns() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

Event EventLog::log_add(int64_t order_id,
                        const std::string& side,
                        std::optional<int64_t> price,
                        int64_t qty,
                        std::optional<int64_t> best_bid,
                        std::optional<int64_t> best_ask) {
    Event ev{
        next_id(),
        now_ns(),
        EventType::ADD,
        order_id,
        side,
        price,
        qty,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        best_bid,
        best_ask,
    };
    events_.push_back(ev);
    return ev;
}

Event EventLog::log_cancel(int64_t order_id,
                           std::optional<int64_t> best_bid,
                           std::optional<int64_t> best_ask) {
    Event ev{
        next_id(),
        now_ns(),
        EventType::CANCEL,
        order_id,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        best_bid,
        best_ask,
    };
    events_.push_back(ev);
    return ev;
}

Event EventLog::log_modify(int64_t order_id,
                           int64_t new_qty,
                           std::optional<int64_t> best_bid,
                           std::optional<int64_t> best_ask) {
    Event ev{
        next_id(),
        now_ns(),
        EventType::MODIFY,
        order_id,
        std::nullopt,
        std::nullopt,
        new_qty,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        best_bid,
        best_ask,
    };
    events_.push_back(ev);
    return ev;
}

Event EventLog::log_fill(int64_t order_id,
                         int64_t counterparty_id,
                         int64_t trade_qty,
                         int64_t price,
                         const std::string& side,
                         int64_t filled_qty,
                         std::optional<int64_t> best_bid,
                         std::optional<int64_t> best_ask) {
    Event ev{
        next_id(),
        now_ns(),
        EventType::FILL,
        order_id,
        side,
        price,
        std::nullopt,
        filled_qty,
        counterparty_id,
        trade_qty,
        best_bid,
        best_ask,
    };
    events_.push_back(ev);
    return ev;
}

std::vector<std::unordered_map<std::string, std::string>> EventLog::to_dicts() const {
    std::vector<std::unordered_map<std::string, std::string>> result;
    result.reserve(events_.size());
    for (const auto& e : events_) {
        std::unordered_map<std::string, std::string> d;
        d["event_id"] = std::to_string(e.event_id);
        d["timestamp_ns"] = std::to_string(e.timestamp_ns);
        d["event_type"] = to_string(e.event_type);
        d["order_id"] = std::to_string(e.order_id);
        d["side"] = e.side.value_or("");
        d["price"] = e.price ? std::to_string(*e.price) : "";
        d["qty"] = e.qty ? std::to_string(*e.qty) : "";
        d["filled_qty"] = e.filled_qty ? std::to_string(*e.filled_qty) : "";
        d["counterparty_id"] = e.counterparty_id ? std::to_string(*e.counterparty_id) : "";
        d["trade_qty"] = e.trade_qty ? std::to_string(*e.trade_qty) : "";
        d["best_bid"] = e.best_bid ? std::to_string(*e.best_bid) : "";
        d["best_ask"] = e.best_ask ? std::to_string(*e.best_ask) : "";
        result.push_back(std::move(d));
    }
    return result;
}

} // namespace lumina_lob
