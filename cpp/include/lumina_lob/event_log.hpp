#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lumina_lob {

enum class EventType {
    ADD,
    CANCEL,
    MODIFY,
    FILL,
};

inline std::string to_string(EventType type) {
    switch (type) {
        case EventType::ADD: return "ADD";
        case EventType::CANCEL: return "CANCEL";
        case EventType::MODIFY: return "MODIFY";
        case EventType::FILL: return "FILL";
    }
    return "UNKNOWN";
}

struct Event {
    int64_t event_id;
    int64_t timestamp_ns;
    EventType event_type;
    int64_t order_id;
    std::optional<std::string> side;
    std::optional<int64_t> price;
    std::optional<int64_t> qty;
    std::optional<int64_t> filled_qty;
    std::optional<int64_t> counterparty_id;
    std::optional<int64_t> trade_qty;
    std::optional<int64_t> best_bid;
    std::optional<int64_t> best_ask;
};

class EventLog {
public:
    EventLog() = default;

    Event log_add(int64_t order_id,
                  const std::string& side,
                  std::optional<int64_t> price,
                  int64_t qty,
                  std::optional<int64_t> best_bid,
                  std::optional<int64_t> best_ask);

    Event log_cancel(int64_t order_id,
                     std::optional<int64_t> best_bid,
                     std::optional<int64_t> best_ask);

    Event log_modify(int64_t order_id,
                     int64_t new_qty,
                     std::optional<int64_t> best_bid,
                     std::optional<int64_t> best_ask);

    Event log_fill(int64_t order_id,
                   int64_t counterparty_id,
                   int64_t trade_qty,
                   int64_t price,
                   const std::string& side,
                   int64_t filled_qty,
                   std::optional<int64_t> best_bid,
                   std::optional<int64_t> best_ask);

    const std::vector<Event>& events() const noexcept { return events_; }
    std::vector<std::unordered_map<std::string, std::string>> to_dicts() const;

    size_t size() const noexcept { return events_.size(); }

private:
    std::vector<Event> events_;
    int64_t counter_ = 0;

    int64_t next_id();
    int64_t now_ns();
};

} // namespace lumina_lob
