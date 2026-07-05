#pragma once

#include <cstdint>
#include <string>

namespace lumina_lob {

enum class OrderType : int8_t {
    LIMIT = 1,
    MARKET = 2,
    IOC = 3,   // Immediate or Cancel
    FOK = 4,   // Fill or Kill
};

inline std::string to_string(OrderType type) {
    switch (type) {
        case OrderType::LIMIT: return "LIMIT";
        case OrderType::MARKET: return "MARKET";
        case OrderType::IOC: return "IOC";
        case OrderType::FOK: return "FOK";
    }
    return "UNKNOWN";
}

} // namespace lumina_lob
