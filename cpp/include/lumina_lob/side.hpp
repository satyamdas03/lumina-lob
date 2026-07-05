#pragma once

#include <cstdint>
#include <string>

namespace lumina_lob {

enum class Side : int8_t {
    BID = 1,
    ASK = 2,
};

inline std::string to_string(Side side) {
    return side == Side::BID ? "BID" : "ASK";
}

} // namespace lumina_lob
