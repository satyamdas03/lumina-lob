cmake_minimum_required(VERSION 3.15)
project(lumina_lob_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lumina_lob_core STATIC
    src/order.cpp
    src/price_level.cpp
    src/event_log.cpp
    src/book.cpp
    src/matching.cpp
)
target_include_directories(lumina_lob_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

enable_testing()

add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE lumina_lob_core)
add_test(NAME core COMMAND test_core)
