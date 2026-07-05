#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lumina_lob/book.hpp"
#include "lumina_lob/event_log.hpp"
#include "lumina_lob/matching.hpp"
#include "lumina_lob/order.hpp"
#include "lumina_lob/order_type.hpp"
#include "lumina_lob/price_level.hpp"
#include "lumina_lob/side.hpp"

namespace py = pybind11;
using namespace lumina_lob;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++17 core of the Lumina LOB simulator";

    py::enum_<Side>(m, "Side")
        .value("BID", Side::BID)
        .value("ASK", Side::ASK);

    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT", OrderType::LIMIT)
        .value("MARKET", OrderType::MARKET)
        .value("IOC", OrderType::IOC)
        .value("FOK", OrderType::FOK);

    py::enum_<EventType>(m, "EventType")
        .value("ADD", EventType::ADD)
        .value("CANCEL", EventType::CANCEL)
        .value("MODIFY", EventType::MODIFY)
        .value("FILL", EventType::FILL);

    py::class_<Order>(m, "Order")
        .def(py::init<int64_t, Side, std::optional<int64_t>, int64_t, OrderType>(),
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"), py::arg("order_type"))
        .def_readonly("order_id", &Order::order_id)
        .def_readonly("side", &Order::side)
        .def_readonly("price", &Order::price)
        .def_readonly("qty", &Order::qty)
        .def_readonly("order_type", &Order::order_type)
        .def_readonly("filled_qty", &Order::filled_qty)
        .def_property_readonly("remaining_qty", &Order::remaining_qty)
        .def_property_readonly("is_filled", &Order::is_filled);

    py::class_<Event>(m, "Event")
        .def_readonly("event_id", &Event::event_id)
        .def_readonly("timestamp_ns", &Event::timestamp_ns)
        .def_readonly("event_type", &Event::event_type)
        .def_readonly("order_id", &Event::order_id)
        .def_readonly("side", &Event::side)
        .def_readonly("price", &Event::price)
        .def_readonly("qty", &Event::qty)
        .def_readonly("filled_qty", &Event::filled_qty)
        .def_readonly("counterparty_id", &Event::counterparty_id)
        .def_readonly("trade_qty", &Event::trade_qty)
        .def_readonly("best_bid", &Event::best_bid)
        .def_readonly("best_ask", &Event::best_ask);

    py::class_<EventLog>(m, "EventLog")
        .def(py::init<>())
        .def("to_dicts", &EventLog::to_dicts)
        .def("size", &EventLog::size)
        .def_property_readonly("events", [](const EventLog& log) { return log.events(); });

    py::class_<PriceLevel>(m, "PriceLevel")
        .def_readonly("price", &PriceLevel::price)
        .def("total_qty", &PriceLevel::total_qty)
        .def("order_count", &PriceLevel::order_count)
        .def("is_empty", &PriceLevel::is_empty)
        .def("orders",
             [](const PriceLevel& level) {
                 std::vector<Order*> out;
                 out.reserve(static_cast<size_t>(level.order_count()));
                 for (const auto& uptr : level.orders()) {
                     out.push_back(uptr.get());
                 }
                 return out;
             },
             py::return_value_policy::reference);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add",
             [](OrderBook& book, int64_t order_id, Side side, std::optional<int64_t> price, int64_t qty,
                OrderType order_type) -> Order& {
                 auto order = std::make_unique<Order>(order_id, side, price, qty, order_type);
                 return book.add(std::move(order));
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"), py::arg("order_type"),
             py::return_value_policy::reference_internal)
        .def("cancel", &OrderBook::cancel)
        .def("modify", &OrderBook::modify)
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("spread", &OrderBook::spread)
        .def("mid_price", &OrderBook::mid_price)
        .def("depth", &OrderBook::depth, py::arg("side"), py::arg("n") = 5)
        .def("full_depth", &OrderBook::full_depth, py::arg("side"))
        .def("snapshot", &OrderBook::snapshot)
        .def("full_snapshot", &OrderBook::full_snapshot)
        .def("trades", &OrderBook::trades)
        .def("bids",
             [](const OrderBook& book) { return book.bids(); },
             py::return_value_policy::reference_internal)
        .def("asks",
             [](const OrderBook& book) { return book.asks(); },
             py::return_value_policy::reference_internal)
        .def("orders",
             &OrderBook::orders,
             py::return_value_policy::reference_internal)
        .def("event_log",
             &OrderBook::event_log,
             py::return_value_policy::reference_internal)
        .def("__len__", &OrderBook::size);

    py::class_<MatchingEngine>(m, "MatchingEngine")
        .def(py::init<OrderBook&>())
        .def("process",
             [](MatchingEngine& engine, int64_t order_id, Side side, std::optional<int64_t> price,
                int64_t qty, OrderType order_type) {
                 auto order = std::make_unique<Order>(order_id, side, price, qty, order_type);
                 engine.process(std::move(order));
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"), py::arg("order_type"));
}
