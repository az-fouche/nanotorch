#pragma once

#include <pybind11/pybind11.h>                                                                                       
#include <pybind11/stl.h>

#include "storage.h"

constexpr int NT_MAX_DIMS = 32;

struct TensorView {
    std::shared_ptr<Storage> storage;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    py::ssize_t offset;
};

inline void bind_tensor_view_(py::module_& m) {
    py::class_<TensorView>(m, "TensorView")
        .def(py::init<>())
        .def(py::init<
            std::shared_ptr<Storage>,
            std::vector<py::ssize_t>,
            std::vector<py::ssize_t>,
            py::ssize_t>())
        .def_readwrite("storage", &TensorView::storage)
        .def_readwrite("shape", &TensorView::shape)
        .def_readwrite("strides", &TensorView::strides)
        .def_readwrite("offset", &TensorView::offset);
}