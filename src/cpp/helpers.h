#pragma once

#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "tensor_view.h"

namespace py = pybind11;

template<typename T>
std::string vec_to_string(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

inline py::ssize_t numel_from_shape(const std::vector<py::ssize_t>& shape) {
    auto n_axes = static_cast<py::ssize_t>(shape.size());
    py::ssize_t numel = 1;
    for (py::ssize_t i = 0; i < n_axes; ++i)
        numel *= shape[i];
    return numel;
}

inline void requires_cpu(const TensorView& x, const char* op_name) {
    if (x.storage->device() != Device::Cpu) {
        throw std::runtime_error(std::string(op_name) + ": requires Cpu.");
    }
}