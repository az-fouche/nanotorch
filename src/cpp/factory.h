#pragma once

#include <random>

#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace py = pybind11;

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype, Device device);
std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype, Device device);
std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype, Device device);
std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype, Device device);
std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype, Device device);

void manual_seed(uint64_t seed);
std::shared_ptr<Storage> uniform(py::ssize_t n, Dtype dtype, Device device);

void bind_factory_(py::module& m);