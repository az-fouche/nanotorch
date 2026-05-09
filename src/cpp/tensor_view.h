#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "storage.h"

inline constexpr int NT_MAX_DIMS = 32;

struct TensorView {
  std::shared_ptr<Storage> storage;
  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;
  py::ssize_t offset;
};

inline void _require_same_shape(const TensorView &x1, const TensorView &x2,
                                const char *fname) {
  if (x1.shape != x2.shape)
    throw std::runtime_error(std::string(fname) +
                             ": Expected two views of the same shape.");
}

inline py::ssize_t unravel(py::ssize_t i, const TensorView &x) {
  py::ssize_t idx = x.offset;
  for (int j = x.shape.size() - 1; j >= 0; --j) {
    auto coord = i % x.shape[j];
    i /= x.shape[j];
    idx += coord * x.strides[j];
  }
  return idx;
}

inline void bind_tensor_view_(py::module_ &m) {
  py::class_<TensorView>(m, "TensorView")
      .def(py::init<>())
      .def(py::init<std::shared_ptr<Storage>, std::vector<py::ssize_t>,
                    std::vector<py::ssize_t>, py::ssize_t>())
      .def_readwrite("storage", &TensorView::storage)
      .def_readwrite("shape", &TensorView::shape)
      .def_readwrite("strides", &TensorView::strides)
      .def_readwrite("offset", &TensorView::offset);
}