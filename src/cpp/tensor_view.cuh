#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "storage.h"

inline constexpr int NT_MAX_DIMS = 32;

struct TensorView { // For host-side
  std::shared_ptr<Storage> storage;
  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;
  py::ssize_t offset;
};
struct TensorViewStatic { // For device-side
  size_t ndim;
  py::ssize_t offset;
  py::ssize_t shape[NT_MAX_DIMS];
  py::ssize_t strides[NT_MAX_DIMS];
};
inline TensorViewStatic tensor_view_to_static(const TensorView &x) {
  auto view = TensorViewStatic(x.shape.size(), x.offset);
  for (size_t j = 0; j < x.shape.size(); ++j) {
    view.shape[j] = x.shape[j];
    view.strides[j] = x.strides[j];
  }
  return view;
}

inline bool is_contiguous(const TensorView &x) {
  py::ssize_t stride = 1;
  for (py::ssize_t i = static_cast<py::ssize_t>(x.shape.size()) - 1; i >= 0;
       --i) {
    if (stride != x.strides[i])
      return false;
    stride *= x.shape[i];
  }
  return true;
}

// Unravel computes for strided view i + view -> idx
__host__ __device__ inline py::ssize_t unravel(py::ssize_t i,
                                               TensorViewStatic view) {
  py::ssize_t idx = view.offset;
  for (py::ssize_t j = view.ndim - 1; j >= 0; --j) {
    auto coord = i % view.shape[j];
    i /= view.shape[j];
    idx += coord * view.strides[j];
  }
  return idx;
}

inline py::ssize_t unravel(py::ssize_t i, const TensorView &x) {
  py::ssize_t idx = x.offset;
  for (py::ssize_t j = x.shape.size() - 1; j >= 0; --j) {
    auto coord = i % x.shape[j];
    i /= x.shape[j];
    idx += coord * x.strides[j];
  }
  return idx;
}

// Indexer depending on the underlying view (contig much faster)
struct ContigIndexer {
  py::ssize_t offset;
  __host__ __device__ __forceinline__ py::ssize_t
  operator()(py::ssize_t i) const {
    return i + offset;
  }
};

struct StridedIndexer {
  TensorViewStatic view;
  __host__ __device__ __forceinline__ py::ssize_t
  operator()(py::ssize_t i) const {
    return unravel(i, view);
  }
};

template <class L> inline auto with_indexer(const TensorView &x, L &&launch) {
  return (is_contiguous(x)) ? launch(ContigIndexer{x.offset})
                            : launch(StridedIndexer{tensor_view_to_static(x)});
}

inline void _require_same_shape(const TensorView &x1, const TensorView &x2,
                                const char *fname) {
  if (x1.shape != x2.shape)
    throw std::runtime_error(std::string(fname) +
                             ": Expected two views of the same shape.");
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