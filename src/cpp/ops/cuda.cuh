#pragma once

#include "cuda.h"
#include "tensor_view.h"

namespace py = pybind11;

constexpr int BLOCK_SIZE = 256;

template <class Kernel, class... Args>
inline void launch_1d(py::ssize_t n, Kernel kernel, Args... args) {
  if (n == 0)
    return;
  int grid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  kernel<<<grid, BLOCK_SIZE>>>(args...);
  NT_CUDA_CHECK(cudaGetLastError());
}

struct TensorViewStatic {
  size_t n_axes;
  py::ssize_t offset;
  py::ssize_t shape[NT_MAX_DIMS];
  py::ssize_t strides[NT_MAX_DIMS];
};
__device__ inline py::ssize_t unravel(py::ssize_t i, TensorViewStatic view) {
  py::ssize_t idx = view.offset;
  for (int j = view.n_axes - 1; j >= 0; --j) {
    auto coord = i % view.shape[j];
    i /= view.shape[j];
    idx += coord * view.strides[j];
  }
  return idx;
}
inline TensorViewStatic tensor_view_to_static(const TensorView &x) {
  auto view = TensorViewStatic(x.shape.size(), x.offset);
  for (size_t j = 0; j < x.shape.size(); ++j) {
    view.shape[j] = x.shape[j];
    view.strides[j] = x.strides[j];
  }
  return view;
}