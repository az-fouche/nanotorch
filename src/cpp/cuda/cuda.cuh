#pragma once

#include "cuda.h"
#include "tensor_view.h"

namespace py = pybind11;

constexpr int BLOCK_SIZE = 256;

template <class Kernel, class... Args>
inline void launch_1d(py::ssize_t n, Kernel kernel, Args... args) {
    if (n == 0) return;
    int grid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel<<<grid, BLOCK_SIZE>>>(args...);
    NT_CUDA_CHECK(cudaGetLastError());
}

struct StridedView {
    size_t n_axes;
    py::ssize_t offset;
    py::ssize_t shape[NT_MAX_DIMS];
    py::ssize_t strides[NT_MAX_DIMS];
};