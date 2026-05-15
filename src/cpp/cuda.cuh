#pragma once

#include "cuda.h"
#include "tensor_view.cuh"

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

__host__ __device__ inline const float4 *as_f4(const void *ptr,
                                               py::ssize_t offset) {
  return reinterpret_cast<const float4 *>(static_cast<const float *>(ptr) +
                                          offset);
}

__host__ __device__ inline float4 *as_f4(void *ptr, py::ssize_t offset) {
  return reinterpret_cast<float4 *>(static_cast<float *>(ptr) + offset);
}