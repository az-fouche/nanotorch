#pragma once

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define NT_CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) \
        throw std::runtime_error( \
            std::string(#expr " failed: ") + cudaGetErrorString(_e)); \
} while(0)

inline bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

void bind_cuda_(py::module_& m);