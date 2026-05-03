#include "cuda.h"

bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

void bind_cuda_(py::module_& m) {
    m.def("is_cuda_available", &is_cuda_available, "Detect if a CUDA device is present.");
}