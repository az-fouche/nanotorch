#include "cuda.h"

void bind_cuda_(py::module_& m) {
    m.def("is_cuda_available", &is_cuda_available, "Detect if a CUDA device is present.");
}