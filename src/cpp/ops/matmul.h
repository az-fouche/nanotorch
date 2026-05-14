#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.cuh"

namespace py = pybind11;

std::shared_ptr<Storage> matmul(const TensorView &x1, const TensorView &x2);

void bind_matmul_op_(py::module_ &m);