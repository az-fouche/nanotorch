#pragma once

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace nt {
std::shared_ptr<Storage> sum(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop,
                             Dtype dtype);
} // namespace nt

void bind_reduction_ops_(py::module_ &m);