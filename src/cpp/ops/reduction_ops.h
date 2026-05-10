#pragma once

#include <cuda/std/limits>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace nt {
std::shared_ptr<Storage> sum(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop,
                             Dtype dtype);
std::shared_ptr<Storage> min(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop);
std::shared_ptr<Storage> max(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop);
std::shared_ptr<Storage> argmin(const TensorView &x,
                                const std::vector<py::ssize_t> &axis_drop);
std::shared_ptr<Storage> argmax(const TensorView &x,
                                const std::vector<py::ssize_t> &axis_drop);
} // namespace nt

void bind_reduction_ops_(py::module_ &m);