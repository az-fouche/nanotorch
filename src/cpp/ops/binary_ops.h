#pragma once

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

std::shared_ptr<Storage> add(const TensorView &a, const TensorView &b);
std::shared_ptr<Storage> subtract(const TensorView &a, const TensorView &b);
std::shared_ptr<Storage> multiply(const TensorView &a, const TensorView &b);
std::shared_ptr<Storage> divide(const TensorView &a, const TensorView &b);
std::shared_ptr<Storage> pw_equal(const TensorView &a, const TensorView &b);
std::shared_ptr<Storage> pw_greater(const TensorView &a, const TensorView &b);
std::shared_ptr<Storage> pw_greater_eq(const TensorView &a,
                                       const TensorView &b);

void bind_binary_ops_(py::module_ &m);