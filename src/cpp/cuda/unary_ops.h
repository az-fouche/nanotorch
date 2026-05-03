#pragma once

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

std::shared_ptr<Storage> exp(const TensorView& x);
std::shared_ptr<Storage> log(const TensorView& x);
std::shared_ptr<Storage> pow(const TensorView& x, Scalar value);
std::shared_ptr<Storage> relu(const TensorView& x);
std::shared_ptr<Storage> neg(const TensorView& x);

void bind_unary_ops_(py::module& m);