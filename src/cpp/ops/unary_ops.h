#pragma once

#include "helpers.h"
#include "storage.h"
#include "tensor_view.cuh"

std::shared_ptr<Storage> exp(const TensorView &x);
std::shared_ptr<Storage> log(const TensorView &x);
std::shared_ptr<Storage> pow(const TensorView &x, Scalar value);
std::shared_ptr<Storage> relu(const TensorView &x);
std::shared_ptr<Storage> neg(const TensorView &x);
std::shared_ptr<Storage> sqrt(const TensorView &x);
std::shared_ptr<Storage> tanh(const TensorView &x);
std::shared_ptr<Storage> sigmoid(const TensorView &x);

void bind_unary_ops_(py::module_ &m);