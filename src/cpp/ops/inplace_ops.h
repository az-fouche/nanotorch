#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace py = pybind11;

void add_inplace(TensorView &out, const TensorView &other);
void sub_inplace(TensorView &out, const TensorView &other);
void mul_inplace(TensorView &out, const TensorView &other);
void div_inplace(TensorView &out, const TensorView &other);
void copy_inplace(TensorView &out, const TensorView &other);

void bind_inplace_ops_(py::module_ &m);