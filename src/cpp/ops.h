#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace py = pybind11;

// Factory

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype);
std::shared_ptr<Storage> zeros(const std::vector<py::ssize_t>& shape, Dtype dtype);
std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype);
std::shared_ptr<Storage> ones(const std::vector<py::ssize_t>& shape, Dtype dtype);
std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype);
std::shared_ptr<Storage> full(const std::vector<py::ssize_t>& shape, Scalar value, Dtype dtype);
std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype);
std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype);
std::shared_ptr<Storage> arange(py::ssize_t n, Dtype dtype);

// Common ops
bool equals(const TensorView& x1, const TensorView& x2);
double sum(const TensorView& x);

// Core ops
void copy_view(const TensorView& src, const TensorView& dst);
void scatter_to_axes(
    const TensorView& src, 
    const TensorView& dst,
    const std::vector<py::ssize_t>& fancy_dims_in_src,
    const std::vector<TensorView>& fancy_dims_data,
    const std::vector<bool>& out_axis_is_fancy,
    const std::vector<py::ssize_t>& out_axis_target
);
std::shared_ptr<Storage> gather_from_axes(
    const TensorView& x,
    const std::vector<py::ssize_t>& new_shape,
    const std::vector<py::ssize_t>& fancy_dims_in_src,
    const std::vector<TensorView>& fancy_dims_data,
    const std::vector<bool>& out_axis_is_fancy,
    const std::vector<py::ssize_t>& out_axis_target
);

void bind_ops_(py::module_& m);