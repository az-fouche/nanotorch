#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace py = pybind11;

// Factory

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype, Device device);
std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype, Device device);
std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype, Device device);
std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype, Device device);
std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype, Device device);

// Common ops
std::shared_ptr<Storage> sum(const TensorView& x, const std::vector<py::ssize_t>& axis, bool keepdim, Dtype dtype);
std::shared_ptr<Storage> add(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> subtract(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> multiply(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> divide(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> matmul(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> exp(const TensorView& x);
std::shared_ptr<Storage> log(const TensorView& x);
std::shared_ptr<Storage> pow(const TensorView& x, Scalar value);
std::shared_ptr<Storage> relu(const TensorView& x);
std::shared_ptr<Storage> neg(const TensorView& x);
bool equals(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> pw_greater(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> pw_greater_eq(const TensorView& x1, const TensorView& x2);
std::shared_ptr<Storage> pw_equal(const TensorView& x1, const TensorView& x2);

// Inplace
void add_inplace(TensorView& out, const TensorView& other);
void sub_inplace(TensorView& out, const TensorView& other);
void mul_inplace(TensorView& out, const TensorView& other);
void div_inplace(TensorView& out, const TensorView& other);
void copy_inplace(TensorView& out, const TensorView& other);

// Special
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