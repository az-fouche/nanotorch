#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"
#include "tensor_view.h"

namespace py = pybind11;

bool equals(const TensorView& x1, const TensorView& x2);
void copy_view(const TensorView& src, const TensorView& out);
std::shared_ptr<Storage> clone_contiguous_view_from(const TensorView& src);
void scatter_to_axes(
    const TensorView& src, 
    const TensorView& out,
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