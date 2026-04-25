#include "ops.h"

py::ssize_t numel_from_shape(const std::vector<py::ssize_t>& shape) {
    auto n_axes = static_cast<py::ssize_t>(shape.size());
    py::ssize_t numel = 1;
    for (py::ssize_t i = 0; i < n_axes; ++i)
        numel *= shape[i];
    return numel;
}

// Factory

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype) {
    return Storage::allocate(n, dtype); // 0 by default
}

std::shared_ptr<Storage> zeros(const std::vector<py::ssize_t>& shape, Dtype dtype) {
    return zeros(numel_from_shape(shape), dtype); // 0 by default
}

std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype) {
    auto storage = zeros(n, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < storage->size(); ++i)
            data[i] = static_cast<T>(1);
    });
    return storage;
}

std::shared_ptr<Storage> ones(const std::vector<py::ssize_t>& shape, Dtype dtype) {
    return ones(numel_from_shape(shape), dtype);
}

std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype) {
    auto storage = zeros(n, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        auto fill = value.item<T>();
        for (py::ssize_t i = 0; i < storage->size(); ++i)
            data[i] = fill;
    });
    return storage;
}

std::shared_ptr<Storage> full(const std::vector<py::ssize_t>& shape, Scalar value, Dtype dtype) {
    return full(numel_from_shape(shape), value, dtype);
}

std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype) {
    py::ssize_t size = n * n;
    auto storage = zeros(size, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < n; ++i)
            data[i + i * n] = static_cast<T>(1);
    });
    return storage;
}

std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype) {
    auto storage = zeros(n, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < n; ++i)
            data[i] = static_cast<T>(start + i * step);
    });
    return storage;
}

std::shared_ptr<Storage> arange(py::ssize_t n, Dtype dtype) {
    return arange(n, 0, 1, dtype);
}

// Ops

double sum(const TensorView& x) {
    auto storage = x.storage;
    auto n_axes = static_cast<py::ssize_t>(x.shape.size());
    auto numel = numel_from_shape(x.shape);
    return dispatch_dtype(storage->dtype(), [&]<typename T>() {
        auto data = static_cast<const T*>(storage->data());
        double result = 0;
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel; ++i) {
            // Add value at loc
            py::ssize_t idx = x.offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) idx += x.strides[j] * loc[j];
            result += data[idx];

            // Loc update
            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < x.shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
        return result;
    });
}

template <typename F>
std::shared_ptr<Storage> _binary_op_generic(
    const TensorView& x1, const TensorView& x2, F&& func
) {
    auto s1 = x1.storage;
    auto s2 = x2.storage;
    if (s1->dtype() != s2->dtype())
        throw std::invalid_argument(
            "equals: expected homogeneous tensors, got " + dtype_to_format(s1->dtype()) 
            +  " and " + dtype_to_format(s2->dtype()) + "."
        );
    if (x1.shape != x2.shape)
        throw std::invalid_argument(
            "equals: expected same size tensors, got " + vec_to_string(x1.shape)
            +  " and " + vec_to_string(x2.shape) + "."
        );

    auto n_axes = static_cast<py::ssize_t>(x1.shape.size());
    auto numel = numel_from_shape(x1.shape);
    auto new_storage = Storage::allocate(numel, x1.storage->dtype());

    return dispatch_dtype(s1->dtype(), [&]<typename T>() {
        auto* ptr1 = static_cast<const T*>(s1->data());
        auto* ptr2 = static_cast<const T*>(s2->data());
        auto* ptrout = static_cast<T*>(new_storage->data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel; ++i) {
            py::ssize_t idx1 = x1.offset, idx2 = x2.offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) {
                idx1 += x1.strides[j] * loc[j];
                idx2 += x2.strides[j] * loc[j];
            }
            ptrout[i] = func(ptr1[idx1], ptr2[idx2]);

            // Loc update
            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < x1.shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
        return new_storage;
    });
}

std::shared_ptr<Storage> add(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a + b; });
}

std::shared_ptr<Storage> subtract(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a - b; });
}

std::shared_ptr<Storage> multiply(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a * b; });
}

std::shared_ptr<Storage> divide(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a / b; });
}

bool equals(const TensorView& x1, const TensorView& x2) {
    auto s1 = x1.storage;
    auto s2 = x2.storage;
    if (s1->dtype() != s2->dtype())
        throw std::invalid_argument(
            "equals: expected homogeneous tensors, got " + dtype_to_format(s1->dtype()) 
            +  " and " + dtype_to_format(s2->dtype()) + "."
        );
    if (x1.shape != x2.shape)
        throw std::invalid_argument(
            "equals: expected same size tensors, got " + vec_to_string(x1.shape)
            +  " and " + vec_to_string(x2.shape) + "."
        );

    auto n_axes = static_cast<py::ssize_t>(x1.shape.size());
    auto numel = numel_from_shape(x1.shape);
    return dispatch_dtype(s1->dtype(), [&]<typename T>() {
        auto* ptr1 = static_cast<const T*>(s1->data());
        auto* ptr2 = static_cast<const T*>(s2->data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel; ++i) {
            // Current loc eq check
            py::ssize_t idx1 = x1.offset, idx2 = x2.offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) {
                idx1 += x1.strides[j] * loc[j];
                idx2 += x2.strides[j] * loc[j];
            }
            if (ptr1[idx1] != ptr2[idx2]) return false;

            // Loc update
            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < x1.shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
        return true;
    });
}


void copy_view(const TensorView& src, const TensorView& dst) {
    auto src_storage = src.storage;
    auto dst_storage = dst.storage;
    auto n_axes = static_cast<py::ssize_t>(src.shape.size());
    auto numel = numel_from_shape(src.shape);
    dispatch_dtype(dst_storage->dtype(), [&]<typename T>() {
        auto* ptr_src = static_cast<const T*>(src_storage->data());
        auto* ptr_dst = static_cast<T*>(dst_storage->data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel; ++i) {
            py::ssize_t idx_src = src.offset, idx_dst = dst.offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) {
                idx_src += src.strides[j] * loc[j];
                idx_dst += dst.strides[j] * loc[j];
            }
            ptr_dst[idx_dst] = ptr_src[idx_src];

            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < src.shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
    });
}

void scatter_to_axes(
    const TensorView& src, 
    const TensorView& dst,
    const std::vector<py::ssize_t>& fancy_dims_in_src,
    const std::vector<TensorView>& fancy_dims_data,
    const std::vector<bool>& out_axis_is_fancy,
    const std::vector<py::ssize_t>& out_axis_target
) {
    // TODO (optim): optimize with chunk memcopy
    py::ssize_t dst_ndim = dst.shape.size();
    py::ssize_t ind_ndim = fancy_dims_data.empty() ? 0 : (py::ssize_t)fancy_dims_data[0].shape.size();
    py::ssize_t src_ndim = src.shape.size();
    py::ssize_t numel = numel_from_shape(src.shape);

    auto indexers_data = std::vector<const int64_t*>(fancy_dims_data.size());
    for (size_t q = 0; q < indexers_data.size(); ++q) 
        indexers_data[q] = static_cast<const int64_t*>(fancy_dims_data[q].storage->data());

    auto fancy_axes = std::vector<py::ssize_t>(dst_ndim, -1);
    for (py::ssize_t p = 0; p < fancy_dims_in_src.size(); ++p) fancy_axes[fancy_dims_in_src[p]] = p;

    auto loc_src = std::vector<py::ssize_t>(src_ndim);
    auto loc_dst_basic = std::vector<py::ssize_t>(dst_ndim);
    auto loc_dst_ind = std::vector<py::ssize_t>(ind_ndim);
    dispatch_dtype(dst.storage->dtype(), [&]<typename T>() {
        auto dst_data = static_cast<T*>(dst.storage->data());
        auto src_data = static_cast<const T*>(src.storage->data());
        for (py::ssize_t idx = 0; idx < numel; ++idx) {

            // Set output loc wrt basic/fancy index combination
            for (py::ssize_t i = 0; i < src_ndim; ++i) {
                py::ssize_t src_dim = out_axis_target[i];
                if (out_axis_is_fancy[i])
                    loc_dst_ind[src_dim] = loc_src[i];
                else
                    loc_dst_basic[src_dim] = loc_src[i];
            }

            // Compute index in source storage
            py::ssize_t idx_src = src.offset;
            for (py::ssize_t i = 0; i < src_ndim; ++i)
                idx_src += src.strides[i] * loc_src[i];

            // Compute index in destination storage
            py::ssize_t idx_dst = dst.offset;
            for (py::ssize_t i = 0; i < dst_ndim; ++i) {
                py::ssize_t p = fancy_axes[i];
                py::ssize_t src_coord_i;
                if (p == -1) 
                    src_coord_i = loc_dst_basic[i];
                else {
                    const TensorView& indexer = fancy_dims_data[p];
                    py::ssize_t flat_idx = indexer.offset;
                    for (size_t d = 0; d < indexer.shape.size(); ++d)
                        flat_idx += indexer.strides[d] * loc_dst_ind[d];
                    src_coord_i = indexers_data[p][flat_idx];
                }
                idx_dst += dst.strides[i] * src_coord_i;
            }

            dst_data[idx_dst] = src_data[idx_src];

            // Carry over
            py::ssize_t i = src_ndim - 1;
            while (i >= 0) {
                loc_src[i] += 1;
                if (loc_src[i] < src.shape[i]) break;
                loc_src[i] = 0;
                i -= 1;
            }
        }
    });
}

std::shared_ptr<Storage> gather_from_axes(
    const TensorView& x,
    const std::vector<py::ssize_t>& new_shape,
    const std::vector<py::ssize_t>& fancy_dims_in_src,
    const std::vector<TensorView>& fancy_dims_data,
    const std::vector<bool>& out_axis_is_fancy,
    const std::vector<py::ssize_t>& out_axis_target
) {
    // TODO (optim): optimize with chunk memcopy
    py::ssize_t src_ndim = x.shape.size();
    py::ssize_t ind_ndim = fancy_dims_data.empty() ? 0 : (py::ssize_t)fancy_dims_data[0].shape.size();
    py::ssize_t out_ndim = new_shape.size();
    py::ssize_t numel = numel_from_shape(new_shape);
    auto new_storage = Storage::allocate(numel, x.storage->dtype());

    auto indexers_data = std::vector<const int64_t*>(fancy_dims_data.size());
    for (size_t q = 0; q < indexers_data.size(); ++q) 
        indexers_data[q] = static_cast<const int64_t*>(fancy_dims_data[q].storage->data());

    auto fancy_axes = std::vector<py::ssize_t>(src_ndim, -1);
    for (py::ssize_t p = 0; p < fancy_dims_in_src.size(); ++p) fancy_axes[fancy_dims_in_src[p]] = p;

    auto loc_out = std::vector<py::ssize_t>(out_ndim);
    auto loc_src_basic = std::vector<py::ssize_t>(src_ndim);
    auto loc_ind = std::vector<py::ssize_t>(ind_ndim);
    return dispatch_dtype(x.storage->dtype(), [&]<typename T>() {
        auto old_data = static_cast<const T*>(x.storage->data());
        auto new_data = static_cast<T*>(new_storage->data());
        for (py::ssize_t idx_out = 0; idx_out < numel; ++idx_out) {

            // Set output loc wrt basic/fancy index combination
            for (py::ssize_t i = 0; i < out_ndim; ++i) {
                py::ssize_t src_dim = out_axis_target[i];
                if (out_axis_is_fancy[i])
                    loc_ind[src_dim] = loc_out[i];
                else
                    loc_src_basic[src_dim] = loc_out[i];
            }

            // Compute index in source storage
            py::ssize_t idx_src = x.offset;
            for (py::ssize_t i = 0; i < src_ndim; ++i) {
                py::ssize_t p = fancy_axes[i];
                py::ssize_t src_coord_i;
                if (p == -1) 
                    src_coord_i = loc_src_basic[i];
                else {
                    const TensorView& indexer = fancy_dims_data[p];
                    py::ssize_t flat_idx = indexer.offset;
                    for (size_t d = 0; d < indexer.shape.size(); ++d)
                        flat_idx += indexer.strides[d] * loc_ind[d];
                    src_coord_i = indexers_data[p][flat_idx];
                }
                idx_src += x.strides[i] * src_coord_i;
            }

            new_data[idx_out] = old_data[idx_src];

            // Carry over
            py::ssize_t i = out_ndim - 1;
            while (i >= 0) {
                loc_out[i] += 1;
                if (loc_out[i] < new_shape[i]) break;
                loc_out[i] = 0;
                i -= 1;
            }
        }
        return new_storage;
    });
}

void bind_ops_(py::module_& m) {
    m.def(
        "zeros", py::overload_cast<const std::vector<py::ssize_t>&, Dtype>(&zeros), 
        "Initialize a zeros-filled vector", py::arg("shape"), py::arg("dtype")
    );
    m.def(
        "ones", py::overload_cast<const std::vector<py::ssize_t>&, Dtype>(&ones), 
        "Initialize a ones-filled vector", py::arg("shape"), py::arg("dtype")
    );
    m.def(
        "full", py::overload_cast<const std::vector<py::ssize_t>&, Scalar, Dtype>(&full), 
        "Initialize a filled vector", py::arg("shape"), py::arg("value"), py::arg("dtype")
    );
    m.def(
        "eye", py::overload_cast<py::ssize_t, Dtype>(&eye), 
        "Initialize an eye matrix.", py::arg("n"), py::arg("dtype")
    );
    m.def(
        "arange", py::overload_cast<py::ssize_t, py::ssize_t, py::ssize_t, Dtype>(&arange), 
        "Initialize a range vector", py::arg("n"), py::arg("start"), py::arg("step"), py::arg("dtype")
    );
    m.def(
        "cast", [](const Storage& s, Dtype t) { return s.cast(t); }, 
        "Cast a storage to another dtype.", py::arg("storage"), py::arg("dtype")
    );
    m.def("sum", &sum, "Sum all elements in a tensor.", py::arg("x"));
    m.def("add", &add, "Add elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("subtract", &subtract, "Subtract elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("multiply", &multiply, "Multiply elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("divide", &divide, "Divide elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def(
        "equals", &equals, "Test the per-coef equality of two tensors.", py::arg("x1"), py::arg("x2")
    );
    m.def(
        "copy_view", &copy_view, "Copy source view into target view.", py::arg("src"), py::arg("dst")
    );
    m.def(
        "scatter_to_axes", &scatter_to_axes, "Scatter elements along fancy indices.", 
          py::arg("src"), py::arg("dst"), py::arg("fancy_dims_in_src"), py::arg("fancy_dims_data"), 
          py::arg("out_axis_is_fancy"), py::arg("out_axis_target")
    );
    m.def(
        "gather_from_axes", &gather_from_axes, "Extract elements along fancy indices.", 
          py::arg("x"), py::arg("new_sh"), py::arg("fancy_dims_in_src"), py::arg("fancy_dims_data"), 
          py::arg("out_axis_is_fancy"), py::arg("out_axis_target")
    );
}