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

std::shared_ptr<Storage> sum(
    const TensorView& x, const std::vector<py::ssize_t>& axis_drop, bool keepdim, Dtype dtype
) {
    auto n_axes = static_cast<py::ssize_t>(axis_drop.size());
    auto ndim_src = static_cast<py::ssize_t>(x.shape.size());
    auto ndim_dst = static_cast<py::ssize_t>(ndim_src - n_axes);

    // Compute both shapes
    auto axis_keep = std::vector<py::ssize_t>(ndim_dst);
    auto shape_drop = std::vector<py::ssize_t>(n_axes);
    auto shape_keep = std::vector<py::ssize_t>(ndim_dst);
    py::ssize_t posd = 0, posk = 0;
    for (size_t p = 0; p < ndim_src; ++p)
        if (std::find(axis_drop.begin(), axis_drop.end(), p) != axis_drop.end())
            shape_drop[posd++] = x.shape[p];
        else {
            axis_keep[posk] = p;
            shape_keep[posk] = x.shape[p];
            posk++;
        }

    auto numel_drop = numel_from_shape(shape_drop);
    auto numel_keep = numel_from_shape(shape_keep);
    auto dst_storage = Storage::allocate(numel_keep, dtype);

    return dispatch_dtype(x.storage->dtype(), [&]<typename T_src>() {
        return dispatch_dtype(dtype, [&]<typename T_dst>() {
            auto src_data = static_cast<const T_src*>(x.storage->data());
            auto dst_data = static_cast<T_dst*>(dst_storage->data());
            std::vector<py::ssize_t> loc_dst(ndim_dst);
            for (py::ssize_t i = 0; i < numel_keep; ++i) {
                // Sum over all src values to reduce
                T_dst acc = static_cast<T_dst>(0);
                std::vector<py::ssize_t> loc_src(n_axes);
                py::ssize_t offset = x.offset;
                for (py::ssize_t p = 0; p < ndim_dst; ++p) 
                    offset += x.strides[axis_keep[p]] * loc_dst[p];
                for (py::ssize_t j = 0; j < numel_drop; ++j) {
                    py::ssize_t idx_src = offset;
                    for (py::ssize_t p = 0; p < n_axes; ++p) 
                        idx_src += x.strides[axis_drop[p]] * loc_src[p];
                    acc += static_cast<T_dst>(src_data[idx_src]); 
                    // Carry over src
                    py::ssize_t k = n_axes - 1;
                    while (k >= 0) { 
                        loc_src[k] += 1;
                        if (loc_src[k] < x.shape[axis_drop[k]]) break;
                        loc_src[k] = 0;
                        k -= 1;
                    }
                }
                dst_data[i] = acc;
                // Cary over dst
                py::ssize_t k = ndim_dst - 1;
                while (k >= 0) { 
                    loc_dst[k] += 1;
                    if (loc_dst[k] < x.shape[axis_keep[k]]) break;
                    loc_dst[k] = 0;
                    k -= 1;
                }
            }
            return dst_storage;
        });
    });
}

template <typename F>
std::shared_ptr<Storage> _unary_op_generic(const TensorView& x, F&& func) {
    auto n_axes = static_cast<py::ssize_t>(x.shape.size());
    auto numel = numel_from_shape(x.shape);
    auto result = Storage::allocate(numel, x.storage->dtype());
    return dispatch_dtype(x.storage->dtype(), [&]<typename T>() {
        auto* data_in = static_cast<const T*>(x.storage->data());
        auto* data_out = static_cast<T*>(result->data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel; ++i) {
            py::ssize_t idx = x.offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) {
                idx += x.strides[j] * loc[j];
            }
            data_out[i] = func(data_in[idx]);

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
    const TensorView& x1, 
    const TensorView& x2, 
    F&& func, 
    std::optional<Dtype> out_dtype = std::nullopt
) {
    auto s1 = x1.storage;
    auto s2 = x2.storage;
    if (s1->dtype() != s2->dtype())
        throw std::invalid_argument(
            "_binary_op_generic: expected homogeneous tensors, got " + dtype_to_format(s1->dtype()) 
            +  " and " + dtype_to_format(s2->dtype()) + "."
        );
    if (x1.shape != x2.shape)
        throw std::invalid_argument(
            "_binary_op_generic: expected same size tensors, got " + vec_to_string(x1.shape)
            +  " and " + vec_to_string(x2.shape) + "."
        );

    auto n_axes = static_cast<py::ssize_t>(x1.shape.size());
    auto numel = numel_from_shape(x1.shape);
    auto eff_dtype = out_dtype.value_or(s1->dtype());
    auto new_storage = Storage::allocate(numel, eff_dtype);
    return dispatch_dtype(eff_dtype, [&]<typename O>() {
        return dispatch_dtype(s1->dtype(), [&]<typename T>() {
            auto* ptr1 = static_cast<const T*>(s1->data());
            auto* ptr2 = static_cast<const T*>(s2->data());
            auto* ptrout = static_cast<O*>(new_storage->data());
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

std::shared_ptr<Storage> pw_equal(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a == b; }, Dtype::Bool);
}

std::shared_ptr<Storage> pw_greater(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a > b; }, Dtype::Bool);
}

std::shared_ptr<Storage> pw_greater_eq(const TensorView& x1, const TensorView& x2) {
    return _binary_op_generic(x1, x2, []<class T>(T a, T b) { return a >= b; }, Dtype::Bool);
}


std::shared_ptr<Storage> matmul(const TensorView& x1, const TensorView& x2) {
    auto s1 = x1.storage;
    auto s2 = x2.storage;

    // Sanity checks
    if (s1->dtype() != s2->dtype())
        throw std::invalid_argument(
            "matmul: expected homogeneous tensors, got " + dtype_to_format(s1->dtype()) 
            +  " and " + dtype_to_format(s2->dtype()) + "."
        );
    if (x1.shape.size() != x2.shape.size())
        throw std::invalid_argument("matmul: x1 and x2 have different ndim.");
    if (x1.shape.size() < 2)
        throw std::invalid_argument("matmul: kernel should be called with >=2d tensors.");
        
    auto ndim = x1.shape.size();
    auto out_shape = std::vector<py::ssize_t>(ndim);
    for (auto i = 0; i < ndim - 2; ++i) {
        if (x1.shape[i] != x2.shape[i]) 
            throw std::invalid_argument("Invalid broacast shapes.");
        else out_shape[i] = x1.shape[i];
    }
    py::ssize_t lsize = x1.shape[ndim - 2];
    py::ssize_t csize = x1.shape[ndim - 1];
    py::ssize_t rsize = x2.shape[ndim - 1];
    if (csize != x2.shape[ndim - 2])
        throw std::invalid_argument("Invalid matmul shape.");
    out_shape[ndim - 2] = lsize;
    out_shape[ndim - 1] = rsize;

    auto numel = numel_from_shape(out_shape);
    auto new_storage = Storage::allocate(numel, x1.storage->dtype());
    std::vector<py::ssize_t> loc(ndim);
    return dispatch_dtype(s1->dtype(), [&]<typename T>() {
        auto* ptr1 = static_cast<const T*>(s1->data());
        auto* ptr2 = static_cast<const T*>(s2->data());
        auto* prout = static_cast<T*>(new_storage->data());
        for (py::ssize_t prout_idx = 0; prout_idx < numel; ++prout_idx) {
            // B, i, j -> derive B, i, c and B, c, j
            T acc = static_cast<T>(0.0);
            for (int c = 0; c < csize; ++c) {
                py::ssize_t idx1 = x1.offset 
                                    + c * x1.strides[ndim - 1] 
                                    + loc[ndim - 2] * x1.strides[ndim - 2];
                py::ssize_t idx2 = x2.offset 
                                    + c * x2.strides[ndim - 2]
                                    + loc[ndim - 1] * x2.strides[ndim - 1];
                for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
                    idx1 += x1.strides[dim_i] * loc[dim_i];
                    idx2 += x2.strides[dim_i] * loc[dim_i];
                }
                acc += ptr1[idx1] * ptr2[idx2];
            }
            prout[prout_idx] = acc;

            py::ssize_t j = ndim - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < out_shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
        return new_storage;
    });
}

std::shared_ptr<Storage> exp(const TensorView& x) {
    return _unary_op_generic(x, []<class T>(T a) { return std::exp(a); });
}

std::shared_ptr<Storage> log(const TensorView& x) {
    return _unary_op_generic(x, []<class T>(T a) { return std::log(a); });
}

std::shared_ptr<Storage> pow(const TensorView& x, Scalar value) {
    return _unary_op_generic(x, [value]<class T>(T a) { return std::pow(a, value.item<T>()); });
}

std::shared_ptr<Storage> neg(const TensorView& x) {
    return _unary_op_generic(x, []<class T>(T a) { return -a; });
}

std::shared_ptr<Storage> relu(const TensorView& x) {
    return _unary_op_generic(x, []<class T>(T a) { return a > 0 ? a : 0; });
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

// Inplace

template<typename F>
void _inplace_apply(TensorView& out, const TensorView& other, F&& func) {
    if (out.shape != other.shape) {
        throw std::invalid_argument("_inplace_apply: Shape mismatch.");
    }
    auto n_axes = static_cast<py::ssize_t>(out.shape.size());
    auto numel = numel_from_shape(out.shape);
    dispatch_dtype(out.storage->dtype(), [&]<typename T>() {
        auto* data_other = static_cast<const T*>(other.storage->data());
        auto* data_out = static_cast<T*>(out.storage->data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel; ++i) {

            py::ssize_t idx_out = out.offset;
            py::ssize_t idx_other = other.offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) {
                idx_out += out.strides[j] * loc[j];
                idx_other += other.strides[j] * loc[j];
            }
            data_out[idx_out] = func(data_out[idx_out], data_other[idx_other]);

            // Loc update
            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < out.shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
    });
}

void add_inplace(TensorView& out, const TensorView& other) {
    _inplace_apply(out, other, []<typename T>(T a, T b) { return a + b; });
}
void sub_inplace(TensorView& out, const TensorView& other) {
    _inplace_apply(out, other, []<typename T>(T a, T b) { return a - b; });
}
void mul_inplace(TensorView& out, const TensorView& other) {
    _inplace_apply(out, other, []<typename T>(T a, T b) { return a * b; });
}
void div_inplace(TensorView& out, const TensorView& other) {
    _inplace_apply(out, other, []<typename T>(T a, T b) { return a / b; });
}
void copy_inplace(TensorView& out, const TensorView& other) {
    _inplace_apply(out, other, []<typename T>(T a, T b) { return b; });
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
    m.def("sum", &sum, "Sum all elements in a tensor.", py::arg("x"), py::arg("axis"), py::arg("keepdim"), py::arg("dtype"));
    m.def("add", &add, "Add elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("subtract", &subtract, "Subtract elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("multiply", &multiply, "Multiply elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("divide", &divide, "Divide elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("matmul", &matmul, "Matrix multiplication.", py::arg("x1"), py::arg("x2"));
    m.def("exp", [](const TensorView& x) { return exp(x); }, "Component-wise exponentiation.", py::arg("x"));
    m.def("log", [](const TensorView& x) { return log(x); }, "Component-wise log.", py::arg("x"));
    m.def("pow", [](const TensorView& x, Scalar a) { return pow(x, a); }, "Component-wise power.", py::arg("x"), py::arg("a"));
    m.def("neg", &neg, "Component-wise negation.", py::arg("x"));
    m.def("relu", &relu, "Rectified linear unit.", py::arg("x"));
    m.def("pw_greater", &pw_greater, "Per-coefficient strictly greater comparison.", py::arg("x"), py::arg("s"));
    m.def("pw_greater_eq", &pw_greater_eq, "Per-coefficient greater/equals comparison.", py::arg("x"), py::arg("s"));
    m.def("pw_equal", &pw_equal, "Per-coefficient equal comparison.", py::arg("x"), py::arg("s"));
    m.def(
        "equals", &equals, "Test the per-coef equality of two tensors.", py::arg("x1"), py::arg("x2")
    );
    m.def("add_inplace", &add_inplace, "A dd elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("sub_inplace", &sub_inplace, "Subtract elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("mul_inplace", &mul_inplace, "Multiply elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("div_inplace", &div_inplace, "Divide elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("copy_inplace", &copy_inplace, "Copy elements inplace.", py::arg("x1"), py::arg("x2"));
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