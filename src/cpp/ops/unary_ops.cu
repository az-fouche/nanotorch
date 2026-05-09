#include "unary_ops.h"
#include "cuda.cuh"

// Sum

template <class T, class O>
std::shared_ptr<Storage> _cpu_sum(
    const TensorView& input, 
    Dtype out_dtype,
    py::ssize_t n_axes,
    py::ssize_t ndim_out,
    const std::vector<py::ssize_t>& axis_keep,
    const std::vector<py::ssize_t>& axis_drop, 
    py::ssize_t numel_keep,
    py::ssize_t numel_drop
) {
    auto out_storage = Storage::allocate(numel_keep, out_dtype, Device::Cpu);
    auto in_data = static_cast<const T*>(input.storage->data());
    auto out_data = static_cast<O*>(out_storage->data());
    std::vector<py::ssize_t> loc_out(ndim_out);
    for (py::ssize_t i = 0; i < numel_keep; ++i) {
        // Sum over all in values to reduce
        O acc = static_cast<O>(0);
        std::vector<py::ssize_t> loc_in(n_axes);
        py::ssize_t offset = input.offset;
        for (py::ssize_t p = 0; p < ndim_out; ++p) 
            offset += input.strides[axis_keep[p]] * loc_out[p];
        for (py::ssize_t j = 0; j < numel_drop; ++j) {
            py::ssize_t idx_in = offset;
            for (py::ssize_t p = 0; p < n_axes; ++p) 
                idx_in += input.strides[axis_drop[p]] * loc_in[p];
            acc += static_cast<O>(in_data[idx_in]); 
            // Carry over in
            py::ssize_t k = n_axes - 1;
            while (k >= 0) { 
                loc_in[k] += 1;
                if (loc_in[k] < input.shape[axis_drop[k]]) break;
                loc_in[k] = 0;
                k -= 1;
            }
        }
        out_data[i] = acc;
        // Carry over out
        py::ssize_t k = ndim_out - 1;
        while (k >= 0) { 
            loc_out[k] += 1;
            if (loc_out[k] < input.shape[axis_keep[k]]) break;
            loc_out[k] = 0;
            k -= 1;
        }
    }
    return out_storage;
}

template <class T, class O>
__global__ void _sum_kernel(
    py::ssize_t nmax, 
    const T* in_storage, 
    O* out_storage, 
    TensorViewStatic view_keep,
    TensorViewStatic view_drop,
    py::ssize_t numel_drop
) {
    // TODO: optimize the parallelization
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nmax) return;
    py::ssize_t base = unravel(i, view_keep);
    O acc = 0;
    for (py::ssize_t j = 0; j < numel_drop; ++j)
        acc += static_cast<O>(in_storage[base + unravel(j, view_drop)]);
    out_storage[i] = acc;
}

template <class T, class O>
std::shared_ptr<Storage> _cuda_sum(
    const TensorView& input, 
    Dtype out_dtype,
    py::ssize_t n_axes,
    py::ssize_t ndim_out,
    const std::vector<py::ssize_t>& axis_keep,
    const std::vector<py::ssize_t>& axis_drop, 
    py::ssize_t numel_keep,
    py::ssize_t numel_drop
) {
    auto view_keep = TensorViewStatic(axis_keep.size(), input.offset);
    for (size_t p = 0; p < axis_keep.size(); ++p) {
        view_keep.shape[p] = input.shape[axis_keep[p]];
        view_keep.strides[p] = input.strides[axis_keep[p]];
    }
    auto view_drop = TensorViewStatic(axis_drop.size(), 0); // offset already counted in view_keep
    for (size_t p = 0; p < axis_drop.size(); ++p) {
        view_drop.shape[p] = input.shape[axis_drop[p]];
        view_drop.strides[p] = input.strides[axis_drop[p]];
    }
    auto out = Storage::allocate(numel_keep, out_dtype, Device::Cuda);
    launch_1d(
        numel_keep,
        _sum_kernel<T, O>,
        numel_keep,
        static_cast<const T*>(input.storage->data()),
        static_cast<O*>(out->data()),
        view_keep,
        view_drop,
        numel_drop
    );
    return out;
}

std::shared_ptr<Storage> sum(
    const TensorView& x, const std::vector<py::ssize_t>& axis_drop, Dtype dtype
) {
    // Compute both shapes
    auto n_axes = static_cast<py::ssize_t>(axis_drop.size());
    auto ndim_in = static_cast<py::ssize_t>(x.shape.size());
    auto ndim_out = static_cast<py::ssize_t>(ndim_in - n_axes);
    auto axis_keep = std::vector<py::ssize_t>(ndim_out);
    auto shape_drop = std::vector<py::ssize_t>(n_axes);
    auto shape_keep = std::vector<py::ssize_t>(ndim_out);
    py::ssize_t posd = 0, posk = 0;
    for (py::ssize_t p = 0; p < ndim_in; ++p)
        if (std::find(axis_drop.begin(), axis_drop.end(), p) != axis_drop.end())
            shape_drop[posd++] = x.shape[p];
        else {
            axis_keep[posk] = p;
            shape_keep[posk] = x.shape[p];
            posk++;
        }
    auto numel_drop = numel_from_shape(shape_drop);
    auto numel_keep = numel_from_shape(shape_keep);
    return dispatch_dtype(x.storage->dtype(), [&]<typename T>() {
        return DispatchArithmetic::run(dtype, [&]<typename O>() {
            switch (x.storage->device()) {
                case Device::Cpu: return _cpu_sum<T, O>(
                    x, dtype, n_axes, ndim_out, axis_keep, axis_drop, 
                    numel_keep, numel_drop
                );
                case Device::Cuda: return _cuda_sum<T, O>(
                    x, dtype, n_axes, ndim_out, axis_keep, axis_drop, 
                    numel_keep, numel_drop
                );
                default: NT_UNREACHABLE();
            }
        });
    });

}

// Unary ops

template <class T, class Op>
std::shared_ptr<Storage> _cpu_unary_op_generic(const TensorView& x, Op op) {
    auto n_axes = static_cast<py::ssize_t>(x.shape.size());
    auto numel = numel_from_shape(x.shape);
    auto result = Storage::allocate(numel, x.storage->dtype(), x.storage->device());
    auto* data_in = static_cast<const T*>(x.storage->data());
    auto* data_out = static_cast<T*>(result->data());
    std::vector<py::ssize_t> loc(n_axes);
    for (py::ssize_t i = 0; i < numel; ++i) {
        py::ssize_t idx = x.offset;
        for (py::ssize_t j = 0; j < n_axes; ++j) {
            idx += x.strides[j] * loc[j];
        }
        data_out[i] = op(data_in[idx]);

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
}

template <class T, class Op>
__global__ void _unary_kernel(
    const T* in, 
    T* out, 
    py::ssize_t n, 
    Op op, 
    TensorViewStatic view
) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto idx = unravel(i, view);
    out[i] = op(in[idx]);
}

template <class T, class Op>
std::shared_ptr<Storage> _cuda_unary_op_generic(const TensorView& x, Op op) {
    auto n = numel_from_shape(x.shape);
    auto out = Storage::allocate(n, x.storage->dtype(), Device::Cuda);
    launch_1d(
        n, 
        _unary_kernel<T, Op>,
        static_cast<const T*>(x.storage->data()),
        static_cast<T*>(out->data()),
        n,
        op,
        tensor_view_to_static(x)
    );
    return out;
}

template <class Dispatch, class Op>
std::shared_ptr<Storage> _dispatch_unary(const TensorView& x, Op op) {
    return Dispatch::run(x.storage->dtype(), [&]<class T>() {
        switch (x.storage->device()) {
            case Device::Cpu: return _cpu_unary_op_generic<T>(x, op);
            case Device::Cuda: return _cuda_unary_op_generic<T>(x, op);
            default: NT_UNREACHABLE();
        }
    });
}

#define DEFINE_UNARY(name, expr) \
    struct name##Op{ \
        template <class T> __host__ __device__ T operator()(T a) const { return expr; } \
    };


DEFINE_UNARY(Exp, std::exp(a))
std::shared_ptr<Storage> exp(const TensorView& x) {
    return _dispatch_unary<DispatchFloat>(x, ExpOp());
}

DEFINE_UNARY(Log, std::log(a))
std::shared_ptr<Storage> log(const TensorView& x) {
    return _dispatch_unary<DispatchFloat>(x, LogOp());
}

template <typename T>
__host__ __device__ T int_pow(T a, T n) {
    T r = 1;
    while (n > 0) { if (n & 1) r *= a; a *= a; n >>= 1; }
    return r;
}
template <typename T>
struct PowOp{ 
    T value;
    __host__ __device__ T operator()(T a) const { 
        if constexpr (std::is_integral_v<T>) return int_pow(a, value);
        else return std::pow(a, value); 
    } 
};
std::shared_ptr<Storage> pow(const TensorView& x, Scalar value) {
    return dispatch_dtype(x.storage->dtype(), [&]<class U>() {
        return _dispatch_unary<DispatchArithmetic>(x, PowOp<U>{value.item<U>()});
    });
}

DEFINE_UNARY(Neg, -a)
std::shared_ptr<Storage> neg(const TensorView& x) {
    return _dispatch_unary<DispatchArithmetic>(x, NegOp());
}

DEFINE_UNARY(Relu, (a >= 0) ? a : 0)
std::shared_ptr<Storage> relu(const TensorView& x) {
    return _dispatch_unary<DispatchArithmetic>(x, ReluOp());
}

void bind_unary_ops_(py::module& m) {
    m.def("sum", &sum, "Sum all elements in a tensor.", py::arg("x"), py::arg("axis"), py::arg("dtype"));
    m.def(
        "exp", [](const TensorView& x) { return exp(x); }, 
        "Component-wise exponentiation.", py::arg("x")
    );
    m.def(
        "log", [](const TensorView& x) { return log(x); }, 
        "Component-wise log.", py::arg("x")
    );
    m.def(
        "pow", [](const TensorView& x, Scalar a) { return pow(x, a); }, 
        "Component-wise power.", py::arg("x"), py::arg("a")
    );
    m.def("neg", &neg, "Component-wise negation.", py::arg("x"));
    m.def("relu", &relu, "Rectified linear unit.", py::arg("x"));
}