#include "unary_ops.h"

#include "cuda.cuh"

#define DEFINE_UNARY(name, expr) \
    struct name##Op{ \
        template <class T> __host__ __device__ T operator()(T a) const { return expr; } \
    };

template <class Op>
std::shared_ptr<Storage> _cpu_unary_op_generic(const TensorView& x, Op op) {
    auto n_axes = static_cast<py::ssize_t>(x.shape.size());
    auto numel = numel_from_shape(x.shape);
    auto result = Storage::allocate(numel, x.storage->dtype(), x.storage->device());
    return dispatch_dtype(x.storage->dtype(), [&]<typename T>() {
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
    });
}

template <class T, class Op>
__global__ void _unary_kernel(
    const T* in, 
    T* out, 
    py::ssize_t n, 
    Op op, 
    StridedView view
) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto idx = view.offset;
    auto r = i;
    for (int j = view.n_axes - 1; j >= 0; --j) {
        auto coord = r % view.shape[j];
        r /= view.shape[j];
        idx += coord * view.strides[j];
    }
    out[i] = op(in[idx]);
}

template <class Op>
std::shared_ptr<Storage> _cuda_unary_op_generic(const TensorView& x, Op op) {
    auto n = numel_from_shape(x.shape);
    auto out = Storage::allocate(n, x.storage->dtype(), Device::Cuda);
    auto view = StridedView(x.shape.size(), x.offset);
    for (size_t j = 0; j < x.shape.size(); ++j) {
        view.shape[j] = x.shape[j];
        view.strides[j] = x.strides[j];
    }
    dispatch_dtype(x.storage->dtype(), [&]<class T>() {
        launch_1d(
            n, 
            _unary_kernel<T, Op>,
            static_cast<const T*>(x.storage->data()),
            static_cast<T*>(out->data()),
            n,
            op,
            view
        );
    });
    return out;
}

template <class Op>
std::shared_ptr<Storage> _dispatch_unary(const TensorView& x, Op op) {
    switch (x.storage->device()) {
        case Device::Cpu: return _cpu_unary_op_generic(x, op);
        case Device::Cuda: return _cuda_unary_op_generic(x, op);
    }
    NT_UNREACHABLE();
}

DEFINE_UNARY(Exp, std::exp(a))
std::shared_ptr<Storage> exp(const TensorView& x) {
    return _dispatch_unary(x, ExpOp());
}

DEFINE_UNARY(Log, std::log(a))
std::shared_ptr<Storage> log(const TensorView& x) {
    return _dispatch_unary(x, LogOp());
}

template <typename T>
struct PowOp{ 
    T value;
    __host__ __device__ T operator()(T a) const { return std::pow(a, value); } 
};
std::shared_ptr<Storage> pow(const TensorView& x, Scalar value) {
    return dispatch_dtype(x.storage->dtype(), [&]<class U>() {
        return _dispatch_unary(x, PowOp<U>{value.item<U>()});
    });
}

DEFINE_UNARY(Neg, -a)
std::shared_ptr<Storage> neg(const TensorView& x) {
    return _dispatch_unary(x, NegOp());
}

DEFINE_UNARY(Relu, (a >= 0) ? a : 0)
std::shared_ptr<Storage> relu(const TensorView& x) {
    return _dispatch_unary(x, ReluOp());
}

void bind_unary_ops_(py::module& m) {
    m.def("exp", [](const TensorView& x) { return exp(x); }, "Component-wise exponentiation.", py::arg("x"));
    m.def("log", [](const TensorView& x) { return log(x); }, "Component-wise log.", py::arg("x"));
    m.def("pow", [](const TensorView& x, Scalar a) { return pow(x, a); }, "Component-wise power.", py::arg("x"), py::arg("a"));
    m.def("neg", &neg, "Component-wise negation.", py::arg("x"));
    m.def("relu", &relu, "Rectified linear unit.", py::arg("x"));
}