#include "unary_ops.h"
#include "cuda.cuh"



template <typename Dispatch, typename F>
std::shared_ptr<Storage> _cpu_binary_op_generic(
    const TensorView& x1, 
    const TensorView& x2, 
    F&& func, 
    std::optional<Dtype> out_dtype = std::nullopt
) {
    auto n_axes = static_cast<py::ssize_t>(x1.shape.size());
    auto numel = numel_from_shape(x1.shape);
    auto eff_dtype = out_dtype.value_or(x1.storage->dtype());
    auto new_storage = Storage::allocate(numel, eff_dtype, x1.storage->device());
    return Dispatch::run(eff_dtype, [&]<typename O>() {
        return Dispatch::run(x1.storage->dtype(), [&]<typename T>() {
            auto* ptr1 = static_cast<const T*>(x1.storage->data());
            auto* ptr2 = static_cast<const T*>(x2.storage->data());
            auto* ptrout = static_cast<O*>(new_storage->data());
            std::vector<py::ssize_t> loc(n_axes);
            for (py::ssize_t i = 0; i < numel; ++i) {
                py::ssize_t idx1 = x1.offset, idx2 = x2.offset;
                for (py::ssize_t j = 0; j < n_axes; ++j) {
                    idx1 += x1.strides[j] * loc[j];
                    idx2 += x2.strides[j] * loc[j];
                }
                ptrout[i] = static_cast<O>(func(ptr1[idx1], ptr2[idx2]));

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

template <class TIn, class TOut, class Op>
__global__ void _binary_kernel(
    const TIn* in_a, 
    const TIn* in_b, 
    StridedView view_a,
    StridedView view_b,
    TOut* out, 
    py::ssize_t n, 
    Op op
) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    auto idx_a = view_a.offset;
    auto r = i;
    for (int j = view_a.n_axes - 1; j >= 0; --j) {
        auto coord = r % view_a.shape[j];
        r /= view_a.shape[j];
        idx_a += coord * view_a.strides[j];
    }

    auto idx_b = view_b.offset;
    r = i;
    for (int j = view_b.n_axes - 1; j >= 0; --j) {
        auto coord = r % view_b.shape[j];
        r /= view_b.shape[j];
        idx_b += coord * view_b.strides[j];
    }

    out[i] = static_cast<TOut>(op(in_a[idx_a], in_b[idx_b]));
}

template <class Dispatch, class Op>
std::shared_ptr<Storage> _cuda_binary_op_generic(
    const TensorView& a, 
    const TensorView& b,
    Op op,
    std::optional<Dtype> out_dtype = std::nullopt
) {
    auto n = numel_from_shape(a.shape);
    auto eff_dtype = out_dtype.value_or(a.storage->dtype());
    auto out = Storage::allocate(n, eff_dtype, Device::Cuda);

    auto view_a = StridedView(a.shape.size(), a.offset);
    for (size_t j = 0; j < a.shape.size(); ++j) {
        view_a.shape[j] = a.shape[j];
        view_a.strides[j] = a.strides[j];
    }
    auto view_b = StridedView(b.shape.size(), b.offset);
    for (size_t j = 0; j < b.shape.size(); ++j) {
        view_b.shape[j] = b.shape[j];
        view_b.strides[j] = b.strides[j];
    }
    Dispatch::run(eff_dtype, [&]<class O>() {
        Dispatch::run(a.storage->dtype(), [&]<class T>() {
            launch_1d(
                n, 
                _binary_kernel<T, O, Op>,
                static_cast<const T*>(a.storage->data()),
                static_cast<const T*>(b.storage->data()),
                view_a,
                view_b,
                static_cast<O*>(out->data()),
                n,
                op
            );
        });
    });
    return out;
}

template <class Dispatch, class Op>
std::shared_ptr<Storage> _dispatch_binary(
    const TensorView& a, 
    const TensorView& b, 
    Op op,
    std::optional<Dtype> out_dtype = std::nullopt
) {
    auto s1 = a.storage;
    auto s2 = b.storage;
    if (s1->dtype() != s2->dtype())
        throw std::invalid_argument(
            "_dispatch_binary: expected homogeneous tensors, got " + dtype_to_format(s1->dtype()) 
            +  " and " + dtype_to_format(s2->dtype()) + "."
        );
    if (s1->device() != s2->device())
        throw std::invalid_argument("_dispatch_binary: expected tensors on same device.");
    if (a.shape != b.shape)
        throw std::invalid_argument(
            "_dispatch_binary: expected same size tensors, got " + vec_to_string(a.shape)
            +  " and " + vec_to_string(b.shape) + "."
        );
    switch (s1->device()) {
        case Device::Cpu: return _cpu_binary_op_generic<Dispatch>(a, b, op, out_dtype);
        case Device::Cuda: return _cuda_binary_op_generic<Dispatch>(a, b, op, out_dtype);
        default: NT_UNREACHABLE();
    }
    return {};
}

#define DEFINE_BINARY(name, expr) \
    struct name##Op{ \
        template <class T> __host__ __device__ T operator()(T a, T b) const { return expr; } \
    };

DEFINE_BINARY(Add, a + b)
std::shared_ptr<Storage> add(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, AddOp());
}

DEFINE_BINARY(Subtract, a - b)
std::shared_ptr<Storage> subtract(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, SubtractOp());
}

DEFINE_BINARY(Multiply, a * b)
std::shared_ptr<Storage> multiply(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, MultiplyOp());
}

DEFINE_BINARY(Divide, a / b)
std::shared_ptr<Storage> divide(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, DivideOp());
}

DEFINE_BINARY(PwEqual, a == b)
std::shared_ptr<Storage> pw_equal(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, PwEqualOp(), Dtype::Bool);
}

DEFINE_BINARY(PwGreater, a > b)
std::shared_ptr<Storage> pw_greater(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, PwGreaterOp(), Dtype::Bool);
}

DEFINE_BINARY(PwGreaterEq, a >= b)
std::shared_ptr<Storage> pw_greater_eq(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, PwGreaterEqOp(), Dtype::Bool);
}

void bind_binary_ops_(py::module& m) {
    m.def("add", &add, "Add elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("subtract", &subtract, "Subtract elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("multiply", &multiply, "Multiply elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("divide", &divide, "Divide elements pointwise.", py::arg("x1"), py::arg("x2"));
    m.def("pw_greater", &pw_greater, "Per-coefficient strictly greater comparison.", py::arg("x"), py::arg("s"));
    m.def("pw_greater_eq", &pw_greater_eq, "Per-coefficient greater/equals comparison.", py::arg("x"), py::arg("s"));
    m.def("pw_equal", &pw_equal, "Per-coefficient equal comparison.", py::arg("x"), py::arg("s"));
}