#include "binary_ops.h"
#include "cuda.cuh"

template <class T, class O, class F>
std::shared_ptr<Storage> _cpu_binary_op_generic(
    const TensorView& x1, 
    const TensorView& x2, 
    F&& func, 
    Dtype out_dtype
) {
    auto n_axes = static_cast<py::ssize_t>(x1.shape.size());
    auto numel = numel_from_shape(x1.shape);
    auto* ptr_in1 = static_cast<const T*>(x1.storage->data());
    auto* ptr_in2 = static_cast<const T*>(x2.storage->data());
    auto storage_out = Storage::allocate(numel, out_dtype, x1.storage->device());
    auto* ptr_out = static_cast<O*>(storage_out->data());
    std::vector<py::ssize_t> loc(n_axes);
    for (py::ssize_t i = 0; i < numel; ++i) {
        py::ssize_t idx1 = x1.offset, idx2 = x2.offset;
        for (py::ssize_t j = 0; j < n_axes; ++j) {
            idx1 += x1.strides[j] * loc[j];
            idx2 += x2.strides[j] * loc[j];
        }
        ptr_out[i] = static_cast<O>(func(ptr_in1[idx1], ptr_in2[idx2]));

        // Loc update
        py::ssize_t j = n_axes - 1;
        while (j >= 0) { 
            loc[j] += 1;
            if (loc[j] < x1.shape[j]) break;
            loc[j] = 0;
            j -= 1;
        }
    }
    return storage_out;
}

template <class T, class O, class Op>
__global__ void _binary_kernel(
    const T* ptr_in1, 
    const T* ptr_in2, 
    TensorViewStatic view_1,
    TensorViewStatic view_2,
    O* ptr_out, 
    py::ssize_t n, 
    Op op
) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto idx_a = unravel(i, view_1);
    auto idx_b = unravel(i, view_2);
    ptr_out[i] = static_cast<O>(op(ptr_in1[idx_a], ptr_in2[idx_b]));
}

template <class T, class O, class Op>
std::shared_ptr<Storage> _cuda_binary_op_generic(
    const TensorView& x1, 
    const TensorView& x2,
    Op op,
    Dtype out_dtype
) {
    auto n = numel_from_shape(x1.shape);
    auto storage_out = Storage::allocate(n, out_dtype, Device::Cuda);
    launch_1d(
        n, 
        _binary_kernel<T, O, Op>,
        static_cast<const T*>(x1.storage->data()),
        static_cast<const T*>(x2.storage->data()),
        tensor_view_to_static(x1),
        tensor_view_to_static(x2),
        static_cast<O*>(storage_out->data()),
        n,
        op
    );
    return storage_out;
}

template <class Dispatch, class Op>
std::shared_ptr<Storage> _dispatch_binary(
    const TensorView& x1, 
    const TensorView& x2, 
    Op op,
    std::optional<Dtype> out_dtype = std::nullopt
) {
    _require_same_shape(x1, x2, "_dispatch_binary");
    _require_same_device(x1.storage, x2.storage, "_dispatch_binary");
    _require_same_dtype(x1.storage, x2.storage, "_dispatch_binary");

    auto eff_dtype = out_dtype.value_or(x1.storage->dtype());
    return Dispatch::run(eff_dtype, [&]<class O>() {
        return Dispatch::run(x1.storage->dtype(), [&]<class T>() {
            switch (x1.storage->device()) {
                case Device::Cpu: return _cpu_binary_op_generic<T, O>(
                    x1, x2, op, eff_dtype
                );
                case Device::Cuda: return _cuda_binary_op_generic<T, O>(
                    x1, x2, op, eff_dtype
                );
                default: NT_UNREACHABLE();
            }
        });
    });
}

#define DEFINE_BINARY(name, expr) \
    struct name##Op{ \
        template <class T> __host__ __device__ T operator()(T a, T b) const { return expr; } \
    };

DEFINE_BINARY(Add, a + b)
std::shared_ptr<Storage> add(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, AddOp());
}

DEFINE_BINARY(Sub, a - b)
std::shared_ptr<Storage> subtract(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, SubOp());
}

DEFINE_BINARY(Mul, a * b)
std::shared_ptr<Storage> multiply(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, MulOp());
}

DEFINE_BINARY(Div, a / b)
std::shared_ptr<Storage> divide(const TensorView& x1, const TensorView& x2) {
    return _dispatch_binary<DispatchAll>(x1, x2, DivOp());
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