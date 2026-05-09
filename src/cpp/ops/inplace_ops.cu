#include "inplace_ops.h"
#include "cuda.cuh"

// Inplace

template<class T, class Op>
void _cpu_inplace_apply(TensorView& out, const TensorView& other, Op op) {
    auto n_axes = static_cast<py::ssize_t>(out.shape.size());
    auto numel = numel_from_shape(out.shape);
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
        data_out[idx_out] = op(data_out[idx_out], data_other[idx_other]);

        // Loc update
        py::ssize_t j = n_axes - 1;
        while (j >= 0) { 
            loc[j] += 1;
            if (loc[j] < out.shape[j]) break;
            loc[j] = 0;
            j -= 1;
        }
    }
}

template <class T, class Op>
__global__ void _inplace_apply_kernel(
    py::ssize_t nmax, 
    T* out, 
    const T* other, 
    StridedView view_out,
    StridedView view_other,
    Op op
) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nmax) return;
    auto idx_out = unravel(i, view_out);
    out[idx_out] = op(out[idx_out], other[unravel(i, view_other)]);
}

template <class T, class Op>
void _cuda_inplace_apply(
    const TensorView& out, const TensorView& other, Op op
) {
    auto n = numel_from_shape(out.shape);
    auto view_other = StridedView(other.shape.size(), other.offset);
    for (size_t j = 0; j < other.shape.size(); ++j) {
        view_other.shape[j] = other.shape[j];
        view_other.strides[j] = other.strides[j];
    }
    auto view_out = StridedView(out.shape.size(), out.offset);
    for (size_t j = 0; j < out.shape.size(); ++j) {
        view_out.shape[j] = out.shape[j];
        view_out.strides[j] = out.strides[j];
    }
    launch_1d(
        n, 
        _inplace_apply_kernel<T, Op>,
        n,
        static_cast<T*>(out.storage->data()),
        static_cast<const T*>(other.storage->data()),
        view_out,
        view_other,
        op
    );
}

template <class Dispatch, class Op>
void _dispatch_inplace_apply(TensorView& out, const TensorView& other, Op op) {
    if (out.shape != other.shape) {
        throw std::invalid_argument("_dispatch_inplace_apply: Shape mismatch.");
    }
    if (out.storage->device() != other.storage->device()) {
        throw std::invalid_argument("_dispatch_inplace_apply: Device mismatch.");
    }
    if (out.storage->dtype() != other.storage->dtype()) {
        throw std::invalid_argument("_dispatch_inplace_apply: Dtype mismatch.");
    }
    Dispatch::run(out.storage->dtype(), [&]<typename T>() {
        switch (out.storage->device()) {
            case Device::Cpu:  _cpu_inplace_apply<T>(out, other, op); break;
            case Device::Cuda: _cuda_inplace_apply<T>(out, other, op); break;
        }
    });
    out.storage->bump_version();
}

#define DEFINE_INPLACE(name, expr) \
    struct name##Op{ \
        template <class T> __host__ __device__ T operator()(T a, T b) const { return expr; } \
    };

DEFINE_INPLACE(Add, a + b)
void add_inplace(TensorView& out, const TensorView& other) {
    _dispatch_inplace_apply<DispatchAll>(out, other, AddOp());
}

DEFINE_INPLACE(Subtract, a - b)
void sub_inplace(TensorView& out, const TensorView& other) {
    _dispatch_inplace_apply<DispatchAll>(out, other, SubtractOp());
}

DEFINE_INPLACE(Multiply, a * b)
void mul_inplace(TensorView& out, const TensorView& other) {
    _dispatch_inplace_apply<DispatchAll>(out, other, MultiplyOp());
}

DEFINE_INPLACE(Divide, a / b)
void div_inplace(TensorView& out, const TensorView& other) {
    _dispatch_inplace_apply<DispatchArithmetic>(out, other, DivideOp());
}

DEFINE_INPLACE(Copy, b)
void copy_inplace(TensorView& out, const TensorView& other) {
    _dispatch_inplace_apply<DispatchAll>(out, other, CopyOp());
}

void bind_inplace_ops_(py::module_& m) {
    m.def("add_inplace", &add_inplace, "A dd elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("sub_inplace", &sub_inplace, "Subtract elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("mul_inplace", &mul_inplace, "Multiply elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("div_inplace", &div_inplace, "Divide elements inplace.", py::arg("x1"), py::arg("x2"));
    m.def("copy_inplace", &copy_inplace, "Copy elements inplace.", py::arg("x1"), py::arg("x2"));
}