#include "cuda.cuh"
#include "inplace_ops.h"

// Inplace

template <class T, class Op>
void _cpu_inplace_apply(TensorView &out, const TensorView &other, Op op) {
  auto ndim = static_cast<py::ssize_t>(out.shape.size());
  auto numel = numel_from_shape(out.shape);
  auto *ptr_other = static_cast<const T *>(other.storage->data());
  auto *ptr_out = static_cast<T *>(out.storage->data());
  for (py::ssize_t i = 0; i < numel; ++i) {
    py::ssize_t idx_out = unravel(i, out);
    py::ssize_t idx_other = unravel(i, other);
    ptr_out[idx_out] = op(ptr_out[idx_out], ptr_other[idx_other]);
  }
}

template <class T, class Op>
__global__ void _inplace_apply_kernel(py::ssize_t n, T *ptr_out,
                                      const T *ptr_other,
                                      TensorViewStatic view_out,
                                      TensorViewStatic view_other, Op op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  auto idx_out = unravel(i, view_out);
  ptr_out[idx_out] = op(ptr_out[idx_out], ptr_other[unravel(i, view_other)]);
}

template <class T, class Op>
void _cuda_inplace_apply(const TensorView &out, const TensorView &other,
                         Op op) {
  auto n = numel_from_shape(out.shape);
  launch_1d(n, _inplace_apply_kernel<T, Op>, n,
            static_cast<T *>(out.storage->data()),
            static_cast<const T *>(other.storage->data()),
            tensor_view_to_static(out), tensor_view_to_static(other), op);
}

template <class Dispatch, class Op>
void _dispatch_inplace_apply(TensorView &out, const TensorView &other, Op op) {
  _require_same_device(out.storage, other.storage, "_dispatch_inplace_apply");
  _require_same_dtype(out.storage, other.storage, "_dispatch_inplace_apply");
  _require_same_shape(out, other, "_dispatch_inplace_apply");
  Dispatch::run(out.storage->dtype(), [&]<class T>() {
    switch (out.storage->device()) {
    case Device::Cpu:
      _cpu_inplace_apply<T>(out, other, op);
      break;
    case Device::Cuda:
      _cuda_inplace_apply<T>(out, other, op);
      break;
    default:
      NT_UNREACHABLE();
    }
  });
  out.storage->bump_version();
}

#define DEFINE_INPLACE(name, expr)                                             \
  struct name##Op {                                                            \
    template <class T> __host__ __device__ T operator()(T a, T b) const {      \
      return expr;                                                             \
    }                                                                          \
  };

DEFINE_INPLACE(Add, a + b)
void add_inplace(TensorView &out, const TensorView &other) {
  _dispatch_inplace_apply<DispatchAll>(out, other, AddOp());
}

DEFINE_INPLACE(Sub, a - b)
void sub_inplace(TensorView &out, const TensorView &other) {
  _dispatch_inplace_apply<DispatchAll>(out, other, SubOp());
}

DEFINE_INPLACE(Mul, a *b)
void mul_inplace(TensorView &out, const TensorView &other) {
  _dispatch_inplace_apply<DispatchAll>(out, other, MulOp());
}

DEFINE_INPLACE(Div, a / b)
void div_inplace(TensorView &out, const TensorView &other) {
  _dispatch_inplace_apply<DispatchArithmetic>(out, other, DivOp());
}

DEFINE_INPLACE(Copy, b)
void copy_inplace(TensorView &out, const TensorView &other) {
  _dispatch_inplace_apply<DispatchAll>(out, other, CopyOp());
}

void bind_inplace_ops_(py::module_ &m) {
  m.def("add_inplace", &add_inplace, "Add elements inplace.", py::arg("x1"),
        py::arg("x2"));
  m.def("sub_inplace", &sub_inplace, "Subtract elements inplace.",
        py::arg("x1"), py::arg("x2"));
  m.def("mul_inplace", &mul_inplace, "Multiply elements inplace.",
        py::arg("x1"), py::arg("x2"));
  m.def("div_inplace", &div_inplace, "Divide elements inplace.", py::arg("x1"),
        py::arg("x2"));
  m.def("copy_inplace", &copy_inplace, "Copy elements inplace.", py::arg("x1"),
        py::arg("x2"));
}