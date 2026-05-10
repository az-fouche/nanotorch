#include "cuda.cuh"
#include "unary_ops.h"

// Unary ops

template <class T, class Op>
std::shared_ptr<Storage> _cpu_unary_op_generic(const TensorView &x, Op op) {
  auto numel = numel_from_shape(x.shape);
  auto storage_out =
      Storage::allocate(numel, x.storage->dtype(), x.storage->device());
  auto *ptr_in = static_cast<const T *>(x.storage->data());
  auto *ptr_out = static_cast<T *>(storage_out->data());
  for (py::ssize_t i = 0; i < numel; ++i)
    ptr_out[i] = op(ptr_in[unravel(i, x)]);
  return storage_out;
}

template <class T, class Op>
__global__ void _unary_kernel(const T *in, T *out, py::ssize_t n, Op op,
                              TensorViewStatic view) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  auto idx = unravel(i, view);
  out[i] = op(in[idx]);
}

template <class T, class Op>
std::shared_ptr<Storage> _cuda_unary_op_generic(const TensorView &x, Op op) {
  auto n = numel_from_shape(x.shape);
  auto storage_out = Storage::allocate(n, x.storage->dtype(), Device::Cuda);
  launch_1d(n, _unary_kernel<T, Op>, static_cast<const T *>(x.storage->data()),
            static_cast<T *>(storage_out->data()), n, op,
            tensor_view_to_static(x));
  return storage_out;
}

template <class Dispatch, class Op>
std::shared_ptr<Storage> _dispatch_unary(const TensorView &x, Op op) {
  return Dispatch::run(x.storage->dtype(), [&]<class T>() {
    switch (x.storage->device()) {
    case Device::Cpu:
      return _cpu_unary_op_generic<T>(x, op);
    case Device::Cuda:
      return _cuda_unary_op_generic<T>(x, op);
    default:
      NT_UNREACHABLE();
    }
  });
}

template <class T> __host__ __device__ T int_pow(T a, T n) {
  T r = 1;
  while (n > 0) {
    if (n & 1)
      r *= a;
    a *= a;
    n >>= 1;
  }
  return r;
}
template <class T> struct PowOp {
  T value;
  __host__ __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T>)
      return int_pow(a, value);
    else
      return std::pow(a, value);
  }
};
std::shared_ptr<Storage> pow(const TensorView &x, Scalar value) {
  return DispatchArithmetic::run(x.storage->dtype(), [&]<class T>() {
    auto op = PowOp<T>{value.item<T>()};
    switch (x.storage->device()) {
    case Device::Cpu:
      return _cpu_unary_op_generic<T>(x, op);
    case Device::Cuda:
      return _cuda_unary_op_generic<T>(x, op);
    default:
      NT_UNREACHABLE();
    }
  });
}

#define DEFINE_UNARY(name, expr, dispatch)                                     \
  struct name##Op {                                                            \
    template <class T> __host__ __device__ T operator()(T a) const {           \
      return expr;                                                             \
    }                                                                          \
  };                                                                           \
  std::shared_ptr<Storage> name(const TensorView &x) {                         \
    return _dispatch_unary<dispatch>(x, name##Op());                           \
  }

DEFINE_UNARY(exp, std::exp(a), DispatchFloat)
DEFINE_UNARY(log, std::log(a), DispatchFloat)
DEFINE_UNARY(neg, -a, DispatchArithmetic)
DEFINE_UNARY(relu, (a >= 0) ? a : 0, DispatchArithmetic)
DEFINE_UNARY(sqrt, std::sqrt(a), DispatchFloat)
DEFINE_UNARY(tanh, std::tanh(a), DispatchFloat)
DEFINE_UNARY(sigmoid, T(1) / (T(1) + std::exp(-a)), DispatchFloat)

void bind_unary_ops_(py::module_ &m) {
  m.def(
      "exp", [](const TensorView &x) { return exp(x); },
      "Component-wise exponentiation.", py::arg("x"));
  m.def(
      "log", [](const TensorView &x) { return log(x); }, "Component-wise log.",
      py::arg("x"));
  m.def(
      "pow", [](const TensorView &x, Scalar a) { return pow(x, a); },
      "Component-wise power.", py::arg("x"), py::arg("a"));
  m.def("neg", &neg, "Component-wise negation.", py::arg("x"));
  m.def("relu", &relu, "Rectified linear unit.", py::arg("x"));
  m.def(
      "sqrt", [](const TensorView &x) { return sqrt(x); },
      "Rectified linear unit.", py::arg("x"));
  m.def(
      "tanh", [](const TensorView &x) { return tanh(x); },
      "Rectified linear unit.", py::arg("x"));
  m.def("sigmoid", &sigmoid, "Rectified linear unit.", py::arg("x"));
}