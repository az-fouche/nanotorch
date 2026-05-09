#include "cuda.cuh"
#include "unary_ops.h"

// Sum

template <class T, class O>
std::shared_ptr<Storage>
_cpu_sum(const TensorView &x, Dtype out_dtype,
         const TensorViewStatic &view_keep, const TensorViewStatic &view_drop,
         py::ssize_t numel_keep, py::ssize_t numel_drop) {
  auto storage_out = Storage::allocate(numel_keep, out_dtype, Device::Cpu);
  auto ptr_in = static_cast<const T *>(x.storage->data());
  auto ptr_out = static_cast<O *>(storage_out->data());
  for (py::ssize_t i = 0; i < numel_keep; ++i) {
    auto base = unravel(i, view_keep);
    O acc = static_cast<O>(0);
    for (py::ssize_t j = 0; j < numel_drop; ++j) {
      auto idx_in = base + unravel(j, view_drop);
      acc += static_cast<O>(ptr_in[idx_in]);
    }
    ptr_out[i] = acc;
  }
  return storage_out;
}

template <class T, class O>
__global__ void _sum_kernel(py::ssize_t n, const T *ptr_in, O *ptr_out,
                            TensorViewStatic view_keep,
                            TensorViewStatic view_drop,
                            py::ssize_t numel_drop) {
  // TODO: optimize the parallelization
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  py::ssize_t base = unravel(i, view_keep);
  O acc = 0;
  for (py::ssize_t j = 0; j < numel_drop; ++j)
    acc += static_cast<O>(ptr_in[base + unravel(j, view_drop)]);
  ptr_out[i] = acc;
}

template <class T, class O>
std::shared_ptr<Storage>
_cuda_sum(const TensorView &x, Dtype out_dtype,
          const TensorViewStatic &view_keep, const TensorViewStatic &view_drop,
          py::ssize_t numel_keep, py::ssize_t numel_drop) {
  auto storage_out = Storage::allocate(numel_keep, out_dtype, Device::Cuda);
  launch_1d(numel_keep, _sum_kernel<T, O>, numel_keep,
            static_cast<const T *>(x.storage->data()),
            static_cast<O *>(storage_out->data()), view_keep, view_drop,
            numel_drop);
  return storage_out;
}

std::shared_ptr<Storage> sum(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop,
                             Dtype dtype) {
  // Compute both shapes
  auto ndim = static_cast<py::ssize_t>(axis_drop.size());
  auto ndim_in = static_cast<py::ssize_t>(x.shape.size());
  auto ndim_out = static_cast<py::ssize_t>(ndim_in - ndim);
  auto axis_keep = std::vector<py::ssize_t>(ndim_out);
  auto shape_drop = std::vector<py::ssize_t>(ndim);
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

  // Compute static views
  auto numel_drop = numel_from_shape(shape_drop);
  auto numel_keep = numel_from_shape(shape_keep);
  auto view_keep = TensorViewStatic(axis_keep.size(), x.offset);
  for (size_t p = 0; p < axis_keep.size(); ++p) {
    view_keep.shape[p] = x.shape[axis_keep[p]];
    view_keep.strides[p] = x.strides[axis_keep[p]];
  }
  // offset already counted in view_keep
  auto view_drop = TensorViewStatic(axis_drop.size(), 0);
  for (size_t p = 0; p < axis_drop.size(); ++p) {
    view_drop.shape[p] = x.shape[axis_drop[p]];
    view_drop.strides[p] = x.strides[axis_drop[p]];
  }

  return DispatchAll::run(x.storage->dtype(), [&]<class T>() {
    return DispatchArithmetic::run(dtype, [&]<class O>() {
      switch (x.storage->device()) {
      case Device::Cpu:
        return _cpu_sum<T, O>(x, dtype, view_keep, view_drop, numel_keep,
                              numel_drop);
      case Device::Cuda:
        return _cuda_sum<T, O>(x, dtype, view_keep, view_drop, numel_keep,
                               numel_drop);
      default:
        NT_UNREACHABLE();
      }
    });
  });
}

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
DEFINE_UNARY(sqrt, std::sqrt(a), DispatchArithmetic)
DEFINE_UNARY(tanh, std::tanh(a), DispatchArithmetic)
DEFINE_UNARY(sigmoid, 1.0 / (1.0 + std::exp(-a)), DispatchArithmetic)

void bind_unary_ops_(py::module_ &m) {
  m.def("sum", &sum, "Sum all elements in a tensor.", py::arg("x"),
        py::arg("axis"), py::arg("dtype"));
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