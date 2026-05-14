#include "binary_ops.h"
#include "cuda.cuh"

// Pointwise binary

template <class T, class O, class F, class Idx1, class Idx2>
std::shared_ptr<Storage>
_cpu_binary_op_generic(const TensorView &x1, const TensorView &x2, F &&func,
                       Dtype out_dtype, Idx1 indexer_1, Idx2 indexer_2) {
  auto numel = numel_from_shape(x1.shape);
  auto *ptr_in1 = static_cast<const T *>(x1.storage->data());
  auto *ptr_in2 = static_cast<const T *>(x2.storage->data());
  auto storage_out = Storage::allocate(numel, out_dtype, x1.storage->device());
  auto *ptr_out = static_cast<O *>(storage_out->data());
  for (py::ssize_t i = 0; i < numel; ++i) {
    ptr_out[i] =
        static_cast<O>(func(ptr_in1[indexer_1(i)], ptr_in2[indexer_2(i)]));
  }
  return storage_out;
}

template <class T, class O, class Op, class Idx1, class Idx2>
__global__ void _binary_kernel(const T *ptr_in1, const T *ptr_in2,
                               Idx1 indexer_1, Idx2 indexer_2, O *ptr_out,
                               py::ssize_t n, Op op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  ptr_out[i] = static_cast<O>(op(ptr_in1[indexer_1(i)], ptr_in2[indexer_2(i)]));
}

template <class T, class O, class Op, class Idx1, class Idx2>
std::shared_ptr<Storage>
_cuda_binary_op_generic(const TensorView &x1, const TensorView &x2, Op op,
                        Dtype out_dtype, Idx1 indexer_1, Idx2 indexer_2) {
  auto n = numel_from_shape(x1.shape);
  auto storage_out = Storage::allocate(n, out_dtype, Device::Cuda);
  launch_1d(n, _binary_kernel<T, O, Op, Idx1, Idx2>,
            static_cast<const T *>(x1.storage->data()),
            static_cast<const T *>(x2.storage->data()), indexer_1, indexer_2,
            static_cast<O *>(storage_out->data()), n, op);
  return storage_out;
}

template <class Dispatch, class Op>
std::shared_ptr<Storage>
_dispatch_binary(const TensorView &x1, const TensorView &x2, Op op,
                 std::optional<Dtype> out_dtype = std::nullopt) {
  _require_same_shape(x1, x2, "_dispatch_binary");
  _require_same_device(x1.storage, x2.storage, "_dispatch_binary");
  _require_same_dtype(x1.storage, x2.storage, "_dispatch_binary");

  auto eff_dtype = out_dtype.value_or(x1.storage->dtype());
  return Dispatch::run(eff_dtype, [&]<class O>() {
    return Dispatch::run(x1.storage->dtype(), [&]<class T>() {
      return with_indexer(x1, [&](auto indexer_1) {
        return with_indexer(x2, [&](auto indexer_2) {
          switch (x1.storage->device()) {
          case Device::Cpu:
            return _cpu_binary_op_generic<T, O>(x1, x2, op, eff_dtype,
                                                indexer_1, indexer_2);
          case Device::Cuda:
            return _cuda_binary_op_generic<T, O>(x1, x2, op, eff_dtype,
                                                 indexer_1, indexer_2);
          default:
            NT_UNREACHABLE();
          }
        });
      });
    });
  });
}

#define DEFINE_BINARY(name, expr, dispatch, out_dtype)                         \
  struct name##Op {                                                            \
    template <class T> __host__ __device__ T operator()(T a, T b) const {      \
      return expr;                                                             \
    }                                                                          \
  };                                                                           \
  std::shared_ptr<Storage> name(const TensorView &x1, const TensorView &x2) {  \
    return _dispatch_binary<dispatch>(x1, x2, name##Op(), out_dtype);          \
  }

DEFINE_BINARY(add, a + b, DispatchAll, std::nullopt)
DEFINE_BINARY(subtract, a - b, DispatchAll, std::nullopt)
DEFINE_BINARY(multiply, a *b, DispatchAll, std::nullopt)
DEFINE_BINARY(divide, a / b, DispatchArithmetic, std::nullopt)
DEFINE_BINARY(pw_equal, a == b, DispatchAll, Dtype::Bool)
DEFINE_BINARY(pw_greater, a > b, DispatchAll, Dtype::Bool)
DEFINE_BINARY(pw_greater_eq, a >= b, DispatchAll, Dtype::Bool)

// Equals

template <class T>
bool _cpu_equals(const TensorView &x1, const TensorView &x2, const T *ptr_in1,
                 const T *ptr_in2, py::ssize_t numel) {
  for (py::ssize_t i = 0; i < numel; ++i) {
    auto idx1 = unravel(i, x1);
    auto idx2 = unravel(i, x2);
    if (ptr_in1[idx1] != ptr_in2[idx2])
      return false;
  }
  return true;
}

template <class T>
__global__ void _cuda_equals_kernel(py::ssize_t n, TensorViewStatic x1,
                                    TensorViewStatic x2, const T *ptr_in1,
                                    const T *ptr_in2, int *mismatch) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  auto idx1 = unravel(i, x1);
  auto idx2 = unravel(i, x2);
  if (ptr_in1[idx1] != ptr_in2[idx2])
    *mismatch = 1;
}

template <class T>
bool _cuda_equals(const TensorView &x1, const TensorView &x2, const T *ptr_in1,
                  const T *ptr_in2, py::ssize_t numel) {
  int *mismatch_cu = nullptr;
  NT_CUDA_CHECK(cudaMalloc((void **)&mismatch_cu, sizeof(int)));
  NT_CUDA_CHECK(cudaMemset(mismatch_cu, 0, sizeof(int)));

  launch_1d(numel, _cuda_equals_kernel<T>, numel, tensor_view_to_static(x1),
            tensor_view_to_static(x2), ptr_in1, ptr_in2, mismatch_cu);

  int mismatch_cpu = 0;
  NT_CUDA_CHECK(cudaMemcpy(&mismatch_cpu, mismatch_cu, sizeof(int),
                           cudaMemcpyDeviceToHost));
  NT_CUDA_CHECK(cudaFree(mismatch_cu));
  return mismatch_cpu == 0;
}

bool equals(const TensorView &x1, const TensorView &x2) {
  _require_same_shape(x1, x2, "equals");
  _require_same_dtype(x1.storage, x2.storage, "equals");
  _require_same_device(x1.storage, x2.storage, "equals");
  auto numel = numel_from_shape(x1.shape);
  return DispatchAll::run(x1.storage->dtype(), [&]<class T>() {
    auto *ptr_in1 = static_cast<const T *>(x1.storage->data());
    auto *ptr_in2 = static_cast<const T *>(x2.storage->data());
    switch (x1.storage->device()) {
    case Device::Cpu:
      return _cpu_equals(x1, x2, ptr_in1, ptr_in2, numel);
    case Device::Cuda:
      return _cuda_equals(x1, x2, ptr_in1, ptr_in2, numel);
    default:
      NT_UNREACHABLE();
    }
  });
}

void bind_binary_ops_(py::module_ &m) {
  m.def("add", &add, "Add elements pointwise.", py::arg("x1"), py::arg("x2"));
  m.def("subtract", &subtract, "Subtract elements pointwise.", py::arg("x1"),
        py::arg("x2"));
  m.def("multiply", &multiply, "Multiply elements pointwise.", py::arg("x1"),
        py::arg("x2"));
  m.def("divide", &divide, "Divide elements pointwise.", py::arg("x1"),
        py::arg("x2"));
  m.def("pw_greater", &pw_greater,
        "Per-coefficient strictly greater comparison.", py::arg("x"),
        py::arg("s"));
  m.def("pw_greater_eq", &pw_greater_eq,
        "Per-coefficient greater/equals comparison.", py::arg("x"),
        py::arg("s"));
  m.def("pw_equal", &pw_equal, "Per-coefficient equal comparison.",
        py::arg("x"), py::arg("s"));
  m.def("equals", &equals, "Test the global equality of two tensors.",
        py::arg("x1"), py::arg("x2"));
}