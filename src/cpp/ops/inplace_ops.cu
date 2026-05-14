#include "cuda.cuh"
#include "inplace_ops.h"

// Inplace

template <class T, class Op, class IdxOut, class IdxOther>
void _cpu_inplace_apply(TensorView &out, const TensorView &other, Op op,
                        IdxOut indexer_out, IdxOther indexer_other) {
  auto numel = numel_from_shape(out.shape);
  auto *ptr_other = static_cast<const T *>(other.storage->data());
  auto *ptr_out = static_cast<T *>(out.storage->data());
  for (py::ssize_t i = 0; i < numel; ++i) {
    auto idx_out = indexer_out(i);
    ptr_out[idx_out] = op(ptr_out[idx_out], ptr_other[indexer_other(i)]);
  }
}

template <class T, class Op, class IdxOut, class IdxOther>
__global__ void _inplace_apply_kernel(py::ssize_t n, T *ptr_out,
                                      const T *ptr_other, IdxOut indexer_out,
                                      IdxOther indexer_other, Op op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  auto idx_out = indexer_out(i);
  ptr_out[idx_out] = op(ptr_out[idx_out], ptr_other[indexer_other(i)]);
}

template <class T, class Op, class IdxOut, class IdxOther>
void _cuda_inplace_apply(const TensorView &out, const TensorView &other, Op op,
                         IdxOut indexer_out, IdxOther indexer_other) {
  auto n = numel_from_shape(out.shape);
  launch_1d(n, _inplace_apply_kernel<T, Op, IdxOut, IdxOther>, n,
            static_cast<T *>(out.storage->data()),
            static_cast<const T *>(other.storage->data()), indexer_out,
            indexer_other, op);
}

template <class Dispatch, class Op>
void _dispatch_inplace_apply(TensorView &out, const TensorView &other, Op op) {
  _require_same_device(out.storage, other.storage, "_dispatch_inplace_apply");
  _require_same_dtype(out.storage, other.storage, "_dispatch_inplace_apply");
  _require_same_shape(out, other, "_dispatch_inplace_apply");
  Dispatch::run(out.storage->dtype(), [&]<class T>() {
    with_indexer(out, [&](auto indexer_out) {
      with_indexer(other, [&](auto indexer_other) {
        switch (out.storage->device()) {
        case Device::Cpu:
          _cpu_inplace_apply<T>(out, other, op, indexer_out, indexer_other);
          break;
        case Device::Cuda:
          _cuda_inplace_apply<T>(out, other, op, indexer_out, indexer_other);
          break;
        default:
          NT_UNREACHABLE();
        }
      });
    });
  });
  out.storage->bump_version();
}

#define DEFINE_INPLACE(name, expr, dispatch)                                   \
  struct name##Op {                                                            \
    template <class T>                                                         \
    __host__ __device__ T operator()([[maybe_unused]] T a,                     \
                                     [[maybe_unused]] T b) const {             \
      return expr;                                                             \
    }                                                                          \
  };                                                                           \
  void name##_inplace(TensorView &out, const TensorView &other) {              \
    _dispatch_inplace_apply<dispatch>(out, other, name##Op());                 \
  }

DEFINE_INPLACE(add, a + b, DispatchAll)
DEFINE_INPLACE(sub, a - b, DispatchAll)
DEFINE_INPLACE(mul, a *b, DispatchAll)
DEFINE_INPLACE(div, a / b, DispatchArithmetic)
DEFINE_INPLACE(copy, b, DispatchAll)

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