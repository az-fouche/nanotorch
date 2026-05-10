#include "cuda.cuh"
#include "reduction_ops.h"

namespace nt {

template <class O> struct MatItem {
  py::ssize_t idx;
  O value;
};

struct ArgMinOp {
  static constexpr bool returns_idx = true;
  template <class O> static constexpr __host__ __device__ MatItem<O> neutral() {
    return MatItem<O>(-1, cuda::std::numeric_limits<O>::max());
  }
  template <class O>
  static __host__ __device__ MatItem<O> combine(MatItem<O> a, MatItem<O> b) {
    return (a.value <= b.value) ? a : b;
  }
};
struct ArgMaxOp {
  static constexpr bool returns_idx = true;
  template <class O> static constexpr __host__ __device__ MatItem<O> neutral() {
    return MatItem<O>(-1, cuda::std::numeric_limits<O>::lowest());
  }
  template <class O>
  static __host__ __device__ MatItem<O> combine(MatItem<O> a, MatItem<O> b) {
    return (a.value >= b.value) ? a : b;
  }
};
struct MinOp : ArgMinOp {
  static constexpr bool returns_idx = false;
};
struct MaxOp : ArgMaxOp {
  static constexpr bool returns_idx = false;
};
struct SumOp {
  static constexpr bool returns_idx = false;
  template <class O> static constexpr __host__ __device__ MatItem<O> neutral() {
    return MatItem<O>(-1, O(0));
  }
  template <class O>
  static __host__ __device__ MatItem<O> combine(MatItem<O> a, MatItem<O> b) {
    return MatItem<O>(-1, a.value + b.value);
  }
};

template <class T, class O, class Op>
std::shared_ptr<Storage>
_generic_reduction_cpu(const TensorView &x, Dtype dtype,
                       const TensorViewStatic &view_keep,
                       const TensorViewStatic &view_drop,
                       py::ssize_t numel_keep, py::ssize_t numel_drop) {
  Dtype storage_dt = Op::returns_idx ? Dtype::Int64 : dtype;
  auto storage_out = Storage::allocate(numel_keep, storage_dt, Device::Cpu);
  auto ptr_in = static_cast<const T *>(x.storage->data());
  using OutT = std::conditional_t<Op::returns_idx, int64_t, O>;
  auto ptr_out = static_cast<OutT *>(storage_out->data());
  for (py::ssize_t i = 0; i < numel_keep; ++i) {
    auto base = unravel(i, view_keep);
    auto acc = Op::template neutral<O>();
    for (py::ssize_t j = 0; j < numel_drop; ++j) {
      auto idx_in = base + unravel(j, view_drop);
      acc = Op::combine(acc, MatItem<O>(j, static_cast<O>(ptr_in[idx_in])));
    }
    if constexpr (Op::returns_idx)
      ptr_out[i] = static_cast<int64_t>(acc.idx);
    else
      ptr_out[i] = acc.value;
  }
  return storage_out;
}

template <class T, class O, class Op>
__global__ void
_sum_kernel(py::ssize_t n, const T *ptr_in,
            std::conditional_t<Op::returns_idx, int64_t, O> *ptr_out,
            TensorViewStatic view_keep, TensorViewStatic view_drop,
            py::ssize_t numel_drop) {
  // TODO: optimize the parallelization
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  py::ssize_t base = unravel(i, view_keep);
  auto acc = Op::template neutral<O>();
  for (py::ssize_t j = 0; j < numel_drop; ++j) {
    auto offset = unravel(j, view_drop);
    acc =
        Op::combine(acc, MatItem<O>(j, static_cast<O>(ptr_in[base + offset])));
  }
  if constexpr (Op::returns_idx)
    ptr_out[i] = static_cast<int64_t>(acc.idx);
  else
    ptr_out[i] = acc.value;
}

template <class T, class O, class Op>
std::shared_ptr<Storage>
_generic_reduction_cuda(const TensorView &x, Dtype dtype,
                        const TensorViewStatic &view_keep,
                        const TensorViewStatic &view_drop,
                        py::ssize_t numel_keep, py::ssize_t numel_drop) {
  Dtype storage_dt = Op::returns_idx ? Dtype::Int64 : dtype;
  auto storage_out = Storage::allocate(numel_keep, storage_dt, Device::Cuda);
  using OutT = std::conditional_t<Op::returns_idx, int64_t, O>;
  auto ptr_out = static_cast<OutT *>(storage_out->data());
  launch_1d(numel_keep, _sum_kernel<T, O, Op>, numel_keep,
            static_cast<const T *>(x.storage->data()), ptr_out, view_keep,
            view_drop, numel_drop);
  return storage_out;
}

template <class Op>
std::shared_ptr<Storage>
_generic_reduction(const TensorView &x,
                   const std::vector<py::ssize_t> &axis_drop, Dtype dtype) {
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
        return _generic_reduction_cpu<T, O, Op>(x, dtype, view_keep, view_drop,
                                                numel_keep, numel_drop);
      case Device::Cuda:
        return _generic_reduction_cuda<T, O, Op>(x, dtype, view_keep, view_drop,
                                                 numel_keep, numel_drop);
      default:
        NT_UNREACHABLE();
      }
    });
  });
}

std::shared_ptr<Storage> sum(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop,
                             Dtype dtype) {
  return _generic_reduction<SumOp>(x, axis_drop, dtype);
}

std::shared_ptr<Storage> min(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop) {
  return _generic_reduction<MinOp>(x, axis_drop, x.storage->dtype());
}

std::shared_ptr<Storage> max(const TensorView &x,
                             const std::vector<py::ssize_t> &axis_drop) {
  return _generic_reduction<MaxOp>(x, axis_drop, x.storage->dtype());
}

std::shared_ptr<Storage> argmin(const TensorView &x,
                                const std::vector<py::ssize_t> &axis_drop) {
  return _generic_reduction<ArgMinOp>(x, axis_drop, x.storage->dtype());
}

std::shared_ptr<Storage> argmax(const TensorView &x,
                                const std::vector<py::ssize_t> &axis_drop) {
  return _generic_reduction<ArgMaxOp>(x, axis_drop, x.storage->dtype());
}

} // namespace nt

void bind_reduction_ops_(py::module_ &m) {
  m.def("sum", &nt::sum, "Sum all elements in a tensor.", py::arg("x"),
        py::arg("axis"), py::arg("dtype"));
  m.def("min", &nt::min, "Compute the max over all elements in a tensor.",
        py::arg("x"), py::arg("axis"));
  m.def("max", &nt::max, "Compute the min over all elements in a tensor.",
        py::arg("x"), py::arg("axis"));
  m.def("argmin", &nt::argmin,
        "Compute the max index over all elements in a tensor.", py::arg("x"),
        py::arg("axis"));
  m.def("argmax", &nt::argmax,
        "Compute the min index over all elements in a tensor.", py::arg("x"),
        py::arg("axis"));
}