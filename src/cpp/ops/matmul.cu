#include "cuda.cuh"
#include "matmul.h"

template <class T>
std::shared_ptr<Storage>
_cpu_matmul(const TensorView &x1, const TensorView &x2, py::ssize_t ndim,
            const std::vector<py::ssize_t> &out_shape, py::ssize_t csize) {
  auto numel = numel_from_shape(out_shape);
  auto storage_out = Storage::allocate(numel, x1.storage->dtype(), Device::Cpu);
  auto *ptr_in1 = static_cast<const T *>(x1.storage->data());
  auto *ptr_in2 = static_cast<const T *>(x2.storage->data());
  auto ptr_out = static_cast<T *>(storage_out->data());
  std::vector<py::ssize_t> loc(ndim);
  for (py::ssize_t ptr_out_idx = 0; ptr_out_idx < numel; ++ptr_out_idx) {
    // B, i, j -> derive B, i, c and B, c, j
    T acc = static_cast<T>(0.0);
    for (int c = 0; c < csize; ++c) {
      py::ssize_t idx1 = x1.offset + c * x1.strides[ndim - 1] +
                         loc[ndim - 2] * x1.strides[ndim - 2];
      py::ssize_t idx2 = x2.offset + c * x2.strides[ndim - 2] +
                         loc[ndim - 1] * x2.strides[ndim - 1];
      for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
        idx1 += x1.strides[dim_i] * loc[dim_i];
        idx2 += x2.strides[dim_i] * loc[dim_i];
      }
      acc += ptr_in1[idx1] * ptr_in2[idx2];
    }
    ptr_out[ptr_out_idx] = acc;

    py::ssize_t j = ndim - 1;
    while (j >= 0) {
      loc[j] += 1;
      if (loc[j] < out_shape[j])
        break;
      loc[j] = 0;
      j -= 1;
    }
  }
  return storage_out;
}

template <class T>
__global__ void _matmul_kernel(const T *ptr_in1, const T *ptr_in2, T *ptr_out,
                               TensorViewStatic view_1, TensorViewStatic view_2,
                               py::ssize_t numel, py::ssize_t ndim,
                               py::ssize_t csize, py::ssize_t lsize,
                               py::ssize_t rsize) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;

  py::ssize_t loc[NT_MAX_DIMS];
  py::ssize_t r = i;
  loc[ndim - 1] = r % rsize;
  r /= rsize;
  loc[ndim - 2] = r % lsize;
  r /= lsize;
  for (int j = ndim - 3; j >= 0; --j) {
    loc[j] = r % view_1.shape[j];
    r /= view_1.shape[j];
  }

  T acc = 0;
  for (py::ssize_t c = 0; c < csize; ++c) {
    py::ssize_t idx1 = view_1.offset + c * view_1.strides[ndim - 1] +
                       loc[ndim - 2] * view_1.strides[ndim - 2];
    py::ssize_t idx2 = view_2.offset + c * view_2.strides[ndim - 2] +
                       loc[ndim - 1] * view_2.strides[ndim - 1];
    for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
      idx1 += view_1.strides[dim_i] * loc[dim_i];
      idx2 += view_2.strides[dim_i] * loc[dim_i];
    }
    acc += ptr_in1[idx1] * ptr_in2[idx2];
  }

  ptr_out[i] = acc;
}

template <class T>
std::shared_ptr<Storage>
_cuda_matmul(const TensorView &x1, const TensorView &x2, py::ssize_t ndim,
             const std::vector<py::ssize_t> &out_shape, py::ssize_t csize,
             py::ssize_t lsize, py::ssize_t rsize) {
  auto numel = numel_from_shape(out_shape);
  auto new_storage =
      Storage::allocate(numel, x1.storage->dtype(), Device::Cuda);
  auto *ptr_in1 = static_cast<const T *>(x1.storage->data());
  auto *ptr_in2 = static_cast<const T *>(x2.storage->data());
  auto ptr_out = static_cast<T *>(new_storage->data());
  launch_1d(numel, _matmul_kernel<T>, ptr_in1, ptr_in2, ptr_out,
            tensor_view_to_static(x1), tensor_view_to_static(x2), numel, ndim,
            csize, lsize, rsize);

  return new_storage;
}

std::shared_ptr<Storage> matmul(const TensorView &x1, const TensorView &x2) {
  _require_same_device(x1.storage, x2.storage, "matmul");
  _require_same_dtype(x1.storage, x2.storage, "matmul");
  if (x1.shape.size() != x2.shape.size())
    throw std::invalid_argument("matmul: x1 and x2 have different ndim.");
  if (x1.shape.size() < 2)
    throw std::invalid_argument(
        "matmul: kernel should be called with >=2d tensors.");

  auto ndim = x1.shape.size();
  auto out_shape = std::vector<py::ssize_t>(ndim);
  for (auto i = 0; i < ndim - 2; ++i) {
    if (x1.shape[i] != x2.shape[i])
      throw std::invalid_argument("Invalid broacast shapes.");
    else
      out_shape[i] = x1.shape[i];
  }
  py::ssize_t lsize = x1.shape[ndim - 2];
  py::ssize_t csize = x1.shape[ndim - 1];
  py::ssize_t rsize = x2.shape[ndim - 1];
  if (csize != x2.shape[ndim - 2])
    throw std::invalid_argument("Invalid matmul shape.");
  out_shape[ndim - 2] = lsize;
  out_shape[ndim - 1] = rsize;

  return dispatch_dtype_arithmetic(x1.storage->dtype(), [&]<class T>() {
    switch (x1.storage->device()) {
    case Device::Cpu:
      return _cpu_matmul<T>(x1, x2, ndim, out_shape, csize);
    case Device::Cuda:
      return _cuda_matmul<T>(x1, x2, ndim, out_shape, csize, lsize, rsize);
    default:
      NT_UNREACHABLE();
    }
  });
}

void bind_matmul_op_(py::module_ &m) {
  m.def("matmul", &matmul, "Matrix multiplication.", py::arg("x1"),
        py::arg("x2"));
}