#include "cuda.cuh"
#include "matmul.h"

namespace nt {

constexpr int TILE_SIZE = 32;

template <class T>
std::shared_ptr<Storage>
_cpu_matmul(const TensorView &x1, const TensorView &x2, py::ssize_t ndim,
            const std::vector<py::ssize_t> &out_shape, py::ssize_t K) {
  auto numel = numel_from_shape(out_shape);
  auto storage_out = Storage::allocate(numel, x1.storage->dtype(), Device::Cpu);
  auto *A = static_cast<const T *>(x1.storage->data());
  auto *B = static_cast<const T *>(x2.storage->data());
  auto C = static_cast<T *>(storage_out->data());
  std::vector<py::ssize_t> loc(ndim);
  for (py::ssize_t C_idx = 0; C_idx < numel; ++C_idx) {
    // B, i, j -> derive B, i, c and B, c, j
    T acc = static_cast<T>(0.0);
    for (int c = 0; c < K; ++c) {
      py::ssize_t idx1 = x1.offset + c * x1.strides[ndim - 1] +
                         loc[ndim - 2] * x1.strides[ndim - 2];
      py::ssize_t idx2 = x2.offset + c * x2.strides[ndim - 2] +
                         loc[ndim - 1] * x2.strides[ndim - 1];
      for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
        idx1 += x1.strides[dim_i] * loc[dim_i];
        idx2 += x2.strides[dim_i] * loc[dim_i];
      }
      acc += A[idx1] * B[idx2];
    }
    C[C_idx] = acc;

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

// Non-nice tensors layout
template <class T>
__global__ void
_matmul_fallback(const T *A, const T *B, T *C, TensorViewStatic view_1,
                 TensorViewStatic view_2, py::ssize_t numel, py::ssize_t ndim,
                 py::ssize_t K, py::ssize_t M, py::ssize_t N) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;

  py::ssize_t loc[NT_MAX_DIMS];
  py::ssize_t r = i;
  loc[ndim - 1] = r % N;
  r /= N;
  loc[ndim - 2] = r % M;
  r /= M;
  for (int j = ndim - 3; j >= 0; --j) {
    loc[j] = r % view_1.shape[j];
    r /= view_1.shape[j];
  }

  T acc = 0;
  for (py::ssize_t c = 0; c < K; ++c) {
    py::ssize_t idx1 =
        c * view_1.strides[ndim - 1] + loc[ndim - 2] * view_1.strides[ndim - 2];
    py::ssize_t idx2 =
        c * view_2.strides[ndim - 2] + loc[ndim - 1] * view_2.strides[ndim - 1];
    for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
      idx1 += view_1.strides[dim_i] * loc[dim_i];
      idx2 += view_2.strides[dim_i] * loc[dim_i];
    }
    acc += A[idx1] * B[idx2];
  }

  C[i] = acc;
}

// Nice tensor layout (most cases)
template <class T>
__global__ void _matmul_contig(const T *A, const T *B, T *C, py::ssize_t K,
                               py::ssize_t M, py::ssize_t N) {
  auto block_idx_i = blockIdx.x * TILE_SIZE;
  auto block_idx_j = blockIdx.y * TILE_SIZE;
  auto thread_i = threadIdx.x / TILE_SIZE;
  auto thread_j = threadIdx.x % TILE_SIZE;
  auto batch = blockIdx.z;

  A += batch * M * K + block_idx_i * K;
  B += batch * K * N + block_idx_j;
  C += batch * M * N + block_idx_i * N + block_idx_j;

  T acc = 0;
  auto a_i = block_idx_i + thread_i;
  auto b_j = block_idx_j + thread_j;
  __shared__ T As[TILE_SIZE * TILE_SIZE], Bs[TILE_SIZE * TILE_SIZE];
  for (int block_start = 0; block_start < K; block_start += TILE_SIZE) {
    // Fill [TILE_SIZE, TILE_SIZE] block buffers
    auto a_j = block_start + thread_j;
    auto b_i = block_start + thread_i;
    As[thread_i * TILE_SIZE + thread_j] =
        (a_i < M && a_j < K) ? A[thread_i * K + thread_j] : T(0);
    Bs[thread_i * TILE_SIZE + thread_j] =
        (b_i < K && b_j < N) ? B[thread_i * N + thread_j] : T(0);
    __syncthreads();

    // Accumulate partial dot product on cached block
    for (int k = 0; k < TILE_SIZE; ++k)
      acc += As[thread_i * TILE_SIZE + k] * Bs[k * TILE_SIZE + thread_j];

    // Move to next block
    A += TILE_SIZE;
    B += TILE_SIZE * N;
    __syncthreads();
  }
  if (block_idx_i + thread_i < M && block_idx_j + thread_j < N)
    C[thread_i * N + thread_j] = acc;
}

template <class T>
std::shared_ptr<Storage>
_cuda_matmul(const TensorView &x1, const TensorView &x2, py::ssize_t ndim,
             const std::vector<py::ssize_t> &out_shape, py::ssize_t K,
             py::ssize_t M, py::ssize_t N) {
  auto numel = numel_from_shape(out_shape);
  auto new_storage =
      Storage::allocate(numel, x1.storage->dtype(), Device::Cuda);
  auto *A = static_cast<const T *>(x1.storage->data()) + x1.offset;
  auto *B = static_cast<const T *>(x2.storage->data()) + x2.offset;
  auto C = static_cast<T *>(new_storage->data());
  if (is_contiguous(x1) && is_contiguous(x2)) {
    auto n_partial_x = (M + TILE_SIZE - 1) / TILE_SIZE;
    auto n_partial_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    py::ssize_t batch = 1;
    for (py::ssize_t i = static_cast<py::ssize_t>(x1.shape.size() - 3); i >= 0;
         --i)
      batch *= x1.shape[i];
    dim3 grid(n_partial_x, n_partial_y, batch), block(TILE_SIZE * TILE_SIZE);
    _matmul_contig<<<grid, block>>>(A, B, C, K, M, N);
    NT_CUDA_CHECK(cudaGetLastError());
  } else {
    launch_1d(numel, _matmul_fallback<T>, A, B, C, tensor_view_to_static(x1),
              tensor_view_to_static(x2), numel, ndim, K, M, N);
  }

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
  py::ssize_t M = x1.shape[ndim - 2];
  py::ssize_t K = x1.shape[ndim - 1];
  py::ssize_t N = x2.shape[ndim - 1];
  if (K != x2.shape[ndim - 2])
    throw std::invalid_argument("Invalid matmul shape.");
  out_shape[ndim - 2] = M;
  out_shape[ndim - 1] = N;

  return dispatch_dtype_arithmetic(x1.storage->dtype(), [&]<class T>() {
    switch (x1.storage->device()) {
    case Device::Cpu:
      return _cpu_matmul<T>(x1, x2, ndim, out_shape, K);
    case Device::Cuda:
      return _cuda_matmul<T>(x1, x2, ndim, out_shape, K, M, N);
    default:
      NT_UNREACHABLE();
    }
  });
}
} // namespace nt

void bind_matmul_op_(py::module_ &m) {
  m.def("matmul", &nt::matmul, "Matrix multiplication.", py::arg("x1"),
        py::arg("x2"));
}