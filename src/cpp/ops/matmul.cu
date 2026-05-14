#include "cuda.cuh"
#include "matmul.h"

namespace nt {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;
constexpr int THREAD_M = 8;
constexpr int THREAD_N = 8;
constexpr int NUM_THREADS = BLOCK_M * BLOCK_N / (THREAD_M * THREAD_N);
constexpr int THREADS_PER_ROW = BLOCK_N / THREAD_N;
constexpr int STRIDE_A = NUM_THREADS / BLOCK_K;
constexpr int STRIDE_B = NUM_THREADS / BLOCK_N;

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
// Ref: https://siboehm.com/articles/22/CUDA-MMM
template <class T>
__global__ void _matmul_contig(const T *A, const T *B, T *C, py::ssize_t K,
                               py::ssize_t M, py::ssize_t N) {
  auto block_i = blockIdx.x * BLOCK_M; // Global coordinates in A/B/C
  auto block_j = blockIdx.y * BLOCK_N;
  auto thread_i = threadIdx.x / THREADS_PER_ROW; // Local coordinates in block
  auto thread_j = threadIdx.x % THREADS_PER_ROW;
  auto batch = blockIdx.z;

  // Offset pointers to the start of the accumulation
  A += batch * M * K + block_i * K;
  B += batch * K * N + block_j;

  // Set up shared storage
  __shared__ T As[BLOCK_M * BLOCK_K], Bs[BLOCK_K * BLOCK_N];
  auto as_i = threadIdx.x / BLOCK_K; // Fixed storage step coordinates
  auto as_j = threadIdx.x % BLOCK_K;
  auto bs_i = threadIdx.x / BLOCK_N; // Fixed storage step coordinates
  auto bs_j = threadIdx.x % BLOCK_N;

  // Fixed A row/B col offset of the thread
  auto a_i = block_i + as_i;
  auto b_j = block_j + bs_j;

  // Per-block 2D accumulation
  T thread_acc[THREAD_M * THREAD_N] = {};
  T scratch_a[THREAD_M] = {}, scratch_b[THREAD_N] = {};
  for (int block_start = 0; block_start < K; block_start += BLOCK_K) {
    // Storage step: fill [TILE_SIZE, TILE_SIZE] block buffers
    for (int offset = 0; offset < BLOCK_M; offset += STRIDE_A) {
      auto as_ioff = as_i + offset;
      As[as_ioff * BLOCK_K + as_j] =
          (a_i + offset < M && block_start + as_j < K) ? A[as_ioff * K + as_j]
                                                       : T(0);
    }
    for (int offset = 0; offset < BLOCK_K; offset += STRIDE_B) {
      auto bs_ioff = bs_i + offset;
      Bs[bs_ioff * BLOCK_N + bs_j] =
          (block_start + bs_ioff < K && b_j < N) ? B[bs_ioff * N + bs_j] : T(0);
    }
    __syncthreads();

    // Accumulate partial dot product on cached block
    for (int k = 0; k < BLOCK_K; ++k) {
      for (int i = 0; i < THREAD_M; ++i)
        scratch_a[i] = As[(thread_i * THREAD_M + i) * BLOCK_K + k];
      for (int i = 0; i < THREAD_N; ++i)
        scratch_b[i] = Bs[k * BLOCK_N + thread_j * THREAD_N + i];

      for (int pm = 0; pm < THREAD_M; ++pm)
        for (int pn = 0; pn < THREAD_N; ++pn)
          thread_acc[pm * THREAD_N + pn] += scratch_a[pm] * scratch_b[pn];
    }

    // Rebase pointers to next block
    A += BLOCK_K;
    B += BLOCK_K * N;
    __syncthreads();
  }

  // Write the result
  C += batch * M * N;
  auto row = block_i + thread_i * THREAD_M, col = block_j + thread_j * THREAD_N;
  for (int pm = 0; pm < THREAD_M; ++pm) {
    if (row + pm >= M)
      continue;
    for (int pn = 0; pn < THREAD_N; ++pn) {
      if (col + pn < N)
        C[(row + pm) * N + col + pn] = thread_acc[pm * THREAD_N + pn];
    }
  }
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
    auto n_partial_x = (M + BLOCK_M - 1) / BLOCK_M;
    auto n_partial_y = (N + BLOCK_N - 1) / BLOCK_N;
    py::ssize_t batch = 1;
    for (py::ssize_t i = static_cast<py::ssize_t>(x1.shape.size() - 3); i >= 0;
         --i)
      batch *= x1.shape[i];
    dim3 grid(n_partial_x, n_partial_y, batch), block(NUM_THREADS);
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