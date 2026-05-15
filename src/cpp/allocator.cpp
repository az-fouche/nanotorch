#include "allocator.h"

Allocator &Allocator::for_device(Device device) {
  static CpuAllocator cpu;
  static CudaAllocator cuda;
  switch (device) {
  case Device::Cpu:
    return cpu;
  case Device::Cuda:
    return cuda;
  default:
    NT_UNREACHABLE();
  }
}

/* Classical bucket system
  - If no bucket of required size is free, create one, otherwise reuse one.
  - On free(), stock the empty bucket instead of calling cudaFree
*/
void *CudaAllocator::alloc(size_t nbytes) {
  auto n_blocks = to_n_blocks(nbytes);
  auto bucket = buckets_.find(n_blocks); // Avoids empty insert
  if (bucket != buckets_.end() && !bucket->second.empty()) {
    void *p = bucket->second.back();
    bucket->second.pop_back();
    return p;
  }
  void *ptr = nullptr;
  NT_CUDA_CHECK(cudaMalloc(&ptr, n_blocks * BLOCK_SIZE));
  return ptr;
}

void CudaAllocator::free(void *ptr, size_t nbytes) {
  auto n_blocks = to_n_blocks(nbytes);
  buckets_[n_blocks].push_back(ptr);
}
