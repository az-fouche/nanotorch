#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cuda.h"
#include "unreachable.h"

enum class Device : uint8_t { Cpu, Cuda };

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void *alloc(size_t nbytes) = 0;
  virtual void free(void *ptr, size_t nbytes) = 0;
  static Allocator &for_device(Device device);
};

class CpuAllocator : public Allocator {
public:
  void *alloc(size_t nbytes) override { return ::operator new(nbytes); }
  void free(void *ptr, size_t nbytes) override {
    ::operator delete(ptr, nbytes);
  }
};

class CudaAllocator : public Allocator {
public:
  void *alloc(size_t nbytes) override;
  void free(void *ptr, size_t nbytes) override;

private:
  static constexpr size_t BLOCK_SIZE = 256;
  static constexpr size_t to_n_blocks(size_t nbytes) {
    if (nbytes == 0) // Avoids UB territory, 256B is cheap enough
      return 1;
    return (nbytes + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }
  std::unordered_map<size_t, std::vector<void *>> buckets_;
};
