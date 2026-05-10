#pragma once

#include <random>

namespace {
std::mt19937_64 &global_rng() {
  static std::mt19937_64 rng{std::random_device{}()};
  return rng;
}
} // namespace

// FIXME: not thread-safe + doesn't affect CuRand
inline void manual_seed(uint64_t seed) { global_rng().seed(seed); }