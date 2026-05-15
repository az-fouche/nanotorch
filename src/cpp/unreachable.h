#pragma once

/* OS-agnostic unreachable flag. */

[[noreturn]] inline void nt_unreachable() {
#if defined(_MSC_VER) && !defined(__clang__)
  __assume(false);
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_unreachable();
#else
  std::abort();
#endif
}
#define NT_UNREACHABLE() nt_unreachable()