# Nanotorch: a minimal PyTorch clone

Pet project for fun (and to learn!), the goal is to reimplement from scratch a
PyTorch-like autograd library in python/C++, without relying on generative AI
to produce the code.

## Nanotorch scope

- Usable tensor library supporting all common operations.
- Tensors convertible from/to Torch/NumPy/Jax.
- Autograd support.
- Heavy lifting done with C++ kernels.
- Basic CUDA support.
- Basic linear algebra and optimizers tools.
- Comprehensive test coverage and documentation.
- **The project is successful if we can train an MLP on GPU with torch-like code.**

## Project constraints

- No code should be produced with generative AI. It can be used as a helper to
  answer questions or review the code, but all code must be hand-written.
- No `numpy` or `torch` is allowed (except in some unit tests, because that's
  handy). All the logic must be implemented in `nanotorch`. 
- No reliance on native Python's `array.array` module either, we must implement
  the contiguous array logic in C++.
- No overreliance on out-of-the-box algorithms in Python and C++, the goal is
  to reimplement as much things as possible from scratch to fully understand all
  the internals. Exceptions can be made for gnarly operations like RNG.