# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

Educational pet project: a hand-written PyTorch-like tensor + autograd library in Python/C++. **Hard constraints:**

- No code is to be produced by generative AI. Claude's role here is reviewer / explainer / docs assistant — do not edit source files unless the user explicitly asks for the edit.
- No `numpy`, `torch`, or Python `array.array` in library code. The contiguous-array logic must live in C++. (Tests may use numpy/torch as oracles.)
- Out-of-the-box algorithms in C++ and Python are avoided when the point is to learn the internals (with some exceptions, like RNG-class algorithms).

**Also:**

- The user is building a torch clone with C++ and CUDA layers, they are an advanced programmer. Do not be pedantic when there is no need to. When you notice something is off, try to guess the intent: if it's a typo, point it as such, do not write a wall of text. If it suggests that there is a more fundamental logic flaw in the user's understanding of the algorithm, please provide helpful indications.
- Adapt your verbosity to the amount of information you intend to provide, not the opposite.
- Optimize your answer for the reader, they appreciate synthetic, to-the-point answers. If deemed necessary, they will ask clarifications about specific points.
- If asked to suggest code changes, prefer minimal patches and focus on the reasoning so the user can write the code themselves.

## Common commands

```bash
uv sync                           # build the C++ extension + install dev deps
uv run pytest                     # run the full test suite
uv run pytest tests/test_autograd.py::test_name   # single test
```

The build is driven by `scikit-build-core` (see `pyproject.toml`) which invokes CMake. Re-run `uv sync` after C++ changes to rebuild `_C`.

## Architecture

Two layers, glued by pybind11:

**C++ core (`src/cpp/`)** — exposes the `nanotorch._C` extension module:
- `Storage` (`storage.h/.cpp`): owns a contiguous, dtype-tagged byte buffer. The only owner of memory.
- `TensorView` (`tensor_view.h`): non-owning view = `(shared_ptr<Storage>, shape, strides, offset)`. All ops consume `TensorView` and produce a fresh `Storage`. Views are how reshape/transpose/slicing stay zero-copy.
- `Scalar` (`storage.h`): variant<bool, int64, double> with a custom pybind11 `type_caster` so Python scalars land as `Scalar` automatically.
- `Dtype` enum + `dispatch_dtype<F>(dtype, func)`: the standard pattern for templated kernels — `ops.cpp` uses it heavily to instantiate per-dtype code paths.
- `_C.cpp` is just the pybind11 module entry — it calls `bind_storage_`, `bind_tensor_view_`, `bind_ops_`.

**Python layer (`src/nanotorch/`)** — user-facing API:
- `core.py` — `Tensor` class, the high-level wrapper around `TensorView`. Indexing/broadcasting/shape logic lives here.
- `factories.py`, `ops.py` — thin wrappers around the C++ ops.
- `autograd/` — `Function` base class with `apply()` / `forward()` / `backward()` / `save_for_backward()` (PyTorch-style). Subclasses live in `autograd/ops.py`.
- `__init__.py` — **bindings of dunder methods (`__add__`, `__mul__`, `.exp`, `.sum`, …) to autograd ops happen here at import time**, deliberately, to avoid a `core ↔ autograd` circular import. New autograd ops must be wired in this file.
- `_C.pyi` — hand-maintained stubs for the C++ module.