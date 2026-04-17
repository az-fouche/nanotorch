---
name: nanotorch educational scope
description: nanotorch is a from-scratch educational torch-like package — numpy and torch are forbidden as implementation dependencies
type: project
---

nanotorch is being built as an educational reimplementation of PyTorch. The user is **not allowed to use numpy or torch** as implementation dependencies — the point is to reimplement the underlying mechanisms (storage, tensor views, ops, autograd) from scratch.

**Why:** It's a learning project; using numpy/torch would defeat the purpose.

**How to apply:**
- Never suggest `np.asarray`, `np.ndarray`, `torch.tensor`, or any numpy/torch API as the implementation path.
- Use Python stdlib primitives (`array.array`, `ctypes`, `bytearray`, `struct`) for in-memory buffers, and C++ (via pybind11/nanobind) for the compute layer.
- numpy/torch currently appear in `pyproject.toml` dependencies — this is probably leftover scaffolding; flag it when relevant, but they may be kept for testing/reference comparisons.
- They *may* be fine to use in tests (e.g., compare nanotorch output against torch) — confirm with user before assuming.
