"""Tensor factories."""

from nanotorch import _C

from . import _data_type as dt
from .core import DataType, InputType, Tensor, TensorShape


def tensor(data: InputType, dtype: DataType | None = None) -> Tensor:
    """Initialize a new tensor."""
    return Tensor(data, dtype)


def zeros(shape: int | TensorShape, dtype: DataType = dt.float32) -> Tensor:
    """Initialize a new tensor filled with zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.zeros(shape, dtype.cpp_dtype))


def ones(shape: int | TensorShape, dtype: DataType = dt.float32) -> Tensor:
    """Initialize a new tensor filled with ones."""
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.ones(shape, dtype.cpp_dtype))


def full(
    shape: int | TensorShape,
    value: bool | int | float,
    dtype: DataType = dt.float32,
) -> Tensor:
    """Initialize a new tensor filled with set value."""
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.full(shape, value, dtype.cpp_dtype))


def eye(n: int, dtype: DataType = dt.float32) -> Tensor:
    """Initialize a new eye square tensor."""
    return Tensor._new_contiguous(dtype, (n, n), _C.eye(n, dtype.cpp_dtype))


def arange(n: int, start: int = 0, step: int = 1, dtype: DataType = dt.int64) -> Tensor:
    """Initialize a new tensor containing an arithmetic range."""
    return Tensor._new_contiguous(
        dtype, (n,), _C.arange(n, start, step, dtype.cpp_dtype)
    )
