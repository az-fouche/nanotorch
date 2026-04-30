"""Usual tensor factories."""

from nanotorch import _C

from . import _data_type as dt
from .core import DataType, InputType, Tensor, TensorShape, inherit_doc


@inherit_doc(Tensor)
def tensor(
    data: InputType, dtype: DataType | None = None, requires_grad: bool = False
) -> Tensor:
    return Tensor(data, dtype, requires_grad)


def zeros(*shape: int, dtype: DataType = dt.float32) -> Tensor:
    """Initialize a new tensor filled with zeros.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        Tensor shape, single integer will be interpreted as (n,).
    dtype: DataType
        Tensor elements data type.
    """
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.zeros(shape, dtype.cpp_dtype))


def ones(*shape: int, dtype: DataType = dt.float32) -> Tensor:
    """Initialize a new tensor filled with ones.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        Tensor shape, single integer will be interpreted as (n,).
    dtype: DataType
        Tensor elements data type.
    """
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.ones(shape, dtype.cpp_dtype))


def full(
    *shape: int,
    value: bool | int | float,
    dtype: DataType = dt.float32,
) -> Tensor:
    """Initialize a new tensor filled with a set value.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        Tensor shape, single integer will be interpreted as (n,).
    value: bool | int | float
        Value to fill the tensor with.
    dtype: DataType
        Tensor elements data type.
    """
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.full(shape, value, dtype.cpp_dtype))


def eye(n: int, dtype: DataType = dt.float32) -> Tensor:
    """Initialize a new eye matrix of rank n.

    Parameters
    ----------
    n: int
        Matrix rank, output will be (n, n).
    dtype: DataType
        Tensor elements data type.
    """
    return Tensor._new_contiguous(dtype, (n, n), _C.eye(n, dtype.cpp_dtype))


def arange(n: int, start: int = 0, step: int = 1, dtype: DataType = dt.int64) -> Tensor:
    """Initialize a new arithmetic range.

    If `x = arange(n, a0, r)`, `len(x) = n` and `x[i] = a0 + i * r`.

    Parameters
    ----------
    n: int
        Number of elements in the range.
    start: int
        First element in the range.
    step: int
        Increment of the range.
    dtype: DataType
        Tensor elements data type.
    """
    return Tensor._new_contiguous(
        dtype, (n,), _C.arange(n, start, step, dtype.cpp_dtype)
    )
