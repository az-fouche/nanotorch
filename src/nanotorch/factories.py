"""Usual tensor factories."""

from nanotorch import _C

from . import _data_type as dt
from .core import Dtype, InputType, Tensor, inherit_doc


@inherit_doc(Tensor)
def tensor(
    data: InputType, dtype: Dtype | None = None, requires_grad: bool = False
) -> Tensor:
    return Tensor(data, dtype, requires_grad)


def zeros(
    *shape: int, dtype: Dtype = dt.float32, requires_grad: bool = False
) -> Tensor:
    """Initialize a new tensor filled with zeros.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    dtype: Dtype
        Tensor elements data type.
    """
    if isinstance(shape, int):
        shape = (shape,)
    x = Tensor._new_contiguous(dtype, shape, _C.zeros(shape, dtype))
    if requires_grad:
        x.enable_grad()
    return x


def ones(*shape: int, dtype: Dtype = dt.float32) -> Tensor:
    """Initialize a new tensor filled with ones.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    dtype: Dtype
        Tensor elements data type.
    """
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.ones(shape, dtype))


def full(
    *shape: int,
    value: bool | int | float,
    dtype: Dtype = dt.float32,
) -> Tensor:
    """Initialize a new tensor filled with a set value.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    value: bool | int | float
        Value to fill the tensor with.
    dtype: Dtype
        Tensor elements data type.
    """
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._new_contiguous(dtype, shape, _C.full(shape, value, dtype))


def eye(n: int, dtype: Dtype = dt.float32) -> Tensor:
    """Initialize a new eye matrix of rank n.

    Parameters
    ----------
    n: int
        Matrix rank, output will be (n, n).
    dtype: Dtype
        Tensor elements data type.
    """
    return Tensor._new_contiguous(dtype, (n, n), _C.eye(n, dtype))


def arange(n: int, start: int = 0, step: int = 1, dtype: Dtype = dt.int64) -> Tensor:
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
    dtype: Dtype
        Tensor elements data type.
    """
    return Tensor._new_contiguous(dtype, (n,), _C.arange(n, start, step, dtype))


def rand(*shape: int, dtype: Dtype = dt.float32, requires_grad: bool = False) -> Tensor:
    """Initialize a new tensor initialized with random noise.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    dtype: Dtype
        Tensor elements data type.
    requires_grad: bool
        If set to True, this tensor will receive gradients.
    """
    from numpy.random import rand

    return Tensor(rand(*shape), dtype=dtype, requires_grad=requires_grad)  # type: ignore
