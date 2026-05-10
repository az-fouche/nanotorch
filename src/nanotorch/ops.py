"""Tensor operations as non-member functions."""

import math

from nanotorch import _C
from nanotorch._data_type import Dtype
from nanotorch._indexing import TensorShape
from nanotorch.core import Tensor

from . import autograd
from ._data_type import promote_dtypes
from ._indexing import broadcast_shapes
from .core import inherit_doc


# Core ops
@inherit_doc(Tensor.clone)
def clone(x: Tensor) -> Tensor:
    return x.clone()


@inherit_doc(Tensor.reshape)
def reshape(x: Tensor, *dims: int) -> Tensor:
    return x.reshape(*dims)


@inherit_doc(Tensor.transpose)
def transpose(x: Tensor, dim0: int, dim1: int) -> Tensor:
    return x.transpose(dim0, dim1)


@inherit_doc(Tensor.flatten)
def flatten(x: Tensor) -> Tensor:
    return x.flatten()


# Tensor math
@inherit_doc(Tensor.equals)
def equals(x: Tensor, y: Tensor) -> bool:
    return x.equals(y)


@inherit_doc(autograd.AddOp)
def add(x: Tensor, y: Tensor) -> Tensor:
    return x + y


@inherit_doc(autograd.SubOp)
def subtract(x: Tensor, y: Tensor) -> Tensor:
    return x - y


@inherit_doc(autograd.MulOp)
def multiply(x: Tensor, y: Tensor) -> Tensor:
    return x * y


@inherit_doc(autograd.TrueDivOp)
def divide(x: Tensor, y: Tensor) -> Tensor:
    return x / y


def eq(x: Tensor, y: Tensor) -> Tensor:
    """Computes x == y element-wise."""
    return x == y


def gt(x: Tensor, y: Tensor) -> Tensor:
    """Computes x > y element-wise."""
    return x > y


def geq(x: Tensor, y: Tensor) -> Tensor:
    """Computes x >= y element-wise."""
    return x >= y


def lt(x: Tensor, y: Tensor) -> Tensor:
    """Computes x < y element-wise."""
    return x < y


def leq(x: Tensor, y: Tensor) -> Tensor:
    """Computes x <= y element-wise."""
    return x <= y


@inherit_doc(autograd.NegOp)
def negate(x: Tensor) -> Tensor:
    return -x


@inherit_doc(autograd.MatmulOp)
def matmul(x: Tensor, y: Tensor) -> Tensor:
    return x @ y


@inherit_doc(autograd.SumOp)
def sum(
    x: Tensor,
    axis: int | TensorShape | None = None,
    keepdim: bool = False,
    *,
    dtype: Dtype | None = None,
) -> Tensor:
    return x.sum(axis, keepdim, dtype=dtype)


@inherit_doc(autograd.MeanOp)
def mean(
    x: Tensor,
    axis: int | TensorShape | None = None,
    keepdim: bool = False,
    *,
    dtype: Dtype | None = None,
) -> Tensor:
    return x.mean(axis, keepdim, dtype=dtype)


@inherit_doc(autograd.ExpOp)
def exp(x: Tensor) -> Tensor:
    return x.exp()


@inherit_doc(autograd.LogOp)
def log(x: Tensor) -> Tensor:
    return x.log()


@inherit_doc(autograd.PowOp)
def pow(x: Tensor, exponent: Tensor | float | int | bool) -> Tensor:
    return x.pow(exponent)


@inherit_doc(autograd.ReluOp)
def relu(x: Tensor) -> Tensor:
    return autograd.ReluOp.apply(x)


@inherit_doc(autograd.TanhOp)
def tanh(x: Tensor) -> Tensor:
    return autograd.TanhOp.apply(x)


@inherit_doc(autograd.SqrtOp)
def sqrt(x: Tensor) -> Tensor:
    return autograd.SqrtOp.apply(x)


@inherit_doc(autograd.SigmoidOp)
def sigmoid(x: Tensor) -> Tensor:
    return autograd.SigmoidOp.apply(x)


def argmin(x: Tensor, axis: int | None = None, keepdim: bool = False) -> Tensor:
    """Computes the index of the min value along one or more axes.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    axis: int | None
        Axes index to min along, integer is interpreted as (n,), None as all axes.
        Dimensions of these axes are collapsed by the operation.
    keepdim: bool
        Instead of collapsing target axes, keeps the axis with shape[i] = 1.

    Returns
    -------
    Tensor
        New tensor of containing the min coefficients.
    """
    if isinstance(axis, int):
        full_axis = (axis,)
    elif axis is None:
        full_axis = tuple(range(x.ndim))
    else:
        raise ValueError(f"Axis must be int or None, found {type(axis)}")
    if keepdim:
        new_shape = tuple(1 if i in full_axis else s for i, s in enumerate(x.shape))
    else:
        new_shape = tuple(s for i, s in enumerate(x.shape) if i not in full_axis)
    if x.is_empty:
        if math.prod(new_shape) != 0:
            raise IndexError(
                "argmin() expects reduction dim to be specified for numel=0."
            )
        return Tensor._new_contiguous(
            _C.zeros(math.prod(new_shape), x.dtype, x.device), new_shape
        )
    return Tensor._new_contiguous(_C.argmin(x._C_view, full_axis), new_shape)


def argmax(x: Tensor, axis: int | None = None, keepdim: bool = False) -> Tensor:
    """Computes the index of the max value along one or more axes.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    axis: int | None
        Axes index to max along, integer is interpreted as (n,), None as all axes.
        Dimensions of these axes are collapsed by the operation.
    keepdim: bool
        Instead of collapsing target axes, keeps the axis with shape[i] = 1.

    Returns
    -------
    Tensor
        New tensor of containing the max coefficients.
    """
    if isinstance(axis, int):
        full_axis = (axis,)
    elif axis is None:
        full_axis = tuple(range(x.ndim))
    else:
        raise ValueError(f"Axis must be int or None, found {type(axis)}")
    if keepdim:
        new_shape = tuple(1 if i in full_axis else s for i, s in enumerate(x.shape))
    else:
        new_shape = tuple(s for i, s in enumerate(x.shape) if i not in full_axis)
    if x.is_empty:
        if math.prod(new_shape) != 0:
            raise IndexError(
                "argmax() expects reduction dim to be specified for numel=0."
            )
        return Tensor._new_contiguous(
            _C.zeros(math.prod(new_shape), x.dtype, x.device), new_shape
        )
    return Tensor._new_contiguous(_C.argmax(x._C_view, full_axis), new_shape)


# Protected


def equal_op_(x1: Tensor, x2: Tensor) -> Tensor:
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(*shape)
    x2 = x2.to(dtype).expand(*shape)
    return Tensor._new_contiguous(_C.pw_equal(x1._C_view, x2._C_view), x1.shape)


def greater_op_(x1: Tensor, x2: Tensor) -> Tensor:
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(*shape)
    x2 = x2.to(dtype).expand(*shape)
    return Tensor._new_contiguous(_C.pw_greater(x1._C_view, x2._C_view), x1.shape)


def greater_eq_op_(x1: Tensor, x2: Tensor) -> Tensor:
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(*shape)
    x2 = x2.to(dtype).expand(*shape)
    return Tensor._new_contiguous(_C.pw_greater_eq(x1._C_view, x2._C_view), x1.shape)
