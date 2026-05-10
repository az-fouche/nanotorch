"""Tensor operations as non-member functions."""

from nanotorch import _C

from . import autograd
from ._data_type import promote_dtypes
from ._indexing import broadcast_shapes
from .core import Dtype, Tensor, TensorShape, inherit_doc


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
