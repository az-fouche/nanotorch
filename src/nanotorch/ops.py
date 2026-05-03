"""Tensor operations as non-member functions."""

from . import autograd
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
