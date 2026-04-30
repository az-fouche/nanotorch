"""Tensor operations as non-member functions."""

from .core import DataType, Tensor, TensorShape, inherit_doc


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


@inherit_doc(Tensor.__add__)
def add(x: Tensor, y: Tensor) -> Tensor:
    return x + y


@inherit_doc(Tensor.__sub__)
def subtract(x: Tensor, y: Tensor) -> Tensor:
    return x - y


@inherit_doc(Tensor.__mul__)
def multiply(x: Tensor, y: Tensor) -> Tensor:
    return x * y


@inherit_doc(Tensor.__truediv__)
def divide(x: Tensor, y: Tensor) -> Tensor:
    return x / y


@inherit_doc(Tensor.__matmul__)
def matmul(x: Tensor, y: Tensor) -> Tensor:
    return x @ y


@inherit_doc(Tensor.sum)
def sum(
    x: Tensor,
    axis: int | TensorShape | None = None,
    keepdim: bool = False,
    *,
    dtype: DataType | None = None,
) -> Tensor:
    return x.sum(axis, keepdim, dtype=dtype)


@inherit_doc(Tensor.mean)
def mean(
    x: Tensor,
    axis: int | TensorShape | None = None,
    keepdim: bool = False,
    *,
    dtype: DataType | None = None,
) -> Tensor:
    return x.mean(axis, keepdim, dtype=dtype)


@inherit_doc(Tensor.exp)
def exp(x: Tensor) -> Tensor:
    return x.exp()


@inherit_doc(Tensor.log)
def log(x: Tensor) -> Tensor:
    return x.log()


@inherit_doc(Tensor.pow)
def pow(x: Tensor, exponent: Tensor | float | int | bool) -> Tensor:
    return x.pow(exponent)
